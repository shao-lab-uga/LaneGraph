
import math
from collections import defaultdict
from re import T
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import networkx as nx
from shapely.geometry import LineString, Point
from sklearn.cluster import DBSCAN
from sympy import im
from utils.graph_postprocessing_utils import point_distance

# ============================================================
# 0) Small geometry helpers
# ============================================================
def _angdiff(a: float, b: float) -> float:
    d = (a - b + math.pi) % (2*math.pi) - math.pi
    return abs(d)

def _heading_along(geom: LineString, center: Point, approaching_junc: bool, d: float=2.0) -> Optional[float]:
    """Heading (rad) near the junction side; toward_end=True means toward the line END (approach),
       toward_end=False means away from the line START (departure)."""
    L = float(geom.length)
    if L <= 1e-6:
        return None
    flip = False
    if approaching_junc:
        if point_distance(geom.coords[0], center.coords[0]) < point_distance(geom.coords[-1], center.coords[0]):
            flip = True  # reverse direction if start is closer to center
    elif not approaching_junc:
        if point_distance(geom.coords[-1], center.coords[0]) < point_distance(geom.coords[0], center.coords[0]):
            flip = True  # reverse direction if end is closer to center
    if not flip:
        s0, s1 = max(0.0, L-d), L
        a = np.asarray(geom.interpolate(s0).coords[0], float)
        b = np.asarray(geom.interpolate(s1).coords[0], float)
    else:
        s1, s0 = min(d, L), 0.0
        a = np.asarray(geom.interpolate(s1).coords[0], float)
        b = np.asarray(geom.interpolate(s0).coords[0], float)
    v = b - a
    if np.allclose(v, 0.0):
        return None
    return math.atan2(v[1], v[0])



def cluster_intersections_by_roads(
    lane_graph: nx.DiGraph,
    lanes_gdf,                       # pandas.DataFrame with at least ['fid','road_id']
    *,
    eps: float = 12.0,               # spatial radius (your units)
    min_samples: int = 1,            # DBSCAN core size (1 is fine here)
    min_roads: int = 2,              # REQUIRE at least this many DISTINCT roads per cluster
    merge_tol: float = 10.0          # merge nearby clusters (center-to-center) if they are the same intersection
):
    """
    Cluster ONLY 'in'/'out' nodes; keep a cluster as an intersection iff it contains nodes
    from >= min_roads DISTINCT road_ids (using node->fid->road_id). Assign contiguous junc_id
    only to nodes in KEPT clusters. Returns {junc_id: Point} medoid centers.
    """
    # Node attrs
    pos = nx.get_node_attributes(lane_graph, 'pos')
    typ = nx.get_node_attributes(lane_graph, 'type')
    fid = nx.get_node_attributes(lane_graph, 'fid')

    # Map fid -> road_id
    fid2road = dict(zip(lanes_gdf['fid'].tolist(), lanes_gdf['road_id'].tolist()))

    # Collect only 'in'/'out' nodes with positions and known road_ids
    io_nodes, XY, roads = [], [], []
    for n, t in typ.items():
        if t in ('in', 'out') and (n in pos):
            r = fid2road.get(fid.get(n))
            if r is None:
                continue
            io_nodes.append(n)
            XY.append(pos[n])
            roads.append(r)
    if not io_nodes:
        return {}

    XY = np.asarray(XY, float)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(XY)

    # Build raw clusters: nodes, road sets
    clusters = {}
    for node, lab, xy, rd in zip(io_nodes, labels, XY, roads):
        lab = int(lab)
        if lab not in clusters:
            clusters[lab] = {'nodes': [], 'xy': [], 'roads': set()}
        clusters[lab]['nodes'].append(node)
        clusters[lab]['xy'].append(xy)
        clusters[lab]['roads'].add(rd)

    # Compute medoid per raw cluster
    for c in clusters.values():
        P = np.vstack(c['xy'])
        D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2).sum(axis=1)
        c['center'] = P[int(np.argmin(D))]

    # (Optional) merge raw clusters that are actually the same intersection (close centers)
    # Union-Find style merge by proximity
    raw_ids = list(clusters.keys())
    parent = {i: i for i in raw_ids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(raw_ids)):
        for j in range(i+1, len(raw_ids)):
            ci, cj = clusters[raw_ids[i]], clusters[raw_ids[j]]
            if np.linalg.norm(ci['center'] - cj['center']) <= merge_tol:
                union(raw_ids[i], raw_ids[j])

    merged = {}
    for k in raw_ids:
        r = find(k)
        if r not in merged:
            merged[r] = {'nodes': [], 'xy': [], 'roads': set()}
        merged[r]['nodes'].extend(clusters[k]['nodes'])
        merged[r]['xy'].extend(clusters[k]['xy'])
        merged[r]['roads'] |= clusters[k]['roads']

    # Recompute centers after merge
    for m in merged.values():
        P = np.vstack(m['xy'])
        D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2).sum(axis=1)
        m['center'] = P[int(np.argmin(D))]

    # Filter: keep only clusters with >= min_roads distinct road_ids
    kept = [m for m in merged.values() if len(m['roads']) >= min_roads]

    # Assign contiguous junc_id to KEPT nodes; clear any previous junc_id
    for n in lane_graph.nodes:
        if 'junc_id' in lane_graph.nodes[n]:
            del lane_graph.nodes[n]['junc_id']

    centers = {}
    for new_id, c in enumerate(kept):
        for n in c['nodes']:
            lane_graph.nodes[n]['junc_id'] = int(new_id)
        centers[int(new_id)] = Point(float(c['center'][0]), float(c['center'][1]))

    return centers

# ============================================================
# 3) Attach lanes to junctions and build directed legs
# ============================================================
def _lane_endpoints(row: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    geom: LineString = row.geometry
    return np.asarray(geom.coords[0], float), np.asarray(geom.coords[-1], float)

def build_directed_legs_per_junction(
    lanes_gdf: pd.DataFrame,
    lane_graph: nx.DiGraph,
    junction_centers: Dict[int, Point],
    *,
    attach_tol: float=15.0,
):
    """
    Create directed legs per junc_id:
      - approaches[(road_id, lane_dir, j)] : lanes ending at j (end_type=='in'), ordered driver-left→right
      - departures[(road_id, lane_dir, j)]: lanes starting at j (start_type=='out'), ordered driver-left→right
    Also compute an average heading (toward end for approaches; from start for departures).
    """
    pos = nx.get_node_attributes(lane_graph, 'pos')
    typ = nx.get_node_attributes(lane_graph, 'type')
    jid = nx.get_node_attributes(lane_graph, 'junc_id')

    in_nodes  = [(n, np.asarray(pos[n], float), jid[n]) for n in lane_graph.nodes if typ.get(n)=='in'  and n in jid and n in pos]
    out_nodes = [(n, np.asarray(pos[n], float), jid[n]) for n in lane_graph.nodes if typ.get(n)=='out' and n in jid and n in pos]

    def _nearest_j(pt: np.ndarray, lst):
        if not lst:
            return None, None
        X = np.vstack([xy for _, xy, _ in lst])
        d = np.linalg.norm(X - pt[None, :], axis=1)
        k = int(np.argmin(d))
        return lst[k][2], float(d[k])  # (junc_id, distance)

    approaches_raw, departures_raw = defaultdict(list), defaultdict(list)
    approaches_raw_key_set, departures_raw_key_set = set(), set()
    for _, r in lanes_gdf.iterrows():
        rid, ldir = r['road_id'], int(r['lane_dir'])
        node_pos_start, node_pos_end = _lane_endpoints(r)
        junction_id_start, junction_dist_start = _nearest_j(node_pos_start, in_nodes)
        junction_id_end,   junction_dist_end   = _nearest_j(node_pos_end, out_nodes)
        # choose the closer junction node between start and end
        ## if the end node is closer to the junction then its an approach
        ## if the start node is closer to the junction then its a departure
        if junction_dist_end is not None and (junction_dist_start is None or junction_dist_end
            <= junction_dist_start) and junction_dist_end <= attach_tol:
            j = junction_id_end
            approaches_raw[(rid, ldir, j)].append(r)
            approaches_raw_key_set.add((rid, ldir, j))
        elif junction_dist_start is not None and (junction_dist_end is None or junction_dist_start
            < junction_dist_end) and junction_dist_start <= attach_tol:
            j = junction_id_start
            
            departures_raw[(rid, ldir, j)].append(r)
            departures_raw_key_set.add((rid, ldir, j))

    def _pack(buckets, approaching_junc: bool):
        legs = {}
        for key, rows in buckets.items():
            road_id, lane_dir, junction_id = key
            df = pd.DataFrame(rows).reset_index(drop=True)
            # average heading at the junction side
            hs = [h for h in (_heading_along(g, center=junction_centers[junction_id], approaching_junc=approaching_junc) for g in df['geometry']) if h is not None]
            H  = math.atan2(np.mean(np.sin(hs)), np.mean(np.cos(hs))) if hs else None
            df['avg_offset'] = df['avg_offset'].abs()  # use absolute offsets for ordering
            df = df.sort_values('avg_offset', ascending=True).reset_index(drop=True)
            fids = df['fid'].tolist()
            # order lanes by offset (driver-left→right)
            legs[key] = {'heading': H, 'fids': fids, 'df': df}
        return legs

    legs_in  = _pack(approaches_raw, approaching_junc=True)   # approaching junc
    legs_out = _pack(departures_raw, approaching_junc=False)  # leaving junc
    return legs_in, legs_out


# ============================================================
# 4) Pick receiving legs by angle (within same junction)
# ============================================================
def pick_receiving_legs(
    legs_in: Dict[Tuple[str,int,int], dict],
    legs_out: Dict[Tuple[str,int,int], dict],
    key_in: Tuple[str,int,int],
    *,
    angle_tol_deg: float=40.0
):
    """For approach leg with heading H_in, choose receiving departures whose heading is near:
       Through: H_in ; Left: H_in + π/2 ; Right: H_in - π/2.
       Only consider departures from the SAME junction id.
    """
    H = legs_in[key_in]['heading']
    if H is None:
        return None, None, None
    (rid, ldir, j) = key_in
    # TODO: Due to the image coordinate system, we might need to SWAP left and right here
    tgt = {'T': H, 'L': H - math.pi/2.0, 'R': H + math.pi/2.0}
    th  = math.radians(angle_tol_deg)

    best, err = {'T':None,'L':None,'R':None}, {'T':1e9,'L':1e9,'R':1e9}
    for key_out, info in legs_out.items():
        if key_out[2] != j:
            continue  # must be same junction
        if key_out[0] == rid:
            continue  # skip same road_id
        Hout = info['heading']
        if Hout is None:
            continue
        for mv in ('T','L','R'):
            e = _angdiff(Hout, tgt[mv])
            if e < err[mv] and e <= th:
                err[mv] = e
                best[mv] = key_out
    return best['L'], best['T'], best['R']


# ============================================================
# 5) Receive-aware template (by capacities)
# ============================================================
def design_receive_template(n: int, capL: int, capT: int, capR: int) -> List[set]:
    """
    Return per-lane allowed sets (driver-left→right) under common defaults,
    trimmed to the given receiving capacities.
    """
    uses = [set() for _ in range(n)]
    if n == 0:
        return uses

    if n == 1:
        if capL: uses[0].add('L')
        if capT: uses[0].add('T')
        if capR: uses[0].add('R')
        if not uses[0]: uses[0].add('T')
        return uses

    if n == 2:
        uses[0] = {'L'} if capL else {'T'}
        uses[1] = {'T'} | ({'R'} if capR else set())
        # trim through to capacity
        while sum('T' in s for s in uses) > capT:
            if 'T' in uses[1] and len(uses[1])>1: uses[1].remove('T')
            elif 'T' in uses[0]: uses[0].discard('T'); break
        return uses
    if n == 3:
        uses[0] = {'L'} if capL else {'T'}
        uses[1] = {'T'} | ({'L'} if capL else set())
        uses[2] = {'T'} | ({'R'} if capR else set())
        # trim through to capacity
        while sum('T' in s for s in uses) > capT:
            for i in [2,1,0]:
                if 'T' in uses[i] and len(uses[i])>1:
                    uses[i].remove('T'); break
    else:
        # n >= 3
        uses[0]  = {'L'} if capL else {'T'}
        uses[1]  = {'T'} if capT else ({'L'} if capL else set())
        uses[-1] = {'R'} if capR else ({'T'} if capT else set())

    # # common tight-through case on 3-lane approaches
    # if n == 3 and capT == 1:
    #     if capL:
    #         uses[1] |= {'L','T'}   # middle becomes L/T
    #     if capR:
    #         uses[-1] |= {'T','R'}  # curb becomes T/R
    #     # trim T back to capacity=1 (drop shared T first)
    #     while sum('T' in s for s in uses) > capT:
    #         for i in [0,1,2]:
    #             if 'T' in uses[i] and len(uses[i])>1:
    #                 uses[i].remove('T'); break
    #         else:
    #             for i in [0,1,2]:
    #                 if 'T' in uses[i]:
    #                     uses[i].remove('T'); break

    for i in range(n):
        if not uses[i]:
            uses[i].add('T')
    return uses

# ============================================================
# 6) Main filter: receive-aware, order-preserving, per junction
# ============================================================
def filter_connections_receive_aware(
    connections: Dict[Tuple[int,int], Dict],
    lane_graph: nx.DiGraph,
    lanes_gdf: pd.DataFrame,
    *,
    junction_eps: float=150.0,
    attach_tol: float=100.0,
    angle_tol_deg: float=40.0,
    offset_is_driver_signed: bool=True
) -> Dict[Tuple[int,int], Dict]:
    """
    Keeps only connections that:
      - lie within the same junction (clustered over 'in'/'out' nodes),
      - go from an approach (end_type=='in') to a departure (start_type=='out'),
      - match a receive-aware template using per-movement receiving capacities,
      - are assigned order-preservingly (no intra-box weaving).
    """
    # 1) road-aware intersections (only 'in'/'out' nodes; require >=2 distinct roads)

    centers = cluster_intersections_by_roads(
        lane_graph,
        lanes_gdf,
        eps=junction_eps,         # keep your tuning
        min_samples=1,
        min_roads=2,
        merge_tol=8.0             # tweak if intersections split/merge oddly
    )

    node_jid = nx.get_node_attributes(lane_graph, 'junc_id')
    node_fid = nx.get_node_attributes(lane_graph, 'fid')

    # 2) build directed legs per junction (ordered by driver-left→right)
    legs_in, legs_out = build_directed_legs_per_junction(
        lanes_gdf, lane_graph, centers,
        attach_tol=attach_tol,
    )

    # 3) bucket candidates by (junction, approach directed leg, movement)
    cand = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(set)
                )
            )
        )
    for (u, v), meta in connections.items():
        j_u, j_v = node_jid.get(u), node_jid.get(v)
        if j_u is None or j_v is None or j_u != j_v:
            continue
        fu, fv = node_fid.get(u), node_fid.get(v)
        print(f"Connection {u}->{v} with fids {fu}->{fv} at junc {j_u}")
        if fu is None or fv is None:
            continue

        ru = lanes_gdf.loc[lanes_gdf['fid'] == fu]
        rv = lanes_gdf.loc[lanes_gdf['fid'] == fv]
        if ru.empty or rv.empty:
            continue

        # enforce directed semantics
        if not (str(ru.iloc[0]['end_type'])=='out' and str(rv.iloc[0]['start_type'])=='in'):
            continue

        key_in = (ru.iloc[0]['road_id'], int(ru.iloc[0]['lane_dir']), j_u)
        if key_in not in legs_in:
            continue

        mv = meta['connection_type']
        mv = 'L' if mv=='left_turn' else 'R' if mv=='right_turn' else 'T'
        cand[j_u][key_in][mv][fu].add(fv)

    # 4) per approach-leg acceptance using receive-aware template
    accepted = {}
    for j, approaches in cand.items():
        for key_in, by_mv in approaches.items():
            # receiving legs by angle inside this junction
            L_leg, T_leg, R_leg = pick_receiving_legs(legs_in, legs_out, key_in, angle_tol_deg=angle_tol_deg)

            # source order (driver-left→right)
            src_order = legs_in[key_in]['fids']
            n = len(src_order)

            # receiving lane orders (driver-left→right); for R use curb→inward (reverse)
            def _recv_fids(key_out, mv):
                if key_out is None:
                    return []
                fids = legs_out[key_out]['fids']
                return fids

            dstL = _recv_fids(L_leg, 'L'); capL = len(dstL)
            dstT = _recv_fids(T_leg, 'T'); capT = len(dstT)
            dstR = _recv_fids(R_leg, 'R'); capR = len(dstR)

            # build receive-aware template for this approach
            uses = design_receive_template(n, capL, capT, capR)

            # sources per movement
            src_L = [src_order[i] for i in range(n) if 'L' in uses[i]]
            src_T = [src_order[i] for i in range(n) if 'T' in uses[i]]
            src_R = [src_order[i] for i in range(n) if 'R' in uses[i]]

            # order-preserving assignments
            def _assign(src, dst):
                k = min(len(src), len(dst))
                return list(zip(src[:k], dst[:k]))

            pairs = []
            pairs += _assign(src_L, dstL)                 # left: inner→inner
            pairs += _assign(src_T, dstT)                 # through: i→i
            pairs += _assign(list(reversed(src_R)), list(reversed(dstR)))  # right: outside→outside

            allowed = set(pairs)
            for fid_u, fid_v in allowed:
                for (u, v), meta in connections.items():
                    if node_fid.get(u)==fid_u and node_fid.get(v)==fid_v and node_jid.get(u)==j and node_jid.get(v)==j:
                        accepted[(u, v)] = connections[(u, v)]
                        break
                print(f"Warning: missing connection {fid_u}->{fid_v} at junc {j}")
    return accepted
