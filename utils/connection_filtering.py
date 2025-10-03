
import math
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict, Counter
from sklearn.cluster import DBSCAN, KMeans
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from typing import Dict, List, Tuple, Optional
from utils.graph_postprocessing_utils import point_distance



def _unit(v):
    v = np.asarray(v, float); n = np.linalg.norm(v)
    return (v / n) if n > 1e-9 else None

def _geom_median(P, iters: int = 100, tol: float = 1e-6):
    x = P.mean(axis=0)
    for _ in range(iters):
        d = np.linalg.norm(P - x, axis=1)
        w = 1.0 / np.maximum(d, 1e-8)
        x_new = (P * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(x_new - x) < tol: break
        x = x_new
    return x

def _fit_hub_point(points: np.ndarray, dirs):
    """
    Least-squares point minimizing sum of squared perpendicular distances
    to lines (x_i, v_i). Returns (p*, used) where used=#valid dirs.
    """
    I = np.eye(2)
    A = np.zeros((2, 2)); b = np.zeros(2); used = 0
    for x, v in zip(points, dirs):
        v = _unit(v)
        if v is None: continue
        P = I - np.outer(v, v)   # projector onto normal space of v
        A += P; b += P @ x; used += 1
    if used >= 2 and np.linalg.cond(A) < 1e8:
        p = np.linalg.solve(A, b)
        return p, used
    return points.mean(axis=0), 0

def _project_point_to_line(x, v, p):
    """Return the closest point to p on line through x with direction v."""
    v = _unit(v)
    if v is None: return x
    t = np.dot(p - x, v)
    return x + t * v

def _center_from_lines(points: np.ndarray, dirs):
    """
    1) Fit hub by least squares
    2) Project hub to each supporting line
    3) Geometric median of projections (robust 'on-road' snap)
    """
    p_ls, used = _fit_hub_point(points, dirs)
    if used < 2:
        return _geom_median(points)
    projs = np.vstack([_project_point_to_line(x, v, p_ls) for x, v in zip(points, dirs)])
    return _geom_median(projs)

def _estimate_dir_from_graph(G, n, pos):
    """Average unit vectors to successors for 'out', predecessors for 'in'."""
    t = G.nodes[n].get('type')
    vecs = []
    if t == 'in':
        for p in G.predecessors(n):
            if p in pos:
                u = _unit(np.asarray(pos[n], float) - np.asarray(pos[p], float))
                if u is not None: vecs.append(u)
    else:
        for s in G.successors(n):
            if s in pos:
                u = _unit(np.asarray(pos[s], float) - np.asarray(pos[n], float))
                if u is not None: vecs.append(u)
    if not vecs: return None
    return _unit(np.mean(vecs, axis=0))

def _tangent_from_geom(geom: BaseGeometry, near_xy: np.ndarray):
    """Tangent from lane geometry near the endpoint (if you have geometry)."""
    try:
        s = geom.project(Point(float(near_xy[0]), float(near_xy[1])))
        ds = max(geom.length * 1e-4, 0.5)
        p1 = np.array(list(geom.interpolate(max(0.0, s-ds)).coords)[0])
        p2 = np.array(list(geom.interpolate(min(geom.length, s+ds)).coords)[0])
        return _unit(p2 - p1)
    except Exception:
        return None

def _geom_median(P: np.ndarray, iters: int = 100, tol: float = 1e-6) -> np.ndarray:
    x = P.mean(axis=0)
    for _ in range(iters):
        d = np.linalg.norm(P - x, axis=1)
        w = 1.0 / np.maximum(d, 1e-8)
        x_new = (P * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x

def _diameter(P: np.ndarray) -> float:
    if len(P) <= 1: return 0.0
    c = P.mean(axis=0)
    return float(np.linalg.norm(P - c, axis=1).max() * 2.0)

def _split_until_small(
    XY: np.ndarray, idxs: List[int],
    *, max_diam: float, max_endpoints: int, min_sz: int = 4
) -> List[List[int]]:
    """
    Recursively split (k=2) while cluster is too big in size or span.
    Accept split only if both children are meaningfully tighter or reduce size.
    """
    P = XY[idxs]
    D = _diameter(P)
    if (len(idxs) <= max_endpoints and D <= max_diam) or len(idxs) < 2*min_sz:
        return [idxs]

    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(P)
    lab = km.labels_
    g0 = [idxs[i] for i in range(len(P)) if lab[i] == 0]
    g1 = [idxs[i] for i in range(len(P)) if lab[i] == 1]
    if len(g0) < min_sz or len(g1) < min_sz:
        return [idxs]

    P0, P1 = P[lab == 0], P[lab == 1]
    D0, D1 = _diameter(P0), _diameter(P1)

    # accept if we either reduce oversize OR shrink diameter well
    size_ok = (len(g0) <= max_endpoints and len(g1) <= max_endpoints)
    diam_ok = max(D0, D1) <= 0.9 * D
    if not (size_ok or diam_ok):
        return [idxs]

    out = []
    out.extend(_split_until_small(XY, g0, max_diam=max_diam, max_endpoints=max_endpoints, min_sz=min_sz))
    out.extend(_split_until_small(XY, g1, max_diam=max_diam, max_endpoints=max_endpoints, min_sz=min_sz))
    return out

def cluster_intersections_by_roads(
    lane_graph: nx.DiGraph,
    lanes_gdf,                       # DataFrame with ['fid','road_id']
    *,
    eps: float = 220.0,              # your existing scale
    min_samples: int = 4,
    max_diam: float = 450.0,         # cap spatial spread (adjust to your map units)
    max_endpoints: int = 20,         # NEW: hard cap on endpoints per junction (your idea)
    min_roads: int = 2,
    require_io_mix: bool = True,
    per_road_cap: int = 8            # optional: cap endpoints contributed by any single road
) -> Dict[int, Point]:

    pos = nx.get_node_attributes(lane_graph, 'pos')
    typ = nx.get_node_attributes(lane_graph, 'type')
    fid = nx.get_node_attributes(lane_graph, 'fid')

    fid2road = dict(zip(lanes_gdf['fid'].tolist(), lanes_gdf['road_id'].tolist()))

    nodes, XY, roads, io_types = [], [], [], []
    for n, t in typ.items():
        if t in ('in', 'out') and (n in pos):
            r = fid2road.get(fid.get(n))
            if r is None:
                continue
            nodes.append(n)
            XY.append(np.asarray(pos[n], float))
            roads.append(r)
            io_types.append(t)

    if not nodes:
        return {}

    XY = np.asarray(XY, float)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(XY)

    # initial groups (ignore noise = -1)
    groups = {}
    for i, lab in enumerate(labels):
        if lab == -1:
            continue
        groups.setdefault(int(lab), []).append(i)

    # auto-split oversized groups by size/diameter
    final_groups = []
    for _, idxs in groups.items():
        final_groups.extend(_split_until_small(XY, idxs, max_diam=max_diam, max_endpoints=max_endpoints, min_sz=4))

    # structural filters + optional per-road cap
    for n in lane_graph.nodes:
        lane_graph.nodes[n].pop('junc_id', None)

    kept = []
    for g in final_groups:
        # enforce per-road cap to avoid one corridor dominating a plaza
        if per_road_cap is not None and per_road_cap > 0:
            cnt = Counter(roads[i] for i in g)
            # if any road contributes too many endpoints, softly trim by nearest to group center later
            if any(v > per_road_cap for v in cnt.values()):
                # compute provisional center for trimming
                c0 = _geom_median(XY[g])
                d = {i: np.linalg.norm(XY[i] - c0) for i in g}
                g_sorted = sorted(g, key=lambda i: d[i])
                # keep up to per_road_cap per road, closest first
                seen = Counter()
                trimmed = []
                for i in g_sorted:
                    rd = roads[i]
                    if seen[rd] < per_road_cap:
                        trimmed.append(i); seen[rd] += 1
                g = trimmed

        rset = set(roads[i] for i in g)
        if len(rset) < min_roads:
            continue
        if require_io_mix:
            tlist = [io_types[i] for i in g]
            if not (('in' in tlist) and ('out' in tlist)):
                continue
        kept.append(g)

    centers = {}
    for junc_id, g in enumerate(kept):
        pos = nx.get_node_attributes(lane_graph, 'pos')
        fid = nx.get_node_attributes(lane_graph, 'fid')

        # optional: if you have lanes_gdf.geometry
        fid2geom = {}
        if 'geometry' in lanes_gdf.columns:
            fid2geom = dict(zip(lanes_gdf['fid'].tolist(), lanes_gdf['geometry'].tolist()))

        pts = XY[g]
        dirs = []
        for i in g:
            n = nodes[i]
            d = None
            # try lane geometry tangent first (more stable), then graph neighbors
            if fid2geom:
                f = fid.get(n, None)
                geom = fid2geom.get(f, None)
                if geom is not None:
                    d = _tangent_from_geom(geom, np.asarray(pos[n], float))
            if d is None:
                d = _estimate_dir_from_graph(lane_graph, n, pos)
            dirs.append(d)

        c = _center_from_lines(pts, dirs)
        for i in g:
            lane_graph.nodes[nodes[i]]['junc_id'] = int(junc_id)
        centers[int(junc_id)] = Point(float(c[0]), float(c[1]))
    return centers


def _angdiff(a: float, b: float) -> float:
    d = (a - b + math.pi) % (2*math.pi) - math.pi
    return abs(d)

def _heading_along(geom: LineString, center: Point, approaching_junc: bool, d: float = 2.0) -> Optional[float]:
    """
    Heading (rad) at the lane's segment near the junction side.
    - If approaching_junc=True: use the segment that points TOWARD the junction end.
    - If approaching_junc=False: use the segment that points AWAY from the junction end.
    """
    L = float(geom.length)
    if L <= 1e-6:
        return None

    p0 = np.asarray(geom.coords[0], float)
    p1 = np.asarray(geom.coords[-1], float)
    jc = np.asarray(center.coords[0], float)

    # Which end is nearer to the junction center?
    start_is_junc_end = np.linalg.norm(p0 - jc) <= np.linalg.norm(p1 - jc)

    if approaching_junc:
        # we want the vector that heads into the junction end
        if start_is_junc_end:
            s0, s1 = min(d, L), 0.0      # segment pointing toward start
        else:
            s0, s1 = max(0.0, L - d), L  # segment pointing toward end
    else:
        # departing: vector pointing away from the junction end
        if start_is_junc_end:
            s0, s1 = 0.0, min(d, L)      # away from start
        else:
            s0, s1 = L, max(0.0, L - d)  # away from end

    a = np.asarray(geom.interpolate(s0).coords[0], float)
    b = np.asarray(geom.interpolate(s1).coords[0], float)
    v = b - a
    if np.allclose(v, 0.0):
        return None
    return math.atan2(v[1], v[0])
# ============================================================
# 3) Attach lanes to junctions and build directed legs
# ============================================================

def build_directed_legs_per_junction(
    lanes_gdf: pd.DataFrame,
    reference_lines: Dict[int, LineString],
    junction_centers: Dict[int, Point],
    *,
    attach_tol: float = 200.0,
):
    """
    Create directed legs per junction. Each reference line end (start/end)
    can attach to a DIFFERENT junction.
    """
    def _nearest_j(pt: np.ndarray, junction_centers: Dict[int, Point]):
        if not junction_centers:
            return None, None
        d = {k: point_distance(pt, np.asarray(c.coords[0], float)) for k, c in junction_centers.items()}
        k = min(d, key=d.get)
        return k, d[k]

    approaches_raw, departures_raw = defaultdict(list), defaultdict(list)

    for road_id, reference_line in reference_lines.items():
        ref_start_pos = np.asarray(reference_line.coords[0], float)
        ref_end_pos   = np.asarray(reference_line.coords[-1], float)

        j_start, d_start = _nearest_j(ref_start_pos, junction_centers)
        j_end,   d_end   = _nearest_j(ref_end_pos,   junction_centers)

        # nothing close to any junction → skip this road_id
        if ((d_start is None or d_start > attach_tol) and
            (d_end   is None or d_end   > attach_tol)):
            continue

        # pull lanes for this road once
        lanes_fwd = lanes_gdf.loc[(lanes_gdf['road_id'] == road_id) & (lanes_gdf['lane_dir'] == 1)]
        lanes_rev = lanes_gdf.loc[(lanes_gdf['road_id'] == road_id) & (lanes_gdf['lane_dir'] == -1)]

        # Attach START end (if close enough)
        if (d_start is not None) and (d_start <= attach_tol):
            # At the START junction:
            #   lane_dir=+1: DEPARTURE from start (moving away from start)
            #   lane_dir=-1: APPROACH to start (moving toward start)
            for _, r in lanes_fwd.iterrows():
                departures_raw[(road_id,  1, j_start)].append(r)
            for _, r in lanes_rev.iterrows():
                approaches_raw[(road_id, -1, j_start)].append(r)

        # Attach END end (if close enough)
        if (d_end is not None) and (d_end <= attach_tol):
            # At the END junction:
            #   lane_dir=+1: APPROACH to end (moving toward end)
            #   lane_dir=-1: DEPARTURE from end (moving away from end)
            for _, r in lanes_fwd.iterrows():
                approaches_raw[(road_id,  1, j_end)].append(r)
            for _, r in lanes_rev.iterrows():
                departures_raw[(road_id, -1, j_end)].append(r)

        # Note: if both ends connect to the SAME junction, both blocks will run;
        # that’s correct: same lanes serve as approaches from one side and
        # departures from the other w.r.t. the same junc_id.

    def _pack(buckets, approaching_junc: bool):
        legs = {}
        for key, rows in buckets.items():
            road_id, lane_dir, junction_id = key
            df = pd.DataFrame(rows).reset_index(drop=True)

            # average heading near the junction side
            hs = [
                h for h in (
                    _heading_along(g, center=junction_centers[junction_id], approaching_junc=approaching_junc)
                    for g in df['geometry']
                ) if h is not None
            ]
            H = math.atan2(np.mean(np.sin(hs)), np.mean(np.cos(hs))) if hs else None

            # order lanes by absolute lateral offset (driver-left → right if you defined it that way)
            df['avg_offset'] = df['avg_offset'].abs()
            df = df.sort_values('avg_offset', ascending=True).reset_index(drop=True)

            legs[key] = {'heading': H, 'fids': df['fid'].tolist(), 'df': df}
        return legs

    legs_in  = _pack(approaches_raw, approaching_junc=True)    # approaches at each junc
    legs_out = _pack(departures_raw, approaching_junc=False)   # departures at each junc
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
        
        if 'T' in uses[1] and len(uses[1])>1: 
            uses[1].remove('T')
        elif 'T' in uses[0]:
            uses[0].discard('T')
        return uses
    if n == 3:
        uses[0] = {'L'} if capL else {'T'}
        uses[1] = {'T'} | ({'L'} if capL else set())
        uses[2] = {'T'} | ({'R'} if capR else set())
        # trim through to capacity
        
        for i in [2,1,0]:
            if 'T' in uses[i] and len(uses[i])>1:
                uses[i].remove('T'); break
        
    else:
        # n >= 3
        uses[0]  = {'L'} if capL else {'T'}
        uses[1]  = {'T'} if capT else ({'L'} if capL else set())
        uses[-1] = {'R'} if capR else ({'T'} if capT else set())

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
    reference_lines: Dict[int, LineString],
    *,
    junction_eps: float=150.0,
    attach_tol: float=200.0,
    angle_tol_deg: float=60.0,
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
    # from matplotlib import pyplot as plt
    # fig, ax = plt.subplots()
    # for j, c in centers.items():
    #     ax.plot(c.x, c.y, 'go')
    # plt.savefig('detected_junctions.png')
    node_jid = nx.get_node_attributes(lane_graph, 'junc_id')
    node_fid = nx.get_node_attributes(lane_graph, 'fid')

    # 2) build directed legs per junction (ordered by driver-left→right)
    legs_in, legs_out = build_directed_legs_per_junction(
        lanes_gdf, reference_lines, lane_graph, centers,
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
        # print(f"Connection {u}->{v} with fids {fu}->{fv} at junc {j_u}")
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
                        # print(f"ACCEPT {u}->{v} with fids {fid_u}->{fid_v} at junc {j}")
                        accepted[(u, v)] = connections[(u, v)]

    return accepted
