
import numpy as np
import geopandas as gpd
from typing import Dict
from collections import defaultdict
from shapely.ops import substring
from scipy.interpolate import splprep, splev
from shapely.geometry import LineString, MultiLineString, Point, MultiLineString


def densify(line, interval=1.0):
    n = int(line.length / interval)
    return LineString([line.interpolate(i * interval) for i in range(n + 1)])


def refine_linestring(line: LineString, spacing=0.5, smoothing=0.001):
    coords = np.array(line.coords)
    if len(coords) < 3:
        return line  # too short to refine

    # Compute arc-length parameterization
    dists = np.cumsum(np.r_[0, np.linalg.norm(np.diff(coords, axis=0), axis=1)])
    total_length = dists[-1]
    t = dists / total_length

    # Fit B-spline (low smoothing)
    try:
        tck, _ = splprep([coords[:, 0], coords[:, 1]], u=t, s=smoothing)
    except ValueError:
        return line

    # Generate resampled distances, excluding 0 and total_length
    num_pts = max(int(total_length // spacing), 1)
    mid_dists = np.linspace(spacing, total_length - spacing, num_pts - 1)
    mid_t = mid_dists / total_length

    # Evaluate spline
    x_mid, y_mid = splev(mid_t, tck)

    # Combine with endpoints
    new_coords = [tuple(coords[0])] + list(zip(x_mid, y_mid)) + [tuple(coords[-1])]
    return LineString(new_coords)


def resample_line_spacing(line: LineString, spacing: float = 0.5) -> LineString:
    """
    Resample a LineString at ~uniform spacing along arc length.
    Keeps endpoints and inserts intermediate points every `spacing`.
    """
    L = line.length
    if L < spacing:
        return line
    n = int(L // spacing) + 2
    return LineString([line.interpolate(d) for d in np.linspace(0, L, n)])


def resample_line_points(geom:LineString, num_points=50):
    """
    Resample a LineString to a fixed number of points (including endpoints).
    """
    return np.array([
        geom.interpolate(i / (num_points - 1), normalized=True).coords[0]
        for i in range(num_points)
    ])


def _reverse_ls(ls: LineString) -> LineString:
    """Return the reversed polyline (endpoints swapped)."""
    return LineString(list(ls.coords)[::-1])

def _avg_perp_distance(a: LineString, b: LineString, spacing: float = 1.0) -> float:
    """
    Average perpendicular distance from points sampled on `a`
    (every `spacing`) to the polyline `b`.
    """
    if a.length < 1e-6 or b.length < 1e-6:
        return np.inf
    n = max(int(a.length // spacing), 2)
    dists = [b.distance(a.interpolate(i * spacing)) for i in range(n)]
    return float(np.mean(dists)) if dists else np.inf

def _sym_avg_distance(a: LineString, b: LineString, spacing: float = 1.0) -> float:
    """
    Symmetric average distance = max( avg(a→b), avg(b→a) ).
    Using max is conservative and avoids asymmetric false positives.
    """
    return max(_avg_perp_distance(a, b, spacing),
               _avg_perp_distance(b, a, spacing))

def _best_aligned_pair(a: LineString, b: LineString, spacing: float = 1.0):
    """
    Try orientations and pick the pairing with the smallest symmetric distance.
    We consider (a, b), (a, rev(b)), and (rev(a), b).
    (rev(a), rev(b)) is equivalent to (a, b) and omitted.
    Returns: (best_cost, A_aligned, B_aligned)
    """
    candidates = [
        (a, b),
        (a, _reverse_ls(b)),
        (_reverse_ls(a), b),
    ]
    best = None
    best_pair = None
    for A, B in candidates:
        cost = _sym_avg_distance(A, B, spacing)
        if best is None or cost < best:
            best, best_pair = cost, (A, B)
    return best, best_pair[0], best_pair[1]

def _dir_vec(line: LineString, frac: float = 0.05) -> np.ndarray:
    """
    Robust direction estimate using two points away from the ends:
    v = P(1-frac) - P(frac). Helps on curvy polylines vs. using the raw endpoints.
    """
    L = max(line.length, 1e-9)
    p0 = np.array(line.interpolate(L * frac).coords[0])
    p1 = np.array(line.interpolate(L * (1 - frac)).coords[0])
    v = p1 - p0
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def _angle_deg_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle (degrees) between two unit-ish vectors."""
    c = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))

def _endpoint_distances(l1: LineString, l2: LineString):
    """
    Return the four endpoint pair distances:
    start-start, end-end, start-end, end-start.
    """
    a0 = np.asarray(l1.coords[0]); a1 = np.asarray(l1.coords[-1])
    b0 = np.asarray(l2.coords[0]); b1 = np.asarray(l2.coords[-1])
    d_s_s = np.linalg.norm(a0 - b0)
    d_e_e = np.linalg.norm(a1 - b1)
    d_s_e = np.linalg.norm(a0 - b1)
    d_e_s = np.linalg.norm(a1 - b0)
    return d_s_s, d_e_e, d_s_e, d_e_s


# ----------------------------
# Core predicate
# ----------------------------

def lines_are_groupable(
    line1: LineString,
    line2: LineString,
    *,
    avg_tol: float = 3.0,
    endpoint_tol: float = 2.0,
    spacing: float = 1.0,
    angle_tol_deg: float = 25.0,     # max heading difference to be "same direction"
    serial_reject_tol: float = 2.0,  # reject same-direction head↔tail proximity (serial join)
    min_overlap_ratio: float = 0.0,  # optional along-track overlap requirement (0..1)
) -> bool:
    """
    Decide if two lane centerlines should be grouped as adjacent segments in the same corridor.

    Logic:
      1) Geometry distance is computed after automatic direction alignment
         (flip whichever side minimizes symmetric average distance).
      2) Direction-sensitive endpoint rule:
         - Same direction: require min(start↔start, end↔end) <= endpoint_tol
                           AND reject serial joins via min(start↔end, end↔start) > serial_reject_tol.
         - Opposite direction: require min(start↔end, end↔start) <= endpoint_tol.
      3) (Optional) Enforce a minimum along-track overlap between the aligned pair.
    """
    if line1.length < 1e-6 or line2.length < 1e-6:
        return False

    # 1) Use best-aligned orientation to evaluate geometric closeness.
    best_cost, A_dist, B_dist = _best_aligned_pair(line1, line2, spacing)
    if best_cost > avg_tol:
        return False

    # 2) Direction-sensitive endpoint rule (based on original headings).
    v1 = _dir_vec(line1); v2 = _dir_vec(line2)
    ang = _angle_deg_between(v1, v2)
    same_dir = (ang <= angle_tol_deg)   # ~180° is NOT treated as same direction here

    d_s_s, d_e_e, d_s_e, d_e_s = _endpoint_distances(line1, line2)

    if same_dir:
        # Adjacent same-direction lanes should have start~start or end~end close,
        # but should NOT be head-to-tail serial connections.
        if min(d_s_s, d_e_e) > endpoint_tol:
            return False
        if min(d_s_e, d_e_s) <= serial_reject_tol:
            return False  # serial (head↔tail) → reject
    else:
        # Opposite-direction adjacency: start~end or end~start should be close.
        if min(d_s_e, d_e_s) > endpoint_tol:
            return False

    # 3) Optional along-track overlap (computed on aligned pair).
    if min_overlap_ratio > 0.0:
        shorter, longer = (A_dist, B_dist) if A_dist.length <= B_dist.length else (B_dist, A_dist)
        d0 = longer.project(shorter.interpolate(0.0, normalized=True))
        d1 = longer.project(shorter.interpolate(1.0, normalized=True))
        ratio = abs(d1 - d0) / max(shorter.length, 1e-9)
        if ratio < min_overlap_ratio:
            return False

    return True


# ----------------------------
# Grouping over a GeoDataFrame
# ----------------------------

def group_lanes_by_geometry(
    gdf_lanes: gpd.GeoDataFrame,
    *,
    spacing: float = 0.5,       # resampling step for distance robustness
    avg_tol: float = 3.0,       # symmetric avg. distance threshold
    endpoint_tol: float = 2.0,  # endpoint proximity threshold
    angle_tol_deg: float = 25.0,
    serial_reject_tol: float = 2.0,
    min_overlap_ratio: float = 0.0,
    pair_expand: float = 6.0,   # prefilter radius for candidate pairs
) -> gpd.GeoDataFrame:
    """
    Group lane polylines into corridor IDs ("road_id") under direction-sensitive
    adjacency rules. Reverse-direction lanes along the same corridor are allowed,
    but serial connections are excluded.

    Expects gdf_lanes to have:
      - geometry: LineString
      - type == "lane" for lane rows
    """
    gdf = gdf_lanes[gdf_lanes["type"] == "lane"].copy().reset_index(drop=True)
    gdf["geometry_resampled"] = gdf["geometry"].apply(lambda g: resample_line_spacing(g, spacing=spacing))

    # Spatial index on original geometry to reduce pair checks.
    sindex = gdf.sindex

    # Union-Find (Disjoint Set) for grouping
    parent = {}
    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x
    def union(x, y):
        parent[find(x)] = find(y)

    for i, row in gdf.iterrows():
        gi = row["geometry_resampled"]
        # Candidate neighbors: lines whose original geometry intersects the buffer.
        buf = row.geometry.buffer(pair_expand)
        cand_idx = list(sindex.query(buf, predicate="intersects"))
        for j in cand_idx:
            if j <= i:
                continue
            gj = gdf.at[j, "geometry_resampled"]
            if lines_are_groupable(
                gi, gj,
                avg_tol=avg_tol,
                endpoint_tol=endpoint_tol,
                spacing=max(0.5, spacing),
                angle_tol_deg=angle_tol_deg,
                serial_reject_tol=serial_reject_tol,
                min_overlap_ratio=min_overlap_ratio,
            ):
                union(i, j)

    # Assign contiguous corridor IDs
    groups = defaultdict(list)
    for i in range(len(gdf)):
        groups[find(i)].append(i)

    id_map = {}
    for k, (_, members) in enumerate(groups.items()):
        rid = f"R{k:03d}"
        for m in members:
            id_map[m] = rid

    gdf["road_id"] = gdf.index.map(id_map)
    return gdf.drop(columns=["geometry_resampled"])


def get_junction_points(lanes_gdf: gpd.GeoDataFrame) -> list[Point]:
    """
    Collect split and merge points from annotated lane node types.
    """
    points = []
    for _, row in lanes_gdf.iterrows():
        if row["start_type"] in {"merge"}:
            points.append(Point(row.geometry.coords[0]))
        if row["end_type"] in {"split"}:
            points.append(Point(row.geometry.coords[-1]))
    return points


def _snap_point_to_line(line: LineString, pt: Point):
    d = line.project(pt)
    return line.interpolate(d), d


def _unique_sorted_distances(ds, length, min_gap=0.25, eps=1e-9):
    ds = [float(d) for d in ds if (d > eps and d < (length - eps))]
    if not ds:
        return []
    ds = sorted(ds)
    kept = [ds[0]]
    for d in ds[1:]:
        if d - kept[-1] >= max(min_gap, 0.0):
            kept.append(d)
    return kept


def _segments_via_substring(line: LineString, cut_ds, min_seg_len=0.25):
    if not cut_ds:
        return [line]
    length = line.length
    bounds = [0.0] + cut_ds + [length]
    segs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if (b - a) >= min_seg_len:
            segs.append(substring(line, a, b, normalized=False))
    return segs


def split_lines_at_junctions(
    lanes_gdf: gpd.GeoDataFrame,
    junction_points: list[Point],
    tol: float = 1.0,
    min_gap: float = 0.25,
    min_seg_len: float = 0.25,
    keep_cols: list[str] = None,
) -> gpd.GeoDataFrame:
    """
    Split each LineString in gdf_lines at nearby junction points.
    Uses GeoPandas sindex if available; otherwise uses Shapely STRtree
    (without relying on object identity).
    """
    if keep_cols is None:
        keep_cols = [c for c in lanes_gdf.columns if c != "geometry"]

    if not junction_points:
        out = lanes_gdf.copy()
        out["seg_idx"] = 0
        out["n_segs"] = 1
        return out

    # Build a GeoSeries for junction points and try GeoPandas sindex first
    gs_pts = gpd.GeoSeries(junction_points, crs=lanes_gdf.crs)
    sindex = getattr(gs_pts, "sindex", None)

    # Fallback: Shapely STRtree (no id mapping)
    tree = None
    if sindex is None:
        try:
            from shapely.strtree import STRtree
            tree = STRtree(list(gs_pts))
        except Exception:
            tree = None  # final fallback: brute force filter by intersects

    out_rows = []
    for _, row in lanes_gdf.iterrows():
        geom = row.geometry
        attrs = row[keep_cols].to_dict()

        def handle_line(line: LineString):
            if line.is_empty or line.length < 1e-9:
                return [line]

            buf = line.buffer(tol)

            # ---- candidate points near the buffered line ----
            if sindex is not None:
                idxs = list(sindex.query(buf, predicate="intersects"))
                cand_pts = [gs_pts.iloc[i] for i in idxs]
            elif tree is not None:
                cand_pts = list(tree.query(buf))
            else:
                # last-resort: brute force
                cand_pts = [p for p in gs_pts if p.intersects(buf)]

            # snap/filter
            cut_ds = []
            for p in cand_pts:
                if line.distance(p) <= tol:
                    _, d = _snap_point_to_line(line, p)
                    cut_ds.append(d)
            cut_ds = _unique_sorted_distances(cut_ds, line.length, min_gap=min_gap)

            segs = _segments_via_substring(line, cut_ds, min_seg_len=min_seg_len)
            return segs

        if isinstance(geom, MultiLineString):
            parts = []
            for part in geom.geoms:
                parts.extend(handle_line(part))
        elif isinstance(geom, LineString):
            parts = handle_line(geom)
        else:
            parts = [geom]

        n = len(parts)
        for i, seg in enumerate(parts):
            new_row = dict(attrs)
            new_row["geometry"] = seg
            new_row["seg_idx"] = i
            new_row["n_segs"] = n
            out_rows.append(new_row)

    return gpd.GeoDataFrame(out_rows, geometry="geometry", crs=lanes_gdf.crs)



def infer_lane_directions_from_geometry(gdf_lanes, head_segment_length=3):
    """
    Infer lane direction using the signed angle between each lane and a reference lane.
    Assigns lane_dir = 1 if aligned with reference, -1 otherwise.
    """
    gdf_lanes = gdf_lanes.copy()
    gdf_lanes["lane_dir"] = 0  # placeholder

    def get_head_vector(geom, length=3):
        coords = np.array(geom.coords)
        if len(coords) < 2:
            return np.array([0, 0])
        dists = np.cumsum([0] + [np.linalg.norm(coords[i+1] - coords[i]) for i in range(len(coords)-1)])
        idx = np.searchsorted(dists, length)
        if idx >= len(coords): idx = len(coords) - 1
        vec = coords[idx] - coords[0]
        return vec / (np.linalg.norm(vec) + 1e-8)

    def angle_between(u, v):
        cross = u[0]*v[1] - u[1]*v[0]
        dot = np.dot(u, v)
        return np.rad2deg(np.arctan2(cross, dot))

    for road_id, group in gdf_lanes.groupby("road_id"):
        ref_geom = group.iloc[0].geometry
        ref_vec = get_head_vector(ref_geom, head_segment_length)

        for idx, row in group.iterrows():
            lane_vec = get_head_vector(row.geometry, head_segment_length)
            angle = angle_between(lane_vec, ref_vec)
            lane_dir = 1 if abs(angle) < 90 else -1

            gdf_lanes.at[idx, "lane_dir"] = lane_dir
            # print(f"Road ID: {road_id}, Lane idx: {idx}, Angle: {angle:.2f}°, Direction: {lane_dir}")

    return gdf_lanes


def compute_signed_offset(ref_pts, lane_pts, method="mean"):
    """
    Compute average signed lateral offset from lane_pts to ref_pts.
    Ref and lane must be resampled to the same length.
    method: 'mean' | 'median' | 'trimmed'
    """
    offsets = []
    for i in range(len(ref_pts) - 1):
        dx, dy = ref_pts[i + 1] - ref_pts[i]
        norm = np.hypot(dx, dy)
        if norm < 1e-6:
            continue
        nx, ny = -dy / norm, dx / norm  # left-hand normal

        vx, vy = lane_pts[i] - ref_pts[i]
        offset = vx * nx + vy * ny
        offsets.append(offset)

    offsets = np.array(offsets)

    if method == "median":
        return np.median(offsets)
    elif method == "trimmed":
        from scipy.stats import trim_mean
        return trim_mean(offsets, proportiontocut=0.1)
    else:  # default: mean
        return np.mean(offsets)

def shift_reference_line_to_outer_edge(center_line, shift_amount):
    coords = np.array(center_line.coords)
    if len(coords) < 2:
        return center_line
    t = coords[-1] - coords[0]
    t /= (np.linalg.norm(t) + 1e-8)
    normal = np.array([-t[1], t[0]])  # left-hand normal
    shifted_coords = coords + shift_amount * normal
    return LineString(shifted_coords)

def compute_reference_lines_direction_aware(lanes_gdf: gpd.GeoDataFrame, num_points=50, average_lane_width=False):
    """
    Compute reference lines and lane widths for each road_id in gdf_lanes.
    Adds: lane_width, lane_offset, avg_offset
    Returns: dict {road_id: LineString}, updated gdf_lanes
    """
    ref_lines = {}
    lanes_gdf["lane_width"] = 0.0
    lanes_gdf["lane_offset"] = 0.0
    lanes_gdf["avg_offset"] = 0.0

    for road_id, group in lanes_gdf.groupby("road_id"):
        lanes_pos = group[group["lane_dir"] == 1]
        lanes_neg = group[group["lane_dir"] == -1]
        group = group.reset_index()
        is_bidirectional = not lanes_pos.empty and not lanes_neg.empty

        if is_bidirectional:
            # print(f"Processing bidirectional road: {road_id}")

            def resample_and_align(geom, ref_coords):
                coords = resample_line_points(geom, num_points)
                d_forward = np.linalg.norm(ref_coords[0] - coords[0])
                d_reverse = np.linalg.norm(ref_coords[0] - coords[-1])
                return coords if d_forward < d_reverse else coords[::-1]

            ref_coords = resample_line_points(lanes_pos.iloc[0].geometry, num_points)
            best_pair = None
            min_dist = float("inf")

            for pos_geom in lanes_pos.geometry:
                pos_coords = resample_and_align(pos_geom, ref_coords)
                for neg_geom in lanes_neg.geometry:
                    neg_coords = resample_and_align(neg_geom, ref_coords)
                    dist = np.mean(np.linalg.norm(pos_coords - neg_coords, axis=1))
                    if dist < min_dist:
                        best_pair = (pos_coords, neg_coords)
                        min_dist = dist

            center_coords = 0.5 * (best_pair[0] + best_pair[1])
            ref_line = LineString(center_coords)

            # Assign avg_offset
            for _, row in group.iterrows():
                lane_pts = resample_line_points(row.geometry, num_points)
                offset = compute_signed_offset(center_coords, lane_pts, method="trimmed")
                lanes_gdf.at[row["index"], "avg_offset"] = offset

            # Compute left/right separately
            group_idx = group["index"].values
            avg_offsets = lanes_gdf.loc[group_idx, "avg_offset"]
            left_mask = avg_offsets < 0
            right_mask = avg_offsets > 0

            left_indices = group_idx[left_mask.values]
            right_indices = group_idx[right_mask.values]
            
            left_lanes = lanes_gdf.loc[left_indices].copy().sort_values(by="avg_offset", ascending=False)
            right_lanes = lanes_gdf.loc[right_indices].copy().sort_values(by="avg_offset")

            offset_acc = 0.0
            for idx, row in left_lanes.iterrows():
                center = abs(lanes_gdf.at[idx, "avg_offset"])
                width = 2 * (center - offset_acc)
                lanes_gdf.at[idx, "lane_width"] = width
                lanes_gdf.at[idx, "lane_offset"] = offset_acc
                offset_acc += width

            offset_acc = 0.0
            for idx, row in right_lanes.iterrows():
                center = abs(lanes_gdf.at[idx, "avg_offset"])
                width = 2 * (center - offset_acc)
                lanes_gdf.at[idx, "lane_width"] = width
                lanes_gdf.at[idx, "lane_offset"] = offset_acc
                offset_acc += width
            if average_lane_width:
                lanes_gdf["lane_width"] = lanes_gdf.groupby("road_id")["lane_width"].transform("mean")
        else:
            # print(f"Processing one-directional road: {road_id}")
            sampled_lines = []
            ref_coords = resample_line_points(group.iloc[0].geometry, num_points)

            for geom in group.geometry:
                coords = resample_line_points(geom, num_points)
                d_forward = np.linalg.norm(ref_coords[0] - coords[0])
                d_reverse = np.linalg.norm(ref_coords[0] - coords[-1])
                if d_reverse < d_forward:
                    coords = coords[::-1]
                sampled_lines.append(coords)

            center_coords = np.mean(np.stack(sampled_lines), axis=0)
            center_line = LineString(center_coords)

            # Compute avg_offset
            for _, row in group.iterrows():
                lane_pts = resample_line_points(row.geometry, num_points)
                offset = compute_signed_offset(center_coords, lane_pts, method="trimmed")
                lanes_gdf.at[row["index"], "avg_offset"] = offset
                
            group_idx = group["index"].values
            avg_offsets = lanes_gdf.loc[group_idx, "avg_offset"]
            left_mask = avg_offsets < 0
            right_mask = avg_offsets > 0

            left_indices = group_idx[left_mask.values]
            right_indices = group_idx[right_mask.values]
            
            left_lanes = lanes_gdf.loc[left_indices].copy().sort_values(by="avg_offset", ascending=False)
            right_lanes = lanes_gdf.loc[right_indices].copy().sort_values(by="avg_offset")

            offset_acc = 0.0
            for idx, row in left_lanes.iterrows():
                center = abs(lanes_gdf.at[idx, "avg_offset"])
                width = 2 * (center - offset_acc)
                lanes_gdf.at[idx, "lane_width"] = width
                lanes_gdf.at[idx, "lane_offset"] = offset_acc
                offset_acc += width

            offset_acc = 0.0
            for idx, row in right_lanes.iterrows():
                center = abs(lanes_gdf.at[idx, "avg_offset"])
                width = 2 * (center - offset_acc)
                lanes_gdf.at[idx, "lane_width"] = width
                lanes_gdf.at[idx, "lane_offset"] = offset_acc
                offset_acc += width
            if average_lane_width:
                lanes_gdf["lane_width"] = lanes_gdf.groupby("road_id")["lane_width"].transform("mean")
            sorted_lanes = group.reindex(np.argsort(np.abs(group["avg_offset"])))
            offset_acc = 0.0
            # Shift to outer edge
            outermost = sorted_lanes.iloc[-1]
            # print(lanes_gdf.at[outermost["index"], "lane_width"])
            shift_amount = abs(lanes_gdf.at[outermost["index"], "avg_offset"]) + 0.5 * lanes_gdf.at[outermost["index"], "lane_width"]
            ref_line = shift_reference_line_to_outer_edge(center_line, shift_amount)

        ref_lines[road_id] = ref_line

    return lanes_gdf, ref_lines


def assign_lane_ids_per_group(lanes_gdf: gpd.GeoDataFrame, ref_lines: Dict[int, LineString], num_points=50):
    """
    Assign OpenDRIVE-compliant lane IDs (+1, -1, etc.) based on signed lateral offset.
    No flipping of geometry is done here.
    """

    for road_id, group in lanes_gdf.groupby("road_id"):
        ref_line: LineString = ref_lines[road_id]
        ref_pts = np.array([[pt.x, pt.y] for pt in [ref_line.interpolate(d) for d in np.linspace(0, ref_line.length, num_points)]])

        lane_offsets = []

        for idx, row in group.iterrows():
            line: LineString = row.geometry
            lane_pts = np.array([[pt.x, pt.y] for pt in [line.interpolate(d) for d in np.linspace(0, line.length, num_points)]])

            # No flipping — just compute signed lateral offset
            signed_offset = row['avg_offset']
            lane_offsets.append((idx, signed_offset))

        # Sort by offset: sort the positive (left) and negative (right) offsets separately
        left_lanes = [x for x in lane_offsets if x[1] > 0]
        right_lanes = [x for x in lane_offsets if x[1] < 0]
        sorted_left_lanes = sorted(left_lanes, key=lambda x: x[1])
        sorted_right_lanes = sorted(right_lanes, key=lambda x: x[1], reverse=True)

        lane_id_map = {}
        current_left = -1
        current_right = 1

        for idx, offset in sorted_left_lanes:
            lane_id_map[idx] = (current_left, "left")
            current_left -= 1

        for idx, offset in sorted_right_lanes:
            lane_id_map[idx] = (current_right, "right")
            current_right += 1

        for idx, (lid, side) in lane_id_map.items():
            lanes_gdf.at[idx, "lane_id"] = lid
            lanes_gdf.at[idx, "lane_side"] = side
 
    return lanes_gdf