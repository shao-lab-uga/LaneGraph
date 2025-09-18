import math
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point
from typing import Any, Dict, List, Optional, Tuple

# =========================
# Geometry helpers
# =========================

def _bearing_oriented(ls: LineString) -> float:
    """Oriented (0..360) length-weighted resultant bearing of a LineString."""
    coords = list(ls.coords)
    if len(coords) < 2:
        return 0.0
    vx = vy = 0.0
    for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:]):
        dx, dy = (x2 - x1), (y2 - y1)
        vx += dx
        vy += dy
    ang = (math.degrees(math.atan2(vy, vx)) + 360.0) % 360.0
    return ang


def _angle_diff_oriented(a: float, b: float) -> float:
    """Smallest oriented angle diff on [0,180]."""
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d


def _sample_along_line(ls: LineString, spacing: float) -> List[Point]:
    L = ls.length
    if L == 0:
        return [Point(ls.coords[0])]
    n = max(2, int(L / spacing) + 1)
    return [ls.interpolate(i * L / (n - 1)) for i in range(n)]


def _mean_distance_to_line(points: List[Point], line: LineString, pctl: float = 0.9) -> float:
    """Robust mean distance: mean up to a percentile to suppress outliers."""
    dists = np.array([p.distance(line) for p in points], dtype=float)
    if dists.size == 0:
        return float("inf")
    cutoff = np.quantile(dists, pctl)
    use = dists[dists <= cutoff]
    return float(use.mean()) if use.size else float(dists.mean())


def _reverse_if_needed(ls: LineString, start_near: Point) -> LineString:
    """Ensure ls starts near start_near (helpful for stitching)."""
    c0 = Point(ls.coords[0])
    c1 = Point(ls.coords[-1])
    if start_near.distance(c1) < start_near.distance(c0):
        return LineString(list(ls.coords)[::-1])
    return ls


def _stitch_lines_in_order(lines: List[LineString]) -> LineString:
    """Greedy stitch: flip each line to connect to previous, then concatenate."""
    if not lines:
        return LineString()
    out = [LineString(lines[0].coords)]
    for i in range(1, len(lines)):
        prev_end = Point(out[-1].coords[-1])
        ls = _reverse_if_needed(lines[i], prev_end)
        out.append(LineString(ls.coords))
    coords = [out[0].coords[0]]
    for seg in out:
        coords.extend(seg.coords[1:])
    return LineString(coords)


def cut_substring(line: LineString, s0: float, s1: float) -> LineString:
    """Return substring of 'line' between curvilinear distances s0 and s1 (s0<=s1)."""
    if line.length == 0:
        return LineString()
    s0 = max(0.0, min(s0, line.length))
    s1 = max(0.0, min(s1, line.length))
    if s1 <= s0:
        return LineString()
    coords = list(line.coords)
    acc = 0.0
    keep = []
    for (x0, y0), (x1_, y1_) in zip(coords[:-1], coords[1:]):
        seg = LineString([(x0, y0), (x1_, y1_)])
        L = seg.length
        seg_start, seg_end = acc, acc + L
        if seg_end <= s0:
            acc += L
            continue
        if seg_start >= s1:
            break
        a = max(0.0, s0 - seg_start)
        b = min(L,    s1 - seg_start)
        pt_a = seg.interpolate(a)
        pt_b = seg.interpolate(b)
        if not keep:
            keep.append((pt_a.x, pt_a.y))
        keep.append((pt_b.x, pt_b.y))
        acc += L
    return LineString(keep) if len(keep) >= 2 else LineString()


def map_s_to_edge(
    ordered_edge_ids: List[Any],
    edges_gdf: gpd.GeoDataFrame,
    s_on_path: float,
    edge_id_col: str = "edge_id",
) -> Tuple[Optional[Any], Optional[float]]:
    """Given distance s along stitched path, return (edge_id, s_on_that_edge)."""
    if not ordered_edge_ids:
        return None, None
    acc = 0.0
    for eid in ordered_edge_ids:
        ls = edges_gdf.loc[edges_gdf[edge_id_col] == eid, "geometry"].iloc[0]
        L = ls.length
        if s_on_path <= acc + L or abs((acc + L) - s_on_path) < 1e-9:
            return eid, s_on_path - acc
        acc += L
    last_id = ordered_edge_ids[-1]
    last_ls = edges_gdf.loc[edges_gdf[edge_id_col] == last_id, "geometry"].iloc[0]
    return last_id, last_ls.length


# =========================
# Candidate selection & ordering
# =========================

def _spatial_candidates(edges_gdf: gpd.GeoDataFrame, ref_line: LineString, radius: float) -> gpd.GeoDataFrame:
    """Fast prefilter using spatial index and a buffer around ref_line."""
    try:
        if edges_gdf.sindex is None:
            raise AttributeError
        buf = ref_line.buffer(radius)
        idxs = list(edges_gdf.sindex.query(buf, predicate="intersects"))
        return edges_gdf.iloc[idxs].copy() if idxs else edges_gdf.head(0).copy()
    except Exception:
        # no sindex; fallback to all
        return edges_gdf.copy()


def select_directional_candidates(
    edges_gdf: gpd.GeoDataFrame,
    ref_line: LineString,
    direction: str = "forward",         # "forward" or "reverse"
    angle_thresh_deg: float = 25.0,
    dist_thresh: float = 6.0,
    sample_spacing: float = 3.0,
) -> gpd.GeoDataFrame:
    """
    Keep only edges whose oriented bearing matches the reference direction
    AND are close to the reference line.
    """
    assert direction in ("forward", "reverse")
    ref_bearing = _bearing_oriented(ref_line)
    if direction == "reverse":
        ref_bearing = (ref_bearing + 180.0) % 360.0

    # quick spatial prefilter
    pre = _spatial_candidates(edges_gdf, ref_line, radius=max(dist_thresh * 2.0, sample_spacing * 3.0))
    if pre.empty:
        return pre

    edge_bearings = pre.geometry.apply(_bearing_oriented).to_numpy()
    ang_diff = np.array([_angle_diff_oriented(b, ref_bearing) for b in edge_bearings])
    ang_mask = ang_diff <= angle_thresh_deg

    samples = _sample_along_line(ref_line, spacing=sample_spacing)
    mean_dists = np.array([_mean_distance_to_line(samples, g, pctl=0.9) for g in pre.geometry], dtype=float)
    dist_mask = mean_dists <= dist_thresh

    mask = ang_mask & dist_mask
    out = pre.loc[mask].copy()
    if out.empty:
        return out
    out["angle_diff_dir"] = ang_diff[mask]
    out["mean_dist"] = mean_dists[mask]
    return out.sort_values(["mean_dist", "angle_diff_dir"])


def order_candidates_along_reference(
    ref_line: LineString,
    candidates_gdf: gpd.GeoDataFrame,
    edge_id_col: str = "edge_id",
    sample_spacing: float = 3.0,
) -> List[Any]:
    """Sample ref_line; for each sample pick nearest candidate edge; deduplicate (stable)."""
    if candidates_gdf.empty:
        return []
    samples = _sample_along_line(ref_line, spacing=sample_spacing)
    cand_geoms = candidates_gdf.geometry.to_list()
    cand_ids = candidates_gdf[edge_id_col].to_list()

    chosen_ids = []
    for p in samples:
        dists = [p.distance(g) for g in cand_geoms]
        if not dists:
            continue
        j = int(np.argmin(dists))
        eid = cand_ids[j]
        if not chosen_ids or eid != chosen_ids[-1]:
            chosen_ids.append(eid)
    return chosen_ids


def build_path_geometry_from_edges(
    edges_gdf: gpd.GeoDataFrame,
    ordered_edge_ids: List[Any],
    ref_line: LineString,
    edge_id_col: str = "edge_id",
) -> LineString:
    """Orient each edge to connect and stitch."""
    if not ordered_edge_ids:
        return LineString()
    ref_start = Point(ref_line.coords[0])
    lines = []
    prev_end = ref_start
    for i, eid in enumerate(ordered_edge_ids):
        ls = edges_gdf.loc[edges_gdf[edge_id_col] == eid, "geometry"].iloc[0]
        ls = _reverse_if_needed(ls, ref_start if i == 0 else prev_end)
        lines.append(ls)
        prev_end = Point(ls.coords[-1])
    return _stitch_lines_in_order(lines)


# =========================
# Direction pairing & endpoint s
# =========================

def find_directional_companions(
    edges_gdf: gpd.GeoDataFrame,
    edge_ids: List[Any],
    edge_id_col: str = "edge_id",
    from_col: str = "from",
    to_col: str = "to",
) -> Dict[Any, Any]:
    """
    Map each edge_id to its reverse-direction edge_id if possible.
    Priority: swapped ('from','to'); fallback: geometry equals().
    """
    idx = edges_gdf.set_index(edge_id_col)
    has_ft = (from_col in edges_gdf.columns) and (to_col in edges_gdf.columns)
    comp: Dict[Any, Any] = {}
    for eid in edge_ids:
        if eid not in idx.index:
            continue
        row = idx.loc[eid]
        rev_id = None

        if has_ft:
            f, t = row[from_col], row[to_col]
            cand = edges_gdf[(edges_gdf[from_col] == t) & (edges_gdf[to_col] == f)]
            if not cand.empty:
                # prefer exact-geometry equal if multiple
                mask_eq = cand.geometry.apply(lambda g: g.equals(row.geometry))
                chosen = cand.loc[mask_eq]
                rev_id = (chosen.iloc[0][edge_id_col] if not chosen.empty else cand.iloc[0][edge_id_col])

        if rev_id is None:
            geom = row.geometry
            cand = edges_gdf[(edges_gdf[edge_id_col] != eid)]
            mask_eq = cand.geometry.apply(lambda g: g.equals(geom))
            cand = cand.loc[mask_eq]
            if not cand.empty:
                rev_id = cand.iloc[0][edge_id_col]

        if rev_id is not None:
            comp[eid] = rev_id
    return comp


def s_on_reverse_edge(
    forward_edge_geom: LineString,
    reverse_edge_geom: LineString,
    pt: Point,
    s_on_forward: float
) -> float:
    """
    Compute s on the reverse-direction edge for the same physical point.
    Prefer direct projection; mirror if reverse row uses identical (non-reversed) coords.
    """
    s_rev_direct = reverse_edge_geom.project(pt)
    Lf = forward_edge_geom.length
    Lr = reverse_edge_geom.length
    if abs(Lf - Lr) < 1e-6:
        s_mirror = Lf - s_on_forward
        if abs(s_rev_direct - s_mirror) <= 1.0:
            return s_rev_direct
        if abs(s_rev_direct - s_on_forward) < 1.0:
            return s_mirror
    return s_rev_direct


# =========================
# Per-edge segment extraction
# =========================

def per_edge_segments(
    edges_gdf: gpd.GeoDataFrame,
    ordered_edge_ids: List[Any],
    s0: float,
    s1: float,
    edge_id_col: str = "edge_id",
) -> List[Dict[str, Any]]:
    """
    For the stitched path interval [s0, s1], return a list of dicts:
      {'edge_id', 's_start', 's_end', 'geometry'} for each overlapped edge.
    """
    out: List[Dict[str, Any]] = []
    if not ordered_edge_ids or s1 <= s0:
        return out
    acc = 0.0
    for eid in ordered_edge_ids:
        edge_geom = edges_gdf.loc[edges_gdf[edge_id_col] == eid, "geometry"].iloc[0]
        L = edge_geom.length
        e0, e1 = acc, acc + L
        ov0, ov1 = max(s0, e0), min(s1, e1)
        if ov1 > ov0:  # overlap exists
            s_on_edge_start = ov0 - e0
            s_on_edge_end   = ov1 - e0
            seg_geom = cut_substring(edge_geom, s_on_edge_start, s_on_edge_end)
            out.append({
                "edge_id": eid,
                "s_start": float(s_on_edge_start),
                "s_end": float(s_on_edge_end),
                "geometry": seg_geom
            })
        acc += L
    return out


# =========================
# Main wrapper (direction-aware)
# =========================

def match_reference_line_to_sumo_edges_directional(
    edges_gdf: gpd.GeoDataFrame,
    ref_line: LineString,
    *,
    edge_id_col: str = "edge_id",
    from_col: str = "from",
    to_col: str = "to",
    angle_thresh_deg: float = 25.0,
    dist_thresh: float = 6.0,
    sample_spacing: float = 3.0,
    include_reverse_path: bool = True
) -> Dict[str, Any]:
    """
    Direction-aware matching. Returns forward path (aligned with ref_line direction)
    and reverse path (opposite travel), plus endpoint projections and per-edge segments.
    """
    # ---- Forward ----
    cand_fwd = select_directional_candidates(
        edges_gdf, ref_line, "forward",
        angle_thresh_deg=angle_thresh_deg,
        dist_thresh=dist_thresh,
        sample_spacing=sample_spacing
    )
    ordered_fwd = order_candidates_along_reference(
        ref_line, cand_fwd, edge_id_col=edge_id_col, sample_spacing=sample_spacing
    )
    path_fwd = build_path_geometry_from_edges(edges_gdf, ordered_fwd, ref_line, edge_id_col=edge_id_col) \
               if ordered_fwd else LineString()

    p0 = Point(ref_line.coords[0])
    p1 = Point(ref_line.coords[-1])
    s0_f = path_fwd.project(p0) if path_fwd.length > 0 else 0.0
    s1_f = path_fwd.project(p1) if path_fwd.length > 0 else 0.0
    if s1_f < s0_f:
        s0_f, s1_f = s1_f, s0_f
    proj0_f = path_fwd.interpolate(s0_f) if path_fwd.length > 0 else None
    proj1_f = path_fwd.interpolate(s1_f) if path_fwd.length > 0 else None
    start_edge_f, s_on_start_f = map_s_to_edge(ordered_fwd, edges_gdf, s0_f, edge_id_col=edge_id_col)
    end_edge_f,   s_on_end_f   = map_s_to_edge(ordered_fwd, edges_gdf, s1_f, edge_id_col=edge_id_col)
    segment_fwd = cut_substring(path_fwd, s0_f, s1_f) if path_fwd.length > 0 else LineString()
    fwd_segments_per_edge = per_edge_segments(edges_gdf, ordered_fwd, s0_f, s1_f, edge_id_col=edge_id_col)

    # ---- Reverse (optional) ----
    ordered_rev: List[Any] = []
    path_rev = LineString()
    proj0_r = proj1_r = None
    start_edge_r = end_edge_r = None
    s_on_start_r = s_on_end_r = None
    segment_rev = LineString()
    rev_segments_per_edge: List[Dict[str, Any]] = []

    if include_reverse_path:
        ref_line_rev = LineString(list(ref_line.coords)[::-1])
        cand_rev = select_directional_candidates(
            edges_gdf, ref_line, "reverse",
            angle_thresh_deg=angle_thresh_deg,
            dist_thresh=dist_thresh,
            sample_spacing=sample_spacing
        )
        ordered_rev = order_candidates_along_reference(
            ref_line_rev, cand_rev, edge_id_col=edge_id_col, sample_spacing=sample_spacing
        )
        path_rev = build_path_geometry_from_edges(edges_gdf, ordered_rev, ref_line_rev, edge_id_col=edge_id_col) \
                   if ordered_rev else LineString()

        s0_r = path_rev.project(p0) if path_rev.length > 0 else 0.0
        s1_r = path_rev.project(p1) if path_rev.length > 0 else 0.0
        if s1_r < s0_r:
            s0_r, s1_r = s1_r, s0_r
        proj0_r = path_rev.interpolate(s0_r) if path_rev.length > 0 else None
        proj1_r = path_rev.interpolate(s1_r) if path_rev.length > 0 else None
        start_edge_r, s_on_start_r = map_s_to_edge(ordered_rev, edges_gdf, s0_r, edge_id_col=edge_id_col)
        end_edge_r,   s_on_end_r   = map_s_to_edge(ordered_rev, edges_gdf, s1_r, edge_id_col=edge_id_col)
        segment_rev = cut_substring(path_rev, s0_r, s1_r) if path_rev.length > 0 else LineString()
        rev_segments_per_edge = per_edge_segments(edges_gdf, ordered_rev, s0_r, s1_r, edge_id_col=edge_id_col)

    # ---- Companion reverse edges for forward endpoints ----
    start_rev_partner = end_rev_partner = None
    s_on_start_r_from_f = s_on_end_r_from_f = None
    if start_edge_f is not None and end_edge_f is not None:
        comp = find_directional_companions(
            edges_gdf, [start_edge_f, end_edge_f],
            edge_id_col=edge_id_col, from_col=from_col, to_col=to_col
        )
        start_rev_partner = comp.get(start_edge_f)
        end_rev_partner   = comp.get(end_edge_f)
        if proj0_f is not None and start_rev_partner is not None:
            fgeom = edges_gdf.loc[edges_gdf[edge_id_col] == start_edge_f, "geometry"].iloc[0]
            rgeom = edges_gdf.loc[edges_gdf[edge_id_col] == start_rev_partner, "geometry"].iloc[0]
            s_on_start_r_from_f = s_on_reverse_edge(fgeom, rgeom, proj0_f, s_on_start_f if s_on_start_f is not None else 0.0)
        if proj1_f is not None and end_rev_partner is not None:
            fgeom = edges_gdf.loc[edges_gdf[edge_id_col] == end_edge_f, "geometry"].iloc[0]
            rgeom = edges_gdf.loc[edges_gdf[edge_id_col] == end_rev_partner, "geometry"].iloc[0]
            s_on_end_r_from_f = s_on_reverse_edge(fgeom, rgeom, proj1_f, s_on_end_f if s_on_end_f is not None else 0.0)

    return {
        # Forward
        "ordered_edge_ids_forward": ordered_fwd,
        "path_geometry_forward": path_fwd,
        "projected_start_forward": (proj0_f.x, proj0_f.y) if proj0_f else None,
        "projected_end_forward":   (proj1_f.x, proj1_f.y) if proj1_f else None,
        "start_edge_id_forward": start_edge_f,
        "end_edge_id_forward":   end_edge_f,
        "start_s_on_edge_forward": float(s_on_start_f) if s_on_start_f is not None else None,
        "end_s_on_edge_forward":   float(s_on_end_f)   if s_on_end_f   is not None else None,
        "replicated_segment_forward": segment_fwd,
        "per_edge_segments_forward": fwd_segments_per_edge,

        # Reverse
        "ordered_edge_ids_reverse": ordered_rev,
        "path_geometry_reverse": path_rev,
        "projected_start_reverse": (proj0_r.x, proj0_r.y) if proj0_r else None,
        "projected_end_reverse":   (proj1_r.x, proj1_r.y) if proj1_r else None,
        "start_edge_id_reverse": start_edge_r,
        "end_edge_id_reverse":   end_edge_r,
        "start_s_on_edge_reverse": float(s_on_start_r) if s_on_start_r is not None else None,
        "end_s_on_edge_reverse":   float(s_on_end_r)   if s_on_end_r   is not None else None,
        "replicated_segment_reverse": segment_rev,
        "per_edge_segments_reverse": rev_segments_per_edge,

        # Companion reverse edges for forward endpoints
        "start_edge_reverse_partner_of_forward": start_rev_partner,
        "end_edge_reverse_partner_of_forward":   end_rev_partner,
        "start_s_on_reverse_partner_from_forward": float(s_on_start_r_from_f) if s_on_start_r_from_f is not None else None,
        "end_s_on_reverse_partner_from_forward":   float(s_on_end_r_from_f)   if s_on_end_r_from_f   is not None else None,
    }
    
# =========================
# Debug plotting

def debug_plot(edges_gdf: gpd.GeoDataFrame, ref_line: LineString, res: Dict[str, Any], figsize=(8, 8)):
    ax = edges_gdf.plot(color="lightgray", linewidth=1, figsize=figsize)
    gpd.GeoSeries([ref_line]).plot(ax=ax, color="tab:blue", linewidth=2, label="reference")
    if res["path_geometry_forward"] and not res["path_geometry_forward"].is_empty:
        gpd.GeoSeries([res["path_geometry_forward"]]).plot(ax=ax, color="tab:orange", linewidth=2, label="path fwd")
        gpd.GeoSeries([res["replicated_segment_forward"]]).plot(ax=ax, color="tab:green", linewidth=4, alpha=0.5, label="replicated fwd")
    if res["path_geometry_reverse"] and not res["path_geometry_reverse"].is_empty:
        gpd.GeoSeries([res["path_geometry_reverse"]]).plot(ax=ax, color="purple", linewidth=2, linestyle="--", label="path rev")
        gpd.GeoSeries([res["replicated_segment_reverse"]]).plot(ax=ax, color="magenta", linewidth=3, alpha=0.5, label="replicated rev")
    pts = []
    if res["projected_start_forward"]:
        x, y = res["projected_start_forward"]
        pts.append(Point(x, y))
    if res["projected_end_forward"]:
        x, y = res["projected_end_forward"]
        pts.append(Point(x, y))
    if pts:
        gpd.GeoSeries(pts).plot(ax=ax, color="red", markersize=30)
    ax.legend()
    return ax
