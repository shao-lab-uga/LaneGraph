import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
import numpy as np
def resample_line_points(line, num_points=50):
    """Uniformly sample a fixed number of points along the line."""
    return np.array([[pt.x, pt.y] for pt in [line.interpolate(d) for d in np.linspace(0, line.length, num_points)]])
    
def visualize_road_groups_with_reference_lines(gdf_lanes, ref_lines, ax=None, cmap='tab20'):
    """
    Visualize road_id groups in color, and overlay reference lines in bold black.
    
    Parameters:
        gdf_lanes: GeoDataFrame with 'road_id' and 'geometry'
        ref_lines: dict {road_id: LineString}
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))

    unique_ids = sorted(gdf_lanes["road_id"].unique())
    num_ids = len(unique_ids)

    cmap_obj = plt.get_cmap(cmap, num_ids)
    color_map = {rid: mcolors.to_hex(cmap_obj(i)) for i, rid in enumerate(unique_ids)}

    # Plot each lane group
    for rid in unique_ids:
        group = gdf_lanes[gdf_lanes["road_id"] == rid]
        group.plot(ax=ax, color=color_map[rid], linewidth=1.5, label=rid, alpha=0.6)

    # Plot each reference line in black
    for rid, ref_geom in ref_lines.items():
        gpd.GeoSeries([ref_geom]).plot(ax=ax, color='black', linewidth=3, zorder=10)

    ax.set_title("Road Groups with Reference Lines")
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend(loc='best', title="road_id")

    return ax

def visualize_links_with_ref_lines(gdf_links, figsize=(10, 10), save_path=None):
    """
    Visualize link centerlines and their generated reference lines.
    - Link geometry (centerline): solid blue
    - Reference line (OpenDRIVE planView): dashed red
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot all centerlines
    gdf_links.plot(ax=ax, color='blue', linewidth=2, label='Link Centerline')

    # Plot reference lines if available
    for idx, row in gdf_links.iterrows():
        if "reference_geom" in row and row["reference_geom"] is not None:
            ref_line = row["reference_geom"]
            gpd.GeoSeries([ref_line]).plot(ax=ax, color='black', linewidth=1.5, label='Reference Line' if idx == 0 else "")

    # Annotations and legend
    ax.set_title("Link Centerlines vs. Reference Lines", fontsize=14)
    ax.set_aspect("equal")
    ax.legend(loc="upper right")

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[✓] Saved to {save_path}")
    else:
        plt.show()

def visualize_road_groups(gdf_lanes, ax=None, cmap='tab20'):
    """
    Visualize grouped lanes with different colors based on road_id.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))

    unique_ids = sorted(gdf_lanes["road_id"].unique())
    num_ids = len(unique_ids)

    # Create color map
    cmap_obj = plt.get_cmap(cmap, num_ids)
    color_map = {rid: mcolors.to_hex(cmap_obj(i)) for i, rid in enumerate(unique_ids)}

    # Plot each group
    for rid in unique_ids:
        group = gdf_lanes[gdf_lanes["road_id"] == rid]
        group.plot(ax=ax, color=color_map[rid], label=rid, linewidth=2, alpha=0.8)

    ax.set_title("Lane Groups")
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend(loc='best', title="road_id")

    return ax


def compute_avg_lateral_offsets(gdf_lanes, ref_lines, num_points=50):
    """
    Compute average signed lateral offset from each lane centerline to its reference line.
    Adds: 'avg_offset' column to the GeoDataFrame.
    """
    gdf_lanes = gdf_lanes.copy()
    offsets = []

    for idx, row in gdf_lanes.iterrows():
        road_id = row["road_id"]
        lane_line = row["geometry"]
        ref_line = ref_lines[road_id]

        lane_pts = resample_line_points(lane_line, num_points=num_points)
        ref_pts = resample_line_points(ref_line, num_points=num_points)

        signed_dists = []
        for i in range(len(ref_pts) - 1):
            # Reference direction vector
            p0 = ref_pts[i]
            p1 = ref_pts[i + 1]
            t = p1 - p0
            t_norm = np.linalg.norm(t)
            if t_norm < 1e-6:
                continue
            t_unit = t / t_norm
            n = np.array([-t_unit[1], t_unit[0]])  # left-hand normal

            # Corresponding lane point
            q = lane_pts[i]

            # Vector from ref point to lane point
            v = q - p0
            signed_dist = np.dot(v, n)  # projection onto normal
            signed_dists.append(signed_dist)

        avg_offset = np.mean(signed_dists)
        offsets.append(avg_offset)

    gdf_lanes["avg_offset"] = offsets
    return gdf_lanes


def assign_lane_widths_and_offsets(gdf_lanes_with_offsets):
    gdf = gdf_lanes_with_offsets.copy()
    gdf["lane_width"] = 0.0
    gdf["lane_offset"] = 0.0

    for road_id, group in gdf.groupby("road_id"):
        # LEFT lanes (avg_offset < 0): more negative = farther from center
        left_lanes = group[group["lane_side"] == "left"].copy()
        left_lanes = left_lanes.sort_values("avg_offset", ascending=False)

        offset_acc = 0.0
        for idx, row in left_lanes.iterrows():
            center = row["avg_offset"]
            width = 2 * abs(abs(center) - offset_acc)
            gdf.at[idx, "lane_width"] = width
            gdf.at[idx, "lane_offset"] = offset_acc
            offset_acc += width

        # RIGHT lanes (avg_offset > 0): closer to 0 = inner lane
        right_lanes = group[group["lane_side"] == "right"].copy()
        right_lanes = right_lanes.sort_values("avg_offset")  # from 1.8 → 5.4

        offset_acc = 0.0
        for idx, row in right_lanes.iterrows():
            center = row["avg_offset"]
            width = 2 * abs(center - offset_acc)
            gdf.at[idx, "lane_width"] = width
            gdf.at[idx, "lane_offset"] = offset_acc
            offset_acc += width

    return gdf

