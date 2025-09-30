import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from shapely.geometry import LineString


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
    # plot the fid text
    for idx, row in gdf_lanes.iterrows():
        if "fid" in row:
            x, y = row.geometry.interpolate(0.5, normalized=True).xy
            ax.text(x[0], y[0], str(row["fid"]), fontsize=8, color='black', ha='center', va='center')
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
    else:
        plt.show()


def visualize_road_groups(gdf_lanes, ax=None, cmap='tab20', label_col='road_id', save_path=None):
    """
    Visualize grouped lanes with different colors based on label_col.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))

    unique_ids = sorted(gdf_lanes[label_col].unique())
    num_ids = len(unique_ids)

    # Create color map
    cmap_obj = plt.get_cmap(cmap, num_ids)
    color_map = {rid: mcolors.to_hex(cmap_obj(i)) for i, rid in enumerate(unique_ids)}

    # Plot each group
    for rid in unique_ids:
        group = gdf_lanes[gdf_lanes[label_col] == rid]
        group.plot(ax=ax, color=color_map[rid], label=rid, linewidth=2, alpha=0.8)

    ax.set_title("Lane Groups")
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend(loc='best', title=label_col)

    if save_path:
        plt.savefig(os.path.join(save_path, "lane_groups.png"), dpi=300)
    else:
        plt.show()

    return ax


def plot_lane_directions(lanes_gdf:gpd.GeoDataFrame, title="Lane Directions", figsize=(10, 10), arrow_length=5):
    fig, ax = plt.subplots(figsize=figsize)
    
    for _, row in lanes_gdf.iterrows():
        geom = row.geometry
        if not isinstance(geom, LineString) or len(geom.coords) < 2:
            continue

        coords = list(geom.coords)
        x, y = zip(*coords)
        ax.plot(x, y, color='gray', linewidth=1, alpha=0.6)

        # Direction arrow
        start = np.array(coords[0])
        end = np.array(coords[-1])
        dir_vec = end - start
        dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-8) * arrow_length

        # Choose color by direction
        dir_val = row.get("lane_dir", 0)
        color = "blue" if dir_val == 1 else "red" if dir_val == -1 else "black"

        ax.arrow(start[0], start[1], dir_vec[0], dir_vec[1],
                 head_width=1.0, head_length=1.5, fc=color, ec=color)

    ax.set_title(title)
    ax.set_aspect('equal')
    plt.grid(True)
    plt.show()


def plot_lane_cuts(gdf_lanes, junction_points, cut_length=5.0, tol=1.0, figsize=(10, 10)):
    """
    Plot lanes, junction points, and orthogonal cut lines.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot lanes
    for _, row in gdf_lanes.iterrows():
        x, y = row.geometry.xy
        ax.plot(x, y, color="gray", linewidth=1.5)

    # Plot junctions and cut lines
    for jp in junction_points:
        ax.plot(jp.x, jp.y, "ro", markersize=6, label="junction" if "junction" not in ax.get_legend_handles_labels()[1] else "")

        # Find nearest lane for visualization
        nearest_geom = min(gdf_lanes.geometry, key=lambda g: g.distance(jp))
        if nearest_geom.distance(jp) < tol:
            proj_dist = nearest_geom.project(jp)
            nearest_pt = nearest_geom.interpolate(proj_dist)

            # Tangent vector
            delta = 1e-6 * nearest_geom.length
            before = nearest_geom.interpolate(max(proj_dist - delta, 0))
            after = nearest_geom.interpolate(min(proj_dist + delta, nearest_geom.length))
            tangent = np.array(after.coords[0]) - np.array(before.coords[0])
            tangent /= (np.linalg.norm(tangent) + 1e-8)

            # Orthogonal vector
            normal = np.array([-tangent[1], tangent[0]])

            # Cutting line
            cut_line = LineString([
                nearest_pt.coords[0] - cut_length * normal,
                nearest_pt.coords[0] + cut_length * normal
            ])

            x, y = cut_line.xy
            ax.plot(x, y, "b--", linewidth=2, label="cut line" if "cut line" not in ax.get_legend_handles_labels()[1] else "")

    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("Lane cuts at junctions (red=junction, blue=cut line)")
    plt.show()