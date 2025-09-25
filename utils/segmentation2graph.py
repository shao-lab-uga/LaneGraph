import os
import cv2
import numpy as np
import networkx as nx
from pathlib import Path
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt

WINDOW_SIZE = 640

def distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Calculates euclidian distance between 2 points.

    Args:
        a (Tuple[int, int]): Tuple representing coordinates of the first point.
        b (Tuple[int, int]): Tuple representing coordinates of the second point.

    Returns:
        float: Euclidian distance between two points.
    """
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return float(np.sqrt(dx**2 + dy**2))


def point_line_distance(
    point: Tuple[int, int], start: Tuple[int, int], end: Tuple[int, int]
) -> float:
    """Calculate the shortest distance between a point and a line defined by 2 points.

    Args:
        point (Tuple[int, int]): Tuple representing coordinates of point.
        start (Tuple[int, int]): Tuple representing coordinates of start point of line.
        end (Tuple[int, int]): Tuple representing coordinates of end point of line.

    Returns:
        float: Shortest distance from point to line segment.
    """
    if start == end:
        return distance(point, start)
    else:
        n = abs(
            (end[0] - start[0]) * (start[1] - point[1])
            - (start[0] - point[0]) * (end[1] - start[1])
        )
        d = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        return float(n / d)


def rdp(points: List[Tuple[int, int]], epsilon: float) -> List[Tuple[int, int]]:
    """Applies the Ramer-Douglas-Peucker algorithm to simplify a polyline.

    Explanation of the algorithm can be found [here](https://w.wiki/3X7H).

    Args:
        points (List[Tuple[int, int]]): List of tuples representing coordinates of points in a polyline.
        epsilon (float): Threshold difference.

    Returns:
        List[Tuple[int, int]]: List of points representing the simplified polyline.
    """
    # TODO Check if 2 sided RDP does better
    dmax = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        d = point_line_distance(points[i], points[0], points[-1])
        if d > dmax:
            index = i
            dmax = d
    if dmax >= epsilon:
        results = rdp(points[: index + 1], epsilon)[:-1] + rdp(points[index:], epsilon)
    else:
        results = [points[0], points[-1]]
    return results


def find_first_unvisited_white_pixel(im, visited, start_pos=None):
    """Find the next unvisited white pixel in the image."""
    white_pixels = np.where((im > 0) & (~visited))
    if white_pixels[0].size == 0:
        return None
    if start_pos is not None:
        # Find the closest white pixel to the start position
        distances = np.sqrt(
            (white_pixels[0] - start_pos[0]) ** 2 + (white_pixels[1] - start_pos[1]) ** 2
        )
        closest_index = np.argmin(distances)
        return white_pixels[0][closest_index], white_pixels[1][closest_index]
    return white_pixels[0][0], white_pixels[1][0]


def get_8_connected_neighbors(im, visited, row, col):
    """Get 8-connected (cardinal + diagonal) unvisited white neighbors."""
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue  # Skip the center pixel itself
            r, c = row + dr, col + dc
            if 0 <= r < im.shape[0] and 0 <= c < im.shape[1]:
                if (im[r, c] > 0) and (not visited[r, c]):
                    neighbors.append((r, c))
    return neighbors


def trace_path(im, visited, start_row, start_col):
    """Follow a path in the image along 8-connected white pixels. Returns the full traced path and the final pixel."""
    path = [(start_row, start_col)]
    visited[start_row, start_col] = True
    row, col = start_row, start_col

    while True:
        neighbors = get_8_connected_neighbors(im, visited, row, col)
        if len(neighbors) != 1:
            break  # Stop if we reach a junction or dead-end
        row, col = neighbors[0]
        path.append((row, col))
        visited[row, col] = True

    return path, row, col


def extract_graph_from_image(
    im: np.ndarray, start_pos=None, epsilon: float = 2.0
) -> nx.Graph:
    """
    Extracts a graph from a thinned binary image using 8-connectivity and adds virtual
    connections between nearby endpoints to reduce fragmentation.
    """
    visited = np.zeros_like(im, dtype=bool)
    G = nx.Graph()
    queue = []

    while True:
        if not queue:
            start = find_first_unvisited_white_pixel(im, visited, start_pos)
            if start is None:
                break  # No more unvisited white pixels
            queue.append((None, start))  # (previous node ID, (row, col))

        prev_node_id, (row, col) = queue.pop()
        # path from (row, col)
        path, end_row, end_col = trace_path(im, visited, row, col)

        # Simplify the path if long enough
        if len(path) > 1:
            path = rdp(path, epsilon=epsilon)

        prev_node = None

        # Add nodes and edges to the graph
        for idx, (r, c) in enumerate(path):
            node = (r, c)
            if node not in G:
                G.add_node(node, pos=(c, r))  # NetworkX expects (x, y)
            if idx == 0 and prev_node_id is not None:
                G.add_edge(prev_node_id, node)
            elif idx > 0:
                G.add_edge(prev_node, node)
            prev_node = node

        # Queue up unvisited neighbors at the end of the path
        neighbors = get_8_connected_neighbors(im, visited, end_row, end_col)
        for neighbor in neighbors:
            queue.append((prev_node, neighbor))

    return G


def direct_graph_from_direction_map(G: nx.Graph, direction_map: np.ndarray) -> nx.DiGraph:


    # Shifting direction map
    if np.min(direction_map) >= 0:
        direction_map = direction_map.astype(np.float64)
        direction_map = direction_map - 127

    DirG = nx.DiGraph()

    for node in G.nodes():
        DirG.add_node(node, pos=G.nodes[node]["pos"])

    for u, v in G.edges():
        u_row, u_col = u
        v_row, v_col = v

        u_vec = direction_map[u_row, u_col]
        v_vec = direction_map[v_row, v_col]

        avg_vec = (u_vec + v_vec) / 2.0

        edge_vec = np.array([v[1] - u[1], v[0] - u[0]], dtype=np.float64)

        dot_product = np.dot(avg_vec, edge_vec)

        if dot_product >= 0:
            DirG.add_edge(u, v)
        else:
            DirG.add_edge(v, u)

    return DirG

def visualize_lanes_and_links(gdf, ax=None, lane_color='blue', link_color='red', show_labels=True, save_path=None, image_name=None):
    import matplotlib.lines as mlines
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    # Split by type
    gdf_lanes = gdf[gdf['type'] == 'lane']
    gdf_links = gdf[gdf['type'] == 'link']
    # Plot lanes
    if not gdf_lanes.empty:
        gdf_lanes.plot(ax=ax, color=lane_color, linewidth=2, alpha=0.8, label="lane")
        for _, row in gdf_lanes.iterrows():
            coords = list(row.geometry.coords)
            ax.plot(*coords[0], marker='o', color=lane_color, markersize=4)  # start
            ax.plot(*coords[-1], marker='x', color=lane_color, markersize=4)  # end
    # Plot links
    if not gdf_links.empty:
        gdf_links.plot(ax=ax, color=link_color, linewidth=2, alpha=0.8, label="link")
        for _, row in gdf_links.iterrows():
            coords = list(row.geometry.coords)
            ax.plot(*coords[0], marker='o', color=link_color, markersize=4)  # start
            ax.plot(*coords[-1], marker='x', color=link_color, markersize=4)  # end
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title("Visualized Lanes and Links")
    if show_labels:
        lane_line = mlines.Line2D([], [], color=lane_color, label='lane', linewidth=2)
        link_line = mlines.Line2D([], [], color=link_color, label='link', linewidth=2)
        ax.legend(handles=[lane_line, link_line])
    if save_path:
        plt.savefig(os.path.join(save_path, f"lane_and_links_{image_name}.png"))
    return ax

def draw_directed_graph(
    G: nx.DiGraph, save_path: Optional[Path] = None, image_name: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw a directed graph with colored nodes and edges, generating lane mask and normal map.

    Args:
        G (nx.DiGraph): Directed graph where node names are (row, col) coordinates.
        save_path (Optional[Path], optional): Directory path to save output images. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing the lane mask image and normal map image.
    """
    # Initialize output images: lane mask (white edges on black) and normal map (directional vectors)
    out_lane = np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8)
    out_normal = np.full((WINDOW_SIZE, WINDOW_SIZE, 3), 127, dtype=np.uint8)

    # Draw edges as white lines on lane mask and directional colors on normal map
    for u, v in G.edges():
        ux, uy = G.nodes[u].get("pos", (0, 0))  # Convert (row, col) to (x, y)
        vx, vy = G.nodes[v].get("pos", (0, 0))

        # Calculate normalized direction vector for normal map coloring
        dx = vx - ux
        dy = vy - uy
        length = np.sqrt(float(dx**2 + dy**2)) + 1e-6  # Avoid division by zero
        dx /= length
        dy /= length
        # Map direction to color: [-1,1] -> [0,254] + 127 offset
        color = (127 + int(dx * 127), 127 + int(dy * 127), 127)

        # Draw edge lines
        cv2.line(out_lane, (ux, uy), (vx, vy), (255, 255, 255), 2)
        cv2.line(out_normal, (ux, uy), (vx, vy), color, 2)

    # Draw nodes with color coding based on type and annotate with degree information
    for node in G.nodes():
        node_type = G.nodes[node].get("type", "unknown")
        x, y = G.nodes[node].get("pos", (0, 0))  # Convert (row, col) to (x, y)
        in_deg, out_deg = G.in_degree(node), G.out_degree(node)

        # Color coding for different node types
        if node_type == "in":
            color = (0, 255, 0)     # Green for entry points
        elif node_type == "out":
            color = (0, 0, 255)     # Red for exit points
        elif node_type == "lane":
            color = (0, 255, 255)   # Cyan for regular lane nodes
        elif node_type == "split":
            color = (255, 0, 255)   # Magenta for split junctions
        elif node_type == "merge":
            color = (255, 255, 0)   # Yellow for merge junctions
        else:
            color = (200, 200, 200) # Gray for unknown types

        # Draw node as filled circle
        cv2.circle(out_lane, (x, y), 4, color, -1)
        cv2.circle(out_normal, (x, y), 4, color, -1)

        # Add text annotation showing in-degree, out-degree, and node type
        text = f"{in_deg},{out_deg},{node_type}"
        cv2.putText(
            out_lane, text, (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA
        )

    # Save images to disk if path is provided
    if save_path and image_name:
        cv2.imwrite(os.path.join(save_path, f"lane_{image_name}.png"), out_lane)
        cv2.imwrite(os.path.join(save_path, f"normal_{image_name}.png"), out_normal)

    return out_lane, out_normal





