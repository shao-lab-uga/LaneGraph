import os
import cv2
import networkx as nx
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, DefaultDict

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


def draw_inputs(
    G: nx.DiGraph, save_path: Optional[Path] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates inputs for turning lane extraction processes. Creates lane mask and normal map using directed graph.

    Args:
        G (nx.DiGraph): Directed graph where node names are (row, col)
        save_path (Optional[Path], optional): Directory path to save outputs to. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: lane mask and normal map
    """
    out_lane = np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8)
    out_normal = np.full((WINDOW_SIZE, WINDOW_SIZE, 3), 127, dtype=np.uint8)

    # draw edges
    for u, v in G.edges():
        ux, uy = u[1], u[0]
        vx, vy = v[1], v[0]

        dx = vx - ux
        dy = vy - uy
        length = np.sqrt(float(dx**2 + dy**2)) + 1e-6
        dx /= length
        dy /= length
        color = (127 + int(dx * 127), 127 + int(dy * 127), 127)

        cv2.line(out_lane, (ux, uy), (vx, vy), (255, 255, 255), 2)
        cv2.line(out_normal, (ux, uy), (vx, vy), color, 2)

    # draw nodes and annotate with type + degree
    for node in G.nodes():
        node_type = G.nodes[node].get("type", "unknown")
        x, y = node[1], node[0]
        in_deg, out_deg = G.in_degree(node), G.out_degree(node)

        # choose color by type
        if node_type == "in":
            color = (0, 255, 0)     # Green
        elif node_type == "out":
            color = (0, 0, 255)     # Red
        elif node_type == "lane":
            color = (0, 255, 255)   # Cyan
        elif node_type == "split":
            color = (255, 0, 255)   # Magenta
        elif node_type == "merge":
            color = (255, 255, 0)   # Yellow
        else:
            color = (200, 200, 200) # Gray fallback

        # draw node
        cv2.circle(out_lane, (x, y), 4, color, -1)
        cv2.circle(out_normal, (x, y), 4, color, -1)

        # annotate in/out degree
        text = f"{in_deg},{out_deg},{node_type}"
        cv2.putText(
            out_lane, text, (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA
        )

    if save_path:
        cv2.imwrite(os.path.join(save_path, "lane.png"), out_lane)
        cv2.imwrite(os.path.join(save_path, "normal.png"), out_normal)

    return out_lane, out_normal


def draw_output(
    G: nx.DiGraph, save_path: Optional[Path] = None, image_name: Optional[str] = None, draw_nodes: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates inputs for turning lane extraction processes. Creates lane mask and normal map using directed graph.

    Args:
        G (nx.DiGraph): Directed graph where node names are (row, col)
        save_path (Optional[Path], optional): Directory path to save outputs to. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: _description_
    """
    out_lane = np.zeros((WINDOW_SIZE, WINDOW_SIZE, 1))
    out_normal = np.full((WINDOW_SIZE, WINDOW_SIZE, 2), 127, dtype=np.uint8)

    for u, v in G.edges():
        ux = u[1]
        uy = u[0]
        vx = v[1]
        vy = v[0]

    
        dx = vx - ux
        dy = vy - uy

        length = np.sqrt(float((dx**2 + dy**2))) + 0.001
        dx /= length
        dy /= length
        color = (127 + int(dx * 127), 127 + int(dy * 127), 127)

        cv2.line(out_lane, (ux, uy), (vx, vy), (255, 255, 255), 5)
        cv2.line(out_normal, (ux, uy), (vx, vy), color, 5)
        # draw the nodes

    for node in G.nodes():
        x, y = node[1], node[0]
        node_type = G.nodes[node].get("type", "unknown")
        if draw_nodes:
            if node_type == "end":
                cv2.circle(out_lane, (x, y), 5, (255, 0, 0), -1)
                cv2.circle(out_normal, (x, y), 5, (255, 0, 0), -1)
            elif node_type == "in":
                cv2.circle(out_lane, (x, y), 5, (0, 255, 0), -1)
                cv2.circle(out_normal, (x, y), 5, (0, 255, 0), -1)
            elif node_type == "out":
                cv2.circle(out_lane, (x, y), 5, (0, 0, 255), -1)
                cv2.circle(out_normal, (x, y), 5, (0, 0, 255), -1)
            elif node_type == "lane":
                cv2.circle(out_lane, (x, y), 5, (0, 255, 255), -1)
                cv2.circle(out_normal, (x, y), 5, (0, 255, 255), -1)
            elif node_type == "link":
                cv2.circle(out_lane, (x, y), 5, (255, 255, 0), -1)
                cv2.circle(out_normal, (x, y), 5, (255, 255, 0), -1)

    if save_path and image_name:
        cv2.imwrite(os.path.join(save_path, f"{image_name}.png"), out_lane)
        save_normal_arr = np.full(
            (WINDOW_SIZE, WINDOW_SIZE, 3), 127, dtype=out_normal.dtype
        )
        save_normal_arr[:, :, 0:2] = out_normal
        cv2.imwrite(os.path.join(save_path, f"{image_name}_normal.png"), save_normal_arr)

    return out_lane, out_normal

def annotate_node_types(G: nx.DiGraph) -> nx.DiGraph:
    """
    Annotate node types in a directed lane graph:
    - "in":  entry boundary (in_degree=0, out_degree=1)
    - "out": exit boundary (in_degree=1, out_degree=0)
    - "split": junction with more outgoing than incoming
    - "merge": junction with more incoming than outgoing
    - "lane": intermediate lane node
    """
    for node in G.nodes():
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)
        total_degree = in_degree + out_degree

        if total_degree >= 3:
            if in_degree < out_degree:
                G.nodes[node]["type"] = "split"
            elif in_degree > out_degree:
                G.nodes[node]["type"] = "merge"
            else:
                G.nodes[node]["type"] = "lane"
        elif total_degree == 1:  # boundary
            if in_degree == 0 and out_degree == 1:
                G.nodes[node]["type"] = "in"
            elif out_degree == 0 and in_degree == 1:
                G.nodes[node]["type"] = "out"
            else:
                G.nodes[node]["type"] = "lane"
        elif total_degree == 2 and in_degree == 1 and out_degree == 1:
            G.nodes[node]["type"] = "lane"
        else:
            G.nodes[node]["type"] = "lane"

    return G


def get_node_types(G: nx.DiGraph) -> Dict[str, List[Tuple[int, int]]]:
    node_types = DefaultDict(list)
    for node, data in G.nodes(data=True):
        node_type = data.get("type")
        if node_type is None:
            raise ValueError(f"Node {node} does not have an assigned type.")
        node_types[node_type].append(node)
    return node_types

