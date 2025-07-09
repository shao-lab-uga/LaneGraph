from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import networkx as nx
import numpy as np

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


def find_first_unvisited_white_pixel(im, visited):
    """Find the next unvisited white pixel in the image."""
    white_pixels = np.where((im > 0) & (~visited))
    if white_pixels[0].size == 0:
        return None
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
    im: np.ndarray, threshold: int = 50, epsilon: float = 2.0
) -> nx.Graph:
    """
    Extracts a graph from a thinned binary image using 8-connectivity.
    Simplifies paths with RDP. Node names are array references of (row, column)
    while node data "pos" is (x, y) location data
    """
    visited = np.zeros_like(im, dtype=bool)
    G = nx.Graph()
    queue = []

    while True:
        if not queue:
            start = find_first_unvisited_white_pixel(im, visited)
            if start is None:
                break  # No more unvisited white pixels
            queue.append((None, start))  # (previous node ID, (row, col))

        prev_node_id, (row, col) = queue.pop()
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


def direct_graph_from_vector_map(G: nx.Graph, vector_map: np.ndarray) -> nx.DiGraph:
    """show_arr = np.full((WINDOW_SIZE, WINDOW_SIZE, 3), 127, dtype=vector_map.dtype)
    show_arr[:, :, 0:2] = vector_map
    px.imshow(show_arr).show()"""

    # Shifting vector map
    if np.min(vector_map) >= 0:
        vector_map = vector_map.astype(np.float64)
        vector_map = vector_map - 127

    DirG = nx.DiGraph()

    for node in G.nodes():
        DirG.add_node(node, pos=G.nodes[node]["pos"])

    for u, v in G.edges():
        u_row, u_col = u
        v_row, v_col = v

        u_vec = vector_map[u_row, u_col]
        v_vec = vector_map[v_row, v_col]

        avg_vec = (u_vec + v_vec) / 2.0

        edge_vec = np.array([v[1] - u[1], v[0] - u[0]], dtype=np.float64)

        dot_product = np.dot(avg_vec, edge_vec)

        if dot_product >= 0:
            DirG.add_edge(u, v)
        else:
            DirG.add_edge(v, u)
    return DirG


"""
center = (995, 1640)
test_lane_img = cv2.imread(
    "/home/lab/development/lab/final/LaneGraph/app/processing/test/lane.png",
    cv2.IMREAD_GRAYSCALE,
)
test_lane_img = test_lane_img > 100
test_lane_img = test_lane_img[
    center[1] - 320 : center[1] + 320, center[0] - 320 : center[0] + 320
].copy()

thin_img = morphology.thin(test_lane_img)


G = extract_graph_from_image(thin_img, epsilon=5.0)
thin_img = thin_img.astype("uint8") * 255
for r, c in G.nodes():
    cv2.circle(thin_img, (c, r), 5, (255, 255, 255), -1)
# px.imshow(thin_img).show()
test_dir_img = cv2.imread(
    "/Users/agraham/development/lab/clean/LaneGraph/msc_dataset/dataset_unpacked/normal_34.jpg",
)
vector_map = test_dir_img[:, :, 0:2]
vector_map = vector_map[
    center[1] - 320 : center[1] + 320, center[0] - 320 : center[0] + 320
].copy()
res = direct_graph_from_vector_map(G, vector_map)"""


def draw_inputs(
    G: nx.DiGraph, save_path: Optional[Path] = None
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

    if save_path:
        cv2.imwrite(str(save_path / "lane.png"), out_lane)
        save_normal_arr = np.full(
            (WINDOW_SIZE, WINDOW_SIZE, 3), 127, dtype=out_normal.dtype
        )
        save_normal_arr[:, :, 0:2] = out_normal
        cv2.imwrite(str(save_path / "normal.png"), save_normal_arr)

    return out_lane, out_normal


def annotate_node_types(
    G: nx.DiGraph, center: Tuple[int, int] = (320, 320)
) -> nx.DiGraph:
    """Annotates nodes in graph based off distance to center point.

    Annotates nodes in each connected component with the "type" attibute. This is done with the following logic for each node:
    1. All nodes located within 120 pixels of the image edge will be type "end".
    2. The furthest degree 1 node of each connected component will be of type "end".
    3. Remaining degree 1 nodes of each connected component will be of type "in" or "out".
        - If the nearest edge is pointed to the node, the type will be "in".
        - If the nearest edge is pointed away the node, the type will be "out".
    4. All remaining nodes will be of type "lane".

    Args:
        G (nx.DiGraph): Directed graph of intersection lanes.
        center (Tuple[int,int]): Intersection center coordinate tuple of (row, col). Defaults to (320,320).

    Returns:
        nx.DiGraph: Directed graph of intersection lanes with typed nodes.
    """
    EDGE_MARGIN = 120

    def near_edge(node_coord: Tuple[int, int]) -> bool:
        """Checks if a coordinate is within the edge margin."""
        row, col = node_coord
        res = row <= (
            row <= EDGE_MARGIN
            or row >= WINDOW_SIZE - EDGE_MARGIN
            or col <= EDGE_MARGIN
            or col >= WINDOW_SIZE - EDGE_MARGIN
        )
        return res

    for cc in nx.weakly_connected_components(G):
        cc_subgraph = G.subgraph(cc)
        deg_one_nodes = [
            n
            for n in cc_subgraph.nodes
            if (cc_subgraph.in_degree(n) + cc_subgraph.out_degree(n)) == 1
        ]

        furthest_node = None
        max_dist = -1
        for n in deg_one_nodes:
            dist = distance(n, center)
            if dist > max_dist:
                max_dist = dist
                furthest_node = n

        for node in cc_subgraph.nodes:
            if near_edge(node):
                G.nodes[node]["type"] = "end"
                continue

            deg = cc_subgraph.in_degree(node) + cc_subgraph.out_degree(node)
            if deg != 1:
                G.nodes[node]["type"] = "lane"
            else:
                if node == furthest_node:
                    G.nodes[node]["type"] = "end"
                else:
                    indeg = cc_subgraph.in_degree(node)
                    outdeg = cc_subgraph.out_degree(node)

                    if indeg == 1:
                        G.nodes[node]["type"] = "in"
                    elif outdeg == 1:
                        G.nodes[node]["type"] = "out"
                    else:
                        G.nodes[node]["type"] = "lane"

    return G


def get_node_types(G: nx.DiGraph) -> Dict[str, List[Tuple[int, int]]]:
    node_types = defaultdict(list)
    for node, data in G.nodes(data=True):
        node_type = data.get("type")
        if node_type is None:
            raise ValueError(f"Node {node} does not have an assigned type.")
        node_types[node_type].append(node)
    return node_types


"""
a, b = draw_inputs(
    res,
    save_path=Path(
        "/Users/agraham/development/lab/clean/LaneGraph/app/processing/test_im"
    ),
)

show_arr = np.full((WINDOW_SIZE, WINDOW_SIZE, 3), 127, dtype=b.dtype)
show_arr[:, :, 0:2] = b
px.imshow(show_arr).show()
px.imshow(np.squeeze(a)).show()
"""
