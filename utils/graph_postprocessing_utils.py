"""
Post-processing utilities for lane graph refinement and optimization.

This module provides functions for cleaning and refining lane graphs
extracted from segmentation outputs, including:
- Removing isolated components and short spurs
- Connecting nearby dead ends  
- Graph downsampling
- Graph connectivity operations
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Set


def refine_lane_graph(
    lane_graph: nx.DiGraph,
    isolated_threshold: float = 150.0,
    spur_threshold: float = 60.0,
    merge_threshold: float = 10.0,
) -> nx.DiGraph:
    """
    Refine a lane graph by removing isolated components, short spurs,
    and merging very close nodes.
    
    Args:
        lane_graph: Input lane graph
        isolated_threshold: Minimum total length for connected components to keep
        spur_threshold: Maximum length of spurs to remove
        merge_threshold: Distance threshold below which nodes are merged
    
    Returns:
        Refined graph with isolated components removed, spurs removed,
        and close nodes merged.
    """
    # --- Phase 1: remove small components & spurs ---
    component_assignments = {}
    for comp_id, component_nodes in enumerate(nx.weakly_connected_components(lane_graph)):
        for node in component_nodes:
            component_assignments[node] = comp_id
    component_stats = _calculate_component_statistics(lane_graph, component_assignments)

    spur_nodes = _identify_spur_nodes(lane_graph, spur_threshold)

    def should_remove_node(node: Tuple[int, int]) -> bool:
        component_id = component_assignments[node]
        node_count, total_length = component_stats[component_id]
        return (
            node_count <= 1
            or total_length <= isolated_threshold
            or node in spur_nodes
        )

    refined_graph = nx.DiGraph()
    for node in lane_graph.nodes():
        if should_remove_node(node):
            continue
        refined_graph.add_node(node, **lane_graph.nodes[node])
        for neighbor in lane_graph.successors(node):
            if not should_remove_node(neighbor):
                refined_graph.add_edge(node, neighbor)

    # --- Phase 2: merge very close nodes ---
    nodes = list(refined_graph.nodes())
    merged_map = {}  # old_node -> new_node
    for i, n1 in enumerate(nodes):
        if n1 in merged_map:  # already merged
            continue
        pos1 = np.array(refined_graph.nodes[n1].get("pos", (0, 0)))
        for n2 in nodes[i + 1 :]:
            if n2 in merged_map:
                continue
            pos2 = np.array(refined_graph.nodes[n2].get("pos", (0, 0)))
            dist = np.linalg.norm(pos1 - pos2)
            if dist < merge_threshold:
                # Merge n2 into n1
                merged_map[n2] = n1

    # Build final merged graph
    final_graph = nx.DiGraph()
    for node in refined_graph.nodes():
        root = merged_map.get(node, node)
        if root not in final_graph:
            final_graph.add_node(root, **refined_graph.nodes[root])

    for u, v in refined_graph.edges():
        u_root = merged_map.get(u, u)
        v_root = merged_map.get(v, v)
        if u_root != v_root:
            final_graph.add_edge(u_root, v_root)

    return final_graph

def _calculate_component_statistics(
    graph: nx.DiGraph,
    component_assignments: Dict[int, int]
) -> Dict[int, Tuple[int, float]]:
    """
    Calculate node count and total edge length for each component in a DiGraph.

    Args:
        graph: A NetworkX directed graph with node positions in node['pos'] = (x, y).
        component_assignments: Mapping of node_id -> component ID.

    Returns:
        Dict[component_id, (node_count, total_edge_length)].
    """
    component_stats: Dict[int, Tuple[int, float]] = {}
    visited_edges: Set[frozenset] = set()

    for node, comp_id in component_assignments.items():
        if comp_id not in component_stats:
            component_stats[comp_id] = (0, 0.0)

        node_count, total_length = component_stats[comp_id]
        node_count += 1

        for neighbor in graph.neighbors(node):
            edge_key = frozenset((node, neighbor))
            if edge_key in visited_edges:
                continue

            edge_length = _calculate_euclidean_distance(
                graph.nodes[node], graph.nodes[neighbor]
            )

            total_length += edge_length
            visited_edges.add(edge_key)

        component_stats[comp_id] = (node_count, total_length)

    return component_stats


def _identify_spur_nodes(
    graph: nx.DiGraph,
    spur_threshold: float
) -> Set[Tuple[int, int]]:
    """
    Identify spur nodes in a DiGraph that should be removed based on:
    - Having exactly one outgoing neighbor.
    - That neighbor having high degree (>= 3).
    - Distance to neighbor < spur_threshold.
    """
    spur_nodes = set()

    for node in graph.nodes():
        neighbors = list(graph.successors(node))

        if len(neighbors) == 1:
            neighbor = neighbors[0]

            # Total degree: in + out
            if graph.degree(neighbor) >= 3:
                spur_length = _calculate_euclidean_distance(graph.nodes[node], graph.nodes[neighbor])

                if spur_length < spur_threshold:
                    spur_nodes.add(node)

    return spur_nodes


def _calculate_euclidean_distance(node_data1, node_data2):
    x1, y1 = node_data1.get("pos", (0, 0))
    x2, y2 = node_data2.get("pos", (0, 0))
    return ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5


def add_bidirectional_edge(
    graph: Dict[Tuple[int, int], List[Tuple[int, int]]],
    node1: Tuple[int, int],
    node2: Tuple[int, int]
) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    """
    Add a bidirectional edge between two nodes in the graph.
    
    Args:
        graph: The graph to modify
        node1: First node coordinate
        node2: Second node coordinate
        
    Returns:
        Modified graph with new edge added
    """
    if node1 == node2:
        return graph
    

    if node1 in graph:
        if node2 not in graph[node1]:
            graph[node1].append(node2)
    else:
        graph[node1] = [node2]
    

    if node2 in graph:
        if node1 not in graph[node2]:
            graph[node2].append(node1)
    else:
        graph[node2] = [node1]
    
    return graph


def downsample_graph(
    graph: Dict[Tuple[int, int], List[Tuple[int, int]]],
    downsample_rate: int = 2
) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    """
    Downsample graph by quantizing node coordinates to a coarser grid.
    
    Args:
        graph: Input graph to downsample
        downsample_rate: Factor by which to downsample coordinates
        
    Returns:
        Downsampled graph with quantized coordinates
    """
    downsampled_graph = {}
    
    for node, neighbors in graph.items():

        quantized_node = (
            (int(node[0]) // downsample_rate) * downsample_rate,
            (int(node[1]) // downsample_rate) * downsample_rate
        )
        

        for neighbor in neighbors:
            quantized_neighbor = (
                (int(neighbor[0]) // downsample_rate) * downsample_rate,
                (int(neighbor[1]) // downsample_rate) * downsample_rate
            )
            
            downsampled_graph = add_bidirectional_edge(
                downsampled_graph, quantized_node, quantized_neighbor
            )
    
    return downsampled_graph

def connect_nearby_dead_ends(
    graph: nx.DiGraph,
    connection_threshold: float = 30.0
) -> nx.DiGraph:
    """
    Connect nearby 'out' nodes to 'in' nodes in a DiGraph to reduce fragmentation.

    Args:
        graph: The input directed graph. Nodes must have a 'type' attribute (e.g., 'in', 'out').
        connection_threshold: Max Euclidean distance to connect an 'out' to an 'in'.

    Returns:
        The modified graph with new edges added between close 'out' and 'in' nodes.
    """
    out_nodes = [n for n, d in graph.nodes(data=True) if d.get("type") == "out"]
    in_nodes  = [n for n, d in graph.nodes(data=True) if d.get("type") == "in"]

    for out_node in out_nodes:
        closest_in = None
        min_distance = connection_threshold

        for in_node in in_nodes:
            # skip if already connected
            if graph.has_edge(out_node, in_node):
                continue
            dist = _calculate_euclidean_distance(graph.nodes[out_node], graph.nodes[in_node])
            
            if dist < min_distance:
                min_distance = dist
                closest_in = in_node

        # Add directed edge if found
        if closest_in is not None:
            graph.add_edge(out_node, closest_in)

    return graph




def refine_lane_graph_with_curves(
    G_dir: nx.DiGraph,
    simplify_epsilon: float = 2.0,
) -> nx.DiGraph:
    """
    Refine a directed lane graph:
    - Split subgraphs at split/merge nodes into lane segments
    - Each segment = (junction/in) → ... → (junction/out)
    - Simplify with Douglas-Peucker
    - Preserve node attributes
    - Use infer_majority_direction for consistent orientation
    """
    refined_graph = nx.DiGraph()
    junction_nodes = {n for n, d in G_dir.nodes(data=True) if d.get("type") in {"split", "merge"}}

    def process_segment(segment):
        """Simplify and insert segment into refined graph."""
        if len(segment) < 2:
            return
        direction = infer_majority_direction(G_dir, segment)
        simplified = douglas_peucker_int(segment, epsilon=simplify_epsilon)

        for node in simplified:
            if node not in refined_graph:
                refined_graph.add_node(node, **G_dir.nodes[node])
        for u, v in zip(simplified[:-1], simplified[1:]):
            if direction == "forward":
                refined_graph.add_edge(u, v)
            else:
                refined_graph.add_edge(v, u)

    # Walk forward from every junction and "in" node
    start_nodes = [n for n in G_dir.nodes if G_dir.out_degree(n) > 0 and 
                   (n in junction_nodes or G_dir.nodes[n].get("type") == "in")]

    for start in start_nodes:
        for succ in G_dir.successors(start):
            path = [start, succ]
            current = succ
            while current not in junction_nodes and G_dir.out_degree(current) == 1:
                nxt = next(G_dir.successors(current))
                path.append(nxt)
                current = nxt
            process_segment(path)

    # Walk backward from every junction and "out" node (optional if symmetry needed)
    # Could be skipped if only forward segments are required

    return refined_graph



def extract_ordered_path(G: nx.Graph) -> List[Tuple[int, int]]:
    ends = [n for n in G.nodes if G.degree(n) == 1]
    if not ends:
        return list(nx.dfs_preorder_nodes(G, source=list(G.nodes)[0]))

    max_path = []
    for start in ends:
        for end in ends:
            if start == end:
                continue
            try:
                path = nx.shortest_path(G, start, end)
                if len(path) > len(max_path):
                    max_path = path
            except nx.NetworkXNoPath:
                continue
    return max_path


def infer_majority_direction(
    G_dir: nx.DiGraph,
    path: List[Tuple[int, int]]
) -> str:
    """
    Infer majority direction of a path based on edge lengths:
    - 'forward' if most of the path aligns with (u → v) edges in G_dir
    - 'reverse' if majority of aligned edges are (v → u)
    """
    forward_length = 0.0
    reverse_length = 0.0

    for u, v in zip(path[:-1], path[1:]):
        d = _calculate_euclidean_distance(G_dir.nodes[u], G_dir.nodes[v])
        if G_dir.has_edge(u, v):
            forward_length += d
        elif G_dir.has_edge(v, u):
            reverse_length += d

    return 'forward' if forward_length >= reverse_length else 'reverse'


def douglas_peucker_int(points: List[Tuple[int, int]], epsilon: float) -> List[Tuple[int, int]]:
    if len(points) <= 2:
        return points

    max_dist = 0
    index = 1
    for i in range(1, len(points) - 1):
        d = point_to_line_distance(points[i], points[0], points[-1])
        if d > max_dist:
            max_dist = d
            index = i

    if max_dist <= epsilon:
        return [points[0], points[-1]]

    left = douglas_peucker_int(points[:index + 1], epsilon)
    right = douglas_peucker_int(points[index:], epsilon)
    return left[:-1] + right


def point_to_line_distance(p, a, b):
    a, b, p = np.array(a), np.array(b), np.array(p)
    ab = b - a
    ap = p - a
    if np.all(ab == 0):
        return np.linalg.norm(ap)
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = np.clip(t, 0, 1)
    proj = a + t * ab
    return np.linalg.norm(p - proj)


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