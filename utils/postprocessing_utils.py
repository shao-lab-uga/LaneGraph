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
    three_edge_loop_threshold: float = 70.0
) -> nx.DiGraph:
    """
    Refine a lane graph by removing isolated components and short spurs.
    
    Args:
        lane_graph: Input lane graph
        isolated_threshold: Minimum total length for connected components to keep
        spur_threshold: Maximum length of spurs to remove
        three_edge_loop_threshold: Threshold for three-edge loop detection (currently unused)
        
    Returns:
        Refined graph with isolated components and spurs removed
    """
    # Find connected components using BFS
    component_assignments = {}
    for comp_id, component_nodes in enumerate(nx.weakly_connected_components(lane_graph)):
        for node in component_nodes:
            component_assignments[node] = comp_id
    component_stats = _calculate_component_statistics(lane_graph, component_assignments)

    # Identify short spurs for removal
    spur_nodes = _identify_spur_nodes(lane_graph, spur_threshold)
    
    # Filter nodes based on component size and spur status
    def should_remove_node(node: Tuple[int, int]) -> bool:
        component_id = component_assignments[node]
        node_count, total_length = component_stats[component_id]
        
        return (
            node_count <= 1 or 
            total_length <= isolated_threshold or
            node in spur_nodes
        )
    

    refined_graph = nx.DiGraph()
    nodes_removed = 0
    
    for node in lane_graph.nodes():
        if should_remove_node(node):
            nodes_removed += 1
            continue

        # Copy node to new graph
        refined_graph.add_node(node)

        # Filter outgoing neighbors
        for neighbor in lane_graph.successors(node):
            if not should_remove_node(neighbor):
                refined_graph.add_edge(node, neighbor)
    
    return refined_graph

def _calculate_component_statistics(
    graph: nx.DiGraph,
    component_assignments: Dict[Tuple[int, int], int]
) -> Dict[int, Tuple[int, float]]:
    """
    Calculate node count and total edge length for each component in a DiGraph.
    
    Args:
        graph: A NetworkX directed graph.
        component_assignments: Mapping of node -> component ID.
    
    Returns:
        Dict[component_id, (node_count, total_edge_length)].
    """
    component_stats = {}

    for node, component_id in component_assignments.items():
        if component_id not in component_stats:
            component_stats[component_id] = (0, 0.0)

        node_count, total_length = component_stats[component_id]
        node_count += 1  # Increment node count

        # Add edge lengths for all neighbors
        for neighbor in graph.neighbors(node):
            edge_length = _calculate_euclidean_distance(node, neighbor)
            total_length += edge_length / 2.0  # Divide by 2 to avoid double counting

        component_stats[component_id] = (node_count, total_length)

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
                spur_length = _calculate_euclidean_distance(node, neighbor)

                if spur_length < spur_threshold:
                    spur_nodes.add(node)

    return spur_nodes


def _calculate_euclidean_distance(
    point1: Tuple[int, int], 
    point2: Tuple[int, int]
) -> float:
    dx = point1[0] - point2[0]
    dy = point1[1] - point2[1]
    return np.sqrt(dx * dx + dy * dy)


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
    Connect nearby dead-end nodes in a DiGraph to reduce fragmentation.

    Args:
        graph: The input directed graph.
        connection_threshold: Max Euclidean distance to connect two dead ends.

    Returns:
        The modified graph with new edges added between close dead ends.
    """
    # Find dead-end nodes: nodes with exactly one outgoing edge
    dead_end_nodes = [node for node in graph.nodes if graph.out_degree(node) == 1]

    for dead_end1 in dead_end_nodes:
        closest_dead_end = None
        min_distance = connection_threshold

        for dead_end2 in dead_end_nodes:
            if dead_end1 == dead_end2:
                continue

            distance = _calculate_euclidean_distance(dead_end1, dead_end2)
            if distance < min_distance:
                min_distance = distance
                closest_dead_end = dead_end2

        # If a close enough dead-end node was found, connect them bidirectionally
        if closest_dead_end is not None and not graph.has_edge(dead_end1, closest_dead_end):
            graph.add_edge(dead_end1, closest_dead_end)
            graph.add_edge(closest_dead_end, dead_end1)

    return graph



def refine_lane_graph_with_curves(
    G_dir: nx.DiGraph,
    simplify_epsilon: float = 2.0,
) -> nx.DiGraph:
    """
    Refine a directed lane graph:
    - Extract paths per connected component (undirected)
    - Simplify using Douglas-Peucker
    - Output a clean DiGraph with consistent direction
    """
    G_undirected = G_dir.to_undirected()
    components = list(nx.connected_components(G_undirected))

    refined_graph = nx.DiGraph()

    for comp_nodes in components:
        subgraph_u = G_undirected.subgraph(comp_nodes)
        path = extract_ordered_path(subgraph_u)
        if len(path) < 2:
            continue

        direction = infer_majority_direction(G_dir, path)
        simplified = douglas_peucker_int(path, epsilon=simplify_epsilon)


        for u, v in zip(simplified[:-1], simplified[1:]):
            if direction == 'forward':
                refined_graph.add_edge(u, v)
            else:
                refined_graph.add_edge(v, u)

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
        d = _calculate_euclidean_distance(u, v)
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


