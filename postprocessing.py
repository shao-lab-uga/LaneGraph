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
from typing import Dict, List, Tuple, Set


def refine_lane_graph(
    lane_graph: Dict[Tuple[int, int], List[Tuple[int, int]]],
    isolated_threshold: float = 150.0,
    spur_threshold: float = 60.0,
    three_edge_loop_threshold: float = 70.0
) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    """
    Refine a lane graph by removing isolated components and short spurs.
    
    Args:
        lane_graph: Dictionary mapping node coordinates to list of neighbor coordinates
        isolated_threshold: Minimum total length for connected components to keep
        spur_threshold: Maximum length of spurs to remove
        three_edge_loop_threshold: Threshold for three-edge loop detection (currently unused)
        
    Returns:
        Refined graph with isolated components and spurs removed
    """
    # Find connected components using BFS
    connected_components = _find_connected_components(lane_graph)
    component_stats = _calculate_component_statistics(lane_graph, connected_components)
    
    # Identify short spurs for removal
    spur_nodes = _identify_spur_nodes(lane_graph, spur_threshold)
    
    # Filter nodes based on component size and spur status
    def should_remove_node(node: Tuple[int, int]) -> bool:
        component_id = connected_components[node]
        node_count, total_length = component_stats[component_id]
        
        return (
            node_count <= 1 or 
            total_length <= isolated_threshold or
            node in spur_nodes
        )
    

    refined_graph = {}
    nodes_removed = 0
    
    for node, neighbors in lane_graph.items():
        if should_remove_node(node):
            nodes_removed += 1
            continue
            

        filtered_neighbors = [
            neighbor for neighbor in neighbors 
            if not should_remove_node(neighbor)
        ]
        
        if filtered_neighbors:
            refined_graph[node] = filtered_neighbors
    
    return refined_graph


def _find_connected_components(
    graph: Dict[Tuple[int, int], List[Tuple[int, int]]]
) -> Dict[Tuple[int, int], int]:
    """Find connected components in the graph using BFS."""
    component_assignments = {}
    component_id = 0
    
    for node in graph.keys():
        if node not in component_assignments:

            queue = [node]
            
            while queue:
                current_node = queue.pop(0)
                
                if current_node not in component_assignments:
                    component_assignments[current_node] = component_id
                    

                    for neighbor in graph[current_node]:
                        if neighbor not in component_assignments:
                            queue.append(neighbor)
            
            component_id += 1
    
    return component_assignments


def _calculate_component_statistics(
    graph: Dict[Tuple[int, int], List[Tuple[int, int]]],
    component_assignments: Dict[Tuple[int, int], int]
) -> Dict[int, Tuple[int, float]]:
    """Calculate node count and total edge length for each component."""
    component_stats = {}
    
    for node, component_id in component_assignments.items():
        if component_id not in component_stats:
            component_stats[component_id] = (0, 0.0)
        
        node_count, total_length = component_stats[component_id]
        component_stats[component_id] = (node_count + 1, total_length)
        

        for neighbor in graph[node]:
            edge_length = _calculate_euclidean_distance(node, neighbor)
            total_length += edge_length / 2.0
            
        component_stats[component_id] = (node_count + 1, total_length)
    
    return component_stats


def _identify_spur_nodes(
    graph: Dict[Tuple[int, int], List[Tuple[int, int]]],
    spur_threshold: float
) -> Set[Tuple[int, int]]:
    """Identify short spur nodes that should be removed."""
    spur_nodes = set()
    
    for node, neighbors in graph.items():

        if len(neighbors) == 1:
            neighbor = neighbors[0]
            

            if len(graph[neighbor]) >= 3:
                spur_length = _calculate_euclidean_distance(node, neighbor)
                
                if spur_length < spur_threshold:
                    spur_nodes.add(node)
    
    return spur_nodes


def _calculate_euclidean_distance(
    point1: Tuple[int, int], 
    point2: Tuple[int, int]
) -> float:
    """Calculate Euclidean distance between two points."""
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
    graph: Dict[Tuple[int, int], List[Tuple[int, int]]],
    connection_threshold: float = 30.0
) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    """
    Connect nearby dead end nodes to reduce fragmentation.
    
    Args:
        graph: Input graph to process
        connection_threshold: Maximum distance for connecting dead ends
        
    Returns:
        Graph with nearby dead ends connected
    """

    dead_end_nodes = [
        node for node, neighbors in graph.items() 
        if len(neighbors) == 1
    ]
    

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
        

        if closest_dead_end is not None:
            graph = add_bidirectional_edge(graph, dead_end1, closest_dead_end)
    
    return graph



def graph_refine(graph, isolated_thr=150, spurs_thr=60, three_edge_loop_thr=70):

    return refine_lane_graph(graph, isolated_thr, spurs_thr, three_edge_loop_thr)


def graphInsert(node_neighbor, n1key, n2key):

    return add_bidirectional_edge(node_neighbor, n1key, n2key)


def downsample(graph, rate=2):

    return downsample_graph(graph, rate)


def connectDeadEnds(graph, thr=30):

    return connect_nearby_dead_ends(graph, thr)
