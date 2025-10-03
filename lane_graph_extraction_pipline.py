import os
import cv2
import torch
import einops
import argparse
import scipy.ndimage
from PIL import Image 
import numpy as np
import networkx as nx
import geopandas as gpd
from pyproj import CRS
from typing import List, Tuple
from itertools import product
from skimage import morphology
import imageio.v3 as imageio
import matplotlib.pyplot as plt
from shapely.affinity import scale
from scipy.spatial import KDTree
from shapely.geometry import LineString
import utils.segmentation2graph as segmentation2graph
from utils.config_utils import load_config
from utils.inference_utils import load_model
from turingLaneExtraction.model import LaneExtractionModel
from reachableLaneValidation.model import ReachableLaneValidationModel
from laneAndDirectionExtraction.model import LaneAndDirectionExtractionModel
from utils.graph_postprocessing_utils import (refine_lane_graph,
                                              connect_nearby_dead_ends,
                                              refine_lane_graph_with_curves,
                                              annotate_node_types,
                                              get_node_types,
                                              get_corresponding_lane_segment,
                                              get_segment_average_angle,
                                              intersection_of_extended_segments,
                                              sample_bezier_curve
                                              )

from utils.lane_process_utils import (get_junction_points, 
                                      split_lines_at_junctions,
                                      group_lanes_by_geometry,
                                      infer_lane_directions_from_geometry,
                                      compute_reference_lines_direction_aware,
                                      assign_lane_ids_per_group,
                                      extract_lanes_and_links_with_fids
                                      )

from utils.connection_filtering import filter_connections_receive_aware
from utils.gdf_visualize_utils import visualize_road_groups
from utils.image_postprocessing_utils import encode_direction_vectors_to_image


class LaneGraphExtraction():
    

    def __init__(self, config, gpu_id=0):
        self.image_size = config.dataset_config.data_attributes.input_image_size
        self.config = config
        self.gpu_id = gpu_id
        self.models_config = config.models

        self.lane_and_direction_extraction_model_config = self.models_config.lane_and_direction_extraction_model
        self.lane_and_direction_extraction_model = LaneAndDirectionExtractionModel(
            self.lane_and_direction_extraction_model_config
        )

        self.reachable_lane_extraction_validation_model_config = self.models_config.reachable_lane_extraction_validation_model
        self.reachable_lane_extraction_validation_model = ReachableLaneValidationModel(
            self.reachable_lane_extraction_validation_model_config.reachable_lane_extraction_model,
            self.reachable_lane_extraction_validation_model_config.reachable_lane_validation_model
        )
        
        self.lane_extraction_model_config = self.models_config.lane_extraction_model
        self.lane_extraction_model = LaneExtractionModel(self.lane_extraction_model_config)

        self._load_weights(self.gpu_id)
        
        self.position_code = np.zeros(
            (1, self.image_size, self.image_size, 2), dtype=np.float32
        )
        for i in range(self.image_size):
            self.position_code[:, i, :, 0] = float(i) / self.image_size
            self.position_code[:, :, i, 0] = float(i) / self.image_size

        self.poscode = np.zeros(
            (self.image_size * 2, self.image_size * 2, 2), dtype=np.float32
        )
        for i in range(self.image_size * 2):
            self.poscode[i, :, 0] = float(i) / self.image_size - 1.0
            self.poscode[:, i, 1] = float(i) / self.image_size - 1.0

    def _load_weights(self, gpu_id=0):
        load_model(
            self.lane_and_direction_extraction_model,
            self.lane_and_direction_extraction_model_config.weight_path
        )
        self.lane_and_direction_extraction_model.to(gpu_id)
        load_model(
            self.reachable_lane_extraction_validation_model,
            self.reachable_lane_extraction_validation_model_config.weight_path
        )
        self.reachable_lane_extraction_validation_model.to(gpu_id)
        load_model(
            self.lane_extraction_model,
            self.lane_extraction_model_config.weight_path
        )
        self.lane_extraction_model.to(gpu_id)

    def _extract_lane_and_direction(self, input_satellite_image: np.ndarray, gpu_id=0, output_path=None, image_name=None) -> nx.DiGraph:
        """
        Extracts lane and direction from the satellite image.
        
        Args:
            input_satellite_image (np.ndarray): Input satellite image of shape (H, W, 3).
        
        Returns:
            np.ndarray: Output of lane and direction extraction model.
        """
        print("Extracting lane and direction from the satellite image...")
        # normalize the input image
        input_satellite_image = (input_satellite_image.astype(np.float64) / 255.0 - 0.5)
        input_satellite_image = torch.from_numpy(input_satellite_image).float().to(gpu_id)  # [H, W, 3]
        input_satellite_image = einops.rearrange(input_satellite_image, 'h w c -> 1 c h w')  # [1, 3, H, W]

        with torch.no_grad():
            lane_predicted, direction_predicted = self.lane_and_direction_extraction_model(input_satellite_image) # [B, 4, H, W]
        lane_predicted = lane_predicted.cpu().numpy()
        direction_predicted = direction_predicted.cpu().numpy()

        lane_predicted = einops.rearrange(lane_predicted, 'b c h w -> b h w c')
        direction_predicted = einops.rearrange(direction_predicted, 'b c h w -> b h w c')

        lane_predicted_image = np.zeros((self.image_size, self.image_size))
        # soft max to get lane probability
        expsum = np.exp(lane_predicted[0, :, :, 0]) + np.exp(lane_predicted[0, :, :, 1])
        lane_predicted_image = np.clip(np.exp(lane_predicted[0, :, :, 0]) / expsum, 0, 1) * 255
        
        direction_predicted_image = np.zeros((self.image_size, self.image_size, 2))
        direction_predicted_image[:,:,0] = np.clip(direction_predicted[0,:,:,0],-1,1) * 127 + 127
        direction_predicted_image[:,:,1] = np.clip(direction_predicted[0,:,:,1],-1,1) * 127 + 127
        def margin_mask_bool(img, margin=20):
            """True on the margins, False elsewhere."""
            h, w = img.shape[:2]
            y = np.arange(h)[:, None]
            x = np.arange(w)[None, :]
            m = (y < margin) | (y >= h - margin) | (x < margin) | (x >= w - margin)
            return m  # shape (H, W), dtype=bool

        border_mask = margin_mask_bool(lane_predicted_image, margin=20)
        lane_predicted_image[border_mask] = 0
        direction_predicted_image[border_mask] = 127

        
        Image.fromarray(lane_predicted_image.astype(np.uint8)).save(os.path.join(output_path, f"lane_predicted_{image_name}.jpg"))
        lane_predicted_image = scipy.ndimage.grey_closing(lane_predicted_image, size=(6,6))
        
        threshold = 64
        lane_predicted_image = lane_predicted_image >= threshold
        Image.fromarray(encode_direction_vectors_to_image(direction_predicted[0])).save(os.path.join(output_path, f"direction_predicted_{image_name}.jpg"))
        
        lane_predicted_image = morphology.thin(lane_predicted_image)
        #
        lane_graph = segmentation2graph.extract_graph_from_image(lane_predicted_image)
        lane_graph = segmentation2graph.direct_graph_from_direction_map(lane_graph, direction_predicted_image)

        return lane_graph

    def _make_reachable_lane_validaton_input(
        self, node_pair: Tuple[Tuple[int, int], Tuple[int, int]], input_satellite_image: np.ndarray, direction_context: np.ndarray, gpu_id=0
    ) -> Tuple[np.ndarray, np.ndarray]:
        connector1 = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        connector2 = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        connectorlink = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        y1, x1 = node_pair[0]
        y2, x2 = node_pair[1]

        cv2.circle(connector1, (x1, y1), 12, (255,), -1)
        cv2.circle(connector2, (x2, y2), 12, (255,), -1)
        connector_features = np.zeros((1, 640, 640, 7), dtype=np.float32)
        # Dummy connectorlink
        connector_features[0,:,:,6] = np.copy(connectorlink) / 255.0 - 0.5

        connector_features[0, :, :, 0] = np.copy(connector1) / 255.0 - 0.5
        connector_features[0, :, :, 1:3] = self.poscode[
            self.image_size - x1 : self.image_size * 2 - x1,
            self.image_size - y1 : self.image_size * 2 - y1,
        ]

        connector_features[0, :, :, 3] = np.copy(connector2) / 255.0 - 0.5
        connector_features[0, :, :, 4:6] = self.poscode[
            self.image_size - x2 : self.image_size * 2 - x2,
            self.image_size - y2 : self.image_size * 2 - y2,
        ]

        input_image = np.expand_dims(input_satellite_image / 255.0 - 0.5, axis=0)
        direction_context = np.expand_dims((direction_context - 127.0) / 127.0, axis=0)

        input_features_node_a = np.concatenate(
            (input_image, connector_features[:, :, :, 0:3], direction_context, self.position_code), axis=3
        )
        input_features_node_b = np.concatenate(
            (input_image, connector_features[:, :, :, 3:6], direction_context, self.position_code), axis=3
        )

        input_features_validation = np.concatenate(
            (connector_features, direction_context, self.position_code), axis=3)
        
        input_features_node_a = torch.from_numpy(input_features_node_a).float().to(gpu_id)
        input_features_node_a = einops.rearrange(input_features_node_a, 'b h w c -> b c h w')

        input_features_node_b = torch.from_numpy(input_features_node_b).float().to(gpu_id)
        input_features_node_b = einops.rearrange(input_features_node_b, 'b h w c -> b c h w')

        input_features_validation = torch.from_numpy(input_features_validation).float().to(gpu_id)
        input_features_validation = einops.rearrange(input_features_validation, 'b h w c -> b c h w')

        return input_features_node_a, input_features_node_b, input_features_validation

    
    def _make_lane_extraction_input(
        self, node_pair: Tuple[Tuple[int, int], Tuple[int, int]], input_satellite_image, direction_context, gpu_id=0
    ) -> Tuple[np.ndarray, np.ndarray]:
        connector1 = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        connector2 = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        connectorlink = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        y1, x1 = node_pair[0]
        y2, x2 = node_pair[1]

        cv2.circle(connector1, (x1, y1), 12, (255,), -1)
        cv2.circle(connector2, (x2, y2), 12, (255,), -1)
        connector_features = np.zeros((1, 640, 640, 7), dtype=np.float32)
        # Dummy connectorlink
        connector_features[0,:,:,6] = np.copy(connectorlink) / 255.0 - 0.5

        connector_features[0, :, :, 0] = np.copy(connector1) / 255.0 - 0.5
        connector_features[0, :, :, 1:3] = self.poscode[
            self.image_size - x1 : self.image_size * 2 - x1,
            self.image_size - y1 : self.image_size * 2 - y1,
        ]

        connector_features[0, :, :, 3] = np.copy(connector2) / 255.0 - 0.5
        connector_features[0, :, :, 4:6] = self.poscode[
            self.image_size - x2 : self.image_size * 2 - x2,
            self.image_size - y2 : self.image_size * 2 - y2,
        ]

        input_image = np.expand_dims(input_satellite_image / 255.0 - 0.5, axis=0)
        direction_context = np.expand_dims((direction_context - 127.0) / 127.0, axis=0)

        input_features = np.concatenate(
            (input_image, connector_features, direction_context, self.position_code), axis=3)
        
        input_features = torch.from_numpy(input_features).float().to(gpu_id)
        input_features = einops.rearrange(input_features, 'b h w c -> b c h w')

        return input_features


    def extract_valid_turning_pairs_model_based(self, lane_graph: nx.DiGraph, input_satellite_image, direction_context, gpu_id=0):
        """
        Extracts valid turning pairs from the lane graph.
        
        Args:
            lane_graph (nx.DiGraph): Input lane graph.
        
        Returns:
            List[Tuple[int, int]]: List of valid turning pairs.
        """
        print("Extracting valid turning pairs from the lane graph...")
        # Step 2: Reachable Lane Extraction

        
        node_types = get_node_types(lane_graph)

        
        out_nodes = node_types.get("out", [])
        in_nodes = node_types.get("in", [])
        
        node_pairs = list(product(out_nodes, in_nodes))
        reachable_node_pairs = []
        debug_image = input_satellite_image.copy()
        for in_node in in_nodes:
            x1, y1 = lane_graph.nodes[in_node]['pos']
            cv2.circle(debug_image, (x1, y1), 12, (255, 0, 0), -1)
        for out_node in out_nodes:
            x2, y2 = lane_graph.nodes[out_node]['pos']
            cv2.circle(debug_image, (x2, y2), 12, (0, 0, 255), -1)
        for idx, node_pair in enumerate(node_pairs):
            input_features_node_a, input_features_node_b, input_features_validation = self._make_reachable_lane_validaton_input(
                node_pair, input_satellite_image, direction_context, gpu_id
            )
            with torch.no_grad():
                reachable_lane_predicted_node_a, reachable_lane_predicted_node_b, reachable_label_predicted = self.reachable_lane_extraction_validation_model(
                    input_features_node_a, input_features_node_b, input_features_validation
                )
            predicted_label = (reachable_label_predicted[:, 0] > reachable_label_predicted[:, 1]).long()
            # 
            output_image = input_satellite_image.copy()
            x1, y1 = lane_graph.nodes[node_pair[0]]['pos']
            x2, y2 = lane_graph.nodes[node_pair[1]]['pos']
            color = (255, 0, 0) if predicted_label.item() == 1 else (0, 0, 255)
            cv2.circle(output_image, (x1, y1), 12, color, -1)
            cv2.circle(output_image, (x2, y2), 12, color, -1)
            
            # save the output image
            if predicted_label.item() == 1:
                reachable_node_pairs.append(node_pair)
        # Image.fromarray(debug_image.astype(np.uint8)).save("debug_reachable_lane_pairs.jpg")
        return reachable_node_pairs

    
    def extract_connections_rule_based(self, lane_graph: nx.DiGraph, segment_max_length=5, distance_threshold=250, turning_threshold=-0.8, topology_threshold=0.8):
        """
        Extracts valid connections from the lane graph.

        Args:
            lane_graph (nx.DiGraph): Input lane graph.
        
        Returns:
            List[Tuple[int, int]]: List of valid connections.
        """
        print("Extracting valid connections from the lane graph...")
        # Step 2: Reachable Lane Extraction

        
        node_types = get_node_types(lane_graph)

        
        out_nodes = node_types.get("out", [])
        in_nodes = node_types.get("in", [])
        
        node_pairs = list(product(out_nodes, in_nodes))
        
        connections = {}
        # plt.figure(figsize=(10,10))
        idx = 0
        for out_node_id, in_node_id in node_pairs:
            
            # nodes near in_node based on distance
            in_node_neighbors = [
                n for n in lane_graph.nodes
                if np.linalg.norm(np.array(lane_graph.nodes[n].get("pos", (0, 0))) - np.array(lane_graph.nodes[in_node_id].get("pos", (0, 0)))) < 100 and lane_graph.nodes[n].get("type") == "in"
            ]
            out_node_neighbors = [
                n for n in lane_graph.nodes
                if np.linalg.norm(np.array(lane_graph.nodes[n].get("pos", (0, 0))) - np.array(lane_graph.nodes[out_node_id].get("pos", (0, 0)))) < 100 and lane_graph.nodes[n].get("type") == "out"
            ]
            
            # if there is a path from out_node_id to in_node_id, skip
            skipped = False
            for in_node_neighbor in in_node_neighbors:
                for out_node_neighbor in out_node_neighbors:
                    
                    if nx.has_path(lane_graph.to_undirected(), out_node_id, in_node_neighbor) or nx.has_path(lane_graph.to_undirected(), out_node_neighbor, in_node_id):
                        skipped = True
                        break
            if skipped:
                continue
            out_node_x, out_node_y = lane_graph.nodes[out_node_id].get("pos", (0, 0))
            in_node_x, in_node_y = lane_graph.nodes[in_node_id].get("pos", (0, 0))
            # plt.scatter(out_node_x, out_node_y, c='r', label='out' if idx == 0 else "")
            # plt.scatter(in_node_x, in_node_y, c='b', label='in' if idx == 0 else "")
            # if idx == 0:
            #     idx += 1
            #     plt.legend()
            
            between_nodes_distance = np.linalg.norm(np.array([out_node_x, out_node_y]) - np.array([in_node_x, in_node_y]))
            if between_nodes_distance > distance_threshold:
                continue
            # out <- prv1 <- prv2 <- ... <- prvN
            out_segment = get_corresponding_lane_segment(lane_graph, out_node_id, segment_max_length=segment_max_length)
            # in -> nxt1 -> nxt2 -> ... -> nxtN
            in_segment = get_corresponding_lane_segment(lane_graph, in_node_id, segment_max_length=segment_max_length)
            
            if len(out_segment) < 2 or len(in_segment) < 2:
                print(f"Warning: segment too short for out_node {out_node_id} or in_node {in_node_id}")
                continue
            out_angle_rad = get_segment_average_angle(lane_graph, out_segment)
            in_angle_rad = get_segment_average_angle(lane_graph, in_segment)
            # print(f"Out node {out_node_id} angle: {np.degrees(out_angle_rad):.2f} degrees, In node {in_node_id} angle: {np.degrees(in_angle_rad):.2f} degrees")
            starting_node_vector = np.array([np.cos(out_angle_rad), np.sin(out_angle_rad), 0.0])
            ending_node_vector   = np.array([np.cos(in_angle_rad), np.sin(in_angle_rad), 0.0])
            dot_val   = np.dot(starting_node_vector, ending_node_vector)
            cross_val = np.cross(starting_node_vector, ending_node_vector)[-1]  # take z component
            
            connected = False
            turning = False
            if dot_val <= turning_threshold:
                connected = False
                turning = False
            elif dot_val > turning_threshold and dot_val <= topology_threshold:
                connected = True
                turning = True
            elif dot_val > topology_threshold:
                connected = True
                turning = False

            # print(f'cos angle: {np.rad2deg(np.arccos(topology_score)):.2f}, distance: {between_nodes_distance:.2f}, connected: {connected}, turning: {turning}')
            if connected:
                if turning:
                    intersection_point = intersection_of_extended_segments(lane_graph, out_segment, in_segment, length=300.0)
                    if intersection_point is None or intersection_point[0] < 0 or intersection_point[0] >= self.image_size or intersection_point[1] < 0 or intersection_point[1] >= self.image_size:
                        continue
                    p0_pos = (out_node_x, out_node_y)
                    p1_pos = (out_node_x + np.cos(out_angle_rad), out_node_y + np.sin(out_angle_rad))
                    p2_pos = (intersection_point[0] + np.cos(in_angle_rad), intersection_point[1] + np.sin(in_angle_rad))
                    p3_pos = (in_node_x, in_node_y)
                    connection_type = "left_turn" if cross_val > 0 else "right_turn"
                    connections[(out_node_id, in_node_id)] = {
                        "connection_type": connection_type,
                        "points": [p0_pos, p1_pos, p2_pos, p3_pos]
                    }
                    # if connection_type == "left_turn":
                    #     plt.plot([p0_pos[0], p1_pos[0], p2_pos[0], p3_pos[0]], [p0_pos[1], p1_pos[1], p2_pos[1], p3_pos[1]], c='r')
                    # else:
                    #     plt.plot([p0_pos[0], p1_pos[0], p2_pos[0], p3_pos[0]], [p0_pos[1], p1_pos[1], p2_pos[1], p3_pos[1]], c='m')
                else:
                    connection_type = "straight"
                    p0_pos = (out_node_x, out_node_y)
                    p1_pos = (in_node_x, in_node_y)
                    connections[(out_node_id, in_node_id)] = {
                        "connection_type": connection_type,
                        "points": [p0_pos, p1_pos]
                    }
                    # plt.plot([out_node_x, in_node_x], [out_node_y, in_node_y], c='g')
                
        # plt.title(f"Extracted Connections: {len(connections)}")
        # plt.savefig("extracted_connections.png")
        return connections
    
    
    
    def _extract_turning_lane(self, lane_graph: nx.DiGraph, reachable_node_pairs: List[Tuple[int, int]], input_satellite_image: np.ndarray, direction_context: np.ndarray, gpu_id: int = 0, radius: float = 5.0):
        # Build KDTree for efficient nearest-neighbor search
        out_nodes = [n for n, d in lane_graph.nodes(data=True) if d.get("type") == "out"]
        in_nodes = [n for n, d in lane_graph.nodes(data=True) if d.get("type") == "in"]

        out_kdtree = KDTree(out_nodes) if out_nodes else None
        in_kdtree = KDTree(in_nodes) if in_nodes else None

        for idx, reachable_node_pair in enumerate(reachable_node_pairs):
            input_features = self._make_lane_extraction_input(
                reachable_node_pair, input_satellite_image, direction_context, gpu_id
            )

            with torch.no_grad():
                outputs = self.lane_extraction_model(input_features)

            outputs = outputs.cpu().numpy()
            lane_predicted = outputs[:, 0:2, :, :]  # [B, 2, H, W]
            lane_predicted = einops.rearrange(lane_predicted, 'b c h w -> b h w c')

            lane_predicted_image = (np.clip(lane_predicted[0, :, :, 0], 0, 1) * 255).astype(np.uint8)
            lane_predicted_image = scipy.ndimage.grey_closing(lane_predicted_image, size=(6, 6))
            lane_predicted_image = lane_predicted_image > 20
            lane_predicted_image = morphology.thin(lane_predicted_image)

            # Extract graph from thinned binary mask
            start_node, end_node = reachable_node_pair
            link_graph = segmentation2graph.extract_graph_from_image(
                lane_predicted_image, start_pos=start_node
            )

            link_nodes = list(link_graph.nodes)
            if len(link_nodes) == 0:
                print(f"Warning: link_graph is empty for pair {reachable_node_pair}")
                continue

            # Add link_graph's nodes and edges into lane_graph
            for node in link_nodes:
                if node not in lane_graph.nodes:
                    lane_graph.add_node(node, pos=link_graph.nodes[node]["pos"], type='link')
            for u, v in link_graph.edges:
                lane_graph.add_edge(u, v)

            # Check and connect start if not exactly equal to reachable_node_pair[0]
            first_node = link_nodes[0]
            if first_node != start_node and out_kdtree:
                dist, idx = out_kdtree.query(first_node, distance_upper_bound=radius)
                if dist < radius:
                    nearest_out = out_nodes[idx]
                    lane_graph.add_edge(nearest_out, first_node)
                else:
                    print(f"No 'out' node found near {first_node} within radius {radius}")

            # Check and connect end if not exactly equal to reachable_node_pair[1]
            last_node = link_nodes[-1]
            if last_node != end_node and in_kdtree:
                dist, idx = in_kdtree.query(last_node, distance_upper_bound=radius)
                if dist < radius:
                    nearest_in = in_nodes[idx]
                    lane_graph.add_edge(last_node, nearest_in)
                else:
                    print(f"No 'in' node found near {last_node} within radius {radius}")

        return lane_graph

    def _build_connecting_lanes(self, lane_graph: nx.DiGraph, connections: dict):
        for (out_node_id, in_node_id), conn_info in connections.items():
            points = conn_info["points"]
            connection_type = conn_info["connection_type"]
            # building the turning lane
            if connection_type in ["left_turn", "right_turn"]:
                p0_pos, p1_pos, p2_pos, p3_pos = points
                sampled_points = sample_bezier_curve(p0_pos, p1_pos, p2_pos, p3_pos, delta=25, oversample=10)
                previous_node = out_node_id
                for point in sampled_points[1:-1]:
                    x, y = int(point[0]), int(point[1])
                    if (x, y) not in lane_graph.nodes:
                        lane_graph.add_node((x, y), type='link', pos=(x, y))
                    lane_graph.add_edge(previous_node, (x, y), type='link')
                    previous_node = (x, y)
                lane_graph.add_edge(previous_node, in_node_id, type='link')
            else:
            # build the straight lane
                p0_pos, p1_pos = points
                lane_graph.add_edge(out_node_id, in_node_id, type='link')

        return lane_graph





    def extract_lane_graph(self, input_satellite_image, output_path=None, image_name=None, mode='rule_based'):
        """
        Extracts lane graph from the satellite image.
        
        Args:
            satellite_image (np.ndarray): Input satellite image of shape (H, W, 3).
        
        Returns:
            nx.DiGraph: Extracted lane graph.
        """
        print("Extracting lane graph from the satellite image...")

        if isinstance(input_satellite_image, str):
            image_name = os.path.basename(input_satellite_image).split('.')[0]
            print(f"Reading image from {image_name}...")
            input_satellite_image: np.ndarray = imageio.imread(input_satellite_image)
        elif isinstance(input_satellite_image, np.ndarray):
            if image_name is None:
                image_name = "input_image"
            
        else:
            raise ValueError("input_satellite_image should be a file path or a numpy array.")
        
        # # Step 1: Lane and Direction Extraction
        lane_graph = self._extract_lane_and_direction(input_satellite_image, self.gpu_id, output_path, image_name)
        lane_graph = annotate_node_types(lane_graph)
        lane_graph = connect_nearby_dead_ends(lane_graph, connection_threshold=10.0, topology_threshold=0.8)
        lane_graph = annotate_node_types(lane_graph)
        lane_graph = refine_lane_graph(lane_graph, isolated_threshold=30, spur_threshold=10)
        lane_graph = annotate_node_types(lane_graph)
        lane_graph = refine_lane_graph_with_curves(lane_graph)
        # lane_graph = annotate_node_types(lane_graph)
        lane_graph_non_intersection = lane_graph.copy()
        lane_predicted, direction_predicted = segmentation2graph.draw_inputs(lane_graph)

        if mode == 'rule_based':
            # assign the lane idx
            lanes_and_links_gdf, lane_graph_with_fids = self.extract_lanes_and_links_geojson(lane_graph,
                origin=(0, 0),
                resolution=(0.125, -0.125),
                output_path=None,
                crs_proj=None
            )
            if lanes_and_links_gdf is None:
                print("Warning: No lanes extracted.")
                connections = self.extract_connections_rule_based(lane_graph_non_intersection, 
                                                              segment_max_length=5, 
                                                              distance_threshold=400,
                                                              turning_threshold=-0.8, 
                                                              topology_threshold=0.9)
                lane_graph_final = self._build_connecting_lanes(lane_graph, connections)
            else:
                lanes_gdf = lanes_and_links_gdf[lanes_and_links_gdf['type'] == 'lane'].reset_index(drop=True)
                # get lane information (e.g., lane id, direction, road id)
                lanes_gdf, reference_lines, fids_to_remove = self.extract_lane_info(lanes_gdf, save_path=output_path)
                # remove nodes and edges with fids_to_remove
                lanes_gdf = lanes_gdf[~lanes_gdf['fid'].isin(fids_to_remove)].reset_index(drop=True)
                reference_lines = {road_id: line for road_id, line in reference_lines.items() if road_id in lanes_gdf['road_id'].values}
                lane_graph_with_fids.remove_nodes_from([n for n, d in lane_graph_with_fids.nodes(data=True) if d.get("fid") in fids_to_remove])
                lane_graph_with_fids.remove_edges_from([ (u, v) for u, v, d in lane_graph_with_fids.edges(data=True) if d.get("fid") in fids_to_remove])
                # [ ] convert the geometry back to image coordinates
                lanes_gdf['geometry'] = lanes_gdf['geometry'].apply(lambda line: LineString([(x / 0.125, y / -0.125) for x, y in line.coords]))
                # also the reference lines
                reference_lines = {road_id: LineString([(x / 0.125, y / -0.125) for x, y in line.coords]) for road_id, line in reference_lines.items()}
                # get possible connections
                connections = self.extract_connections_rule_based(lane_graph_non_intersection, 
                                                                segment_max_length=5, 
                                                                distance_threshold=400,
                                                                turning_threshold=-0.8, 
                                                                topology_threshold=0.8)
                # print(f"Number of connections before filtering by lane templates: {len(connections)}")
                
                connections_filtered = filter_connections_receive_aware(connections=connections, 
                                                                        lane_graph=lane_graph_with_fids, 
                                                                        lanes_gdf=lanes_gdf,
                                                                        reference_lines=reference_lines,
                                                                        attach_tol=220.0
                                                                        )
                # print(f"Number of connections after filtering by lane templates: {len(connections_filtered)}")
                lane_graph_final = self._build_connecting_lanes(lane_graph, connections_filtered)
        elif mode == 'model_based':
            # Step 2: Reachable Lane Extraction
            reachable_node_pairs = self.extract_valid_turning_pairs_model_based(lane_graph, 
                                                              input_satellite_image, 
                                                              direction_predicted[..., 0:2], 
                                                              self.gpu_id)
            print(f"Number of reachable node pairs: {len(reachable_node_pairs)}")
            # Step 3: Turning Lane Extraction
            lane_graph_final = self._extract_turning_lane(lane_graph, 
                                                     reachable_node_pairs, 
                                                     input_satellite_image, 
                                                     direction_predicted[..., 0:2], 
                                                     gpu_id=self.gpu_id,
                                                     radius=5.0)


        if output_path is not None:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            # save the raw image
            Image.fromarray(input_satellite_image.astype(np.uint8)).save(os.path.join(output_path, f"{image_name}_input.png"))
            # save the lane graphs
            segmentation2graph.draw_directed_graph(lane_graph_non_intersection, save_path=output_path, image_name=f"non_intersection_{image_name}")
            segmentation2graph.draw_directed_graph(lane_graph_final, save_path=output_path, image_name=f"final_{image_name}")
            
            
        return lane_graph_non_intersection, lane_graph_final
    
    
    def extract_lanes_and_links_geojson(self, lane_graph, origin=(0, 0), resolution=(0.125, 0.125), output_path='./', crs_proj=CRS.from_epsg(3857)):
        """
        Extracts lanes and links from the lane graph and saves them as a GeoJSON file.
        
        Args:
            lane_graph (nx.DiGraph): Input lane graph.
            origin (Tuple[float, float]): Origin of the image in geographic (left-top corner).
            resolution (Tuple[float, float]): Resolution of the image in meters/pixel.
            output_path (str): Path to save the GeoJSON file.
            crs_proj (CRS): Coordinate reference system for the output GeoDataFrame.

        """
        lanes_and_links_gdf, lane_graph_with_fids = extract_lanes_and_links_with_fids(lane_graph,
            origin=origin,
            resolution=resolution,
            output_path=output_path,
            crs_proj=crs_proj
        )
        if output_path is not None:
            segmentation2graph.visualize_lanes_and_links(lanes_and_links_gdf, save_path=output_path, image_name=f"lane_links")
        return lanes_and_links_gdf, lane_graph_with_fids

    def extract_lane_info(self, lanes_gdf: gpd.GeoDataFrame, save_path: str = None) -> gpd.GeoDataFrame:
        """
        Annotates road information to the lane graph based on the lanes GeoDataFrame.

        Args:
            lane_graph (nx.DiGraph): Input lane graph.
            lanes_gdf (gpd.GeoDataFrame): GeoDataFrame containing lanes information.

        """
        junction_points = get_junction_points(lanes_gdf)
        lanes_gdf_splitted = split_lines_at_junctions(lanes_gdf, junction_points)
        lanes_gdf_grouped = group_lanes_by_geometry(
            lanes_gdf_splitted,
            spacing=0.5,          # Resample step along each line (in CRS units, e.g., meters).
            avg_tol=10,          # Max symmetric avg. perpendicular distance to treat lanes as the same corridor.
            endpoint_tol=10,     # Endpoint proximity threshold: at least one valid endpoint pair must be within this.
            angle_tol_deg=25.0,   # Max heading difference to be considered "same direction" (excludes ~180° flips).
            serial_reject_tol=1.5,# If start-end or end-start is <= this (when same direction), reject as serial (head-to-tail).
            min_overlap_ratio=0.2,# Min fraction of the shorter line that must overlap the longer one (0..1).
            pair_expand=6.0       # Search radius to gather candidate neighbors before detailed checks.
            )
        if save_path is not None:
            visualize_road_groups(lanes_gdf_grouped, label_col='road_id', save_path=save_path)
        lanes_gdf_grouped = infer_lane_directions_from_geometry(lanes_gdf_grouped)
        lanes_gdf, reference_lines = compute_reference_lines_direction_aware(lanes_gdf_grouped)
        lanes_gdf = assign_lane_ids_per_group(lanes_gdf, reference_lines)
        fids_to_remove = []
        for road_id, line in reference_lines.items():
            # if only one lane for this road_id and the reference line is short, then consider removing it
            lanes_in_road = lanes_gdf[lanes_gdf['road_id'] == road_id]
            if len(lanes_in_road) == 1 and line.length < 20:
                print(f"Warning: road_id {road_id} has only one lane and short reference line, consider removing it.")
                fid = lanes_in_road.iloc[0]['fid']
                fids_to_remove.append(fid)
        
        return lanes_gdf, reference_lines, fids_to_remove

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="options")
    parser.add_argument("--config", type=str, default="configs/lane_graph_extraction_pipline.py", help="config file")
    args = parser.parse_args()
    # ============= Load Configuration =============
    config = load_config(args.config)
    lane_graph_extraction = LaneGraphExtraction(config, gpu_id=0)
    input_satellite_img_path = "test_intersection.jpg"  # Path to the input satellite image
    # lane_graph_non_intersection, lane_graph_final = lane_graph_extraction.extract_lane_graph(input_satellite_img_path, mode="rule_based",output_path='./')
    lane_graph_non_intersection, lane_graph_final = lane_graph_extraction.extract_lane_graph(input_satellite_img_path, mode="rule_based",output_path='./')
