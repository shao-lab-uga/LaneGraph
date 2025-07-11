
import cv2
import argparse
import torch
import scipy.ndimage
from PIL import Image
import einops
import numpy as np
from typing import Tuple
from itertools import product
from skimage import morphology
import imageio.v3 as imageio
from scipy.spatial import KDTree
import geopandas as gpd
from shapely.geometry import LineString
import pandas as pd
import networkx as nx
import os
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import utils.segmentation2graph as segmentation2graph
from utils.config_utils import load_config
from utils.inference_utils import load_model

from turingLaneExtraction.model import LaneExtractionModel
from reachableLaneValidation.model import ReachableLaneValidationModel
from laneAndDirectionExtraction.model import LaneAndDirectionExtractionModel



class LaneGraphExtraction():
    

    def __init__(self, config):
        self.image_size = config.dataset_config.data_attributes.input_image_size
        self.config = config
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

        self._load_weights()
        
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

    def _extract_lane_and_direction(self, input_satellite_image, gpu_id=0):
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
            outputs = self.lane_and_direction_extraction_model(input_satellite_image) # [B, 4, H, W]
        outputs = outputs.cpu().numpy()
        lane_predicted = outputs[:, 0:2, :, :]  # [B, 2, H, W]
        direction_predicted = outputs[:, 2:4, :, :] # [B, 2, H, W]

        lane_predicted = einops.rearrange(lane_predicted, 'b c h w -> b h w c')
        direction_predicted = einops.rearrange(direction_predicted, 'b c h w -> b h w c')

        lane_predicted_image = np.zeros((self.image_size, self.image_size))
        lane_predicted_image[:, :] = np.clip(lane_predicted[0,:,:,0], 0, 1) * 255
        direction_predicted_image = np.zeros((self.image_size, self.image_size, 2))
        direction_predicted_image[:,:,0] = np.clip(direction_predicted[0,:,:,0],-1,1) * 127 + 127
        direction_predicted_image[:,:,1] = np.clip(direction_predicted[0,:,:,1],-1,1) * 127 + 127
        Image.fromarray(lane_predicted_image.astype(np.uint8)).save("lane_predicted.jpg")

        lane_predicted_image = scipy.ndimage.grey_closing(lane_predicted_image, size=(6,6))
        threshold = 64
        lane_predicted_image = lane_predicted_image >= threshold
        lane_predicted_image = morphology.thin(lane_predicted_image)
        lane_graph = segmentation2graph.extract_graph_from_image(lane_predicted_image)
        lane_graph = segmentation2graph.direct_graph_from_vector_map(lane_graph, direction_predicted_image)

        return lane_graph

    def _make_reachable_lane_validaton_input(
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


    def extract_valid_turining_pairs(self, lane_graph, input_satellite_image, direction_context, intersection_center, gpu_id=0):
        """
        Extracts valid turning pairs from the lane graph.
        
        Args:
            lane_graph (nx.DiGraph): Input lane graph.
            intersection_center (Tuple[int, int]): Center of the intersection.
        
        Returns:
            List[Tuple[int, int]]: List of valid turning pairs.
        """
        print("Extracting valid turning pairs from the lane graph...")
        # Step 2: Reachable Lane Extraction

        lane_graph = segmentation2graph.annotate_node_types(lane_graph, intersection_center)
        node_types = segmentation2graph.get_node_types(lane_graph)

        
        out_nodes = node_types.get("out", [])
        in_nodes = node_types.get("in", [])
        
        node_pairs = list(product(out_nodes, in_nodes))
        reachable_node_paris = []
        for idx, node_pair in enumerate(node_pairs):
            input_features_node_a, input_features_node_b, input_features_validation = self._make_reachable_lane_validaton_input(
                node_pair, input_satellite_image, direction_context, gpu_id
            )
            with torch.no_grad():
                # Extract reachable lane and validation
                reachable_lane_predicted_node_a, reachable_lane_predicted_node_b, reachable_label_predicted = self.reachable_lane_extraction_validation_model(
                    input_features_node_a, input_features_node_b, input_features_validation
                )
            
            predicted_label = (reachable_label_predicted[:, 0] > reachable_label_predicted[:, 1]).long()
            # 
            output_image = input_satellite_image.copy()
            y1, x1 = node_pair[0]
            y2, x2 = node_pair[1]
            color = (255, 0, 0) if predicted_label.item() == 1 else (0, 0, 255)
            cv2.circle(output_image, (x1, y1), 12, color, -1)
            cv2.circle(output_image, (x2, y2), 12, color, -1)
            
            # save the output image
            # Image.fromarray(output_image.astype(np.uint8)).save(f"reachable_lane_predicted_{idx}.jpg")
            if predicted_label.item() == 1:
                reachable_node_paris.append(node_pair)

        return reachable_node_paris
    

    def _extract_turning_lane(self, lane_graph, reachable_node_pairs, input_satellite_image, direction_context, gpu_id=0, radius=5.0):
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
                    lane_graph.add_node(node, type='link')
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

    


    def pixel_to_geo(self, path, origin=(0, 0), resolution=(1.0, -1.0)):
        """Convert a pixel path to geographic coordinates (lon, lat or meters)."""
        lon0, lat0 = origin
        dx, dy = resolution
        return [(lon0 + x * dx, lat0 + y * dy) for (x, y) in path]

    def extract_lanes(self, lane_graph, origin=(0, 0), resolution=(1.0, -1.0), crs="EPSG:4326",
                      lane_path="lanes.geojson", turning_path="turning_links.geojson"):
        import geopandas as gpd
        from shapely.geometry import LineString
        import pandas as pd
        import networkx as nx

        def pixel_to_geo(path, origin, resolution):
            lon0, lat0 = origin
            dx, dy = resolution
            return [(lon0 + x * dx, lat0 + y * dy) for (x, y) in path]

        lane_rows = []
        turning_rows = []

        valid_lane_combinations = {
            ("end", "out"),
            ("in", "end")
        }
        valid_link_combinations = {
            ("out", "in"),
        }

        for src, src_data in lane_graph.nodes(data=True):
            for tgt, tgt_data in lane_graph.nodes(data=True):
                if src == tgt:
                    continue

                src_type = src_data.get("type")
                tgt_type = tgt_data.get("type")

                is_lane = (src_type, tgt_type) in valid_lane_combinations
                is_turn = (src_type, tgt_type) in valid_link_combinations

                if is_lane or is_turn:
                    try:
                        path = nx.shortest_path(lane_graph, src, tgt)
                    except nx.NetworkXNoPath:
                        continue

                    # Convert to geo coordinates
                    geo_path = pixel_to_geo(path, origin=origin, resolution=resolution)
                    geometry = LineString(geo_path)

                    row = {
                        "start_x": geo_path[0][0],
                        "start_y": geo_path[0][1],
                        "end_x": geo_path[-1][0],
                        "end_y": geo_path[-1][1],
                        "type": f"{src_type}->{tgt_type}",
                        "geometry": geometry
                    }

                    if is_lane:
                        lane_rows.append(row)
                    elif is_turn:
                        turning_rows.append(row)

        gdf_lanes = gpd.GeoDataFrame(lane_rows, geometry="geometry", crs=crs)
        gdf_turning = gpd.GeoDataFrame(turning_rows, geometry="geometry", crs=crs)

        gdf_lanes.to_file(lane_path, driver="GeoJSON")
        gdf_turning.to_file(turning_path, driver="GeoJSON")

        return gdf_lanes, gdf_turning


    def visualize_lanes(self, gdf, ax=None, color='blue', label='lane'):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        gdf.plot(ax=ax, color=color, linewidth=2, alpha=0.7, label=label)

        ax.set_aspect('equal')
        ax.set_title(f"Visualized {label}s")
        ax.grid(True)

        # Optional: Add start/end dots
        for _, row in gdf.iterrows():
            coords = list(row.geometry.coords)
            ax.plot(*coords[0], marker='o', color=color, markersize=4)   # start
            ax.plot(*coords[-1], marker='x', color=color, markersize=4)  # end

        return ax

    def extract_lane_graph(self, input_satellite_image_path, gpu_id=0):
        """
        Extracts lane graph from the satellite image.
        
        Args:
            satellite_image (np.ndarray): Input satellite image of shape (H, W, 3).
        
        Returns:
            nx.DiGraph: Extracted lane graph.
        """
        print("Extracting lane graph from the satellite image...")
        image_name = os.path.basename(input_satellite_image_path)
        input_satellite_image = imageio.imread(input_satellite_image_path)
        # # Step 1: Lane and Direction Extraction
        lane_graph = self._extract_lane_and_direction(input_satellite_image)
        lane_prediced, direction_predicted = segmentation2graph.draw_inputs(lane_graph, '.')
        # Step 2: Reachable Lane Extraction
        intersection_center = (self.image_size // 2, self.image_size // 2)
        reachable_node_paris = self.extract_valid_turining_pairs(lane_graph, 
                                                          input_satellite_image, 
                                                          direction_predicted, 
                                                          intersection_center, 
                                                          gpu_id)
        # Step 3: Lane Extraction
        lane_graph = self._extract_turning_lane(lane_graph, 
                                                reachable_node_paris, 
                                                input_satellite_image, 
                                                direction_predicted, 
                                                gpu_id,
                                                radius=5.0)
        for node in lane_graph.nodes:
            print(f"Node: {node}, Type: {lane_graph.nodes[node]['type']}")
        segmentation2graph.draw_output(lane_graph, '.')

        gdf_lanes, gdf_turning = self.extract_lanes(lane_graph,
            origin=(-122.0, 37.0),  # lon, lat
            resolution=(0.00001, -0.00001),
            lane_path="lanes.geojson",
            turning_path="turning_links.geojson"
        )

        ax = self.visualize_lanes(gdf_lanes, color='gray', label='lane')
        self.visualize_lanes(gdf_turning, ax=ax, color='red', label='turning')
        plt.savefig(f"lanes_{image_name}.png")
        return 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="options")
    parser.add_argument("--config", type=str, default="configs/lane_graph_extraction_pipline.py", help="config file")
    args = parser.parse_args()
    # ============= Load Configuration =============
    config = load_config(args.config)
    lane_graph_extraction = LaneGraphExtraction(config)
    input_satellite_image_path = "test_image_1.png"
    lane_graph_extraction.extract_lane_graph(input_satellite_image_path)