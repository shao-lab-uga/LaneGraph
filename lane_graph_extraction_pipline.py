
import cv2
import argparse
import torch
import scipy.ndimage
import matplotlib.lines as mlines
from PIL import Image
import einops
import numpy as np
from typing import Tuple
from itertools import product
from skimage import morphology
import imageio.v3 as imageio
from scipy.spatial import KDTree
import os
import matplotlib.pyplot as plt
from pyproj import CRS
import utils.segmentation2graph as segmentation2graph
from utils.config_utils import load_config
from utils.inference_utils import load_model
import geopandas as gpd
from shapely.geometry import LineString
import pandas as pd
import networkx as nx
import pyproj
from turingLaneExtraction.model import LaneExtractionModel
from reachableLaneValidation.model import ReachableLaneValidationModel
from laneAndDirectionExtraction.model import LaneAndDirectionExtractionModel

from image_postprocessing import (
    normalize_image_for_model_input,
    post_process_model_output,
    encode_direction_vectors_to_image,
    denormalize_model_output
)
								  
									
							  
									  
							
 


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
            nx.DiGraph: Extracted lane graph with direction information.
        """
        print("Extracting lane and direction from the satellite image...")
        
																						
																									   
																										   

        normalized_image = normalize_image_for_model_input(input_satellite_image)
        normalized_image = torch.from_numpy(normalized_image).float().to(gpu_id)  # [H, W, 3]
        normalized_image = einops.rearrange(normalized_image, 'h w c -> 1 c h w')  # [1, 3, H, W]
									 
							
									 
												   
		   

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
        # Image.fromarray(lane_predicted_image.astype(np.uint8)).save("lane_predicted.jpg")

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

        connector_features[0,:,:,6] = normalize_image_for_model_input(connectorlink)

        connector_features[0, :, :, 0] = normalize_image_for_model_input(connector1)
        connector_features[0, :, :, 1:3] = self.poscode[
            self.image_size - x1 : self.image_size * 2 - x1,
            self.image_size - y1 : self.image_size * 2 - y1,
        ]

        connector_features[0, :, :, 3] = normalize_image_for_model_input(connector2)
        connector_features[0, :, :, 4:6] = self.poscode[
            self.image_size - x2 : self.image_size * 2 - x2,
            self.image_size - y2 : self.image_size * 2 - y2,
        ]

        input_image = np.expand_dims(normalize_image_for_model_input(input_satellite_image), axis=0)
        direction_context = np.expand_dims(denormalize_model_output(direction_context), axis=0)

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
        connector_features[0,:,:,6] = normalize_image_for_model_input(connectorlink)

        connector_features[0, :, :, 0] = normalize_image_for_model_input(connector1)
        connector_features[0, :, :, 1:3] = self.poscode[
            self.image_size - x1 : self.image_size * 2 - x1,
            self.image_size - y1 : self.image_size * 2 - y1,
        ]

        connector_features[0, :, :, 3] = normalize_image_for_model_input(connector2)
        connector_features[0, :, :, 4:6] = self.poscode[
            self.image_size - x2 : self.image_size * 2 - x2,
            self.image_size - y2 : self.image_size * 2 - y2,
        ]

        input_image = np.expand_dims(normalize_image_for_model_input(input_satellite_image), axis=0)
        direction_context = np.expand_dims(denormalize_model_output(direction_context), axis=0)

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
        reachable_node_pairs = []
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
            y1, x1 = node_pair[0]
            y2, x2 = node_pair[1]
            color = (255, 0, 0) if predicted_label.item() == 1 else (0, 0, 255)
            cv2.circle(output_image, (x1, y1), 12, color, -1)
            cv2.circle(output_image, (x2, y2), 12, color, -1)
            

																										
            if predicted_label.item() == 1:
                reachable_node_pairs.append(node_pair)

        return reachable_node_pairs

    def _extract_turning_lane(self, lane_graph, reachable_node_pairs, input_satellite_image, direction_context, gpu_id=0, radius=5.0):

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


            lane_predicted_image, _ = post_process_model_output(
                outputs.cpu().numpy(), 
                threshold=20, 
                morphology_size=(6, 6),
                apply_thinning=True
            )

		# outputs = self.lane_and_direction_extraction_model(normalized_image) # [B, 4, H, W]

        # lane_predicted_image, direction_predicted_image = post_process_model_output(
            # outputs.cpu().numpy(), 
            # threshold=64, 
            # morphology_size=(6, 6),
            # save_debug_image="lane_predicted.jpg"
        # )																									 
																								
															
																		

													
            start_node, end_node = reachable_node_pair
            link_graph = segmentation2graph.extract_graph_from_image(
                lane_predicted_image, start_pos=start_node
            )

            link_nodes = list(link_graph.nodes)
            if len(link_nodes) == 0:
                print(f"Warning: link_graph is empty for pair {reachable_node_pair}")
                continue


            for node in link_nodes:
                if node not in lane_graph.nodes:
                    lane_graph.add_node(node, type='link')
            for u, v in link_graph.edges:
                lane_graph.add_edge(u, v)


            first_node = link_nodes[0]
            if first_node != start_node and out_kdtree:
                dist, idx = out_kdtree.query(first_node, distance_upper_bound=radius)
                if dist < radius:
                    nearest_out = out_nodes[idx]
                    lane_graph.add_edge(nearest_out, first_node)
                else:
                    print(f"No 'out' node found near {first_node} within radius {radius}")


																				  
            last_node = link_nodes[-1]
            if last_node != end_node and in_kdtree:
                dist, idx = in_kdtree.query(last_node, distance_upper_bound=radius)
                if dist < radius:
                    nearest_in = in_nodes[idx]
                    lane_graph.add_edge(last_node, nearest_in)
                else:
                    print(f"No 'in' node found near {last_node} within radius {radius}")

        return lane_graph

    


    def pixel_to_geo(self, path, origin=(0, 0), resolution = (0.125, -0.125)):
        """Convert a pixel path to geographic coordinates (lon, lat or meters)."""
        lon0, lat0 = origin
        dx, dy = resolution
        return [(lon0 + x * dx, lat0 + y * dy) for (x, y) in path]

    def extract_lanes_and_links_with_fids(self, lane_graph, origin=(0, 0), resolution=(0.125, -0.125), output_path="lane_links.geojson"):

        proj = pyproj.Proj(proj="utm", zone=51, datum="WGS84")  # use proper UTM zone
        origin_lon, origin_lat = origin
        origin_x, origin_y = proj(origin_lon, origin_lat)  # Convert origin to UTM coordinates
        origin_m = (origin_x, origin_y)
        def pixel_to_meter(path, origin_m, resolution):
            x0, y0 = origin_m
            dx, dy = resolution
            return [(x0 + x * dx, y0 + y * dy) for (y, x) in path]
    
        rows = []
        fid_counter = 0
        node_to_fid = {}  # Maps (start_node, end_node) to fid for later reverse lookups
    
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
                is_link = (src_type, tgt_type) in valid_link_combinations
    
                if is_lane or is_link:
                    try:
                        path = nx.shortest_path(lane_graph, src, tgt)
                    except nx.NetworkXNoPath:
                        continue
                       
                    geo_path = pixel_to_meter(path, origin_m=origin_m, resolution=resolution)
                    geometry = LineString(geo_path)
    
                    fid = fid_counter
                    fid_counter += 1
    
                    row = {
                        "fid": str(fid),
                        "type": "lane" if is_lane else "link",
                        "subtype": f"{src_type}->{tgt_type}",
                        "start_node_id": str(src),
                        "end_node_id": str(tgt),
                        "start_type": src_type,
                        "end_type": tgt_type,
                        "start_x": geo_path[0][0],
                        "start_y": geo_path[0][1],
                        "end_x": geo_path[-1][0],
                        "end_y": geo_path[-1][1],
                        "edge_nodes": str(path),
                        "geometry": geometry,
                        "from_fid": None,  # To be filled in second pass
                        "to_fid": None     # To be filled in second pass
                    }
    
                    node_to_fid[(str(src), str(tgt))] = fid
                    rows.append(row)
    
        # --- Second pass: assign from_fid and to_fid for links ---
        for row in rows:
            if row["type"] == "link":
                src = row["start_node_id"]
                tgt = row["end_node_id"]
    
                # Find lanes that connect to this link
                for candidate in rows:
                    if candidate["type"] == "lane":
                        # Upstream lane → this link
                        if candidate["end_node_id"] == src:
                            row["from_fid"] = str(candidate["fid"])
                        # This link → downstream lane
                        if candidate["start_node_id"] == tgt:
                            row["to_fid"] = str(candidate["fid"])
         # --- Second pass: assign from_fid and to_fid for lanes ---
        for row in rows:
            if row["type"] == "lane":
                src = row["start_node_id"]
                tgt = row["end_node_id"]
    
                # Find links that connect to this lane
                for candidate in rows:
                    if candidate["type"] == "link":
                        # Upstream link → this lane
                        if candidate["end_node_id"] == src:
                            row["from_fid"] = str(candidate["fid"])
                        # This lane → downstream link
                        if candidate["start_node_id"] == tgt:
                            row["to_fid"] = str(candidate["fid"])
        
        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=proj.srs)
        gdf.drop(columns=["start_node_id", "end_node_id", "subtype", "edge_nodes"], inplace=True)
        gdf.drop(columns=["start_x", "start_y", "end_x", "end_y"], inplace=True)
        gdf.to_file(output_path, driver="GeoJSON")
    
        return gdf


    def visualize_lanes_and_links(self, gdf, ax=None, lane_color='blue', link_color='red', show_labels=True):

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
        lane_prediced, direction_predicted = segmentation2graph.draw_inputs(lane_graph, save_path=None)
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
        # for node in lane_graph.nodes:
        #     print(f"Node: {node}, Type: {lane_graph.nodes[node]['type']}")
        segmentation2graph.draw_output(lane_graph, '.')

        lanes_and_links_gdf = self.extract_lanes_and_links_with_fids(lane_graph,
            origin=(0, -87), # Assuming origin is (0, -87) for simplicity.
            resolution=(0.125, -0.125), # 0.125 meters/pixel.
            output_path="lanes_and_links.geojson"
        )

        ax = self.visualize_lanes_and_links(lanes_and_links_gdf)
        plt.savefig(f"lanes_and_links_{image_name}.png")
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