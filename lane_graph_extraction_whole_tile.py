import os
import re
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
from utils.connection_filtering import cluster_intersections_by_roads

class LaneGraphExtraction():
    

    def __init__(self, config, gpu_id=0):
        self.regionsize = config.dataset_config.data_attributes.regionsize
        # the input image size for the lane and direction extraction model
        self.image_size = config.dataset_config.data_attributes.input_image_size
        self.windowsize1 = 256
        self.windowsize2 = 512
        self.margin = (self.image_size - self.windowsize1) // 2
        self.margin2 = (self.image_size - self.windowsize2) // 2

        self.config = config
        self.gpu_id = gpu_id
        self.models_config = config.models

        self.lane_and_direction_extraction_model_config = self.models_config.lane_and_direction_extraction_model
        self.lane_and_direction_extraction_model = LaneAndDirectionExtractionModel(
            self.lane_and_direction_extraction_model_config
        )

       
        self._load_weights(self.gpu_id)
        
    def _load_weights(self, gpu_id=0):
        load_model(
            self.lane_and_direction_extraction_model,
            self.lane_and_direction_extraction_model_config.weight_path
        )
        self.lane_and_direction_extraction_model.to(gpu_id)


    def _extract_lane_and_direction(self, input_satellite_image: np.ndarray, gpu_id=0):

        print("Extracting lane and direction from the satellite image...")
        # normalize the input image
        input_satellite_image = torch.from_numpy(input_satellite_image).float().to(gpu_id)  # [H, W, 3]
        input_satellite_image = einops.rearrange(input_satellite_image, 'h w c -> 1 c h w')  # [1, 3, H, W]

        with torch.no_grad():
            lane_predicted, direction_predicted = self.lane_and_direction_extraction_model(input_satellite_image) # [B, 4, H, W]
        lane_predicted = lane_predicted.cpu().numpy()
        direction_predicted = direction_predicted.cpu().numpy()

        lane_predicted = einops.rearrange(lane_predicted, 'b c h w -> b h w c')
        
        direction_predicted = einops.rearrange(direction_predicted, 'b c h w -> b h w c')
        return lane_predicted, direction_predicted

    def extract_lane_and_direction_from_tile(self, input_satellite_image_tile_path: str, gpu_id=0, output_path=None):
        input_satellite_image_tile: np.ndarray = imageio.imread(input_satellite_image_tile_path)
        image_name = os.path.basename(input_satellite_image_tile_path).split('.')[0]
        # normalize the input image
        input_satellite_image_tile = (input_satellite_image_tile / 255.0 - 0.5) * 0.81
        satellite_image_tile_height, satellite_image_tile_width, _ = np.shape(input_satellite_image_tile)
        input_satellite_image_tile = np.pad(input_satellite_image_tile, ((self.margin, self.margin),(self.margin, self.margin),(0,0)), 'constant')

        mask = np.zeros((self.image_size,self.image_size,3)) 
        for i in range((self.windowsize2 - self.windowsize1) // 2 ):
            r = i / float((self.windowsize2 - self.windowsize1) // 2)
            mask[self.margin2+i:-(self.margin2+i-1),self.margin2+i:-(self.margin2+i-1),:] = r 

        output_lane_mask_image = np.zeros((satellite_image_tile_height + 2 * self.margin, satellite_image_tile_width + 2 * self.margin, 2))
        output_direction_image = np.zeros((satellite_image_tile_height + 2 * self.margin, satellite_image_tile_width + 2 * self.margin, 3))
        weights = np.zeros((satellite_image_tile_height + 2 * self.margin, satellite_image_tile_width + 2 * self.margin, 3)) + 0.0001

        for i in range(satellite_image_tile_height // self.windowsize1):
            for j in range(satellite_image_tile_width // self.windowsize1):

                r = i * self.windowsize1
                c = j * self.windowsize1
                input_satellite_image_patch = input_satellite_image_tile[r:r+self.image_size, c:c+self.image_size, :]
                lane_predicted, direction_predicted = self._extract_lane_and_direction(input_satellite_image_patch, gpu_id)

                output_lane_mask_image[r:r+self.image_size, c:c+self.image_size, :] += lane_predicted[0, :, :, :] * mask[:, :, 0:2]
                output_direction_image[r:r+self.image_size, c:c+self.image_size, 0:2] += direction_predicted[0, :, :, :] * mask[:, :, 0:2]
                weights[r:r+self.image_size, c:c+self.image_size, :] += mask[:, :, 0:1]

        output_lane_mask_image = np.divide(output_lane_mask_image, weights[:, :, 0:2])
        output_direction_image = np.divide(output_direction_image, weights)
        output_lane_mask_image = output_lane_mask_image[self.margin:-self.margin, self.margin:-self.margin, :]
        output_direction_image = output_direction_image[self.margin:-self.margin, self.margin:-self.margin, :]

        expsum = np.exp(output_lane_mask_image[:, :, 0]) + np.exp(output_lane_mask_image[:, :, 1])
        output_lane_mask_image = (np.clip(np.exp(output_lane_mask_image[:, :, 0]) / expsum, 0, 1) * 255)
        output_lane_mask_image = scipy.ndimage.grey_closing(output_lane_mask_image, size=(6,6))
        threshold = 64
        output_lane_mask_image = output_lane_mask_image >= threshold
        Image.fromarray(((output_lane_mask_image) * 255).astype(np.uint8)).save(os.path.join(output_path, f"lane_mask_{image_name}.png"))


        output_direction_image[:,:,2] = np.clip(output_direction_image[:,:,0],-1,1) * 127 + 127
        output_direction_image[:,:,1] = np.clip(output_direction_image[:,:,1],-1,1) * 127 + 127
        output_direction_image[:,:,0] = 127
        Image.fromarray(output_direction_image.astype(np.uint8)).save(os.path.join(output_path, f"direction_{image_name}.png"))

        

    def extract_lane_graph_from_lane_and_direction(self, lane_mask_path: str, direction_field_path: str):
        print("Extracting lane graph from the lane mask and direction image...")
        lane_mask_image = imageio.imread(lane_mask_path)
        direction_image = imageio.imread(direction_field_path)
        satellite_image_tile_height, satellite_image_tile_width = np.shape(lane_mask_image)
        lane_mask_image = morphology.thin(lane_mask_image)
        non_directed_lane_graph = segmentation2graph.extract_graph_from_image(lane_mask_image)
        direction_map = np.zeros((satellite_image_tile_height, satellite_image_tile_width, 2))
        direction_map[:,:,0] = (direction_image[:,:,2].astype(np.float32) - 127) / 127.0
        direction_map[:,:,1] = (direction_image[:,:,1].astype(np.float32) - 127) / 127.0
        directed_lane_graph = segmentation2graph.direct_graph_from_direction_map(non_directed_lane_graph, direction_map)
        return non_directed_lane_graph, directed_lane_graph
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
        # filter node pairs based on distance
        print(f"Filtering node pairs based on distance threshold: {distance_threshold}... Original pairs: {len(node_pairs)}")
        node_pairs = [ (out_node, in_node) for out_node, in_node in node_pairs
            if np.linalg.norm(np.array(lane_graph.nodes[out_node].get("pos", (0, 0))) - np.array(lane_graph.nodes[in_node].get("pos", (0, 0)))) < distance_threshold
        ]
        print(f"Total node pairs to evaluate: {len(node_pairs)}")
        connections = {}
        plt.figure(figsize=(40,40))
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
            plt.scatter(out_node_x, out_node_y, c='r', label='out' if idx == 0 else "")
            plt.scatter(in_node_x, in_node_y, c='b', label='in' if idx == 0 else "")
            if idx == 0:
                idx += 1
                plt.legend()
            
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
                    plt.plot([out_node_x, in_node_x], [out_node_y, in_node_y], c='g')
                
        plt.title(f"Extracted Connections: {len(connections)}")
        plt.savefig("extracted_connections.png")
        return connections
    
    def extract_lane_info(self, lanes_gdf: gpd.GeoDataFrame, save_path: str = None) -> gpd.GeoDataFrame:
        """
        Annotates road information to the lane graph based on the lanes GeoDataFrame.

        Args:
            lane_graph (nx.DiGraph): Input lane graph.
            lanes_gdf (gpd.GeoDataFrame): GeoDataFrame containing lanes information.

        """
        junction_points = get_junction_points(lanes_gdf)
        lanes_gdf_splitted = split_lines_at_junctions(
            lanes_gdf,
            junction_points,
            tol=20.0,        # tolerance in your CRS units
            min_gap=0.5,    # avoid over-splitting at near-duplicate cuts
            min_seg_len=0.5 # drop tiny fragments
        )
        lanes_gdf_grouped = group_lanes_by_geometry(
            lanes_gdf_splitted,
            spacing=0.5,          # Resample step along each line (in CRS units, e.g., meters).
            avg_tol=30,          # Max symmetric avg. perpendicular distance to treat lanes as the same corridor.
            endpoint_tol=10,     # Endpoint proximity threshold: at least one valid endpoint pair must be within this.
            angle_tol_deg=20.0,   # Max heading difference to be considered "same direction" (excludes ~180° flips).
            serial_reject_tol=1.5,# If start↔end or end↔start is <= this (when same direction), reject as serial (head-to-tail).
            min_overlap_ratio=0.2,# Min fraction of the shorter line that must overlap the longer one (0..1).
            pair_expand=10.0,     # Search radius to gather candidate neighbors before detailed checks.
            angle_avg_tol_deg=20.0 # Mean heading difference threshold
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
    # input_satellite_img_path = "test_tile.jpg"  # Path to the input satellite image
    # lane_graph_extraction.extract_lane_and_direction_from_tile(input_satellite_img_path, gpu_id=0, output_path='./')
    lane_mask_path = "lane_mask_test_tile.png"
    direction_field_path = "direction_test_tile.png"
    _, directed_lane_graph = lane_graph_extraction.extract_lane_graph_from_lane_and_direction(lane_mask_path, direction_field_path)
    # directed_lane_graph = annotate_node_types(directed_lane_graph)
    directed_lane_graph = annotate_node_types(directed_lane_graph)
    directed_lane_graph = connect_nearby_dead_ends(directed_lane_graph, connection_threshold=16)
    directed_lane_graph = refine_lane_graph(directed_lane_graph, isolated_threshold=30, spur_threshold=0)
    directed_lane_graph = annotate_node_types(directed_lane_graph)
    directed_lane_graph = refine_lane_graph_with_curves(directed_lane_graph)
    directed_lane_graph = annotate_node_types(directed_lane_graph)
    directed_lane_graph = refine_lane_graph(directed_lane_graph, isolated_threshold=30, spur_threshold=0)
    directed_lane_graph = annotate_node_types(directed_lane_graph)
    # segmentation2graph.draw_directed_graph(directed_lane_graph, save_path='./', image_name=f"test_tile_graph", region_size=4096)
    lanes_and_links_gdf, lane_graph_with_fids = lane_graph_extraction.extract_lanes_and_links_geojson(directed_lane_graph,
                origin=(0, 0),
                resolution=(0.125, -0.125),
                output_path='./',
                crs_proj=None
            )
    lanes_gdf = lanes_and_links_gdf[lanes_and_links_gdf['type'] == 'lane'].reset_index(drop=True)
    # get lane information (e.g., lane id, direction, road id)
    lanes_gdf, reference_lines, fids_to_remove = lane_graph_extraction.extract_lane_info(lanes_gdf, save_path=None)
    # remove nodes and edges with fids_to_remove
    lanes_gdf = lanes_gdf[~lanes_gdf['fid'].isin(fids_to_remove)].reset_index(drop=True)
    reference_lines = {road_id: line for road_id, line in reference_lines.items() if road_id in lanes_gdf['road_id'].values}
    lane_graph_with_fids.remove_nodes_from([n for n, d in lane_graph_with_fids.nodes(data=True) if d.get("fid") in fids_to_remove])
    lane_graph_with_fids.remove_edges_from([ (u, v) for u, v, d in lane_graph_with_fids.edges(data=True) if d.get("fid") in fids_to_remove])
    # [ ] convert the geometry back to image coordinates
    lanes_gdf['geometry'] = lanes_gdf['geometry'].apply(lambda line: LineString([(x / 0.125, y / -0.125) for x, y in line.coords]))
    # also the reference lines
    reference_lines = {road_id: LineString([(x / 0.125, y / -0.125) for x, y in line.coords]) for road_id, line in reference_lines.items()}
    # connections = lane_graph_extraction.extract_connections_rule_based(directed_lane_graph, 
    #                                                             segment_max_length=5, 
    #                                                             distance_threshold=400,
    #                                                             turning_threshold=-0.8, 
    #                                                             topology_threshold=0.8)
    centers = cluster_intersections_by_roads(
        lane_graph_with_fids,
        lanes_gdf,
    )
    print(f"Total clustered intersections: {len(centers)}")
    fig, ax = plt.subplots(figsize=(20,20))
    segmentation2graph.draw_directed_graph(lane_graph_with_fids, ax=ax, region_size=4096)
    for center_id, center in centers.items():
        ax.scatter(center.x, center.y, c='r', s=200, marker='x')
    plt.savefig("clustered_intersections.png")