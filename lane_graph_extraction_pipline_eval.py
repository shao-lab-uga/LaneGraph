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
                                      assign_lane_ids_per_group
                                      )

from utils.connection_filtering import filter_connections_receive_aware
from utils.gdf_visualize_utils import visualize_road_groups
from utils.image_postprocessing_utils import encode_direction_vectors_to_image


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
        Image.fromarray(((output_lane_mask_image) * 255).astype(np.uint8) ).save(os.path.join(output_path, f"lane_mask_{image_name}.png"))


        output_direction_image[:,:,2] = np.clip(output_direction_image[:,:,0],-1,1) * 127 + 127
        output_direction_image[:,:,1] = np.clip(output_direction_image[:,:,1],-1,1) * 127 + 127
        output_direction_image[:,:,0] = 127

        Image.fromarray(output_direction_image.astype(np.uint8)).save(os.path.join(output_path, f"direction_{image_name}.png"))
        output_lane_mask_image = morphology.thin(output_lane_mask_image)
        #
        lane_graph = segmentation2graph.extract_graph_from_image(output_lane_mask_image)
        direction_map = np.zeros((satellite_image_tile_height, satellite_image_tile_width, 2))
        direction_map[:,:,0] = (output_direction_image[:,:,2].astype(np.float32) - 127) / 127.0
        direction_map[:,:,1] = (output_direction_image[:,:,1].astype(np.float32) - 127) / 127.0
        lane_graph = segmentation2graph.direct_graph_from_direction_map(lane_graph, direction_map)
        return lane_graph
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="options")
    parser.add_argument("--config", type=str, default="configs/lane_graph_extraction_pipline.py", help="config file")
    args = parser.parse_args()
    # ============= Load Configuration =============
    config = load_config(args.config)
    lane_graph_extraction = LaneGraphExtraction(config, gpu_id=0)
    input_satellite_img_path = "test_tile.jpg"  # Path to the input satellite image
    lane_graph = lane_graph_extraction.extract_lane_and_direction_from_tile(input_satellite_img_path, gpu_id=0, output_path='./')
    segmentation2graph.draw_directed_graph(lane_graph, save_path='./', image_name=f"test_tile_graph", region_size=4096)