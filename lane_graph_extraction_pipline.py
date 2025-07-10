
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
import os
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

        
        in_nodes = node_types.get("in", [])
        out_nodes = node_types.get("out", [])
        
        node_pairs = list(product(in_nodes, out_nodes))
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
    
    def _extract_turning_lane(self, lane_graph, reachable_node_pairs, input_satellite_image, direction_context, gpu_id=0):

        for idx, reachable_node_pair in enumerate(reachable_node_pairs):
            input_features = self._make_lane_extraction_input(
                reachable_node_pair, input_satellite_image, direction_context, gpu_id
            )
            with torch.no_grad():
                # Extract lane
                outputs = self.lane_extraction_model(input_features)
            outputs = outputs.cpu().numpy()
            lane_predicted = outputs[:, 0:2, :, :]  # [B, 2, H, W]
            lane_predicted = einops.rearrange(lane_predicted, 'b c h w -> b h w c')
            lane_predicted_image = np.zeros((self.image_size, self.image_size))
            lane_predicted_image[:, :] = (np.clip(lane_predicted[0,:,:,0], 0, 1) * 255).astype(np.uint8)
            # lane_predicted_image = scipy.ndimage.grey_closing(lane_predicted_image, size=(6,6))
            lane_predicted_image = lane_predicted_image > 20
            lane_predicted_image = morphology.thin(lane_predicted_image)
            # Image.fromarray(lane_predicted_image).save(f"turning_lane_predicted_{idx}.jpg")
            
            link_graph = segmentation2graph.extract_graph_from_image(lane_predicted_image, start_pos=reachable_node_pair[0])
            # add the drived link to the lane graph (which is a directed graph), start from the first node

            start_node, end_node = reachable_node_pair

            # Add start and end to lane_graph if missing
            if start_node not in lane_graph.nodes:
                lane_graph.add_node(start_node, type='in')
            if end_node not in lane_graph.nodes:
                lane_graph.add_node(end_node, type='out')

            # Copy all nodes and edges from link_graph to lane_graph
            for node in link_graph.nodes:
                if node not in lane_graph.nodes:
                    lane_graph.add_node(node, type='link')
            for u, v in link_graph.edges:
                lane_graph.add_edge(u, v)

            # Build KDTree to find closest/farthest nodes
            link_nodes = np.array(list(link_graph.nodes))
            if len(link_nodes) == 0:
                print("Warning: link_graph is empty.")
                return


        return lane_graph
    
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
                                                gpu_id)
        segmentation2graph.draw_output(lane_graph, '.')
        return 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="options")
    parser.add_argument("--config", type=str, default="configs/lane_graph_extraction_pipline.py", help="config file")
    args = parser.parse_args()
    # ============= Load Configuration =============
    config = load_config(args.config)
    lane_graph_extraction = LaneGraphExtraction(config)
    input_satellite_image_path = "image.png"
    lane_graph_extraction.extract_lane_graph(input_satellite_image_path)