import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from turingLaneExtraction.model_segmentation import UnetResnet34

class LaneExtractionModel(nn.Module):
    def __init__(self, lane_extraction_model_config):
        super().__init__()
        self.lane_extraction_model = UnetResnet34(lane_extraction_model_config)
        

    def forward(self, input_features_node):
        """
        Forward pass for the Lane Extraction model.
        Args:
            input_features_node_a: Input features for node A.
            input_features_node_b: Input features for node B.
        Returns:
            lane_predicted_a: Predicted lanes for node A.
            lane_predicted_b: Predicted lanes for node B.
        """
        lane_predicted = self.lane_extraction_model(input_features_node)


        return lane_predicted