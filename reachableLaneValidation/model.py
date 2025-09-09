import torch
import torch.nn as nn
import torch.nn.functional as F
from reachableLaneValidation.model_segmentation import UnetResnet34
from reachableLaneValidation.model_classification import Resnet34Classifier

class ReachableLaneValidationModel(nn.Module):
    def __init__(self, reachable_lane_extraction_model_config, reachable_lane_validation_model_config):
        super().__init__()
        self.reachable_lane_extraction_model = UnetResnet34(reachable_lane_extraction_model_config)
        self.reachable_lane_validation_model = Resnet34Classifier(reachable_lane_validation_model_config)
        

    def forward(self, input_features_node_a, input_features_node_b, input_features_validation):
        """
        Forward pass for the Reachable Lane Extraction and Validation model.
        Args:
            input_features_node_a: Input features for node A.
            input_features_node_b: Input features for node B.
        Returns:
            lane_predicted_a: Predicted lanes for node A.
            lane_predicted_b: Predicted lanes for node B.
            reachable_label_predicted: Predicted reachable label.
        """
        reachable_lane_predicted_node_a = self.reachable_lane_extraction_model(input_features_node_a)
        reachable_lane_predicted_node_b = self.reachable_lane_extraction_model(input_features_node_b)

        input_features_validation = torch.cat([
            F.softmax(reachable_lane_predicted_node_a, dim=1)[:, 0:1, :, :],
            F.softmax(reachable_lane_predicted_node_b, dim=1)[:, 0:1, :, :],
            input_features_validation
        ], dim=1)

        reachable_label_predicted = self.reachable_lane_validation_model(input_features_validation)

        return reachable_lane_predicted_node_a, reachable_lane_predicted_node_b, reachable_label_predicted