import torch.nn as nn
import torch.nn.functional as F
from laneAndDirectionExtraction.model_segmentation import UnetResnet34

class LaneAndDirectionExtractionModel(nn.Module):
    def __init__(self, lane_direction_extraction_model_config):
        super().__init__()
        self.lane_and_direction_extraction_model = UnetResnet34(lane_direction_extraction_model_config)


    def forward(self, x):
        out = self.lane_and_direction_extraction_model(x)

        return out