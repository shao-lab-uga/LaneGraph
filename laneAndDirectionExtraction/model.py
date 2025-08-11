import torch.nn as nn
import torch.nn.functional as F
from laneAndDirectionExtraction.HRNetW32_OCR import HRNetW32_OCR

class LaneAndDirectionExtractionModel(nn.Module):
    def __init__(self, lane_direction_extraction_model_config):
        super().__init__()
        self.lane_and_direction_extraction_model = HRNetW32_OCR(lane_direction_extraction_model_config)


    def forward(self, x):
        out = self.lane_and_direction_extraction_model(x)

        return out