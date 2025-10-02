import os
from utils.config_utils import load_config
# ============= Dataset Parameters=================
dataset_config = load_config("configs/dataset.py")
paths_config = dataset_config.paths
data_attributes_config = dataset_config.data_attributes
dataset_image_size = data_attributes_config.dataset_image_size
input_image_size = data_attributes_config.input_image_size



# ============= Config ===================
config = dict(
    dataset_config=dataset_config,


    models=dict(
        lane_and_direction_extraction_model=dict(
            in_channels = 3,  # Input channels for RGB images
            num_classes = 4,  # Output dimension for lane and direction extraction
            weight_path="exp/LaneAndDirectionExtraction/checkpoints/epoch_250.pth",
        ),
        reachable_lane_extraction_validation_model=dict(
            reachable_lane_extraction_model=dict(
                in_channels = 10,  
                num_classes = 2, 
            ),
            reachable_lane_validation_model=dict(
                in_channels = 13,  
                num_classes = 2,  
            ),
            weight_path="exp/ReachableLaneValidation/checkpoints/epoch_150.pth",
        ),
        lane_extraction_model=dict(
            in_channels = 14,  
            num_classes = 2, 
            weight_path="exp/LaneExtraction/checkpoints/epoch_220.pth",
        ),
    )   

)