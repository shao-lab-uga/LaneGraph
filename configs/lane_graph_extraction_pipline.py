import os
from utils.config_utils import load_config
# ============= Dataset Parameters=================
dataset_config = load_config("configs/dataset.py")


paths_config = dataset_config.paths
data_attributes_config = dataset_config.data_attributes
processed_data_path = paths_config.processed_data_path
dataset_image_size = data_attributes_config.dataset_image_size
input_image_size = data_attributes_config.input_image_size

training_range = data_attributes_config.training_range
testing_range = data_attributes_config.testing_range
validation_range = data_attributes_config.validation_range

batch_size = 8
preload_tiles=4

# ============= Config ===================
config = dict(
    dataset_config=dataset_config,
    dataloaders=dict(
        train=dict(
            data_path=processed_data_path,
            image_size=input_image_size,
            dataset_image_size=dataset_image_size,
            preload_tiles=preload_tiles,
            batch_size=batch_size,
            indrange=training_range,
            training=True,  # Indicates this is for training
        ),
        test=dict(
            data_path=processed_data_path,
            image_size=input_image_size,
            dataset_image_size=dataset_image_size,
            preload_tiles=preload_tiles,
            batch_size=1,
            indrange=testing_range,
            training=False,  # Indicates this is for testing
        ),
        validate=dict(
            data_path=processed_data_path,
            image_size=input_image_size,
            dataset_image_size=dataset_image_size,
            preload_tiles=preload_tiles,
            batch_size=batch_size,
            indrange=validation_range,
            training=False,  # Indicates this is for validation
        ),
    ),

    models=dict(
        lane_and_direction_extraction_model=dict(
            in_channels = 3,  # Input channels for RGB images
            num_classes = 4,  # Output dimension for lane and direction extraction
            weight_path="exp/LaneAndDirectionExtraction/checkpoints/epoch_310.pth",
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
            weight_path="exp/ReachableLaneValidation/checkpoints/epoch_400.pth",
        ),
        lane_extraction_model=dict(
            in_channels = 14,  
            num_classes = 2, 
            weight_path="exp/LaneExtraction/checkpoints/epoch_150.pth",
        ),
    )   

)