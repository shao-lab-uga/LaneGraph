import os
from utils.config_utils import load_config
# ============= Seed ===================
random_seed = 42
# ============= Path ===================
project_name = 'LaneAndDirectionExtractionEvaluation'  # Name of the project
exp_dir = './exp/'  # PATH TO YOUR EXPERIMENT FOLDER
project_dir = os.path.join(exp_dir, project_name)
# ============= Dataset Parameters=================
dataset_config = load_config("configs/dataset_lane_and_direction_extraction.py")


paths_config = dataset_config.paths
data_attributes_config = dataset_config.data_attributes
processed_data_path = paths_config.processed_data_path
dataset_image_size = data_attributes_config.dataset_image_size
input_image_size = data_attributes_config.input_image_size

training_range = data_attributes_config.training_range
testing_range = data_attributes_config.testing_range
validation_range = data_attributes_config.validation_range
# ============= Model Parameters =================

# ============= Test Parameters =================
num_machines = 1
gpu_ids = [0]
batch_size = 6
preload_tiles=4
guidance_scale = 7.5
epoch_sisze = len(training_range) * dataset_image_size * dataset_image_size // (batch_size * input_image_size * input_image_size)
max_epochs = 1  # Total number of epochs to test
weight_path = 'exp/LaneAndDirectionExtractionTraining/checkpoints/epoch_150.pth'
# ============= Config ===================
config = dict(
    project_name=project_name,
    project_dir=project_dir,
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
            num_classes = 4,  # Output dimension for lane and direction extraction
        )
    ),
    test=dict(
        checkpoint_path=weight_path,  # Path to the model checkpoint
        max_epochs=max_epochs,
        epoch_size=epoch_sisze,
        visualize_output_path=os.path.join(project_dir, 'visualizations'),
        log_interval=10,
    ),

)