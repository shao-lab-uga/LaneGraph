import os
from utils.config_utils import load_config
# ============= Seed ===================
random_seed = 42
# ============= Dataset Parameters=================
dataset_config = load_config("configs/dataset.py")


paths_config = dataset_config.paths
data_attributes_config = dataset_config.data_attributes
testing_data_path = paths_config.testing_data_path
dataset_image_size = data_attributes_config.dataset_image_size
input_image_size = data_attributes_config.input_image_size

testing_range = data_attributes_config.testing_range
num_machines = 1
gpu_ids = [0]
batch_size = 1
preload_tiles=4
epoch_size = len(testing_range) * dataset_image_size * dataset_image_size // (batch_size * input_image_size * input_image_size)
max_epochs = 5

# ============= Config ===================
config = dict(

    dataset_config=dataset_config,
    test=dict(
        random_seed=random_seed,
        batch_size=batch_size,
        epoch_size=epoch_size,
        max_epochs=max_epochs,  
    ),
    dataloaders=dict(
        test=dict(
            data_path=testing_data_path,
            image_size=input_image_size,
            dataset_image_size=dataset_image_size,
            preload_tiles=preload_tiles,
            batch_size=1,
            indrange=testing_range,
            training=False,  # Indicates this is for testing
        ),

    ),

)