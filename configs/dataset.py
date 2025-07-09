import os
import json
dataset = 'SatelliteLane'

# ============= Path ===================
dataset_root_dir = "/mnt/c/Users/hg25079/Documents/GitHub/LaneGraph"
raw_data_path = os.path.join(dataset_root_dir, 'raw_data')
processed_data_path = os.path.join(dataset_root_dir, 'processed_data')
dataset_split_path = os.path.join(dataset_root_dir, 'dataset_split.json')
with open(dataset_split_path, 'r') as f:
    dataset_split = json.load(f)

# ============= Data Parameters =================
regionsize = 4096
stride = 4096
tilesize = 4096
res = 8
dataset_image_size = 2048  # the size of the image in pixels (2048x2048 which is cropped from 4096x4096)
input_image_size = 640 # the size of the input image in pixels (640x640 which is cropped from 2048x2048)
training_range = [] # This is the ids of the training data, which is used to load the data in the training process.
testing_range = [] # This is the ids of the testing data, which is used to load the data in the testing process.
validation_range = [] # This is the ids of the validation data, which is used to load the data in the validation process.

# [ ]: Right now, there are 35 satellite images in the dataset, and each image is divided into 9 tiles.
# So the total number of tiles is 35 * 9 = 315. Below is the code to generate the training, testing, and validation ranges based on the dataset split.
# Load the dataset split from the json file
for tile_id in dataset_split["training"]:
    for i in range(9):
        training_range.append("%d" % (tile_id*9+i))
for tile_id in dataset_split["testing"]:
    for i in range(9):
        testing_range.append("%d" % (tile_id*9+i))
for tile_id in dataset_split["validation"]:
    for i in range(9):
        validation_range.append("%d" % (tile_id*9+i))
# ============= Task General Parameters =================

        
# ============= Config ===================
config = dict(
    paths=dict(
        dataset_root_dir=dataset_root_dir,
        raw_data_path=os.path.join(dataset_root_dir, raw_data_path),
        processed_data_path=os.path.join(dataset_root_dir, processed_data_path),
    ),
    data_attributes=dict(
        regionsize=regionsize,
        stride=stride,
        tilesize=tilesize,
        res=res,
        dataset_image_size=dataset_image_size,
        input_image_size=input_image_size,
        training_range=training_range,
        testing_range=testing_range,
        validation_range=validation_range,
    ),
)

