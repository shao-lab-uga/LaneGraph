import sys 
from subprocess import Popen 

Popen("python hdmapeditor/create_dataset_for_training.py ./raw_data/regions.json ./raw_data/ ./processed_data/", shell=True).wait()
Popen("python hdmapeditor/create_dataset_for_training_vectors.py ./raw_data/regions.json ./raw_data/ ./processed_data/", shell=True).wait()
