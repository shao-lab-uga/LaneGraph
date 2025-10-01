from subprocess import Popen 

Popen("python hdmapeditor/create_dataset_for_evaluation.py ./raw_data/regions.json ./raw_data/ ./evaluation_data/", shell=True).wait()
