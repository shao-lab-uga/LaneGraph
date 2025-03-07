import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[3]))

import json

import cv2
import numpy as np
from app.new_code_copy.framework import base_classes
from dataloader import TurnExDataloader
from model_manager import TurnExModelManager


class TurnExTrainer(base_classes.Trainer):
    def __init__(
        self,
        config: dict,
        dataloader: base_classes.DataLoader,
        model_manager: base_classes.ModelManager,
    ) -> None:
        super().__init__(config, dataloader, model_manager)

    def _visualize(self, epoch: int, step: int, batch: tuple, result: tuple):
        path = self.visualization_folder

        image_size = batch[0].shape[1]
        batch_size = batch[0].shape[0]

        direction_image = np.zeros((image_size, image_size, 3))

        for i in range(batch_size):
            id_str = f"{epoch}_{step}_{i}"
            cv2.imwrite(
                str(path / f"{id_str}_input.jpg"),
                ((batch[0][i, :, :, :] + 0.5) * 255).astype(np.uint8),
            )
            cv2.imwrite(
                str(path / f"{id_str}_connector1.jpg"),
                ((batch[1][i, :, :, 0:3]) * 127 + 127).astype(np.uint8),
            )
            cv2.imwrite(
                str(path / f"{id_str}_connector2.jpg"),
                ((batch[1][i, :, :, 3:6]) * 127 + 127).astype(np.uint8),
            )
            cv2.imwrite(
                str(path / f"{id_str}_target.jpg"),
                ((batch[2][i, :, :, 0]) * 255).astype(np.uint8),
            )
            cv2.imwrite(
                str(path / f"{id_str}_output1.jpg"),
                ((result[1][i, :, :, 0]) * 255).astype(np.uint8),
            )

            direction_image[:, :, 2] = np.clip(batch[4][i, :, :, 0], -1, 1) * 127 + 127
            direction_image[:, :, 1] = np.clip(batch[4][i, :, :, 1], -1, 1) * 127 + 127
            direction_image[:, :, 0] = 127

            direction_image[:, :, 0] += batch[1][i, :, :, 0] * 255 + 127
            direction_image[:, :, 1] += batch[1][i, :, :, 3] * 255 + 127
            direction_image[:, :, 2] += batch[1][i, :, :, 6] * 255 + 127

            direction_image = np.clip(direction_image, 0, 255)

            cv2.imwrite(
                str(path / f"{id_str}_direction.jpg"), direction_image.astype(np.uint8)
            )
        return


dataset_path = Path(
    "/home/lab/development/lab/modular/LaneGraph/msc_dataset/dataset_unpacked"
)
split_file = "/home/lab/development/lab/modular/LaneGraph/app/code_torch/split_all.json"
training_range = []

config = {
    "tag": "testing turnval",
    "data_config": {
        "batch_size": 2,
        "preload_size": 2,
        "image_size": 640,
        "dataset_image_size": 4096,
        "testing": False,
        "num_loaders": 2,
        "dataloader": TurnExDataloader,
        "dataset_folder": dataset_path,
    },
    "learning_rate": 0.001,
    "lr_decay": [0.1, 0.1],
    "lr_decay_ep": [350, 450],
    "ep_max": 500,
    "step_init": 0,
}

with open(split_file, "r") as json_file:
    dataset_split = json.load(json_file)
for tid in dataset_split["training"]:
    training_range.append(f"_{tid}")

dataloader = base_classes.ParallelDataLoader(
    2,
    2,
    2,
    TurnExDataloader,
    training_range=training_range,
    dataset_folder=dataset_path,
)

model_manager = TurnExModelManager(batch_size=2)

trainer = TurnExTrainer(config, dataloader, model_manager)
trainer.run()
