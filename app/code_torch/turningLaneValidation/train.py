import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[3]))

import json

import cv2
import numpy as np
import torch
from app.code_torch.framework import base_classes
from app.code_torch.TurnVal.model_manager import TurnValModelManager
from dataloader import TurnValDataloader

torch.autograd.set_detect_anomaly(True)

class TurnValTrainer(base_classes.Trainer):
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

            # Saving Batch
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
                str(path / f"{id_str}_target1.jpg"),
                ((batch[2][i, :, :, 1]) * 255).astype(np.uint8),
            )
            cv2.imwrite(
                str(path / f"{id_str}_target2.jpg"),
                ((batch[2][i, :, :, 2]) * 255).astype(np.uint8),
            )

            # Saving Results
            # sw ind
            cv2.imwrite(
                str(path / f"{id_str}_output1.jpg"),
                ((result[3][i, 0, :, :]) * 255).astype(np.uint8),
            )
            cv2.imwrite(
                str(path / f"{id_str}_output2.jpg"),
                ((result[3][i, 1, :, :]) * 255).astype(np.uint8),
            )
            with open(path / f"{id_str}_label.txt", "w") as txt_file:
                txt_file.write(f"{batch[3][i, 0]} {result[4][i, 0]} \n")

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


dataset_path = Path(
    "/Users/agraham/development/lab/dist/LaneGraph/msc_dataset/dataset_unpacked"
)
split_file = Path("/Users/agraham/development/lab/dist/LaneGraph/app/code/split_all.json")

config = base_classes.Config(1,1, dataset_path,split_file)

dataloader = base_classes.ParallelDataLoader(
    1,
    1,
    2,
    TurnValDataloader,
    training_range=config.training_range,
    dataset_folder=config.dataset_folder,
)

model_manager = TurnValModelManager(config.batch_size)

trainer = TurnValTrainer(config, dataloader, model_manager)
trainer.run()