import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[3]))


import cv2
import numpy as np
from dataloader import CenteredLaneExDataLoader

from app.code_torch.framework import base_classes, utils
from app.code_torch.laneAndDirectionExtraction.model_manager import LaneExModelManager

dataset_path = Path(
    "/home/lab/development/lab/modular/LaneGraph/msc_dataset/dataset_unpacked"
)

config = {
    "tag": "testing laneex",
    "data_config": {
        "batch_size": 4,
        "preload_size": 4,
        "image_size": 640,
        "dataset_image_size": 4096,
        "testing": False,
        "num_loaders": 2,
        "dataloader": CenteredLaneExDataLoader,
        "dataset_folder": dataset_path,
    },
    "learning_rate": 0.001,
    "lr_decay": [0.1, 0.1],
    "lr_decay_ep": [350, 450],
    "ep_max": 500,
    "step_init": 0,
}


class LaneExTrainer(base_classes.Trainer):
    def __init__(
        self,
        config: base_classes.Config,
        dataloader: base_classes.DataLoader,
        model_manager: base_classes.ModelManager,
    ) -> None:
        super().__init__(config, dataloader, model_manager)

    def _add_log(self, step: int, results: tuple) -> None:
        logs = {
            "loss": results[0],
        }
        for key, value in logs.items():
            if key in self.logs:
                self.logs[key].append(value)
            else:
                self.logs[key] = [value]
        return

    def _visualize(self, epoch: int, step: int, batch: tuple, result: tuple):
        # Arrays are RGB, cv2 does BGR
        path = self.config.visualization_folder

        image_size = batch[0].shape[1]  # Image will be square
        batch_size = batch[0].shape[0]
        direction_img = np.zeros((image_size, image_size, 3), dtype=np.float32)

        for i in range(batch_size):
            id_str = f"{epoch}_{step}_{i}"
            # Batch
            cv2.imwrite(
                str(path / f"{id_str}_input.jpg"),
                ((batch[0][i, :, :, ::-1] + 0.5) * 255).astype(np.uint8),
            )
            cv2.imwrite(
                str(path / f"{id_str}_mask.jpg"),
                (batch[1][i, :, :, 0] * 255).astype(np.uint8),
            )
            cv2.imwrite(
                str(path / f"{id_str}_target.jpg"),
                ((batch[2][i, :, :, 0]) * 255).astype(np.uint8),
            )

            direction_img[:, :, 0] = batch[3][i, :, :, 0] * 127 + 127
            direction_img[:, :, 1] = batch[3][i, :, :, 1] * 127 + 127
            direction_img[:, :, 2] = 127

            cv2.imwrite(
                str(path / f"{id_str}_target_dir.jpg"),
                direction_img.astype(np.uint8),
            )

            if len(batch) == 5:
                cv2.imwrite(
                    str(path / f"{id_str}_sdmap.jpg"),
                    (batch[4][i, :, :, 0] * 255).astype(np.uint8),
                )

            # Results
            cv2.imwrite(
                str(path / f"{id_str}_output.jpg"),
                (result[1][i, :, :, 0] * 255).astype(np.uint8),
            )

            direction_img[:, :, 0] = np.clip(result[1][i, :, :, 1], -1, 1) * 127 + 127
            direction_img[:, :, 1] = np.clip(result[1][i, :, :, 2], -1, 1) * 127 + 127
            direction_img[:, :, 2] = 127

            cv2.imwrite(
                str(path / f"{id_str}_output_dir.jpg"),
                direction_img.astype(np.uint8),
            )
        return


root_dir = utils.get_git_report()

dataset_path = root_dir / "msc_dataset" / "dataset_unpacked"

split_file = root_dir / "app" / "code" / "split_all.json"

config = base_classes.Config(4, 4, dataset_path, split_file)

config.epoch_size = int(
    (len(config.training_range) * 9 * 2048 * 2048) / (config.batch_size * 640 * 640)
)

dataloader = base_classes.ParallelDataLoader(
    config.batch_size,
    config.preload_size,
    4,
    CenteredLaneExDataLoader,
    training_range=config.training_range,
    dataset_folder=config.dataset_folder,
)

model_manager = LaneExModelManager(config)

trainer = LaneExTrainer(config, dataloader, model_manager)
trainer.run()
