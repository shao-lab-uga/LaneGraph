from pathlib import Path

import cv2
import numpy as np


def lane_ex_visualize(epoch: int, step: int, batch: tuple, result: tuple, path: Path):
    image_size = batch[0].shape[1]  # Image will be square
    batch_size = batch[0].shape[0]
    direction_img = np.zeros((image_size, image_size, 3), dtype=np.float32)

    for i in range(batch_size):
        # Arrays are RGB, cv2 does BGR
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
