import json
import math
import random
from pathlib import Path
from typing import List

import imageio.v3 as imageio
import numpy as np
import scipy

from app.code_torch.framework.base_classes import DataLoader


class CenteredLaneExDataLoader(DataLoader):
    """
    Generates batches centered on intersections. Preprocesses and
    samples images from larger dataset images.

    :param dataset_folder: Location of dataset files
    :type dataset_folder: pathlib.Path
    :param training_range: Elligible file name suffixes to load
    :type training_range: List[str]
    :param batch_size: Number of images per batch
    :type batch_size: int
    :param preload_size: Number of dataset images to preload
    :type preload_size: int
    :param image_size: Batch image dimension, defaults to 640
    :type image_size: int, optional
    :param dataset_image_size: Dataset image dimension, defaults to 4096
    :type dataset_image_size: int, optional
    :param testing: Flag to enable testing mode, defaults to False
    :type testing: bool, optional
    """

    def __init__(
        self,
        dataset_folder: Path,
        training_range: List[str],
        batch_size: int,
        preload_size: int,
        image_size=640,
        dataset_image_size=4096,
        testing=False,
    ):
        super().__init__(batch_size, preload_size)
        self.dataset_folder = dataset_folder
        self.training_range = training_range
        self.image_size = image_size
        self.dataset_image_size = dataset_image_size
        self.preload_size = preload_size
        self.testing = testing

        self.images = np.zeros(
            (preload_size, dataset_image_size, dataset_image_size, 3)
        )
        self.normal = np.zeros(
            (preload_size, dataset_image_size, dataset_image_size, 2)
        )
        self.targets = np.zeros(
            (preload_size, dataset_image_size, dataset_image_size, 1)
        )
        self.masks = np.zeros((preload_size, dataset_image_size, dataset_image_size, 1))
        self.centers = [[None]] * preload_size

        self.image_batch = np.zeros((batch_size, image_size, image_size, 3))
        self.normal_batch = np.zeros((batch_size, image_size, image_size, 2))
        self.target_batch = np.zeros((batch_size, image_size, image_size, 1))
        self.mask_batch = np.zeros((batch_size, image_size, image_size, 1))

    def preload(self, ind=None):
        for i in range(self.preload_size):
            ind = (
                random.choice(self.training_range) if ind is None else ind
            )  # Change? loads the same main image 4 times
            # ind = "_0"
            sat_img = imageio.imread(self.dataset_folder / f"sat{ind}.jpg")
            mask = imageio.imread(self.dataset_folder / f"regionmask{ind}.jpg")
            target = imageio.imread(self.dataset_folder / f"lane{ind}.jpg")
            normal = imageio.imread(self.dataset_folder / f"normal{ind}.jpg")
            with open(self.dataset_folder / f"link{ind}.json", "r") as json_file:
                centers = json.load(json_file)[3]

            if len(np.shape(mask)) == 3:
                mask = mask[:, :, 0]

            if len(np.shape(target)) == 3:
                target = target[:, :, 0]

            angle = 0
            if not self.testing and random.randint(0, 5) < 4:  # rotates 2/3 of time
                angle = random.randint(0, 359)

                sat_img = scipy.ndimage.rotate(sat_img, angle, reshape=False)
                mask = scipy.ndimage.rotate(mask, angle, reshape=False)
                target = scipy.ndimage.rotate(target, angle, reshape=False)
                normal = scipy.ndimage.rotate(normal, angle, reshape=False, cval=127)

            im_center = (
                np.array((self.dataset_image_size, self.dataset_image_size)) // 2
            )
            rot_centers = []
            margin = 200
            angle_rad = np.radians(-angle)
            for x, y in centers:
                x_rot = round(
                    (x - im_center[0]) * np.cos(angle_rad)
                    - (y - im_center[1]) * np.sin(angle_rad)
                    + im_center[0]
                )
                y_rot = round(
                    (x - im_center[0]) * np.sin(angle_rad)
                    + (y - im_center[1]) * np.cos(angle_rad)
                    + im_center[1]
                )
                if (
                    0 + margin <= x_rot < self.dataset_image_size - margin - 1
                    and 0 + margin <= y_rot < self.dataset_image_size - margin - 1
                ):
                    rot_centers.append((x_rot, y_rot))
            self.centers[i] = rot_centers

            normal = (normal.astype(float) - 127) / 127.0
            normal = normal[:, :, 1:3]

            normal_x = normal[:, :, 1]
            normal_y = normal[:, :, 0]

            new_normal_x = normal_x * math.cos(
                math.radians(-angle)
            ) - normal_y * math.sin(math.radians(-angle))
            new_normal_y = normal_x * math.sin(
                math.radians(-angle)
            ) + normal_y * math.cos(math.radians(-angle))

            normal[:, :, 0] = new_normal_x
            normal[:, :, 1] = new_normal_y
            normal = np.clip(normal, -0.9999, 0.9999)

            sat_img = sat_img.astype(np.float64) / 255.0 - 0.5
            mask = mask.astype(np.float64) / 255.0
            target = target.astype(np.float64) / 255.0

            self.images[i, :, :, :] = sat_img
            self.masks[i, :, :, 0] = mask
            self.targets[i, :, :, 0] = target
            self.normal[i, :, :, :] = normal

            if not self.testing:
                self.images[i, :, :, :] = self.images[i, :, :, :] * (
                    0.8 + 0.2 * random.random()
                ) - (random.random() * 0.4 - 0.2)
                self.images[i, :, :, :] = np.clip(self.images[i, :, :, :], -0.5, 0.5)

                self.images[i, :, :, 0] = self.images[i, :, :, 0] * (
                    0.8 + 0.2 * random.random()
                )
                self.images[i, :, :, 1] = self.images[i, :, :, 1] * (
                    0.8 + 0.2 * random.random()
                )
                self.images[i, :, :, 2] = self.images[i, :, :, 2] * (
                    0.8 + 0.2 * random.random()
                )

    def get_batch(self):
        self.image_batch.fill(0)  # do for other batches too?

        available_coords = []
        for list_index, sublist in enumerate(self.centers):
            for coord in sublist:
                available_coords.append((coord, list_index))
        if len(available_coords) < self.batch_size:
            return None

        selected_coords = []
        while len(selected_coords) < self.batch_size:
            if not available_coords:
                return None

            coord, tile_id = random.choice(available_coords)
            available_coords.remove((coord, tile_id))

            x, y = coord
            noise = 100  # plus or minus # ADD CONFIG
            x += random.randint(-noise, noise) - (self.image_size // 2)
            y += random.randint(-noise, noise) - (self.image_size // 2)

            x = np.clip(x, 0, self.dataset_image_size - self.image_size - 1)
            y = np.clip(y, 0, self.dataset_image_size - self.image_size - 1)

            if (
                np.sum(
                    self.targets[
                        tile_id,
                        x + 64 : x + self.image_size - 64,
                        y + 64 : y + self.image_size - 64,
                        :,
                    ]
                )
                < 100
            ):
                continue

            if (
                np.sum(
                    self.masks[
                        tile_id,
                        x + 64 : x + self.image_size - 64,
                        y + 64 : y + self.image_size - 64,
                        :,
                    ]
                )
                < 50 * 50
            ):
                continue

            selected_coords.append(((x, y), tile_id))

        batch_index = 0
        for (x, y), tile_id in selected_coords:
            self.image_batch[batch_index, :, :, :] = self.images[
                tile_id, y : y + self.image_size, x : x + self.image_size, :
            ]
            self.mask_batch[batch_index, :, :, :] = self.masks[
                tile_id, y : y + self.image_size, x : x + self.image_size, :
            ]
            self.target_batch[batch_index, :, :, :] = self.targets[
                tile_id, y : y + self.image_size, x : x + self.image_size, :
            ]
            self.normal_batch[batch_index, :, :, :] = self.normal[
                tile_id, y : y + self.image_size, x : x + self.image_size, :
            ]
            batch_index += 1
        return (
            self.image_batch[:, :, :, :],
            self.mask_batch[:, :, :, :],
            self.target_batch[:, :, :, :],
            self.normal_batch[:, :, :, :],
        )
