import imageio.v3 as imageio
import numpy as np
import threading
import random
import time
import json
import scipy
import math

class Dataloader:
    def __init__(
        self,
        folder,
        indrange,
        image_size=640,
        dataset_image_size=4096,
        batch_size=8,
        preload_tiles=4,
        testing=False,
    ):
        self.folder = folder
        self.indrange = indrange
        self.image_size = image_size
        self.dataset_image_size = dataset_image_size
        self.preload_tiles = preload_tiles
        self.testing = testing

        self.images = np.zeros(
            (preload_tiles, dataset_image_size, dataset_image_size, 3)
        )
        self.normal = np.zeros((preload_tiles, dataset_image_size, dataset_image_size, 2))
        self.targets = np.zeros((preload_tiles, dataset_image_size, dataset_image_size, 1))
        self.masks = np.zeros((preload_tiles, dataset_image_size, dataset_image_size, 1))
        self.centers = [[None]] * preload_tiles

        self.image_batch = np.zeros((batch_size, image_size, image_size, 3))
        self.normal_batch = np.zeros((batch_size, image_size, image_size, 2))
        self.target_batch = np.zeros((batch_size, image_size, image_size, 1))
        self.mask_batch = np.zeros((batch_size, image_size, image_size, 1))

    def preload(self, ind=None):
        for i in range(self.preload_tiles):
            ind = random.choice(self.indrange) if ind is None else ind # Change? loads the same main image 4 times
            #ind = "_0"
            sat_img = imageio.imread(self.folder / f"sat{ind}.jpg")
            mask = imageio.imread(self.folder / f"regionmask{ind}.jpg")
            target = imageio.imread(self.folder / f"lane{ind}.jpg")
            normal = imageio.imread(self.folder / f"normal{ind}.jpg")
            with open(self.folder / f"link{ind}.json", "r") as json_file:
                centers = json.load(json_file)[3]


            if len(np.shape(mask)) == 3:
                mask = mask[:, :, 0]

            if len(np.shape(target)) == 3:
                target = target[:, :, 0]

            angle = 0
            if self.testing == False and random.randint(0, 5) < 4: # rotates 2/3 of time
                angle = random.randint(0, 359)

                sat_img = scipy.ndimage.rotate(sat_img, angle, reshape=False)
                mask = scipy.ndimage.rotate(mask, angle, reshape=False)
                target = scipy.ndimage.rotate(target, angle, reshape=False)
                normal = scipy.ndimage.rotate(normal, angle, reshape=False, cval=127)

            im_center = np.array((self.dataset_image_size, self.dataset_image_size)) // 2
            rot_centers = []
            margin = 200
            angle_rad = np.radians(-angle)
            for (x, y) in centers:
                x_rot = round(
                    (x - im_center[0]) * np.cos(angle_rad) - (y - im_center[1]) * np.sin(angle_rad) + im_center[0])
                y_rot = round(
                    (x - im_center[0]) * np.sin(angle_rad) + (y - im_center[1]) * np.cos(angle_rad) + im_center[1])
                if (
                        0 + margin <= x_rot < self.dataset_image_size - margin - 1 and 0 + margin <= y_rot < self.dataset_image_size - margin - 1):
                    rot_centers.append((x_rot, y_rot))
            self.centers[i] = rot_centers

            normal = (normal.astype(float) - 127) / 127.0
            normal = normal[:, :, 1:3]

            normal_x = normal[:, :, 1]
            normal_y = normal[:, :, 0]

            new_normal_x = normal_x * math.cos(math.radians(-angle)) - normal_y * math.sin(math.radians(-angle))
            new_normal_y = normal_x * math.sin(math.radians(-angle)) + normal_y * math.cos(math.radians(-angle))

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

    def get_batch(self, batch_size):

        self.image_batch.fill(0) # do for other batches too?

        available_coords = []
        for list_index, sublist in enumerate(self.centers):
            for coord in sublist:
                available_coords.append((coord, list_index))
        if len(available_coords) < batch_size:
            return None

        selected_coords = []
        while len(selected_coords) < batch_size:
            if not available_coords:
                return None

            coord, tile_id = random.choice(available_coords)
            available_coords.remove((coord, tile_id))

            x, y = coord
            noise = 100 #plus or minus # ADD CONFIG
            x += random.randint(-noise, noise) - (self.image_size // 2)
            y += random.randint(-noise, noise) - (self.image_size // 2)

            x = np.clip(x, 0, self.dataset_image_size - self.image_size - 1)
            y = np.clip(y, 0, self.dataset_image_size - self.image_size - 1)

            if (
                    np.sum(
                        self.targets[
                        tile_id,
                        x + 64: x + self.image_size - 64,
                        y + 64: y + self.image_size - 64,
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
                        x + 64: x + self.image_size - 64,
                        y + 64: y + self.image_size - 64,
                        :,
                        ]
                    )
                    < 50 * 50
            ):
                continue

            selected_coords.append(((x, y), tile_id))

        batch_index = 0
        for (x, y), tile_id in selected_coords:
            self.image_batch[batch_index, :, :, :] = self.images[tile_id, y: y + self.image_size, x: x + self.image_size, :]
            self.mask_batch[batch_index, :, :, :] = self.masks[tile_id, y: y + self.image_size, x: x + self.image_size, :]
            self.target_batch[batch_index, :, :, :] = self.targets[tile_id, y: y + self.image_size, x: x + self.image_size, :]
            self.normal_batch[batch_index, :, :, :] = self.normal[tile_id, y: y + self.image_size, x: x + self.image_size, :]
            batch_index += 1
        return (
            self.image_batch[:batch_size, :, :, :],
            self.mask_batch[:batch_size, :, :, :],
            self.target_batch[:batch_size, :, :, :],
            self.normal_batch[:batch_size, :, :, :],
        )


class ParallelDataLoader:
    def __init__(self, *args, **kwargs):
        self.n = 4  # number of subloaders
        self.subloader = []
        self.subloaderReadyEvent = []
        self.subloaderWaitEvent = []

        self.current_loader_id = 0

        for i in range(self.n):
            self.subloader.append(Dataloader(*args, **kwargs))
            self.subloaderReadyEvent.append(threading.Event())
            self.subloaderWaitEvent.append(threading.Event())

        for i in range(self.n):
            self.subloaderReadyEvent[i].clear()
            self.subloaderWaitEvent[i].clear()
            x = threading.Thread(target=self.daemon, args=(i,), daemon=True)
            x.start()

    def daemon(self, tid):
        while True:
            print("thread-%d starts preloading" % tid)
            t0 = time.time()
            self.subloader[tid].preload()
            print(f"thread-{tid} finished preloading (time = {time.time() - t0:.2f})")
            self.subloaderReadyEvent[tid].set()

            self.subloaderWaitEvent[tid].wait()
            self.subloaderWaitEvent[tid].clear()

    def preload(self):

        self.subloaderWaitEvent[self.current_loader_id].set()

        self.current_loader_id = (self.current_loader_id + 1) % self.n

        self.subloaderReadyEvent[self.current_loader_id].wait()
        self.subloaderReadyEvent[self.current_loader_id].clear()

    def get_batch(self, batch_size):
        attempts = 0
        while attempts < self.n:
            batch = self.subloader[self.current_loader_id].get_batch(batch_size)

            if batch is not None:
                return batch

            print(f"thread-{self.current_loader_id} exausted")
            self.subloaderWaitEvent[self.current_loader_id].set()

            self.current_loader_id = (self.current_loader_id + 1) % self.n

            self.subloaderReadyEvent[self.current_loader_id].wait()
            self.subloaderReadyEvent[self.current_loader_id].set()

            attempts += 1

        print("Waiting for subloader to finish preloading")
        while True:
            for i in range(self.n):
                if self.subloaderReadyEvent[i].is_set():
                    self.current_loader_id = i
                    self.subloaderReadyEvent[i].clear()
                    return self.get_batch(batch_size)

    def current(self):
        return self.subloader[self.current_loader_id]


if __name__ == "__main__":
    folder = ""
    training_range = []
    image_size = 640

    parallel_loader = ParallelDataLoader(
        folder,
        training_range,
        image_size,
    )
    for _ in range(100):
        batch = parallel_loader.get_batch(4)
        print(batch)

