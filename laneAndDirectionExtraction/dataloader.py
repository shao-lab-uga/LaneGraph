import imageio.v3 as imageio
import numpy as np
import threading
import random
import time
import json
import scipy
import math
import os
from pathlib import Path
import easydict
from utils.config_utils import load_config

class LaneAndDirectionDataloader:
    def __init__(
        self,
        data_path,
        indrange,
        image_size=640,
        dataset_image_size=2048,
        batch_size=8,
        preload_tiles=4,
        training=True,
    ):
        self.data_path = data_path
        self.indrange = indrange
        self.image_size = image_size
        self.dataset_image_size = dataset_image_size
        self.batch_size = batch_size
        self.preload_tiles = preload_tiles
        self.training = training
        
        # Preload
        self.images = np.zeros((preload_tiles, dataset_image_size, dataset_image_size, 3))
        self.normal = np.zeros((preload_tiles, dataset_image_size, dataset_image_size, 2))
        self.targets = np.zeros((preload_tiles, dataset_image_size, dataset_image_size, 1))
        self.masks = np.zeros((preload_tiles, dataset_image_size, dataset_image_size, 1))
        self.centers = [[] for _ in range(preload_tiles)]
        
        # Batch
        self.image_batch = np.zeros((batch_size, image_size, image_size, 3))
        self.normal_batch = np.zeros((batch_size, image_size, image_size, 2))
        self.target_batch = np.zeros((batch_size, image_size, image_size, 1))
        self.mask_batch = np.zeros((batch_size, image_size, image_size, 1))

    def _load_image_data(self, ind):
        """Load all image data for a given index."""
        sat_img = imageio.imread(os.path.join(self.data_path, f"sat_{ind}.jpg"))
        mask = imageio.imread(os.path.join(self.data_path, f"regionmask_{ind}.jpg"))
        target = imageio.imread(os.path.join(self.data_path, f"lane_{ind}.jpg"))
        normal = imageio.imread(os.path.join(self.data_path, f"normal_{ind}.jpg"))
        
        with open(os.path.join(self.data_path, f"link_{ind}.json"), "r") as json_file:
            nidmap, localnodes, locallinks, centers = json.load(json_file)
            
        return sat_img, mask, target, normal, centers

    def _convert_to_grayscale(self, image):
        """Convert image to grayscale if needed."""
        return image[:, :, 0] if len(image.shape) == 3 else image

    def _apply_rotation(self, sat_img, mask, target, normal, angle):
        """Apply rotation to all images."""
        sat_img = scipy.ndimage.rotate(sat_img, angle, reshape=False)
        mask = scipy.ndimage.rotate(mask, angle, reshape=False)
        target = scipy.ndimage.rotate(target, angle, reshape=False)
        normal = scipy.ndimage.rotate(normal, angle, reshape=False, cval=127)
        return sat_img, mask, target, normal

    def _rotate_centers(self, centers, angle):
        """Rotate center coordinates and filter valid ones."""
        im_center = np.array([self.dataset_image_size, self.dataset_image_size]) // 2
        angle_rad = np.radians(-angle)
        margin = 200
        
        rot_centers = []
        for x, y in centers:
            x_rot = round((x - im_center[0]) * np.cos(angle_rad) - (y - im_center[1]) * np.sin(angle_rad) + im_center[0])
            y_rot = round((x - im_center[0]) * np.sin(angle_rad) + (y - im_center[1]) * np.cos(angle_rad) + im_center[1])
            
            if (margin <= x_rot < self.dataset_image_size - margin and 
                margin <= y_rot < self.dataset_image_size - margin):
                rot_centers.append((x_rot, y_rot))
                
        return rot_centers

    def _process_normal_map(self, normal, angle):
        """Process normal map with rotation compensation."""
        normal = (normal.astype(float) - 127) / 127.0
        normal = normal[:, :, 1:3]
        
        normal_x, normal_y = normal[:, :, 1], normal[:, :, 0]
        angle_rad = math.radians(-angle)
        
        new_normal_x = normal_x * math.cos(angle_rad) - normal_y * math.sin(angle_rad)
        new_normal_y = normal_x * math.sin(angle_rad) + normal_y * math.cos(angle_rad)
        
        normal[:, :, 0] = new_normal_x
        normal[:, :, 1] = new_normal_y
        return np.clip(normal, -0.9999, 0.9999)

    def encode_direction_to_bin_top_left(self, normal_u, num_bins=36, y_down=True, mask=None):
        ux = normal_u[..., 0]
        uy_img = normal_u[..., 1]

        if mask is None:
            mask = np.ones_like(ux, dtype=bool)

        # normalize to unit length
        mag = np.maximum(np.sqrt(ux**2 + uy_img**2), 1e-6)
        ux /= mag
        uy_img /= mag

        uy_math = -uy_img if y_down else uy_img
        ang = np.arctan2(uy_math, ux)  # standard math coords: 0°=east, CCW

        # shift so that top-left (north-west) = 0°
        angle_offset_rad = math.radians(-135)  # image coords
        ang = (ang - angle_offset_rad + 2*np.pi) % (2*np.pi)

        # binning
        bin_width = 2 * np.pi / num_bins
        bin_idx = np.floor(ang / bin_width).astype(np.int32)
        residual = ang - (bin_idx + 0.5) * bin_width
        bin_idx = bin_idx[..., None]
        residual = residual[..., None]
        return bin_idx, residual
    
    def _decode_bin_to_direction(self, bin_idx, residual=None, num_bins=36):
        """
        Decode angle bins (and optional residuals) back to unit vectors.

        Args:
            bin_idx: int array (H, W) with values in [0, num_bins-1]
            residual: float array (H, W) in radians, optional
            num_bins: number of angle bins

        Returns:
            normal_u: np.ndarray of shape (H, W, 2), unit vectors in [-1, 1].
        """
        bin_width = 2 * np.pi / num_bins
        if residual is None:
            angles = (bin_idx + 0.5) * bin_width
        else:
            angles = (bin_idx + 0.5) * bin_width + residual

        ux = np.cos(angles)
        uy = np.sin(angles)

        normal_u = np.stack([ux, uy], axis=-1)
        return normal_u
    
    def _apply_augmentation(self, images, tile_idx):
        """Apply color augmentation to images."""
        if not self.training:
            return
            
        # Global brightness and contrast
        brightness_factor = 0.8 + 0.2 * random.random()
        brightness_offset = random.random() * 0.4 - 0.2
        self.images[tile_idx] = self.images[tile_idx] * brightness_factor - brightness_offset
        self.images[tile_idx] = np.clip(self.images[tile_idx], -0.5, 0.5)
        
        # Per-channel color adjustment
        for c in range(3):
            color_factor = 0.8 + 0.2 * random.random()
            self.images[tile_idx, :, :, c] *= color_factor

    def preload(self, ind=None):
        """Preload tiles with image data."""
        for i in range(self.preload_tiles):
            current_ind = random.choice(self.indrange) if ind is None else ind
            
            # Load raw data
            sat_img, mask, target, normal, centers = self._load_image_data(current_ind)
            
            # Convert to grayscale if needed
            mask = self._convert_to_grayscale(mask)
            target = self._convert_to_grayscale(target)
            
            # Apply rotation
            angle = 0
            random_rotation_probability = 0.2
            if self.training and random.random() < random_rotation_probability:  # 20% chance of rotation
                angle = random.randint(0, 359)
                sat_img, mask, target, normal = self._apply_rotation(sat_img, mask, target, normal, angle)
            
            # Process centers and normal map
            self.centers[i] = self._rotate_centers(centers, angle)
            normal = self._process_normal_map(normal, angle)
            # Normalize images
            sat_img = sat_img.astype(np.float64) / 255.0 - 0.5
            mask = mask.astype(np.float64) / 255.0
            target = target.astype(np.float64) / 255.0
            
            # Store processed data
            self.images[i] = sat_img
            self.masks[i, :, :, 0] = mask
            self.targets[i, :, :, 0] = target
            self.normal[i] = normal
            # Apply augmentation
            self._apply_augmentation(self.images, i)

    def _get_available_coordinates(self):
        """Get all available coordinates from all tiles."""
        available_coords = []
        for tile_idx, centers in enumerate(self.centers):
            for coord in centers:
                available_coords.append((coord, tile_idx))
        return available_coords

    def _is_valid_crop(self, x, y, tile_id):
        """Check if crop location has sufficient lane and mask coverage."""
        margin = 64
        crop_targets = self.targets[tile_id, x+margin:x+self.image_size-margin, 
                                   y+margin:y+self.image_size-margin, :]
        crop_masks = self.masks[tile_id, x+margin:x+self.image_size-margin, 
                               y+margin:y+self.image_size-margin, :]
        
        return np.sum(crop_targets) >= 100 and np.sum(crop_masks) >= 2500  # 50*50

    def get_batch(self, centered_intersection=False):
        """Generate a batch of cropped images."""
        # Clear batch arrays
        for batch_array in [self.image_batch, self.mask_batch, self.target_batch, self.normal_batch]:
            batch_array.fill(0)
        if centered_intersection:
            available_coords = self._get_available_coordinates()
            if len(available_coords) < self.batch_size:
                return None
        
        selected_coords = []
        
        while len(selected_coords) < self.batch_size:
            
            if centered_intersection:
                # Select random coordinate
                coord, tile_id = random.choice(available_coords)
                available_coords.remove((coord, tile_id))
                x, y = coord
                noise_range = 100
                # Add noise and clamp to valid range
                x += random.randint(-noise_range, noise_range) - (self.image_size // 2)
                y += random.randint(-noise_range, noise_range) - (self.image_size // 2)
                x = np.clip(x, 0, self.dataset_image_size - self.image_size - 1)
                y = np.clip(y, 0, self.dataset_image_size - self.image_size - 1)
            else: 
                tile_id = random.randint(0,self.preload_tiles-1)
                x = random.randint(0, self.dataset_image_size-1-self.image_size)
                y = random.randint(0, self.dataset_image_size-1-self.image_size)
            # Validate crop
            if self._is_valid_crop(x, y, tile_id):
                selected_coords.append(((x, y), tile_id))
        
        # Create batch
        for batch_idx, ((x, y), tile_id) in enumerate(selected_coords):
            self.image_batch[batch_idx] = self.images[tile_id, y:y+self.image_size, x:x+self.image_size]
            self.mask_batch[batch_idx] = self.masks[tile_id, y:y+self.image_size, x:x+self.image_size]
            self.target_batch[batch_idx] = self.targets[tile_id, y:y+self.image_size, x:x+self.image_size]
            self.normal_batch[batch_idx] = self.normal[tile_id, y:y+self.image_size, x:x+self.image_size]
        return (
            self.image_batch[:self.batch_size],
            self.mask_batch[:self.batch_size],
            self.target_batch[:self.batch_size],
            self.normal_batch[:self.batch_size],
        )


class LaneAndDirectionParallelDataLoader:
    """
    A parallel data loader that manages multiple Dataloader instances in separate threads
    to enable concurrent data loading and preprocessing.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize parallel data loader with multiple subloaders."""
        self.num_workers = 4  # Number of worker threads
        self.subloaders = list()
        self.ready_events = []  # Events signaling when subloader has data ready
        self.wait_events = []   # Events to signal subloader to start preloading
        
        self.current_loader_index = 0
        
        # Create subloaders and their associated events
        for i in range(self.num_workers):
            self.subloaders.append(LaneAndDirectionDataloader(*args, **kwargs))
            self.ready_events.append(threading.Event())
            self.wait_events.append(threading.Event())
        
        # Start worker threads
        self._start_worker_threads()
    
    def _start_worker_threads(self):
        """Start daemon threads for each subloader."""
        for worker_id in range(self.num_workers):
            self.ready_events[worker_id].clear()
            self.wait_events[worker_id].clear()
            
            worker_thread = threading.Thread(
                target=self._worker_daemon, 
                args=(worker_id,), 
                daemon=True
            )
            worker_thread.start()
    
    def _worker_daemon(self, worker_id):
        """
        Worker thread that continuously preloads data.
        
        Args:
            worker_id: ID of this worker thread
        """
        while True:
            # print(f"Worker-{worker_id} starts preloading")
            start_time = time.time()
            
            # Preload data
            self.subloaders[worker_id].preload()
            
            elapsed_time = time.time() - start_time
            # print(f"Worker-{worker_id} finished preloading (time = {elapsed_time:.2f}s)")
            
            # Signal that data is ready
            self.ready_events[worker_id].set()
            
            # Wait for signal to start next preload cycle
            self.wait_events[worker_id].wait()
            self.wait_events[worker_id].clear()
    
    def preload(self):
        """Trigger preloading for the next worker and switch to it."""
        # Signal current worker to start next preload cycle
        self.wait_events[self.current_loader_index].set()
        
        # Move to next worker
        self.current_loader_index = (self.current_loader_index + 1) % self.num_workers
        
        # Wait for the next worker to have data ready
        self.ready_events[self.current_loader_index].wait()
        self.ready_events[self.current_loader_index].clear()
    
    def get_batch(self):
        """
        Get a batch of data from available workers.
        
        Args:
            batch_size: Number of samples in the batch
            
        Returns:
            Tuple of batch data or None if no data available
        """
        attempts = 0
        
        # Try each worker once
        while attempts < self.num_workers:
            batch = self.subloaders[self.current_loader_index].get_batch()
            
            if batch is not None:
                return batch
            
            # print(f"Worker-{self.current_loader_index} exhausted")
            
            # Current worker is exhausted, signal it to preload and switch
            self.wait_events[self.current_loader_index].set()
            self.current_loader_index = (self.current_loader_index + 1) % self.num_workers
            
            # Wait for next worker to be ready
            self.ready_events[self.current_loader_index].wait()
            self.ready_events[self.current_loader_index].set()
            
            attempts += 1
        
        # All workers exhausted, wait for any worker to finish preloading
        return self._wait_for_any_worker_ready()
    
    def _wait_for_any_worker_ready(self):
        """
        Wait for any worker to finish preloading and return a batch.
        
        Args:
            batch_size: Number of samples in the batch
            
        Returns:
            Tuple of batch data
        """
        # print("All workers exhausted. Waiting for any worker to finish preloading...")
        
        while True:
            for worker_id in range(self.num_workers):
                if self.ready_events[worker_id].is_set():
                    self.current_loader_index = worker_id
                    self.ready_events[worker_id].clear()
                    return self.get_batch()
            
            # Small sleep to avoid busy waiting
            time.sleep(0.001)
    
    def get_current_loader(self):
        """Get the currently active subloader."""
        return self.subloaders[self.current_loader_index]


def get_dataloaders(dataloaders_config):
    """
    Create training, validation, and testing dataloaders based on the provided configuration.
    Args:
        dataloaders_config: Configuration dictionary containing paths and ranges for each dataset split.
    Returns:
        Tuple of training, validation, and testing dataloaders.
    """
    train_config = dataloaders_config.train
    validate_config = dataloaders_config.validate
    test_config = dataloaders_config.test
    
    def get_dataloader(dataloader_config):
        """
        Create a Dataloader instance based on the provided configuration.
        Args:
            config: Configuration dictionary for the dataloader.
            mode: Mode of the dataloader (train, validate, test).
        Returns:
            Dataloader instance.
        """
        return LaneAndDirectionParallelDataLoader(
            data_path=dataloader_config.data_path,
            indrange=dataloader_config.indrange,
            image_size=dataloader_config.image_size,
            dataset_image_size=dataloader_config.dataset_image_size,
            batch_size=dataloader_config.batch_size,
            preload_tiles=dataloader_config.preload_tiles,
            training=dataloader_config.training
        )
    train_dataloader = get_dataloader(train_config)
    validate_dataloader = get_dataloader(validate_config)
    test_dataloader = get_dataloader(test_config)
    train_dataloader.preload()
    validate_dataloader.preload()
    test_dataloader.preload()
    return train_dataloader, validate_dataloader, test_dataloader


if __name__ == "__main__":
    config = load_config("configs/train_lane_and_direction_extraction.py")
    dataloaers_config = config.dataloaders
    train_dataloader, validate_dataloader, test_dataloader = get_dataloaders(dataloaers_config)

