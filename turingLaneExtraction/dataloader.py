import threading
import time
import random
import cv2
import json
import math
import os
import scipy.ndimage
import numpy as np
import imageio.v3 as imageio
from utils.config_utils import load_config


def rotate(pos, angle, size):
    x = pos[0] - int(size / 2)
    y = pos[1] - int(size / 2)

    new_x = x * math.cos(math.radians(angle)) - y * math.sin(math.radians(angle))
    new_y = x * math.sin(math.radians(angle)) + y * math.cos(math.radians(angle))

    return (int(new_x + int(size / 2)), int(new_y + int(size / 2)))


class Dataloader:
    def __init__(
        self,
        data_path,
        indrange,
        image_size=640,
        dataset_image_size=2048,
        batch_size=8,
        preload_tiles=4,
        training=True,
        maxbatchsize=32,
    ):
        self.data_path = data_path
        self.indrange = indrange
        self.image_size = image_size
        self.dataset_image_size = dataset_image_size
        self.batch_size = batch_size
        self.preload_tiles = preload_tiles
        self.training = training
        self.crop_padding = 8
        self.images = np.zeros((preload_tiles, dataset_image_size, dataset_image_size, 3))
        self.normal = np.zeros((preload_tiles, dataset_image_size, dataset_image_size, 2))
        self.targets = np.zeros((preload_tiles, dataset_image_size, dataset_image_size, 1))
        self.masks = np.ones((preload_tiles, dataset_image_size, dataset_image_size, 1))
        self.links = []
        self.maxbatchsize = maxbatchsize
        self.image_batch = np.zeros((self.maxbatchsize, image_size, image_size, 3))
        self.normal_batch = np.zeros((self.maxbatchsize, image_size, image_size, 2))
        self.target_batch = np.zeros((self.maxbatchsize, image_size, image_size, 1))
        self.connector_batch = np.zeros((self.maxbatchsize, image_size, image_size, 7))


        self.poscode = np.zeros((image_size * 2, image_size * 2, 2))
        for i in range(image_size * 2):
            self.poscode[i, :, 0] = float(i) / image_size - 1.0
            self.poscode[:, i, 1] = float(i) / image_size - 1.0

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
    def _load_image_data(self, ind):
        """Load all image data for a given index."""
        sat_img = imageio.imread(os.path.join(self.data_path, f"sat_{ind}.jpg"))
        mask = imageio.imread(os.path.join(self.data_path, f"regionmask_{ind}.jpg"))
        target = imageio.imread(os.path.join(self.data_path, f"lane_{ind}.jpg"))
        normal = imageio.imread(os.path.join(self.data_path, f"normal_{ind}.jpg"))
        
        return sat_img, mask, target, normal
    
    def _load_link_data(self, ind):
        """Load link data for a given index."""
        with open(os.path.join(self.data_path, f"link_{ind}.json"), "r") as json_file:
            links = json.load(json_file)
        return links
    
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
    

    def preload(self, ind=None):

        self.links = []
        for i in range(self.preload_tiles if ind is None else 1):
            while True:
                current_ind = random.choice(self.indrange) if ind is None else ind
                
                sat_img, mask, target, normal = self._load_image_data(current_ind)
                links = self._load_link_data(current_ind)

                if len(links[2]) == 0:
                    continue

                # Convert to grayscale if needed
                mask = self._convert_to_grayscale(mask)
                target = self._convert_to_grayscale(target)
                # Apply rotation
                angle = 0
                random_rotation_probability = 0.2
                if self.training and random.random() < random_rotation_probability:  # 20% chance of rotation

                    angle = random.randint(0, 3) * 90 + random.randint(-30, 30)
                    sat_img, mask, target, normal = self._apply_rotation(sat_img, mask, target, normal, angle)

                    # Rotate and filter links
                    nidmap, nodes, locallinks, centers = links
                    newlocallinks = []
                    for locallink in locallinks:
                        oor = False
                        newlocallink = []
                        for k in range(len(locallink)):
                            pos = [locallink[k][0], locallink[k][1]]
                            pos = rotate(pos, -angle, self.dataset_image_size) 
                            if pos[0] < 0 or pos[0] > self.dataset_image_size or pos[1] < 0 or pos[1] > self.dataset_image_size:
                                oor = True
                                break
                            
                            newlocallink.append(pos)

                        if oor == False:
                            newlocallinks.append(newlocallink)
                    
                    if len(newlocallinks) == 0:
                        continue 
                    
                    links[2] = newlocallinks
                            
                    new_nodes = {}
                    for k in nodes.keys():
                        pos = nodes[k]
                        pos = rotate(pos, -angle, self.dataset_image_size)
                        if pos[0] < 0 or pos[0] > self.dataset_image_size or pos[1] < 0 or pos[1] > self.dataset_image_size:
                            continue
                        new_nodes[k] = pos

                    if len(new_nodes) == 0:
                        continue 

                    links[1] = new_nodes

                normal = self._process_normal_map(normal, angle)
                # Normalize images
                sat_img = sat_img.astype(np.float64) / 255.0 - 0.5
                mask = mask.astype(np.float64) / 255.0
                target = target.astype(np.float64) / 255.0
                
                self.links.append(links)
                self.images[i, :, :, :] = sat_img
                self.masks[i, :, :, 0] = mask
                self.targets[i, :, :, 0] = target
                # self.targets_t[i,:,:,0] = target_t
                self.normal[i, :, :, :] = normal

                # Apply augmentation
                self._apply_augmentation(self.images, i)

                break

        self.get_internal_batch(self.maxbatchsize)

    def get_internal_batch(self, batchsize):
        #print("getting batch")

        img = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        connector1 = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        connector2 = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        connectorlink = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        
        for i in range(batchsize):
            while True:
                tile_id = random.randint(0,self.preload_tiles-1)
                nidmap, nodes, locallinks, centers = self.links[tile_id]
                if len(locallinks) == 0:
                    continue

                # sample two in-connected points
                locallink = random.choice(locallinks)
                vertices = locallink
                sr = (vertices[0][1] + vertices[-1][1]) // 2
                sc = (vertices[0][0] + vertices[-1][0]) // 2
                
                sr -= self.image_size // 2
                sc -= self.image_size // 2
                if sr < 0: 
                    sr = 0 
                if sr + self.image_size >= self.dataset_image_size:
                    sr = self.dataset_image_size - self.image_size
                if sc < 0: 
                    sc = 0 
                if sc + self.image_size >= self.dataset_image_size:
                    sc = self.dataset_image_size - self.image_size
                
                img = img * 0
                connector1 = connector1 * 0
                connector2 = connector2 * 0
                
                for k in range(len(vertices)-1):
                    x1 = vertices[k][0] - sc 
                    y1 = vertices[k][1] - sr 
                    x2 = vertices[k+1][0] - sc 
                    y2 = vertices[k+1][1] - sr 
                    cv2.line(img, (x1,y1), (x2,y2), (255), 5)
                    if k == 0:
                        cv2.circle(connector1, (x1,y1), 12, (255), -1)
                        xx1, yy1 = x1, y1
                        #print(x1,y1)
                    if k == len(vertices)-2:
                        xx2, yy2 = x2, y2
                        cv2.circle(connector2, (x2,y2), 12, (255), -1)
                        #print(x2,y2)
                x1,y1 = xx1,yy1
                x2,y2 = xx2,yy2
                if x1 < 0 or x1 >= self.image_size or x2 < 0 or x2 >= self.image_size:
                    continue
                if y1 < 0 or y1 >= self.image_size or y2 < 0 or y2 >= self.image_size:
                    continue
            
                self.target_batch[i,:,:,0] = np.copy(img) / 255.0
                self.connector_batch[i,:,:,0] = np.copy(connector1) / 255.0 - 0.5
                self.connector_batch[i,:,:,3] = np.copy(connector2) / 255.0 - 0.5
                self.connector_batch[i,:,:,1:3] = self.poscode[self.image_size - y1:self.image_size*2 - y1, self.image_size - x1:self.image_size*2 - x1, :]
                self.connector_batch[i,:,:,4:6] = self.poscode[self.image_size - y2:self.image_size*2 - y2, self.image_size - x2:self.image_size*2 - x2, :]
                self.connector_batch[i,:,:,6] = np.copy(connectorlink) / 255.0 - 0.5
                
                self.image_batch[i,:,:,:] = self.images[tile_id, sr:sr+self.image_size, sc:sc+self.image_size, :]
                
                self.normal_batch[i,:,:,:] = self.normal[tile_id, sr:sr+self.image_size, sc:sc+self.image_size, :]
                
                break
        
        #print("getting batch done")
        return self.image_batch[:batchsize, :,:,:], self.connector_batch[:batchsize,:,:,:], self.target_batch[:batchsize, :,:,:], self.normal_batch[:batchsize,:,:,:]


    def get_batch(self):
        batchsize = self.batch_size
        st = random.randint(0, self.maxbatchsize - batchsize - 1)

        return (
            self.image_batch[st : st + batchsize, :, :, :],
            self.connector_batch[st : st + batchsize, :, :, :],
            self.target_batch[st : st + batchsize, :, :, :],
            self.normal_batch[st : st + batchsize, :, :, :],
        )


class ParallelDataLoader:
    """
    A parallel data loader that manages multiple Dataloader instances in separate threads
    to enable concurrent data loading and preprocessing.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize parallel data loader with multiple subloaders."""
        self.num_workers = 4  # Number of worker threads
        self.subloaders = []
        self.ready_events = []  # Events signaling when subloader has data ready
        self.wait_events = []   # Events to signal subloader to start preloading
        
        self.current_loader_index = 0
        
        # Create subloaders and their associated events
        for i in range(self.num_workers):
            self.subloaders.append(Dataloader(*args, **kwargs))
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
        return ParallelDataLoader(
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
    config = load_config("configs/train_lane_validation.py")
    dataloaers_config = config.dataloaders
    train_dataloader, validate_dataloader, test_dataloader = get_dataloaders(dataloaers_config)
    train_dataloader.get_batch()