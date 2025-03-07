import json
import sys
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from time import time
from typing import Tuple, Type, Union, List
from dataclasses import dataclass, field
import numpy as np

@dataclass
class Config:
    batch_size: int
    preload_size: int
    dataset_folder: Path
    dataset_split_file: Path
    image_size: int=640
    dataset_image_size: int=4096
    testing: bool=False
    learning_rate: float=0.001
    lr_decay: List[float]=field(default_factory=lambda: [0.1, 0.1])
    lr_decay_ep: List[int]=field(default_factory=lambda: [350, 450])
    ep_max: int=500
    epoch_size: int=500 # change to lambda later
    step_init: int=0
    model_save_ep_int: int=10
    viz_save_ep_int: int=1
    tag: str=field(default_factory=lambda: input("Enter run tag: "))
    training_range: List[str] = field(init=False)

    def __post_init__(self):
        self.training_range = []
        with open(self.dataset_split_file, "r") as json_file:
            dataset_split = json.load(json_file)
        for tid in dataset_split["training"]:
            self.training_range.append(f"_{tid}")
    

class DataLoader(ABC):
    @abstractmethod
    def __init__(self, batch_size: int, preload_size: int) -> None:
        self.batch_size = batch_size
        self.preload_size = preload_size

    @abstractmethod
    def preload(self) -> None:
        pass

    @abstractmethod
    def get_batch(self) -> Union[Tuple[np.ndarray, ...], None]:
        pass


class ParallelDataLoader(DataLoader):
    def __init__(
        self,
        batch_size: int,
        preload_size: int,
        num_loaders: int,
        dataloader: Type[DataLoader],
        **kwargs,
    ) -> None:
        super().__init__(batch_size, preload_size)
        self.num_loaders = num_loaders

        self.subloader = [
            dataloader(batch_size=batch_size, preload_size=preload_size, **kwargs)
            for _ in range(num_loaders)
        ]
        self.subloader_ready_event = [threading.Event() for _ in range(num_loaders)]
        self.subloader_wait_event = [threading.Event() for _ in range(num_loaders)]

        self.current_loader_id = 0
        self.lock = threading.Lock()

        for i in range(self.num_loaders):
            threading.Thread(target=self._daemon, args=(i,), daemon=True).start()
        self._wait_for_preload()

    def _wait_for_preload(self):
        with self.lock:
            print("Waiting for all threads to complete preloading...")
        for event in self.subloader_ready_event:
            event.wait()

    def _daemon(self, tid: int) -> None:
        while True:
            with self.lock:
                print(f"Thread-{tid} starts preloading")
            t0 = time()
            self.subloader[tid].preload()
            print(f"Thread-{tid} finished preloading (time = {time() - t0:.2f}s)")
            self.subloader_ready_event[tid].set()
            self.subloader_wait_event[tid].wait()
            self.subloader_wait_event[tid].clear()

    def preload(self):
        with self.lock:
            self.subloader_wait_event[self.current_loader_id].set()

            self.current_loader_id = (self.current_loader_id + 1) % self.num_loaders

        self.subloader_ready_event[self.current_loader_id].wait()
        self.subloader_ready_event[self.current_loader_id].clear()

    def get_batch(self):
        attempts = 0
        while attempts < self.num_loaders:
            batch = self.subloader[self.current_loader_id].get_batch()

            if batch is not None:
                return batch

            print(f"Thread-{self.current_loader_id} exausted")
            self.subloader_wait_event[self.current_loader_id].set()

            with self.lock:
                self.current_loader_id = (self.current_loader_id + 1) % self.num_loaders

            self.subloader_ready_event[self.current_loader_id].wait()
            self.subloader_ready_event[self.current_loader_id].set()

            attempts += 1

        print("Waiting for subloader to finish preloading...")
        while True:
            for i in range(self.num_loaders):
                if self.subloader_ready_event[i].is_set():
                    with self.lock:
                        self.current_loader_id = i
                    self.subloader_ready_event[i].clear()
                    return self.get_batch()


class ModelManager(ABC):
    @abstractmethod
    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size

    @abstractmethod
    def train(self, batch: Tuple[np.ndarray, ...], lr: float) -> Tuple:
        # TODO Change return to better hint [batch, loss, idk]
        pass

    @abstractmethod
    def infer(self, input) -> np.ndarray:
        pass

    @abstractmethod
    def save_model(self, path: Path) -> None:
        pass

    @abstractmethod
    def restore_model(self, path: Path) -> None:
        pass


class Trainer(ABC):
    def __init__(
        self,
        config: Config,
        dataloader: DataLoader,
        model_manager: ModelManager,
    ) -> None:
        default_config = {
            "data_config": {
                "batch_size": 4,
                "preload_size": 4,
                "image_size": 640,
                "dataset_image_size": 4096,
                "testing": False,
                "dataset_folder": Path(
                    "/home/lab/development/lab/modular/LaneGraph/msc_dataset/dataset_unpacked"
                ),
                "training_range": None,
            },
            "dataset_split_file": "/home/lab/development/lab/modular/LaneGraph/app/code_torch/split_all.json",
            "step_init": 0,
            "model_save_ep_int": 10,  # epochs
            "image_save_int": 500,  # step
        }
        self.config = config
        self.tag = self.config.tag
        self.epoch_size = self.config.epoch_size
        self.logs = {}
        self.training_range = self.config.training_range

        self.model_manager = model_manager
        self.dataloader = dataloader

        cur_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.model_folder = (
            Path(__file__).parents[3] / "models" / f"{cur_datetime}_{self.tag}"
        ) # move to config?
        self.visualization_folder = self.model_folder / "visualization"
        self.visualization_folder.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        lr = self.config.learning_rate
        lr_decay = self.config.lr_decay
        lr_decay_step = [i * self.epoch_size for i in self.config.lr_decay_ep]
        max_step = self.config.ep_max * self.epoch_size + 1
        step = self.config.step_init
        last_step = -1

        loss = 0
        t_load = 0
        t_preload = 0
        t_train = 0
        t_other = 0
        t0 = time()

        epoch = 0.0
        last_epoch = -1.0
        print("Starting Training:{}")
        while True:
            t_other += time() - t0
            t0 = time()

            if step % 50 == 0:
                self.dataloader.preload()
            t1 = time()

            batch = self.dataloader.get_batch()  # TODO Add batchsize config
            if batch is None:
                raise ValueError("Batch Is None")
            t2 = time()

            result = self.model_manager.train(batch, lr)
            t3 = time()

            t_preload += t1 - t0
            t_train += t2 - t1
            t_load += t3 - t2

            if np.isnan(result[0]):
                raise ValueError("Loss is nan")
            loss += result[0]

            if step % 10 == 0:
                t_misc = t_other - t_preload - t_load - t_train
                epoch = step / float(self.epoch_size)
                p = int((epoch - int(epoch)) * 50)
                loss /= 10

                self._add_log("loss", step, loss)

                sys.stdout.write(
                    f"\rstep {step} epoch: {epoch:.2f}"
                    + ">" * p
                    + "-" * (51 - p)
                    + f" loss {loss:f} "
                    + f"time {t_preload:f} {t_load:f} {t_train:f} {t_misc:f} "
                )
                sys.stdout.flush()

                if int(epoch) != int(last_epoch):
                    print("\ntime per epoch", t_other)
                    print(
                        "eta",
                        t_other
                        * (max_step - step)
                        / max(1, (step - last_step))
                        / 3600.0,
                    )
                    last_step = step
                    t_load = 0
                    t_preload = 0
                    t_train = 0
                    t_other = 0

                loss = 0
                last_epoch = epoch
            if epoch % self.config.model_save_ep_int == 0:
                self.model_manager.save_model(self.model_folder / f"ep{epoch}.pth")
                with open(self.model_folder / "logs.json", "w") as jsonfile:
                    json.dump(self.logs, jsonfile, indent=4)

            if step % self.config.viz_save_ep_int == 0:
                self._visualize(int(epoch), step, batch, result)

            # Updating learning rate for next loop
            for i in range(len(lr_decay_step)):
                if step == lr_decay_step[i]:
                    lr = lr * lr_decay[i]

            if step == max_step + 1:
                break
            step += 1

    @abstractmethod
    def _visualize(self, epoch: int, step: int, batch: tuple, result: tuple):
        """
        Abstract Method to save images to visualization folder

        :param epoch: Current Epoch
        :type epoch: int
        :param step: Current Step
        :type step: int
        :param batch: Tuple of loaded input data
        :type batch: tuple
        :param result: Tuple of outputs from model
        :type result: tuple
        :param save_folder: pathlib Path for save directory
        :type save_folder: Path
        """
        pass

    def _add_log(self, key: str, *args) -> None:
        """
        Adds a new log. Args are logged to the list indexed py their position.

        :param key: Key to associate the log entry with.
        :type key: str
        :param args: Variable length arguments, where each is logged to its own list.
        :type args: tuple
        """
        if key not in self.logs:
            self.logs[key] = [[] for _ in args]
        for i, arg in enumerate(args):
            self.logs[key][i].append(arg)
