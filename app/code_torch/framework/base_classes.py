import json
import sys
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import time
from typing import List, Tuple, Type, Union

import numpy as np


@dataclass
class Config:
    batch_size: int
    preload_size: int
    dataset_folder: Path
    dataset_split_file: Path  # Move to dataset folder?
    image_size: int = 640
    dataset_image_size: int = 4096

    testing: bool = False

    learning_rate: float = 0.001
    lr_decay: List[float] = field(default_factory=lambda: [0.1, 0.1])
    lr_decay_ep: List[int] = field(default_factory=lambda: [350, 450])
    ep_max: int = 500
    epoch_size: int = field(init=False)
    step_init: int = 0
    model_save_ep_int: int = 50
    viz_save_ep_int: int = 10
    tag: str = field(default_factory=lambda: input("Enter run tag: "))
    model_folder: Path = field(init=False)
    visualization_folder: Path = field(init=False)

    def __post_init__(self):
        self.training_range = []
        with open(self.dataset_split_file, "r") as json_file:
            dataset_split = json.load(json_file)
        for tid in dataset_split["training"]:
            self.training_range.append(f"_{tid}")

        self.epoch_size = (self.dataset_image_size**2 * len(self.training_range)) // (
            self.batch_size * self.image_size**2
        )  # Ratio of dataset pixels to batch pixes

        # Setting up folder paths for model
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.model_folder = (
            Path(__file__).parents[3] / "models" / f"{current_datetime}_{self.tag}"
        )
        self.visualization_folder = self.model_folder / "visualization"
        self.visualization_folder.mkdir(parents=True, exist_ok=True)


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
    def __init__(self, config: Config) -> None:
        self.config = config

    @abstractmethod
    def train(self, batch: Tuple[np.ndarray, ...], lr: float) -> Tuple:
        # TODO Change return to better hint [batch, loss, idk]
        pass

    @abstractmethod
    def infer(self, input) -> np.ndarray:
        pass

    @abstractmethod
    def save_model(self, ep: int) -> None:
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
        self.config = config
        self.model_manager = model_manager
        self.dataloader = dataloader

        self.logs = {}

    def run(self) -> None:
        lr = self.config.learning_rate
        lr_decay = self.config.lr_decay
        lr_decay_step = [i * self.config.epoch_size for i in self.config.lr_decay_ep]
        max_step = self.config.ep_max * self.config.epoch_size + 1
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
            self._add_log(step, result)
            loss += result[0]

            if step % 10 == 0:
                t_misc = t_other - t_preload - t_load - t_train
                epoch = step / float(self.config.epoch_size)
                p = int((epoch - int(epoch)) * 50)
                loss /= 10

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
            if step % (self.config.epoch_size * self.config.model_save_ep_int) == 0:
                self.model_manager.save_model(int(epoch))
                self._save_logs()

            if step % (self.config.epoch_size * self.config.viz_save_ep_int) == 0:
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

    @abstractmethod
    def _add_log(self, step: int, results: tuple) -> None:
        """
        Adds values to self.logs based on their keyword. Values will be appended to the
        list of their keyword. If a list does not already exist for a keyword, a list
        will be created.

        :param kwargs: Values to be logged at keyword
        :type kwargs: dict
        """
        return

    def _save_logs(self) -> None:
        """
        Save self.logs log dictionary to logs.json in model folder
        """
        save_path = self.config.model_folder / "logs.json"
        with save_path.open("w", encoding="utf-8") as json_file:
            json.dump(self.logs, json_file)
        return
