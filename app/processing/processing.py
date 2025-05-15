# input image file path (pathlib.path)
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))

from itertools import product
from pathlib import Path
from typing import Tuple

import cv2
import networkx as nx
import numpy as np
import plotly.express as px  # noqa
import s2g
from skimage import morphology

from app.code_torch.framework.base_classes import SimpleConfig
from app.code_torch.laneAndDirectionExtraction.model_manager import LaneExModelManager
from app.code_torch.turningPairExtraction.model_manager import TurnValModelManager

MODEL_WINDOW = 640


class InferenceEngine:
    def __init__(
        self,
        lane_ex_model_path: Path,
        turn_val_model_path: Path,
        save_images: bool = False,
    ):
        self.window_size = 640
        self.lane_ex_model_path = lane_ex_model_path
        self.turn_val_model_path = turn_val_model_path

        self.config = SimpleConfig(1, 640)

        self.lane_ex_model = LaneExModelManager(self.config)
        self.lane_ex_model.restore_model(self.lane_ex_model_path)

        self.turn_val_model = TurnValModelManager(self.config)
        self.turn_val_model.restore_model(self.turn_val_model_path)

        self.position_code = np.zeros(
            (1, self.window_size, self.window_size, 2), dtype=np.float32
        )
        for i in range(self.window_size):
            self.position_code[:, i, :, 0] = float(i) / self.window_size
            self.position_code[:, :, i, 0] = float(i) / self.window_size

        self.poscode = np.zeros(
            (self.window_size * 2, self.window_size * 2, 2), dtype=np.float32
        )
        for i in range(self.window_size * 2):
            self.poscode[i, :, 0] = float(i) / self.window_size - 1.0
            self.poscode[:, i, 1] = float(i) / self.window_size - 1.0

        self.save_images = save_images

    def process_lane_ex(self, img_path: Path) -> nx.DiGraph:
        """Processes aerial image into lane and normal mapping using Lane Extraction CNN.

        Args:
            img_path (Path): Input aerial image of size 640x640

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of raw lane and direction segmentation.
        """
        in_img = cv2.imread(str(img_path))

        raw_res = self.lane_ex_model.infer(in_img)

        # Removing bin complement
        lane_out = (
            np.clip(
                raw_res[:, :, 0],
                0,
                1,
            )
            * 255
        )
        normal_out = (
            np.clip(
                raw_res[:, :, 1:3],
                -1,
                1,
            )
            * 127
            + 127
        )

        lane_out = lane_out > 100

        lane_out = morphology.thin(lane_out)

        lane_graph = s2g.extract_graph_from_image(lane_out, epsilon=5.0)
        lane_graph = s2g.direct_graph_from_vector_map(lane_graph, normal_out)

        return lane_graph

    def make_turn_ex_input(
        self, node_pair: Tuple[Tuple[int, int], Tuple[int, int]], aerial_img, normal
    ) -> Tuple[np.ndarray, np.ndarray]:
        connector1 = np.zeros((self.window_size, self.window_size), dtype=np.uint8)
        connector2 = np.zeros((self.window_size, self.window_size), dtype=np.uint8)

        y1, x1 = node_pair[0]
        y2, x2 = node_pair[1]

        cv2.circle(connector1, (x1, y1), 12, (255,), -1)
        cv2.circle(connector2, (x2, y2), 12, (255,), -1)
        connector = np.zeros((1, 640, 640, 6), dtype=np.float32)

        connector[0, :, :, 0] = np.copy(connector1) / 255.0 - 0.5
        connector[0, :, :, 1:3] = self.poscode[
            self.window_size - y1 : self.window_size * 2 - y1,
            self.window_size - x1 : self.window_size * 2 - x1,
        ]
        # Not sure why this needs to be flipped but it does.
        connector[0, :, :, [1, 2]] = connector[0, :, :, [2, 1]]

        connector[0, :, :, 3] = np.copy(connector2) / 255.0 - 0.5
        connector[0, :, :, 4:6] = self.poscode[
            self.window_size - y2 : self.window_size * 2 - y2,
            self.window_size - x2 : self.window_size * 2 - x2,
        ]
        connector[0, :, :, [4, 5]] = connector[0, :, :, [5, 4]]

        in_img = np.expand_dims(aerial_img / 255.0 - 0.5, axis=0)
        normal = np.expand_dims((normal - 127.0) / 127.0, axis=0)
        normal[0, :, :, [0, 1]] = normal[0, :, :, [1, 0]]

        input_1 = np.concatenate(
            (in_img, connector[:, :, :, 0:3], normal, self.position_code), axis=3
        )
        input_2 = np.concatenate(
            (in_img, connector[:, :, :, 3:6], normal, self.position_code), axis=3
        )
        return input_1, input_2

    def process_turn_ex(
        self,
        input_arr: Tuple[np.ndarray, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        # each input is bat0, bat1[0:3] / bat1[3:6], bat[4], poscode

        res_1 = self.turn_val_model.infer(input_arr[0])
        res_2 = self.turn_val_model.infer(input_arr[1])

        return res_1, res_2

    def process(
        self,
        intersection_image_path: Path,
        intersection_center: Tuple[int, int] = (320, 320),
    ):
        in_img = cv2.imread(str(intersection_image_path))

        lane_graph = self.process_lane_ex(intersection_image_path)
        lane_out, normal_out = s2g.draw_inputs(lane_graph)

        lane_graph = s2g.annotate_node_types(lane_graph, intersection_center)
        node_types = s2g.get_node_types(lane_graph)

        in_nodes = node_types.get("in", [])
        out_nodes = node_types.get("out", [])

        node_pairs = list(product(in_nodes, out_nodes))

        out_ints = []
        for node_pair in node_pairs:
            turn_ex_inputs = self.make_turn_ex_input(node_pair, in_img, normal_out)

            out_1, out_2 = self.process_turn_ex(turn_ex_inputs)
            out_1 = out_1 > 0.4
            out_2 = out_2 > 0.4

            out = np.logical_and(out_1, out_2)
            out_ints.append(out)
        return out_ints


lane_ex_model_path = Path(
    "/home/lab/development/lab/final/LaneGraph/models/LaneEx500ep/LaneEx_ep_500"
)
turn_val_model_path = Path(
    "/home/lab/development/lab/final/LaneGraph/models/TurnPair500ep/TurnPair_ep_500"
)
ie = InferenceEngine(lane_ex_model_path, turn_val_model_path)

test_sat_img = Path(
    "/home/lab/development/lab/final/LaneGraph/app/processing/test/0_0_0_input.jpg"
)


print(ie.process(test_sat_img))
