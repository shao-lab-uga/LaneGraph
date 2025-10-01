import json
import os
import random
import pickle
import numpy as np
import networkx as nx
import imageio.v3 as imageio
import matplotlib.pyplot as plt
import torch
from lane_graph_extraction_pipline import LaneGraphExtraction
from utils import segmentation2graph
from utils.config_utils import load_config
from evaluator.evaluator import GraphEvaluator
from utils.graph_postprocessing_utils import annotate_node_types
    
def adjust_node_positions(G, x_offset=0, y_offset=0):
    for n in G.nodes:
        node_position = G.nodes[n].get('pos', None)
        if node_position is None:
            continue
        node_position = (node_position[0] - x_offset, node_position[1] - y_offset)
        G.nodes[n]['pos'] = node_position
    return G


def crop_graph(G: nx.Graph, x_min, x_max, y_min, y_max, pos_attr="pos"):
    """
    Crop a networkx graph based on node coordinates.

    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
        The input graph.
    x_min, x_max, y_min, y_max : float
        Bounding box range.
    pos_attr : str
        Node attribute name storing (x, y) coordinates, default 'pos'.

    Returns
    -------
    nx.Graph or nx.DiGraph
        Cropped subgraph containing only nodes inside the bounding box,
        along with edges between them.
    """
    # Keep only nodes whose coordinates are inside the bounding box
    nodes_in_box = [
        n for n, d in G.nodes(data=True)
        if pos_attr in d and
           x_min <= d[pos_attr][0] <= x_max and
           y_min <= d[pos_attr][1] <= y_max
    ]
    
    # Induce subgraph
    return G.subgraph(nodes_in_box).copy()


class EvaluationDataloader:
    def __init__(
        self,
        data_path,
        indrange,
        image_size=640,
        dataset_image_size=2048,
        preload_tiles=4,
    ):
        self.data_path = data_path
        self.indrange = indrange
        self.image_size = image_size
        self.dataset_image_size = dataset_image_size
        self.preload_tiles = preload_tiles
        self.images = np.zeros((preload_tiles, dataset_image_size, dataset_image_size, 3))
        self.undirected_nonintersection_graphs = [ None for _ in range(preload_tiles) ]
        self.directed_nonintersection_graphs = [ None for _ in range(preload_tiles) ]
        self.full_directed_graphs = [ None for _ in range(preload_tiles) ]
        self.centers = [[] for _ in range(preload_tiles)]
        self.links = []
        self.nid2links = []
        self.pos2nid = []
        self.batch_size = 1
        self.image_batch = np.zeros((image_size, image_size, 3))
        self.undirected_nonintersection_graph = nx.Graph()
        self.directed_nonintersection_graph = nx.DiGraph()
        self.full_directed_graph = nx.DiGraph()

    def _load_sat_image_data(self, ind):
        """Load all image data for a given index."""
        sat_img = imageio.imread(os.path.join(self.data_path, f"sat_{ind}.jpg"))
        with open(os.path.join(self.data_path, f"link_{ind}.json"), "r") as json_file:
            _, _, _, centers = json.load(json_file)
        return sat_img, centers

    def _load_graph_data(self, ind):
        """Load all graph data for a given index."""
        undirected_nonintersection_graph: nx.Graph
        directed_nonintersection_graph: nx.DiGraph
        full_directed_graph: nx.DiGraph

        with open(os.path.join(self.data_path, f"undirected_nonintersection_graph_{ind}.gpickle"), 'rb') as f:
            undirected_nonintersection_graph = pickle.load(f)
        with open(os.path.join(self.data_path, f"directed_nonintersection_graph_{ind}.gpickle"), 'rb') as f:
            directed_nonintersection_graph = pickle.load(f)
        with open(os.path.join(self.data_path, f"full_directed_graph_{ind}.gpickle"), 'rb') as f:
            full_directed_graph = pickle.load(f)

        return undirected_nonintersection_graph, directed_nonintersection_graph, full_directed_graph

    
    def preload(self, ind=None):

        self.links = []
        self.nid2links = []
        self.pos2nid = []
        for idx in range(self.preload_tiles if ind is None else 1):

            current_ind = random.choice(self.indrange) if ind is None else ind
            sat_img, centers = self._load_sat_image_data(current_ind)
            self.images[idx, :, :, :] = sat_img
            undirected_nonintersection_graph, directed_nonintersection_graph, full_directed_graph = self._load_graph_data(current_ind)
            self.undirected_nonintersection_graphs[idx] = undirected_nonintersection_graph
            self.directed_nonintersection_graphs[idx] = directed_nonintersection_graph
            self.full_directed_graphs[idx] = full_directed_graph
            self.centers[idx] = centers

    def _get_available_coordinates(self):
        """Get all available coordinates from all tiles."""
        available_coords = []
        for tile_idx, centers in enumerate(self.centers):
            for coord in centers:
                available_coords.append((coord, tile_idx))
        return available_coords

    def get_batch(self, margin = 20, centered_intersection=False):
        if centered_intersection:
            available_coords = self._get_available_coordinates()
            if len(available_coords) < self.batch_size:
                return None
        while True:
            if centered_intersection:
                center_coord, tile_id = random.choice(available_coords)
                available_coords.remove((center_coord, tile_id))
                # the bottom left corner of the crop
                center_x, center_y = center_coord
                x_min = center_x - (self.image_size // 2)
                y_min = center_y - (self.image_size // 2)
                x_min = np.clip(x_min, 0, self.dataset_image_size - self.image_size - 1)
                y_min = np.clip(y_min, 0, self.dataset_image_size - self.image_size - 1)
            else:
                tile_id = random.randint(0, self.preload_tiles - 1)
                x_min = random.randint(0, self.dataset_image_size - self.image_size - 1)
                y_min = random.randint(0, self.dataset_image_size - self.image_size - 1)
            x_max = x_min + self.image_size
            y_max = y_min + self.image_size
            # crop the graphs
            undirected_nonintersection_graph = crop_graph(
                self.undirected_nonintersection_graphs[tile_id],
                x_min + margin, x_max - margin, y_min + margin, y_max - margin,
                pos_attr="pos"
            )
            directed_nonintersection_graph = crop_graph(
                self.directed_nonintersection_graphs[tile_id],
                x_min + margin, x_max - margin, y_min + margin, y_max - margin,
                pos_attr="pos"
            )
            full_directed_graph = crop_graph(
                self.full_directed_graphs[tile_id],
                x_min + margin, x_max - margin, y_min + margin, y_max - margin,
                pos_attr="pos"
            )
            # adjust the node positions
            self.undirected_nonintersection_graph:nx.Graph = adjust_node_positions(undirected_nonintersection_graph, x_offset=x_min, y_offset=y_min)
            self.directed_nonintersection_graph:nx.DiGraph = adjust_node_positions(directed_nonintersection_graph, x_offset=x_min, y_offset=y_min)
            self.full_directed_graph:nx.DiGraph = adjust_node_positions(full_directed_graph, x_offset=x_min, y_offset=y_min)
            # if edges total length (calcuated by distance between nodes) is 0, then re-sample
            total_edge_length = 0
            for u, v in self.full_directed_graph.edges():
                pos_u = self.full_directed_graph.nodes[u].get('pos', None)
                pos_v = self.full_directed_graph.nodes[v].get('pos', None)
                if pos_u is None or pos_v is None:
                    continue
                total_edge_length += np.linalg.norm(np.array(pos_u) - np.array(pos_v))

            
            # crop the image
            self.image_batch[:,:,:] = self.images[tile_id, y_min:y_max, x_min:x_max]
            break

        return self.image_batch[:,:,:], self.undirected_nonintersection_graph, self.directed_nonintersection_graph, self.full_directed_graph


def get_test_dataloader(dataloaders_config):
    """
    Create a testing dataloader based on the provided configuration.
    Args:
        dataloaders_config: Configuration dictionary containing paths and ranges for each dataset split.
    Returns:
        Tuple of training, validation, and testing dataloaders.
    """

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
        return EvaluationDataloader(
            data_path=dataloader_config.data_path,
            indrange=dataloader_config.indrange,
            image_size=dataloader_config.image_size,
            dataset_image_size=dataloader_config.dataset_image_size,
            preload_tiles=dataloader_config.preload_tiles,
        )

    test_dataloader = get_dataloader(test_config)

    test_dataloader.preload()
    return test_dataloader

if __name__ == "__main__":
    # ============= Load Configuration =============
    evaluation_config = load_config("configs/evaluation.py")
    pipline_config = load_config("configs/lane_graph_extraction_pipline.py")
    lane_graph_extraction = LaneGraphExtraction(pipline_config, gpu_id=0)
    dataloaders_config = evaluation_config.dataloaders
    test_config = evaluation_config.test
    random_seed = test_config.random_seed
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    epoch_size = test_config.epoch_size
    max_epochs = test_config.max_epochs
    max_epochs = 1
    
    
    evaluator = GraphEvaluator()
    test_dataloader = get_test_dataloader(dataloaders_config)
    
    step = 0
    directed_nonintersection_graph_metrics_dicts = []
    final_directed_graph_metrics_dicts = []
    for epoch in range(max_epochs):
        for batch_idx in range(10):
            sat_image, undirected_nonintersection_graph_groundtruth, directed_nonintersection_graph_groundtruth, final_directed_graph_groundtruth = test_dataloader.get_batch()
            directed_nonintersection_graph_predicted, final_directed_graph_predicted = lane_graph_extraction.extract_lane_graph(sat_image, mode="rule_based", output_path="evaluation_output", image_name=f"epoch_{epoch+1}_batch_{batch_idx+1}.png")
            
            # evaluate the directed_nonintersection_graph_predicted against directed_nonintersection_graph_groundtruth
            metrics_dict = {"directed_nonintersection_graph":None, "final_directed_graph":None}


            directed_nonintersection_graph_metrics_dict = evaluator.evaluate_graph(directed_nonintersection_graph_groundtruth, directed_nonintersection_graph_predicted, area_size=[640,640], lane_width=10)
            

            final_directed_graph_metrics_dict = evaluator.evaluate_graph(final_directed_graph_groundtruth, final_directed_graph_predicted, area_size=[640,640], lane_width=10)

            directed_nonintersection_graph_metrics_dicts.append(directed_nonintersection_graph_metrics_dict)

            print(f"Directed Non-Intersection Graph Metrics: {directed_nonintersection_graph_metrics_dict}")
            print(f"Final Directed Graph Metrics: {final_directed_graph_metrics_dict}")

            final_directed_graph_metrics_dicts.append(final_directed_graph_metrics_dict)
            print(f"Epoch {epoch+1}/{max_epochs}, Batch {batch_idx+1}/{epoch_size}")
            
            sat_image_vis = sat_image.astype(np.uint8)
            fig, axes = plt.subplots(1,3,figsize=(24,8))
            # disable labels for all axes, keep the boder
            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(2)
            axes[0].imshow(sat_image_vis)
            axes[0].set_aspect('auto')
            axes[0].autoscale(False)
            # tight layout
            plt.tight_layout()
            final_directed_graph_groundtruth = annotate_node_types(final_directed_graph_groundtruth)
            segmentation2graph.draw_directed_graph(final_directed_graph_groundtruth, ax=axes[1])

            segmentation2graph.draw_directed_graph(final_directed_graph_predicted, ax=axes[2])
            plt.savefig(os.path.join("evaluation_output", f"epoch_{epoch+1}_batch_{batch_idx+1}.png"))
            if (step + 1) % 10 == 0:
                test_dataloader.preload()

            step += 1
    
# get the average metrics
    total_steps = len(directed_nonintersection_graph_metrics_dicts)
    avg_directed_nonintersection_graph_metrics = {}
    avg_final_directed_graph_metrics = {}
    for key in directed_nonintersection_graph_metrics_dicts[0].keys():
        avg_directed_nonintersection_graph_metrics[key] = np.mean([metrics_dict[key] for metrics_dict in directed_nonintersection_graph_metrics_dicts])
    for key in final_directed_graph_metrics_dicts[0].keys():
        avg_final_directed_graph_metrics[key] = np.mean([metrics_dict[key] for metrics_dict in final_directed_graph_metrics_dicts])
    print("Average Directed Non-Intersection Graph Metrics over {} steps:".format(total_steps))
    for key, value in avg_directed_nonintersection_graph_metrics.items():
        print(f"{key}: {value:.4f}")
    print("Average Final Directed Graph Metrics over {} steps:".format(total_steps))
    for key, value in avg_final_directed_graph_metrics.items():
        print(f"{key}: {value:.4f}")