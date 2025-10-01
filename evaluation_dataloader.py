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
        self.links = []
        self.nid2links = []
        self.pos2nid = []

        self.image_batch = np.zeros((image_size, image_size, 3))
        self.undirected_nonintersection_graph = nx.Graph()
        self.directed_nonintersection_graph = nx.DiGraph()
        self.full_directed_graph = nx.DiGraph()

    def _load_sat_image_data(self, ind):
        """Load all image data for a given index."""
        sat_img = imageio.imread(os.path.join(self.data_path, f"sat_{ind}.jpg"))
        
        return sat_img

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
            sat_img = self._load_sat_image_data(current_ind)
            self.images[idx, :, :, :] = sat_img
            undirected_nonintersection_graph, directed_nonintersection_graph, full_directed_graph = self._load_graph_data(current_ind)
            self.undirected_nonintersection_graphs[idx] = undirected_nonintersection_graph
            self.directed_nonintersection_graphs[idx] = directed_nonintersection_graph
            self.full_directed_graphs[idx] = full_directed_graph



    def get_batch(self):
        
        while True:
            tile_id = random.randint(0,self.preload_tiles-1)
            # the bottom left corner of the crop
            x_min = random.randint(0, self.dataset_image_size-1-self.image_size)
            y_min = random.randint(0, self.dataset_image_size-1-self.image_size)
            x_max = x_min + self.image_size
            y_max = y_min + self.image_size
            # crop the graphs
            undirected_nonintersection_graph = crop_graph(
                self.undirected_nonintersection_graphs[tile_id],
                x_min, x_max, y_min, y_max,
                pos_attr="pos"
            )
            directed_nonintersection_graph = crop_graph(
                self.directed_nonintersection_graphs[tile_id],
                x_min, x_max, y_min, y_max,
                pos_attr="pos"
            )
            full_directed_graph = crop_graph(
                self.full_directed_graphs[tile_id],
                x_min, x_max, y_min, y_max,
                pos_attr="pos"
            )
            # adjust the node positions
            self.undirected_nonintersection_graph:nx.Graph = adjust_node_positions(undirected_nonintersection_graph, x_offset=x_min, y_offset=y_min)
            self.directed_nonintersection_graph:nx.DiGraph = adjust_node_positions(directed_nonintersection_graph, x_offset=x_min, y_offset=y_min)
            self.full_directed_graph:nx.DiGraph = adjust_node_positions(full_directed_graph, x_offset=x_min, y_offset=y_min)
            # if too few nodes, then resample
            if self.undirected_nonintersection_graph.number_of_nodes()<100:
                
                continue
            
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
    
    
    
    evaluator = GraphEvaluator()
    test_dataloader = get_test_dataloader(dataloaders_config)
    metrics_dicts = []
    
    for epoch in range(max_epochs):
        for batch_idx in range(epoch_size):
            sat_image, undirected_nonintersection_graph_groundtruth, directed_nonintersection_graph_groundtruth, final_directed_graph_groundtruth = test_dataloader.get_batch()
            directed_nonintersection_graph_predicted, final_directed_graph_predicted = lane_graph_extraction.extract_lane_graph(sat_image, mode="rule_based")
            
            # evaluate the directed_nonintersection_graph_predicted against directed_nonintersection_graph_groundtruth
            metrics_dict = {"directed_nonintersection_graph":None, "final_directed_graph":None}


            directed_nonintersection_graph_metrics_dict = evaluator.evaluate_graph(directed_nonintersection_graph_groundtruth, directed_nonintersection_graph_predicted, area_size=[640,640], lane_width=10)
            metrics_dict["directed_nonintersection_graph"] = directed_nonintersection_graph_metrics_dict

            final_directed_graph_metrics_dict = evaluator.evaluate_graph(final_directed_graph_groundtruth, final_directed_graph_predicted, area_size=[640,640], lane_width=10)
            metrics_dict["final_directed_graph"] = final_directed_graph_metrics_dict

            metrics_dicts.append(metrics_dict)
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
            segmentation2graph.draw_directed_graph(final_directed_graph_groundtruth, ax=axes[1])
            segmentation2graph.draw_directed_graph(final_directed_graph_predicted, ax=axes[2])
            plt.savefig(os.path.join("evaluation_output", f"epoch_{epoch+1}_batch_{batch_idx+1}.png"))
    
    
    
    
    
    

