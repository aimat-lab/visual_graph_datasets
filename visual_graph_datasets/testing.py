"""
Functionality to support unittests.
"""
import os
import itertools
import tempfile
import typing as t

import matplotlib.pyplot as plt
import numpy as np
from visual_graph_datasets.data import VisualGraphDatasetWriter
import visual_graph_datasets.typing as tv

import visual_graph_datasets.typing as tv
from visual_graph_datasets.util import TEMPLATE_ENV
from visual_graph_datasets.config import Config
from visual_graph_datasets.processing.base import ProcessingBase
from visual_graph_datasets.generation.graph import GraphGenerator


def generate_random_graph(num_nodes: int = 10,
                          num_node_features: int = 3,
                          num_edge_features: int = 1,
                          ) -> tv.GraphDict:
    """
    This method will create a randomly generated graph dict representation of a graph with the 
    given number ``num_nodes`` of nodes, the given number ``num_node_features`` of features per 
    node and ``num_edge_features`` nodes per edge.
    
    This graph will have no cycles. The node and edge features will be randomly generated.
    
    :param num_nodes: The nodes that the graph should have
    :param num_node_features: The number of features that the nodes of the graph should have
    :param num_edge_features: The number of featuers that the edges of the graph should have 
    
    :returns: A graph dict representation of the randomly generated graph.
    """
    generator = GraphGenerator(
        num_nodes=num_nodes,
        num_additional_edges=0,
        node_attributes_cb=lambda *args, **kwargs: np.random.random(size=(num_node_features, )),
        edge_attributes_cb=lambda *args, **kwargs: np.random.random(size=(num_edge_features, ))
    )
    generator.reset()
    graph = generator.generate()
    return graph


def clear_files(file_paths: t.List[str]):
    """
    Small utility function, which will make sure that all the files with the given ``file_paths`` are
    definitely removed from the filesystem. Used for testing purposes.
    """
    for file_path in file_paths:
        if os.path.exists(file_path) and os.path.isfile(file_path):
            os.remove(file_path)


class IsolatedConfig:
    """
    This is a context manager which will create a fresh config file from the default template within
    it's context. Furthermore, this config file will be within a temporary folder, into which the
    default datasets path will be configured to as well.
    """
    def __init__(self):
        self.dir = tempfile.TemporaryDirectory()
        self.folder_path: t.Optional[str] = None
        self.datasets_folder: t.Optional[str] = None
        self.config_path: t.Optional[str] = None
        self.config = Config()

    def __enter__(self):
        # First of all we need to init the temporary directory
        self.folder_path = self.dir.__enter__()

        # Then we need to create the folder which will contain all the datasets that are potentially
        # downloaded:
        self.datasets_folder = os.path.join(self.folder_path, 'datasets')
        os.mkdir(self.datasets_folder)

        # Next we need to create the config file from the template. In this step it is important that we
        # pass in the custom dataset folder to be used in the config
        self.config_path = os.path.join(self.folder_path, 'config.yaml')
        template = TEMPLATE_ENV.get_template('config.yaml.j2')
        with open(self.config_path, mode='w') as file:
            content = template.render(datasets_path=self.datasets_folder)
            file.write(content)

        # And finally we load that config file into the config singleton
        self.config.load(self.config_path)

        return self.config

    def __exit__(self, *args, **kwargs):
        # In the end we need to make sure to destroy the temporary directory again
        self.dir.__exit__(*args, **kwargs)



class MockProcessing(ProcessingBase):
    """
    This is a mock implementation for a processing class for testing purposes.
    
    The domain representation for this processing class is a string which represents an integer. That 
    integer will be used as the number of nodes for the generated graph. The graph will be fully 
    connected, meaning that every node will be connected to every other node. The nodes and edges of 
    the graph will have attribute feature vectors with constant feature values of 1.0
    """
    
    def process(self,
                value: str,
                num_node_attributes: int = 10,
                num_edge_attributes: int = 10,
                ):
        num_nodes = int(value)
        
        node_indices = list(range(num_nodes))
        node_attributes = [[1.0] * num_node_attributes for _ in node_indices]
        
        # The graph will also be fully-connected, which means that all nodes are connected to all 
        # other nodes with an edge.
        edge_indices = list(itertools.permutations(node_indices, 2))
        edge_attributes = [[1.0] * num_edge_attributes for _ in edge_indices]
        
        graph = {
            'node_indices':         np.array(node_indices, dtype=int),
            'node_attributes':      np.array(node_attributes, dtype=float),
            'edge_indices':         np.array(edge_indices, dtype=int),
            'edge_attributes':      np.array(edge_attributes, dtype=float)
        }
        return graph
    
    def visualize(self, 
                  value: tv.DomainRepr, 
                  width: int, 
                  height: int, 
                  **kwargs) -> np.ndarray:
        pass
    
    def create(self,
               value: tv.DomainRepr, 
               index: str, 
               width: int,
               height: int, 
               output_path: str, 
               additional_graph_data: dict, 
               additional_metadata: dict, 
               writer: VisualGraphDatasetWriter | None = None, 
               *args, 
               **kwargs
               ) -> tv.MetadataDict:
        pass
