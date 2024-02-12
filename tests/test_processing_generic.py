import os
import tempfile
import pytest

import numpy as np
import matplotlib.pyplot as plt
import visual_graph_datasets.typing as tv
from visual_graph_datasets.testing import generate_random_graph
from visual_graph_datasets.data import VisualGraphDatasetWriter
from visual_graph_datasets.processing.generic import GenericProcessing

from .util import ARTIFACTS_PATH


def test_generate_random_graph():
    """
    "generate_random_graph" is supposed to generate a random graph dict with a view pre-determinable 
    characteristics. Simply tests whether that function can be called with default parameters without 
    causing an exception.
    """
    for _ in range(10):
        graph = generate_random_graph()
        assert isinstance(graph, dict)
        tv.assert_graph_dict(graph)


def test_generic_processing_basically_works():
    """
    The "GenericProcessing" class implements the "BaseProcessing" interface and should be able to handle 
    generic graphs whose domain representation is simply their json encoded graph dict and whose visualization 
    is simple uniform nodes without any additional indications of node or edge attributes. 
    """
    # At first, we generate a random graph with which we will do the exemplary testing.
    num_nodes = 10
    num_edges = 2 * (num_nodes - 1)
    num_node_features = 7
    num_edge_features = 5
    graph = generate_random_graph(
        num_nodes=num_nodes,
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
    )
    
    # Then we need to setup the processing object itself.
    processing = GenericProcessing()
    
    value = processing.unprocess(graph)
    assert isinstance(value, str)
    assert value
    
    graph_ = processing.process(value)
    assert isinstance(graph_, dict)
    tv.assert_graph_dict(graph_)
    assert graph_['node_attributes'].shape == (num_nodes, num_node_features)
    assert graph_['edge_attributes'].shape == (num_edges, num_edge_features)
    
    fig, node_positions = processing.visualize_as_figure(value, graph_)
    assert isinstance(fig, plt.Figure)
    assert node_positions.shape == (num_nodes, 2)
    
    fig_path = os.path.join(ARTIFACTS_PATH, 'generic_processing_basically_works.png')
    fig.savefig(fig_path)
    
    
def test_generic_processing_create():
    """
    The "GenericProcessing" class implements the "BaseProcessing" interface and should be able to handle
    generic graphs whose domain representation is simply their json encoded graph dict and whose visualization
    is simple uniform nodes without any additional indications of node or edge attributes.
    """
    # At first, we generate a random graph with which we will do the exemplary testing.
    num_nodes = 10
    num_edges = 2 * (num_nodes - 1)
    num_node_features = 7
    num_edge_features = 5
    graph = generate_random_graph(
        num_nodes=num_nodes,
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
    )
    
    # Then we need to setup the processing object itself.
    processing = GenericProcessing()
    
    with tempfile.TemporaryDirectory() as path:
        
        assert os.listdir(path) == []
        
        # For the writing to the disk we also need to setup the writer class
        writer = VisualGraphDatasetWriter(
            path=path,
        )
        processing.create(
            index=0,
            value=None,
            graph=graph,
            writer=writer,
        )
        
        assert len(os.listdir(path)) != 0
        assert len(os.listdir(path)) == 2
        