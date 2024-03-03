import os
import tempfile
import pytest

import numpy as np
import matplotlib.pyplot as plt
import visual_graph_datasets.typing as tv
from imageio.v2 import imread
from visual_graph_datasets.testing import generate_random_graph
from visual_graph_datasets.util import dynamic_import
from visual_graph_datasets.data import VisualGraphDatasetWriter
from visual_graph_datasets.data import load_visual_graph_element
from visual_graph_datasets.visualization.base import draw_image
from visual_graph_datasets.processing.base import create_processing_module
from visual_graph_datasets.processing.base import ProcessingBase
from visual_graph_datasets.processing.generic import GenericProcessing

from .util import ARTIFACTS_PATH


# This is only a mock class, which we need to be defined in the upper most level of the module so 
# that the "create_processing_module" function can find it for one of the unittests.
class CustomGenericProcessing(GenericProcessing):
    pass



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

        # Now we load the element from the disk and check this persistent representation
        data = load_visual_graph_element(path=path, name='0000000')
        
        # ~ Checking the visualization and the node positions
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
        node_positions = data['metadata']['graph']['node_positions']
        print(node_positions)
        draw_image(ax, data['image_path'])
        
        # Then we also want to plot the index of each node at that node location so that in the 
        # visualization image we can then see whether or not the position is correct.
        for i, (x, y) in enumerate(node_positions):
            plt.text(x, y, str(i), fontsize=12, ha='center')
        
        fig_path = os.path.join(ARTIFACTS_PATH, 'test_generic_visualization.png')
        fig.savefig(fig_path)

        
def test_create_processing_module():
    """
    It should be possible to use the "create_processing_module" function to create a standalone module 
    from a processing instance. This module should be able to be imported and used as a standalone module.
    This test checks if that is the case for the GenericProcessing class.
    """
    processing = CustomGenericProcessing()

    with tempfile.TemporaryDirectory() as path:
        
        # This function should dynamically create code to turn the processing instance into an 
        # independent module. This module should be able to be imported and used as a standalone
        # module.
        string = create_processing_module(processing)
        # The least we can assume is that this string is not empty
        assert string != ''
        # another easy check is if the name of the class is contained in the code, which it should
        assert processing.__class__.__name__ in string
        
        module_path = os.path.join(path, 'processing.py')
        with open(module_path, mode='w') as file:
            file.write(string)
            
        assert os.path.exists(module_path)

        # Now we can attempt to import the module and actually use the processing class from that 
        # imported version
        module = dynamic_import(module_path)
        processing = module.processing
        assert isinstance(processing, ProcessingBase)