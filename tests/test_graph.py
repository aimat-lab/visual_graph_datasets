import os
import pytest
import numpy as np
import matplotlib.pyplot as plt

from visual_graph_datasets.graph import copy_graph_dict
from visual_graph_datasets.graph import graph_expand_mask
from visual_graph_datasets.graph import extract_subgraph
from visual_graph_datasets.graph import graph_find_connected_regions
from visual_graph_datasets.graph import graph_remove_node
from visual_graph_datasets.processing.colors import ColorProcessing

from .util import ARTIFACTS_PATH


# This is the test graph that will be used for all the operations in this unittest module. 
# this is a COLOR graph that was handcrafted for the testing.
TEST_GRAPH = {
    'node_indices': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    'node_attributes': np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 1, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
    ]),
    'edge_indices': np.array([
        [0, 1], [1, 0],
        [1, 2], [2, 1],
        [1, 3], [3, 1],
        [2, 3], [3, 2],
        [3, 4], [4, 3],
        [4, 5], [5, 4],
        [5, 6], [6, 5],
        [5, 7], [7, 5],
        [5, 8], [8, 5],
        [6, 8], [8, 6],
        [7, 8], [8, 7],
        [8, 9], [9, 8],
    ]),
    'edge_attributes': np.array([
        [1], [1],
        [1], [1],
        [1], [1],
        [1], [1],
        [1], [1],
        [1], [1],
        [1], [1],
        [1], [1],
        [1], [1],
        [1], [1],
        [1], [1],
        [1], [1],
    ]),
}


def test_visualize_test_graph():
    """
    This method simply visualizes the TEST_GRAPH that is used in this module to test the various graph 
    operations.
    """
    processing = ColorProcessing()
    fig, node_positions = processing.visualize_as_figure(
        value=None,
        graph=TEST_GRAPH,
        width=1000,
        height=1000,
    )
    ax: plt.Axes = fig.axes[0]
    for i, (x, y) in enumerate(node_positions):
        ax.text(x, y, f'{i}')
    
    fig_path = os.path.join(ARTIFACTS_PATH, 'test_visualize_test_graph.pdf')
    fig.savefig(fig_path)
    
    
def test_graph_expand_mask():
    """
    the method graph_expand_mask is a function that will expand a binary mask to all the adjacent nodes
    """
    
    graph = copy_graph_dict(TEST_GRAPH)
    # This mask covers the first triangle substructure in the graph and we know that if we expand this 
    # it will have to cover indices 0 and 4 as well which then means that there are 5 nodes in total 
    # in the one hop expanded mask.
    mask = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    assert np.sum(mask) == 3

    mask_expanded_1 = graph_expand_mask(
        graph=graph,
        node_mask=mask.copy(),
        num_hops=1,
    )
    # For the one-hop expansion we know that there have to be 5 nodes masked on that particular graph
    assert int(np.sum(mask_expanded_1)) == 5
    
    mask_expanded_2 = graph_expand_mask(
        graph=graph,
        node_mask=mask.copy(),
        num_hops=2,
    )
    # For the two-hop expanstion we know that there have to be 6 nodes masked out
    assert int(np.sum(mask_expanded_2)) == 6
    
    
def test_extract_subgraph():
    """
    The "extract_subgraph" method is supposed to return a derived graph dict if given an original 
    graph dict and a node mask that determines which nodes also to include in the derived structure.
    
    Briefly tests the functionality based on a single example case.
    """
    processing = ColorProcessing()
    
    graph = copy_graph_dict(TEST_GRAPH)
    # This mask defines a triangle sub structure in the test graph which should then 
    # subsequently be masked out.
    mask = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    
    # Besides the actual sub graph, this function will also return the node and edge 
    # index mapping dicts.
    sub_graph, node_index_map, edge_index_map = extract_subgraph(
        graph=graph,
        node_mask=mask,
    )
    print(sub_graph)
    # Now for this particular sub graph we can compare with what we know it must look
    # like
    # The triangle sub graph consists of 3 nodes
    assert len(sub_graph['node_indices']) == 3
    # Since the edges are directed, there should be a 6 edges between those nodes in general
    assert len(sub_graph['edge_indices']) == 6
    
    # Generally, the extraction should also have copied the node_attributes and edge_attributes
    # properties of the graph dict correctly.
    assert 'node_attributes' in sub_graph
    assert len(sub_graph['node_attributes']) == 3
    
    assert 'edge_attributes' in sub_graph
    assert len(sub_graph['edge_attributes']) == 6
    
    # As another heuristic check we can visualize the sub graph to confirm that the function 
    # is working correctly.
    fig, node_positions = processing.visualize_as_figure(
        value=None,
        graph=sub_graph,
        width=1000,
        height=1000,
    )
    fig_path = os.path.join(ARTIFACTS_PATH, 'test_extract_subgraph.pdf')
    fig.savefig(fig_path)
    
    
def test_graph_find_connected_regions():
    """
    The "graph_find_connected_regions" function given a node mask will find out which of the nodes 
    of that mask are connected into regions and return a region mask where that is defined.
    """
    graph = copy_graph_dict(TEST_GRAPH)
    # On the given test graph, this mask consists of two separate substructures 
    mask = np.array([0, 1, 1, 1, 1, 0, 0, 1, 1, 0])
    
    region_mask = graph_find_connected_regions(
        graph=graph,
        node_mask=mask,
    )
    # In the given test graph this region mask should evaluate to exactly two regions (indices 0 and 1)
    # as well as the -1 special indices which indicate that those elements of the graph were not part of 
    # the original mask to be evaluated.
    assert region_mask.shape == mask.shape
    assert set(region_mask) == set([-1, 0, 1])
    
    
def test_graph_find_connected_regions_empty_mask():
    """
    tests the edge case when an empty mask is given to the "graph_find_connected_regions" function
    """
    graph = copy_graph_dict(TEST_GRAPH)
    # On the given test graph, this mask consists of two separate substructures 
    mask = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    region_mask = graph_find_connected_regions(
        graph=graph,
        node_mask=mask,
    )
    assert region_mask.shape == mask.shape
    assert set(region_mask) == set([-1])


def test_graph_remove_node():
    
    graph = copy_graph_dict(TEST_GRAPH)
    # For the original graph we know that it has 10 nodes
    assert len(graph['node_indices']) == 10
    
    node_index = 0
    mod_graph = graph_remove_node(
        graph=graph,
        node_index=node_index,
    )
    assert len(mod_graph['node_indices']) == 9