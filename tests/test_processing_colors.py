import os
import json
import pytest
import tempfile
import typing as t

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from imageio.v2 import imread
from mpl_toolkits.mplot3d import Axes3D

from visual_graph_datasets.typing import assert_graph_dict
from visual_graph_datasets.data import VisualGraphDatasetReader
from visual_graph_datasets.graph import graph_add_edge
from visual_graph_datasets.graph import nx_from_graph
from visual_graph_datasets.visualization.base import layout_node_positions
from visual_graph_datasets.visualization.base import draw_image
from visual_graph_datasets.visualization.colors import visualize_color_graph, colors_layout
from visual_graph_datasets.processing.colors import ColorProcessing

from .util import ARTIFACTS_PATH, ASSETS_PATH
from .util import load_mock_color_graph


def test_color_processing_visualize_as_figure_3d():
    """
    In the new version it should be possible to visualize 3d color graphs as well by using the 
    "dim" parameter of the visualize_as_figure method of the ColorProcessing class.
    """
    width = 1000
    height = 1000
    
    processing = ColorProcessing()
    graph = load_mock_color_graph()
    
    def visualize_color_graph_3d(ax: Axes3D, 
                                 g: dict, 
                                 node_positions: np.ndarray
                                 ) -> None:
        for attr, pos in zip(g['node_attributes'], node_positions):
            ax.scatter(
                pos[0], pos[1], pos[2],
                color=attr,
                marker='o',
                s=100,
            )
    
    fig, image_node_positions = processing.visualize_as_figure(
        value=None,
        graph=graph,  # based just on the graph dict
        width=width,
        height=height,
        node_positions=None,   # all node positions will be determined internally
        # This parameter can be used to specify the dimensionality of the visualization.
        # default is 2 dimensions. When set to 3, the corresponding axis will be 3 dimensional
        # and the node positions will be 3 dimensional as well.
        dim=3,    
        # It is possible to supply a custom visualization function that will be called to actually 
        # draw the color nodes onto the Axes object. This function receices the Axes object, the 
        # graph dict and the node_positions as arguments.
        visualize_func=visualize_color_graph_3d,
    )
    
    # For visual verification that it works.
    fig_path = os.path.join(ARTIFACTS_PATH, 'test_color_processing_visualize_as_figure_3d.png')
    fig.savefig(fig_path)
    
    # A very simple test we can do for the node positions which supposedly in the pixel-space of the 
    # generated image is that none of them can be negative
    for pos in image_node_positions:
        assert (pos >= 0).all()
        assert (0 <= pos[0] <= width).all()
        assert (0 <= pos[1] <= height).all()
        
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
    draw_image(ax, fig_path)
    for x, y in image_node_positions:
        ax.text(x, y, s=f'({int(x)},{int(y)})', fontsize=14)
        
    fig_path = os.path.join(ARTIFACTS_PATH, 'test_color_processing_visualize_as_figure_3d_loaded.pdf')
    fig.savefig(fig_path)


def test_bug_layout_with_partial_node_positions_real_graph():
    """
    27.03.24 - There was previously a bug where passing a partially filled "node_positions" array to the 
    "colors_layout" function would cause an exception with the specific graph configuration given in this 
    test.
    """
    
    # This is the json representation of a specific graph for which the bug previously appeared 
    # The bug did not appear for all graphs but rather only specific configurations.
    string = '{"index": 1, "name": "test", "value": 0, "target": [-0.026701535786801955], "graph": {"node_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22], "node_positions": [null, null, null, [68, 0], [26, 0], null, null, null, null, null, null, null, null, [-131, 229], null, [115, -18], null, null, [60, 86], [102, 178], null, null, null], "node_attributes": [[0.0, 0.0, 1.0], [0.800000011920929, 0.800000011920929, 0.800000011920929], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.800000011920929, 0.800000011920929, 0.800000011920929], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.800000011920929, 0.800000011920929, 0.800000011920929], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.800000011920929, 0.800000011920929, 0.800000011920929], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], "edge_indices": [[3, 19], [3, 15], [19, 3], [19, 15], [15, 19], [15, 3], [4, 18], [4, 13], [18, 4], [18, 13], [13, 18], [13, 4], [17, 13], [13, 17], [16, 19], [19, 16], [11, 13], [13, 11], [2, 19], [19, 2], [1, 11], [11, 1], [6, 16], [16, 6], [14, 2], [2, 14], [21, 18], [18, 21], [20, 17], [17, 20], [0, 21], [21, 0], [12, 3], [3, 12], [10, 15], [15, 10], [8, 19], [19, 8], [5, 1], [1, 5], [7, 2], [2, 7], [22, 15], [15, 22], [9, 18], [18, 9], [2, 13], [13, 2], [12, 16], [16, 12], [9, 22], [22, 9], [17, 10], [10, 17], [14, 8], [8, 14], [22, 19], [19, 22], [4, 17], [17, 4], [10, 20], [20, 10], [14, 5], [5, 14]], "edge_attributes": [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]], "seed_graph_indices": [-1, -1, -1, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 0, -1, -1, 1, 0, -1, -1, -1], "node_importances_1": [[0.0], [0.0], [0.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [0.0], [1.0], [0.0], [0.0], [1.0], [1.0], [0.0], [0.0], [0.0]], "node_importances_2": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], "graph_labels": [-0.026701535786801955], "graph_labels_raw": [0]}}'
    metadata = json.loads(string)
    graph = metadata['graph']
    for key, value in graph.items():
        if key != 'node_positions':
            graph[key] = np.array(value)
    node_positions = graph['node_positions']

    # In this section the bug would cause an exception.
    g = nx_from_graph(graph)
    node_positions = colors_layout(g, node_positions=graph['node_positions'], k=50)
    node_positions = np.array([node_positions[i].tolist() for i in graph['node_indices']])
    graph['node_positions'] = node_positions

    processing = ColorProcessing()
    fig, _ = processing.visualize_as_figure(
        value=None,
        graph=graph,
        node_positions=node_positions,
    )
    
    image_path = os.path.join(ARTIFACTS_PATH, 'color_layout_partial_positions_real_graph.pdf')
    fig.savefig(image_path)


def test_layout_with_partial_node_positions():
    
    graph = {
        'node_indices': [0, 1, 2, 3, 4, 5],
        'node_attributes': [
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
        ],
        'edge_indices': [
            [0, 1], [1, 0],
            [1, 2], [2, 1],
            [0, 2], [2, 0],
            [2, 3], [3, 2],
            [3, 4], [4, 3],
            [3, 5], [5, 3],
        ],
        'edge_attributes': [
            [1], [1],
            [1], [1],
            [1], [1],
            [1], [1],
            [1], [1],
            [1], [1],
        ],
        'node_positions': [
            None,
            [0, 2],
            [6, 1],
            [0, 0],
            None,
            None,
        ]
    }
    
    g = nx_from_graph(graph)
    node_positions = colors_layout(g, node_positions=graph['node_positions'])
    node_positions = np.array([node_positions[i] for i in graph['node_indices']])
    assert len(node_positions) == len(graph['node_indices'])
    assert np.allclose(node_positions[3], [0, 0])
    assert np.allclose(node_positions[1], [0, 2])
    assert np.allclose(node_positions[2], [6, 1])
    
    print(node_positions)
    processing = ColorProcessing()
    fig, _ = processing.visualize_as_figure(
        value=None,
        graph=graph,
        node_positions=node_positions,
    )
    
    image_path = os.path.join(ARTIFACTS_PATH, 'color_layout_partial_positions.pdf')
    fig.savefig(image_path)
    


@pytest.mark.parametrize('value', [
    'RRR(G)(YGG)RR',
    'RR(G)(YG)',
])
def test_color_processing_visualize_with_external_node_positions(value: str):
    """
    The visualize_as_figure method of the color processing instance should be able to visualize
    a graph with external node positions that are not calculated by the processing instance itself.
    """    
    processing = ColorProcessing()
    graph = processing.process(value)
    # generating random 2D noe positions for each node in the graph:
    node_positions = np.random.rand(len(graph['node_indices']), 2)
    
    fig, _ = processing.visualize_as_figure(
        value=None,
        graph=graph,
        node_positions=node_positions
    )
    
    image_path = os.path.join(ARTIFACTS_PATH, 'color_processing_visualize_with_external_node_positions.pdf')
    fig.savefig(image_path)


def test_bug_cogiles_encoding_edge_duplication():
    # This is a weird bug because it also only occurs for very specific graphs and certain edge 
    # indice combinations such as the example used in this test case!
    index_1, index_2 = 0, 2
    cogiles = 'RRRRR'
    processing = ColorProcessing()
    
    graph = processing.process(cogiles)
    num_edges = len(graph['edge_indices'])
    assert num_edges == 8
    
    graph = graph_add_edge(
        graph=graph,
        node_index_1=0,
        node_index_2=2,
        directed=False
    )
    # This always still works because we are literally just adding the two edge entries to the 
    # list
    assert len(graph['edge_indices']) == num_edges + 2
    
    # We do the process and un-process chain, which should technically result in the very same graph, but 
    # with the bug there are actually two edges too many!
    # I have traced the problem down to the cogiles encoding in that case.
    _cogiles = processing.unprocess(graph)
    _graph = processing.process(_cogiles)
    print(cogiles, _cogiles)
    assert len(_graph['edge_indices']) == num_edges + 2
    


def test_bug_color_processing_edge_attributes_wrong():
    """
    12.06.23 - There was a bug where the COGILES decoding scheme produced graph dicts where the 
    edge_attributes array had the wrong datatype and the wrong shape.
    """
    processing = ColorProcessing()
    cogiles = 'RRR(G)(YGG)RR'
    
    graph = processing.process(cogiles)
    assert 'edge_attributes' in graph
    assert isinstance(graph['edge_attributes'], np.ndarray)
    # These test the bug
    assert len(graph['edge_attributes'].shape) == 2
    assert graph['edge_attributes'].shape[1] == 1
    assert graph['edge_attributes'].dtype == np.float64


def test_bug_color_processing_create_without_name():
    """
    12.06.23 - There was a bug where the ColorProcessing.create method did not properly create the 
    metadata fields "value" and "name" which caused problems down the line as they are expected for a 
    valid visual graph element.
    """
    processing = ColorProcessing()
    cogiles = 'RRR(G)(YGG)RR'
    
    with tempfile.TemporaryDirectory() as path:
        assert len(os.listdir(path)) == 0
        processing.create(
            value=cogiles,
            index=0,
            output_path=path
        )

        data = VisualGraphDatasetReader.read_element(path, '0')
        assert 'image_path' in data
        assert os.path.exists(data['image_path'])
        # The following ones check the bug
        assert 'name' in data['metadata'] 
        assert data['metadata']['name'] == cogiles
        assert 'value' in data['metadata']
        assert data['metadata']['value'] == cogiles
        

def test_color_processing_canonical_representation_with_extract():
    """
    10.06.23 - Using the "extract" method of the color processing instance with a complete node mask 
    it should be possible to simply convert an existing graph into it's "canonical" representation, which 
    should not change the underlying graph at all. That is being tested here.
    """
    processing = ColorProcessing()
    graph_original = {
        'node_indices': [0, 1, 2, 3, 4],
        'node_attributes': [
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [1, 0, 0],
            [1, 0, 1],
        ],
        'edge_indices': [
            [0, 1], [1, 0],
            [1, 2], [2, 1],
            [3, 4], [4, 3],
            [2, 3], [3, 2],
            [1, 4], [4, 1],
            [2, 4], [4, 2],
        ],
        'edge_attributes': [
            [1], [1],
            [1], [1],
            [1], [1],
            [1], [1],
            [1], [1],
            [1], [1],
        ]
    }
    graph_original['node_positions'] = layout_node_positions(graph_original)

    # By using a mask that identifies ALL the nodes of the graph we can use the "extract" method 
    # to essentially just convert the graph dict into it's canonical graph representation as defined 
    # by the COGILES encoding scheme.
    cogiles, graph_canon = processing.extract(
        graph_original,
        mask=np.ones_like(graph_original['node_indices'])
    )
    assert_graph_dict(graph_canon)
    assert len(graph_canon['node_indices']) == len(graph_original['node_indices'])
    assert 'node_positions' in graph_canon

    # VISUAL VERIFICATION
    # What we would like to see here is that the visualization of the canonical and the original 
    # graph representation are exactly the same as it should be, because the representation in which 
    # they are in should not make a difference for the actual graph that they both represent.
    fig, (ax_org, ax_canon) = plt.subplots(ncols=2, nrows=1, figsize=(12, 6))

    ax_org.set_title('original graph')
    visualize_color_graph(
        ax=ax_org,
        g=graph_original,
        node_positions=graph_original['node_positions']
    )
    for i, (x, y) in zip(graph_original['node_indices'], graph_original['node_positions']):
        ax_org.text(x, y, s=f'{i}', fontsize=14)

    ax_canon.set_title('canonized graph')
    visualize_color_graph(
        ax=ax_canon,
        g=graph_canon,
        node_positions=graph_canon['node_positions'],
    )
    for i, (x, y) in zip(graph_canon['node_indices'], graph_canon['node_positions']):
        ax_canon.text(x, y, s=f'{i}:{graph_canon["node_indices_original"][i]}', fontsize=14)

    fig_path = os.path.join(ARTIFACTS_PATH, 'color_processing_canonical_representation_with_extract.pdf')
    fig.savefig(fig_path)


def test_color_processing_extract_basically_works():
    """
    10.06.23 - The basic functionality of the "extract" method of the color processing instance 
    is that given a node mask it should extract a new domain repr. and graph dict for just the 
    the specified sub graph structure.
    """
    processing = ColorProcessing()
    cogiles = 'HHY-3(RR-3)(R)H-1HH-1-2H-1HH-2'
    graph = processing.process(cogiles)
    graph['node_positions'] = layout_node_positions(graph)
    mask = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

    value, sub_graph = processing.extract(graph, mask)
    sub_graph['node_positions'] = layout_node_positions(sub_graph)

    assert_graph_dict(sub_graph)
    assert len(sub_graph['node_indices']) == 4
    assert len(sub_graph['edge_indices']) == 8

    # Visual Test
    # What we want to see here is that the subgraph is actually the one that is specified in the 
    # main graph by the graph mask and also for the node index annotations we want to see that the 
    # original node indices are correctly attributed to the subgraph as well.
    fig, (ax, ax_sub) = plt.subplots(ncols=2, nrows=1, figsize=(12, 6))
    visualize_color_graph(
        ax=ax,
        g=graph,
        node_positions=graph['node_positions']
    )
    for i, (x, y) in zip(graph['node_indices'], graph['node_positions']):
        ax.text(x, y, s=f'{i}', fontsize=14)

    visualize_color_graph(
        ax=ax_sub,
        g=sub_graph,
        node_positions=sub_graph['node_positions'],
    )
    for i, (x, y) in zip(sub_graph['node_indices'], sub_graph['node_positions']):
        ax_sub.text(x, y, s=f'{i}:{sub_graph["node_indices_original"][i]}', fontsize=14)

    fig_path = os.path.join(ARTIFACTS_PATH, 'color_processing_extract_basically_works.pdf')
    fig.savefig(fig_path)
