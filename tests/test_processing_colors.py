import os
import tempfile
import typing as t

import numpy as np
import matplotlib.pyplot as plt

from visual_graph_datasets.typing import assert_graph_dict
from visual_graph_datasets.data import VisualGraphDatasetReader
from visual_graph_datasets.graph import graph_add_edge
from visual_graph_datasets.visualization.base import layout_node_positions
from visual_graph_datasets.visualization.colors import visualize_color_graph, colors_layout
from visual_graph_datasets.processing.colors import ColorProcessing

from .util import ARTIFACTS_PATH, ASSETS_PATH


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
