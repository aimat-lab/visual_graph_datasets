import os
import numpy as np
import matplotlib.pyplot as plt

from visual_graph_datasets.typing import assert_graph_dict
from visual_graph_datasets.visualization.base import layout_node_positions
from visual_graph_datasets.visualization.colors import visualize_color_graph, colors_layout
from visual_graph_datasets.processing.colors import ColorProcessing

from .util import ARTIFACTS_PATH, ASSETS_PATH


def test_color_processing_canonical_representation_with_extract():
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

    cogiles, graph_canon = processing.extract(
        graph_original,
        mask=np.ones_like(graph_original['node_indices'])
    )
    assert_graph_dict(graph_canon)
    assert len(graph_canon['node_indices']) == len(graph_original['node_indices'])
    assert 'node_positions' in graph_canon

    # VISUAL VERIFICATION
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
