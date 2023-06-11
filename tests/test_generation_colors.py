"""
Unittests for ``generation.colors``
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from visual_graph_datasets.typing import assert_graph_dict
from visual_graph_datasets.visualization.base import layout_node_positions
from visual_graph_datasets.visualization.base import create_frameless_figure
from visual_graph_datasets.visualization.colors import visualize_color_graph
from visual_graph_datasets.generation.colors import RED, GREEN, BLUE
from visual_graph_datasets.generation.colors import make_star_motif
from visual_graph_datasets.generation.colors import make_grid_motif
from visual_graph_datasets.generation.colors import cogiles_grammar
from visual_graph_datasets.generation.colors import CogilesVisitor
from visual_graph_datasets.generation.colors import graph_to_cogiles, graph_from_cogiles
from visual_graph_datasets.data import extract_graph_mask

from .util import ARTIFACTS_PATH


def test_bug_cogiles_encoding_node_duplication():
    """
    06.06.23 - I have discovered an example where the cogiles encoding fails. In this specific example
    the encoding process produces one node too many.
    """
    cogiles_1 = "Y-3(RR-3)(R)"
    graph_1 = graph_from_cogiles(cogiles_1)

    cogiles_2 = graph_to_cogiles(graph_1)
    graph_2 = graph_from_cogiles(cogiles_2)

    print(cogiles_1, cogiles_2)
    # this assertion would fail with the bug, but should theoretically be true if everything worked as
    # it should.
    assert len(graph_1['node_indices']) == len(graph_2['node_indices'])


def test_extract_graph_mask_color_graph_works():
    cogiles = "R-1R(GY-1)BRRR(R)RB"
    graph = graph_from_cogiles(cogiles)
    # A mask with 6 nodes from the original graph which are most importantly not connected!
    mask = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0])

    sub_graph = extract_graph_mask(graph, mask)
    assert_graph_dict(sub_graph)
    assert len(sub_graph['node_indices']) == 6

    # Here we want to visually confirm that the extraction process in fact works
    fig, (ax, ax_sub) = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
    visualize_color_graph(
        ax=ax,
        g=graph,
        node_positions=layout_node_positions(graph)
    )
    visualize_color_graph(
        ax=ax_sub,
        g=sub_graph,
        node_positions=layout_node_positions(sub_graph)
    )
    fig_path = os.path.join(ARTIFACTS_PATH, 'extract_graph_mask_color_graph')
    fig.savefig(fig_path)


def test_cogiles_grammar_basically_works():
    """
    05.06.23 - I created a new type of string representation for colored graphs called COGILES. This
    essentially works by creating a new regex grammar with parsimonious. The goal is to be able to convert
    a string into a graph and that graph back into a corresponding string and have that be a consistent
    graph. This is what this function tests.
    """
    # First we parse a COGILES string here
    tree = cogiles_grammar.parse('RRGR-1(GGH)Y(GGGHHR(BBR-2)YYC-1)RR-2.RR(YY(CC)Y)G')
    visitor = CogilesVisitor()
    visitor.visit(tree)
    graph = visitor.process()
    # We roughly know that the result has to be a valid GraphDict and the graph has 30 nodes
    assert_graph_dict(graph)
    assert len(graph['node_indices']) == 30

    # Then we plot this to visually confirm that this is in fact the correct graph. One important visual
    # detail is that it is actually two disconnected graphs because of the colon break symbol.
    fig, ax = create_frameless_figure(width=500, height=500)
    visualize_color_graph(
        ax=ax,
        g=graph,
        node_positions=layout_node_positions(graph)
    )
    path = os.path.join(ARTIFACTS_PATH, 'cogiles_grammer_1.png')
    fig.savefig(path)

    # In this step we encode the graph again into a cogiles string which should not cause any errors and
    # then we decode that string again back into a graph dict and plot it again!
    # The two visualizations should be the same graph.
    cogiles_encoded = graph_to_cogiles(graph)
    assert isinstance(cogiles_encoded, str)
    assert cogiles_encoded != ''

    graph2 = graph_from_cogiles(cogiles_encoded)
    fig, ax = create_frameless_figure(width=500, height=500)
    visualize_color_graph(
        ax=ax,
        g=graph2,
        node_positions=layout_node_positions(graph2)
    )
    path = os.path.join(ARTIFACTS_PATH, 'cogiles_grammer_2.png')
    fig.savefig(path)


def test_make_star_motif():
    """
    21.03.2023 - the function ``make_star_motif`` should create a color graph motif which is essentially a
    star consisting of a center node of one color and several outer nodes of another color.
    """
    g = make_star_motif(
        inner_color=GREEN,
        outer_color=RED,
        k=3
    )
    # A function which tests for valid graph dict in-depth
    assert_graph_dict(g)

    # We visualize the graph as a testing artifact
    fig, ax = create_frameless_figure(width=500, height=500)
    visualize_color_graph(
        ax=ax,
        g=g,
        node_positions=layout_node_positions(g)
    )
    path = os.path.join(ARTIFACTS_PATH, 'star.png')
    fig.savefig(path)
