"""
Unittests for ``generation.colors``
"""
import os

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

from .util import ARTIFACTS_PATH


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


def test_make_star_motif():
    """
    21.03.2023 - the function ``make_star_motif`` should create a color graph motif which is essentially a
    star consisting of a center node of one color and several outer nodes of another color.
    """
    g = make_grid_motif(
        color_1=GREEN,
        color_2=RED,
        m=3,
        n=3
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
    path = os.path.join(ARTIFACTS_PATH, 'grid.png')
    fig.savefig(path)


def test_cogiles_grammar():

    tree = cogiles_grammar.parse('RRGR-1(GGH-1)(GGGHHR(BBR-2)YYC-1)RR-2')
    visitor = CogilesVisitor()
    visitor.visit(tree)
    graph = visitor.process()

    print(graph)

    fig, ax = create_frameless_figure(width=500, height=500)
    visualize_color_graph(
        ax=ax,
        g=graph,
        node_positions=layout_node_positions(graph)
    )
    path = os.path.join(ARTIFACTS_PATH, 'cogiles_grammer.png')
    fig.savefig(path)

    graph2 = graph_from_cogiles(graph_to_cogiles(graph))
    fig, ax = create_frameless_figure(width=500, height=500)
    visualize_color_graph(
        ax=ax,
        g=graph2,
        node_positions=layout_node_positions(graph)
    )
    path = os.path.join(ARTIFACTS_PATH, 'cogiles_grammer_2.png')
    fig.savefig(path)