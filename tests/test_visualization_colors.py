import os
import json
import random

import numpy as np
import matplotlib.pyplot as plt

import visual_graph_datasets.typing as tc
from visual_graph_datasets.visualization.base import create_frameless_figure
from visual_graph_datasets.visualization.base import layout_node_positions
from visual_graph_datasets.visualization.colors import visualize_grayscale_graph
from visual_graph_datasets.visualization.colors import visualize_color_graph

from.util import ASSETS_PATH, ARTIFACTS_PATH


def test_visualize_grayscale_graph_basically_works():
    # ~ loading the graph
    json_path = os.path.join(ASSETS_PATH, 'g_grayscale.json')
    with open(json_path) as file:
        g = json.load(file)
        tc.assert_graph_dict(g)

    # ~ creating the figure
    fig, ax = create_frameless_figure(1000, 1000)
    ax.patch.set_facecolor('white')
    node_positions = layout_node_positions(g)
    visualize_grayscale_graph(ax, g, node_positions)

    # ~ saving the figure
    vis_path = os.path.join(ARTIFACTS_PATH, 'grayscale_graph_visualization.png')
    fig.savefig(vis_path)


def test_visualize_color_graph_basically_works():
    json_path = os.path.join(ASSETS_PATH, 'g_color.json')
    with open(json_path) as file:
        g = json.load(file)
        tc.assert_graph_dict(g)

    # ~ creating the figure
    fig, ax = create_frameless_figure(1000, 1000)
    ax.patch.set_facecolor('white')
    node_positions = layout_node_positions(g)
    visualize_color_graph(ax, g, node_positions)

    # ~ saving the figure
    vis_path = os.path.join(ARTIFACTS_PATH, 'color_graph_visualization.png')
    fig.savefig(vis_path)