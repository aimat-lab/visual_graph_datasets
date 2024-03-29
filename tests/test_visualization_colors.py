import os
import json
import random

import numpy as np
import matplotlib.pyplot as plt

import visual_graph_datasets.typing as tc
from visual_graph_datasets.graph import nx_from_graph
from visual_graph_datasets.visualization.base import create_frameless_figure
from visual_graph_datasets.visualization.base import layout_node_positions
from visual_graph_datasets.visualization.colors import visualize_grayscale_graph
from visual_graph_datasets.visualization.colors import visualize_color_graph
from visual_graph_datasets.visualization.colors import colors_layout

from .util import ASSETS_PATH, ARTIFACTS_PATH
from .util import load_mock_color_graph


def test_colors_layout_3d():
    """
    The "visualize_color_graph" function is supposed to accept a nx.Graph representation as input and 
    create the corresponding node_positions using the networkx spring layouting algorithm. When 
    given the "dim=3" argument the node positions should be generated in 3D space.
    """
    # the layout method needs the graph as a nx.Graph instance
    graph = load_mock_color_graph()
    
    # The layout method needs the graph as a nx.Graph instance
    graph_nx = nx_from_graph(graph)
    node_positions = colors_layout(
        graph_nx,
        dim=3,
    )
    assert isinstance(node_positions, dict)
    
    # For the further processing we need the node_positions as a numpy array.
    node_positions = np.array(list(node_positions.values()))
    assert node_positions.shape == (len(graph['node_indices']), 3)
    
    # Now we create a new figure and plot the graph onto it.
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        node_positions[:, 0], node_positions[:, 1], node_positions[:, 2],
        color='black',
        marker='o',
    )
    
    fig_path = os.path.join(ARTIFACTS_PATH, 'test_colors_layout_3d.pdf')
    fig.savefig(fig_path)


def test_colors_layout_2d():
    """
    The "visualize_color_graph" function is supposed to accept a nx.Graph representation as input and 
    create the corresponding node positions using the networkx spring layouting algorithm. 
    visualize_color_graph_2d basically works for this.
    """
    graph = load_mock_color_graph()
    
    # The layout method needs the graph as a nx.Graph instance
    graph_nx = nx_from_graph(graph)
    node_positions = colors_layout(
        graph_nx,
    )
    assert isinstance(node_positions, dict)
    
    # For the further processing we need the node_positions as a numpy array.
    node_positions = np.array(list(node_positions.values()))
    assert node_positions.shape == (len(graph['node_indices']), 2)
    
    # Now we create a new figure and plot the graph onto it.
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    visualize_color_graph(ax, graph, node_positions)
    
    fig_path = os.path.join(ARTIFACTS_PATH, 'test_colors_layout_2d.pdf')
    fig.savefig(fig_path)


def test_visualize_color_graph_basically_works():
    """
    The "visualize_color_graph" function is supposed to plot a color graph, given as a graph dict, to 
    the given axis using the given node_positions.
    """
    json_path = os.path.join(ASSETS_PATH, 'g_color.json')
    g = load_mock_color_graph()

    # ~ creating the figure
    fig, ax = create_frameless_figure(1000, 1000)
    ax.patch.set_facecolor('white')
    node_positions = layout_node_positions(g)
    visualize_color_graph(ax, g, node_positions)

    # ~ saving the figure
    vis_path = os.path.join(ARTIFACTS_PATH, 'color_graph_visualization.png')
    fig.savefig(vis_path)


def test_visualize_grayscale_graph_basically_works():
    """
    The "visualize_grayscale_graph" function is supposed to plot a grayscale graph, given as a graph 
    dict, to the given axis using the given node_positions.
    """
    # ~ loading the graph
    json_path = os.path.join(ASSETS_PATH, 'g_grayscale.json')
    g = load_mock_color_graph()

    # ~ creating the figure
    fig, ax = create_frameless_figure(1000, 1000)
    ax.patch.set_facecolor('white')
    node_positions = layout_node_positions(g)
    visualize_grayscale_graph(ax, g, node_positions)

    # ~ saving the figure
    vis_path = os.path.join(ARTIFACTS_PATH, 'grayscale_graph_visualization.png')
    fig.savefig(vis_path)