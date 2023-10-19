import os
import json
import random

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from imageio.v2 import imread

import visual_graph_datasets.typing as tc
from visual_graph_datasets.data import load_visual_graph_dataset
from visual_graph_datasets.data import load_visual_graph_element
from visual_graph_datasets.visualization.base import draw_image
from visual_graph_datasets.visualization.importances import plot_node_importances_border
from visual_graph_datasets.visualization.importances import plot_edge_importances_border
from visual_graph_datasets.visualization.importances import plot_node_importances_background
from visual_graph_datasets.visualization.importances import plot_edge_importances_background
from visual_graph_datasets.visualization.importances import create_importances_pdf

from .util import ASSETS_PATH, ARTIFACTS_PATH

mpl.use('TkAgg')


def test_plot_importances_background_basically_works():
    """
    The functions "plot_node_importances_background" and "plot_edge_importances_background" should be able to 
    visualize a graph importance mask on top of a given graph visualization image as background color circles 
    behind the corresponding nodes.
    """
    # We are going to test the importance visualization using the mock dataset
    dataset_path = os.path.join(ASSETS_PATH, 'mock')
    data = load_visual_graph_element(dataset_path, '0')
    graph = data['metadata']['graph']
    node_positions = np.array(graph['image_node_positions'])
    ni = np.random.random(size=graph['node_importances_2'].shape)
    ei = np.random.random(size=graph['edge_importances_2'].shape)

    fig, (ax_0, ax_1) = plt.subplots(ncols=2, nrows=1, figsize=(20, 10))

    draw_image(ax_0, data['image_path'])
    draw_image(ax_1, data['image_path'])

    plot_node_importances_background(ax_0, graph, node_positions, ni[:, 0])
    plot_edge_importances_background(ax_0, graph, node_positions, ei[:, 0])

    plot_node_importances_background(ax_1, graph, node_positions, ni[:, 1])
    plot_edge_importances_background(ax_1, graph, node_positions, ei[:, 1])

    fig_path = os.path.join(ARTIFACTS_PATH, 'test_plot_importances_background_basically_works.pdf')
    fig.savefig(fig_path)

def test_plot_importances_border_color_weight_works():
    """
    It should be possible to specify an additional parameter "weight" which should control the color of the 
    using a color map.
    """
    # We are going to test the importance visualization using the mock dataset
    dataset_path = os.path.join(ASSETS_PATH, 'mock')
    data = load_visual_graph_element(dataset_path, '0')
    graph = data['metadata']['graph']
    node_positions = np.array(graph['image_node_positions'])
    ni = np.random.random(size=graph['node_importances_2'].shape)
    ei = np.random.random(size=graph['edge_importances_2'].shape)

    fig, (ax_0, ax_1) = plt.subplots(ncols=2, nrows=1, figsize=(20, 10))

    draw_image(ax_0, data['image_path'])
    draw_image(ax_1, data['image_path'])

    plot_node_importances_border(ax_0, graph, node_positions, ni[:, 0], color_map='coolwarm', weight=1.0)
    plot_edge_importances_border(ax_0, graph, node_positions, ei[:, 0], color_map='coolwarm', weight=1.0)

    plot_node_importances_border(ax_1, graph, node_positions, ni[:, 1], color_map='coolwarm', weight=0.1)
    plot_edge_importances_border(ax_1, graph, node_positions, ei[:, 1], color_map='coolwarm', weight=0.1)

    fig_path = os.path.join(ARTIFACTS_PATH, 'test_plot_importances_border_color_weight_works.pdf')
    fig.savefig(fig_path)


def test_plot_importances_border_basically_works():
    """
    The functions "plot_node_importances_border" and "plot_edge_importances_border" should be able to 
    visualize a graph importance mask on top of a given graph visualization image as circles around the 
    corresponding nodes.
    """
    # We are going to test the importance visualization using the mock dataset
    dataset_path = os.path.join(ASSETS_PATH, 'mock')
    data = load_visual_graph_element(dataset_path, '0')
    graph = data['metadata']['graph']
    node_positions = np.array(graph['image_node_positions'])
    ni = np.random.random(size=graph['node_importances_2'].shape)
    ei = np.random.random(size=graph['edge_importances_2'].shape)

    fig, (ax_0, ax_1) = plt.subplots(ncols=2, nrows=1, figsize=(20, 10))

    draw_image(ax_0, data['image_path'])
    draw_image(ax_1, data['image_path'])

    plot_node_importances_border(ax_0, graph, node_positions, ni[:, 0])
    plot_edge_importances_border(ax_0, graph, node_positions, ei[:, 0])

    plot_node_importances_border(ax_1, graph, node_positions, ni[:, 1])
    plot_edge_importances_border(ax_1, graph, node_positions, ei[:, 1])

    fig_path = os.path.join(ARTIFACTS_PATH, 'test_plot_importances_border_basically_works.pdf')
    fig.savefig(fig_path)


def test_create_importances_pdf_basically_works():
    """
    The function "create_importances_pdf" can be used to create a multi page pdf which can display the 
    explanation masks for multiple elements (one per page) and multiple explanation channels and multiple 
    different explanation masks.
    """
    num_examples = 5

    # We are going to test the importance visualization using the mock dataset
    dataset_path = os.path.join(ASSETS_PATH, 'mock')
    _, index_data_map = load_visual_graph_dataset(dataset_path)
    example_indices = random.sample(index_data_map.keys(), k=num_examples)

    data_list = [data for index, data in index_data_map.items() if index in example_indices]

    pdf_path = os.path.join(ARTIFACTS_PATH, 'mock_importances.pdf')
    create_importances_pdf(
        graph_list=[data['metadata']['graph'] for data in data_list],
        image_path_list=[data['image_path'] for data in data_list],
        node_positions_list=[data['metadata']['graph']['image_node_positions'] for data in data_list],
        importances_map={
            'gt1': (
                [data['metadata']['graph']['node_importances_2'] for data in data_list],
                [data['metadata']['graph']['edge_importances_2'] for data in data_list],
            ),
            'gt2': (
                [data['metadata']['graph']['node_importances_2'] for data in data_list],
                [data['metadata']['graph']['edge_importances_2'] for data in data_list],
            ),
        },
        output_path=pdf_path,
    )