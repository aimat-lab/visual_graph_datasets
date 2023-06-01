"""
Generates a mock visual graph dataset.

This mock dataset is not actually used for any meaningful model training but instead it is supposed to be
easy to understand and small in size, such that it can be used for testing purposes.

The idea of this simple mock dataset is to generate small graphs which only have a single node property
which is a float value which will represent the blackness of the node for visualization. Inside about
half of the graphs an explanatory motif will be embedded which is a triangle of completely black nodes,
which will explain a certain regression value.
"""
import os
import json
import random
import shutil
import typing as t

import pycomex
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pycomex.experiment import Experiment
from pycomex.util import Skippable

import visual_graph_datasets.typing as tc
from visual_graph_datasets.generation.graph import GraphGenerator
from visual_graph_datasets.data import NumericJsonEncoder
from visual_graph_datasets.visualization.base import create_frameless_figure
from visual_graph_datasets.visualization.base import layout_node_positions
from visual_graph_datasets.visualization.colors import visualize_grayscale_graph

SHORT_DESCRIPTION = 'Generates a mock visual graph dataset'

# == GENERATION PARAMETERS ==
DATASET_NAME: str = 'mock'
COMPRESS_ZIP: bool = True
IMAGE_WIDTH: int = 1000
IMAGE_HEIGHT: int = 1000

NUM_ELEMENTS: int = 100
TRAIN_RATIO: float = 0.8

SEED_GRAPH = {
    'node_indices': [0, 1, 2],
    'node_attributes': [[1.0], [1.0], [1.0]],
    'edge_indices': [
        [0, 1], [1, 0],
        [1, 2], [2, 1],
        [2, 0], [0, 2],
    ],
    'edge_attributes': [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]
}

# == EVALUATION PARAMETERS ==
LOG_STEP_EVAL = 10

# == EXPERIMENT PARAMETERS ==
DEBUG = True
BASE_PATH = os.getcwd()
NAMESPACE = 'results/generate_synthetic_dataset/mock'
with Skippable(), (e := Experiment(base_path=BASE_PATH, namespace=NAMESPACE, glob=globals())):
    e.info('creating mock dataset...')
    dataset_folder_path = os.path.join(e.path, DATASET_NAME)
    os.mkdir(dataset_folder_path)

    # -- GENERATING THE GRAPHS --
    dataset: t.List[tc.GraphDict] = []

    e.info('starting to generate graphs...')
    while len(dataset) != NUM_ELEMENTS:
        num_nodes = random.randint(10, 20)
        num_additional_edges = random.randint(1, 3)

        # Now with equal probability, we either embed the seed graph or not, which also influences the
        # label of the graph
        if random.randint(0, 1):
            seed_graphs = [SEED_GRAPH]
            graph_labels = [0, 1]
        else:
            seed_graphs = []
            graph_labels = [1, 0]

        try:
            generator = GraphGenerator(
                num_nodes=num_nodes,
                num_additional_edges=num_additional_edges,
                node_attributes_cb=lambda *args: [random.random()],
                edge_attributes_cb=lambda *args: [1.0],
                seed_graphs=seed_graphs,
                prevent_edges_in_seed_graphs=True
            )
            generator.reset()
            g: tc.GraphDict = generator.generate()
        except Exception as exc:
            e.info(f' ! Failed generation: {str(exc)}')
            continue

        # We need to add the label to the graph
        g['graph_labels'] = np.array(graph_labels)

        # And we need to create the node and edge importances accordingly to declare the seed graph as the
        # ground truth explanation for the second class
        node_importances = np.array([
            [0 for _ in g['node_indices']],
            generator.get_seed_graph_node_indication(0)
        ])
        g['node_importances_2'] = np.array(node_importances).T

        edge_importances = np.array([
            [0 for _ in g['edge_indices']],
            generator.get_seed_graph_edge_indication(0)
        ])
        g['edge_importances_2'] = np.array(edge_importances).T

        dataset.append(g)

        if len(dataset) % LOG_STEP_EVAL == 0:
            e.info(f' * ({len(dataset)}/{NUM_ELEMENTS})'
                   f' - nodes: {generator.num_nodes}'
                   f' - edges: {generator.num_edges}')

    e.info(f'generated dataset with {len(dataset)} elements')

    # -- GENERATING THE VISUALIZATIONS --
    # Now we need to create the visualizations for these graphs as they are an integral part of any
    # "VISUAL graph dataset".
    e.info('generating visualizations...')
    for index, g in enumerate(dataset):

        # First of all we need to create a matplotlib figure onto which we can paint the visualization
        # This function creates a special kind of figure which is essentially a blank slate without any
        # kinds of labels, background etc.
        fig, ax = create_frameless_figure(IMAGE_WIDTH, IMAGE_HEIGHT)

        # Then to actually draw the nodes somewhere we need to know the positions within the coordinate
        # system of the figure. This function uses networkx to create such a layout and return the
        # node positions
        node_positions = layout_node_positions(g)

        # Then finally we draw the grayscale visualization using this information
        visualize_grayscale_graph(ax, g, node_positions)

        # Now we can save this figure as an image
        image_path = os.path.join(dataset_folder_path, f'{index}.png')
        fig.savefig(image_path)

        # Now we also need to figure out the coordinates of nodes as pixel values inside the coordinate
        # system of the created image representation of the figure.
        image_node_positions = [[int(v) for v in ax.transData.transform((x, y))]
                                 for x, y in node_positions]
        g['image_node_positions'] = image_node_positions

        plt.close(fig)
        if index % LOG_STEP_EVAL == 0:
            e.info(f'({index}/{NUM_ELEMENTS}) done')

    # -- GENERATING THE METADATA --
    indices = range(len(dataset))
    train_indices = random.sample(indices, k=int(len(indices) * TRAIN_RATIO))
    e.info(f'identified {len(train_indices)} as train set and the rest as test set')

    e.info('generating metadata...')
    for index, g in enumerate(dataset):

        train_split = [1] if index in train_indices else []
        test_split = [1] if index not in train_indices else []

        metadata = {
            'split': 'train' if index in train_indices else 'test',
            'train_split': train_split,
            'test_split': test_split,
            'index': index,
            'target': g['graph_labels'],
            'graph': g,
        }

        metadata_path = os.path.join(dataset_folder_path, f'{index}.json')
        with open(metadata_path, mode='w') as file:
            json.dump(metadata, file, cls=NumericJsonEncoder)

        if index % LOG_STEP_EVAL == 0:
            e.info(f'({index}/{NUM_ELEMENTS}) done')

    # == COMPRESSING DATASET ==
    # In the end we also want to compress the dataset into a zip file which we can use to upload to the
    # remote file share provider
    e.info('compressing dataset into ZIP file...')
    zip_path = os.path.join(e.path, f'{DATASET_NAME}')
    shutil.make_archive(zip_path, 'zip', dataset_folder_path)
