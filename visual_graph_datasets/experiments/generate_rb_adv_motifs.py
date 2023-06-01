import os
import time
import pathlib
import random
import yaml
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from pycomex.experiment import Experiment
from pycomex.util import Skippable

import visual_graph_datasets.typing as tc
from visual_graph_datasets.data import load_visual_graph_dataset
from visual_graph_datasets.data import generate_visual_graph_dataset_metadata
from visual_graph_datasets.generation.graph import GraphGenerator
from visual_graph_datasets.generation.colors import *
from visual_graph_datasets.processing.base import create_processing_module
from visual_graph_datasets.processing.colors import ColorProcessing
from visual_graph_datasets.visualization.importances import create_importances_pdf

PATH = pathlib.Path(__file__).parent.absolute()

# == GENERATION PARAMETERS ==
COLORS = [
    # We want gray nodes to show with double the chance.
    GRAY,
    RED, RED,
    GREEN,
    BLUE, BLUE,
    MAGENTA,
    CYAN,
    YELLOW,
]

EMPTY_MOTIF = {
    'node_indices': [],
    'node_attributes': [],
    'edge_indices': [],
    'edge_attributes': []
}

TARGET_MOTIFS = [
    make_star_motif(YELLOW, RED, k=4),
    # make_star_motif(YELLOW, MAGENTA, k=3)
]
NON_TARGET_MOTIFS = [
    make_ring_motif(GREEN, RED, k=4),
    # make_ring_motif(GREEN, MAGENTA, k=3),
]

ADVERSARIAL_MOTIFS = [
    make_ring_motif(YELLOW, BLUE, k=4),
    # make_ring_motif(YELLOW, CYAN, k=3),
]
NON_ADVERSARIAL_MOTIFS = [
    make_star_motif(GREEN, BLUE, k=4),
    # make_star_motif(GREEN, CYAN, k=3),
]

RANDOM_MOTIF = make_grid_motif(GREEN, YELLOW, n=2, m=2)

NUM_NODES_RANGE = (20, 50)
NUM_ADDITIONAL_EDGES_RANGE = (3, 6)

# == DATASET PARAMETERS ==
DATASET_NAME = 'rb_adv_motifs'
NUM_ELEMENTS = 5_000
TRAIN_RATIO = 0.8
NUM_EXAMPLES = 500

DATASET_META: t.Optional[dict] = {
    'version': '0.1.0',
    'changelog': [
        '0.1.0 - 23.03.2023 - initial version'
    ],
    'description': (
        'Small dataset consisting of molecular graphs, where the target is the measured logS value of '
        'the molecules solubility in Benzene.'
    ),
    'references': [
        'Generated from the visual graph datasets',
    ],
    'visualization_description': (
        ''
    ),
    'target_descriptions': {
        0: 'one-hot encoding of class "does NOT contain target motif"',
        1: 'one-hot encoding of class "does contain target motif"'
    }
}


# == PROCESSING PARAMETERS ==

class VgdColorProcessing(ColorProcessing):

    pass


PROCESSING = VgdColorProcessing()


# == EVALUATION_PARAMETERS ==

NUM_BINS: int = 10


# == EXPERIMENT PARAMETERS ==
BASE_PATH = PATH
NAMESPACE = 'results/' + os.path.basename(__file__).strip('.py')
DEBUG = True
TESTING = False
with Skippable(), (e := Experiment(BASE_PATH, NAMESPACE, globals())):

    if TESTING:
        NUM_ELEMENTS = 500
        NUM_EXAMPLE = 50

    e.info('starting to generate the red-blue adversarial motifs dataset...')

    # -- determining dataset properties --
    e.info('determining dataset properties...')
    num_half = int(0.5 * NUM_ELEMENTS)
    indices = list(range(NUM_ELEMENTS))
    target_indices = random.sample(indices, k=num_half)
    non_target_indices = list(set(indices).difference(set(target_indices)))

    adversarial_indices = \
        random.sample(target_indices, k=int(0.5*num_half)) + \
        random.sample(non_target_indices, k=int(0.5*num_half))

    random_indices = random.sample(indices, k=num_half)

    e['indices'] = indices
    e['target_indices'] = target_indices
    e['adversarial_indices'] = adversarial_indices
    e['random_indices'] = random_indices

    # Here we randomly decide on a canonical train-test split. Having one as part of the dataset will make
    # the results more comparable.
    train_indices = random.sample(indices, k=int(TRAIN_RATIO * NUM_ELEMENTS))

    index_property_map = defaultdict(lambda: {
        'graph_labels': [1, 0],
        'graph_labels_adverse': [1, 0],
        'motif_target': EMPTY_MOTIF,
        'motif_adverse': EMPTY_MOTIF,
        'motif_random': EMPTY_MOTIF
    })
    for index in indices:
        if index in target_indices:
            index_property_map[index]['motif_target'] = random.choice(TARGET_MOTIFS)
            index_property_map[index]['graph_labels'] = [0, 1]
        else:
            index_property_map[index]['motif_target'] = random.choice(NON_TARGET_MOTIFS)

        if index in adversarial_indices:
            index_property_map[index]['motif_adverse'] = random.choice(ADVERSARIAL_MOTIFS)
            index_property_map[index]['graph_labels_adverse'] = [0, 1]
        else:
            index_property_map[index]['motif_adverse'] = random.choice(NON_ADVERSARIAL_MOTIFS)

        if index in random_indices:
            index_property_map[index]['motif_random'] = RANDOM_MOTIF

        index_property_map[index]['index'] = index

    index_property_map = dict(index_property_map)
    e.info(f'assigned properties to {len(index_property_map)} elements')

    # -- Setting up the dataset folder --
    e.info('setting up the dataset folder...')
    dataset_path = os.path.join(e.path, DATASET_NAME)
    os.mkdir(dataset_path)
    e['dataset_path'] = dataset_path

    # -- generating the graphs --
    e.info('generating graphs...')

    start_time = time.time()
    dataset = []
    for c, (index, data) in enumerate(index_property_map.items()):

        # first thing we want to do is we want to assemble the motifs which are the main thing here
        motifs = [
            data['motif_target'],
            data['motif_adverse'],
            data['motif_random']
        ]

        num_nodes = random.randint(*NUM_NODES_RANGE)
        num_additional_edges = random.randint(*NUM_ADDITIONAL_EDGES_RANGE)

        generator = GraphGenerator(
            num_nodes=num_nodes,
            num_additional_edges=num_additional_edges,
            node_attributes_cb=lambda *args: random.choice(COLORS),
            edge_attributes_cb=lambda *args: [1],
            seed_graphs=motifs,
            is_directed=False,
            prevent_edges_in_seed_graphs=True,
        )

        is_valid = False
        while not is_valid:
            try:
                generator.reset()
                graph = generator.generate()
                tc.assert_graph_dict(graph)
                is_valid = True
            except (AssertionError, IndexError, KeyError) as exc:
                e.info(f' * error: {str(exc)}')

        # The generation of the graph was only the first step however, Now we need to generate and
        # attach additional metadata to that graph structure. One important piece of information for example
        # is to store which nodes represent the various motifs! Also we need to attach the graph labels

        # The labels we can directly use as they are in the data dict
        graph['graph_labels'] = data['graph_labels']
        graph['graph_labels_adverse'] = data['graph_labels_adverse']
        target_class = int(np.argmax(data['graph_labels']))
        adverse_class = int(np.argmax(data['graph_labels_adverse']))

        # From the construction of the "motifs" list we know that the target motif is internally the seed
        # graph with the index 0 - the others accordingly corresponding to their position in the list.
        graph['node_importances_2'] = np.zeros(shape=(generator.num_nodes, 2))
        graph['node_importances_2'][:, target_class] = generator.get_seed_graph_node_indication(0)
        graph['edge_importances_2'] = np.zeros(shape=(generator.num_edges, 2))
        graph['edge_importances_2'][:, target_class] = generator.get_seed_graph_edge_indication(0)

        graph['node_importances_2_adverse'] = np.zeros(shape=(generator.num_nodes, 2))
        graph['node_importances_2_adverse'][:, adverse_class] = generator.get_seed_graph_node_indication(1)
        graph['edge_importances_2_adverse'] = np.zeros(shape=(generator.num_edges, 2))
        graph['edge_importances_2_adverse'][:, adverse_class] = generator.get_seed_graph_edge_indication(1)

        graph['node_importances_2_random'] = np.zeros(shape=(generator.num_nodes, 2))
        graph['node_importances_2_random'][:, 1] = generator.get_seed_graph_node_indication(2)
        graph['edge_importances_2_random'] = np.zeros(shape=(generator.num_edges, 2))
        graph['edge_importances_2_random'][:, 1] = generator.get_seed_graph_edge_indication(2)

        is_train_index = index in train_indices
        metadata = {
            'name':             str(index),
            'target':           data['graph_labels'],
            'adverse':          data['graph_labels_adverse'],
            'train_split':      [0] if is_train_index else [],
            'test_split':       [] if is_train_index else [0]
        }

        # This method will handle the creation of the visual graph dataset file representations (metadata
        # json file and png visualization) completely on its own.
        PROCESSING.create(
            node_attributes=graph['node_attributes'],
            edge_indices=graph['edge_indices'],
            index=index,
            output_path=dataset_path,
            additional_metadata=metadata,
            additional_graph_data=graph,
        )

        if c % 100 == 0:
            elapsed_time = time.time() - start_time
            time_per_element = elapsed_time / (c + 1)
            remaining_time = (NUM_ELEMENTS - c) * time_per_element
            e.info(f'created ({c}/{NUM_ELEMENTS}) graphs'
                   f' - elapsed time: {elapsed_time:.1f}s'
                   f' - remaining time: {remaining_time:.1f}s'
                   f' - index: {index}'
                   f' - num nodes: {len(graph["node_indices"])}'
                   f' - num motifs: {len(motifs)}')

    # Now we need to generate the metadata file and the processing python module for the visual graph
    # dataset folder to be complete
    metadata_map, index_data_map = load_visual_graph_dataset(
        dataset_path,
        logger=e.logger,
        log_step=100,
        metadata_contains_index=True,
    )

    # This function will fill in the procedurally generated information into the metadata such as the
    # number of elements in the dataset and the combined files size
    metadata_map.update(generate_visual_graph_dataset_metadata(index_data_map))
    # This will add the descriptions of the node and edge feature tensors to the metadata (what each of
    # the elements of the feature tensors represent in natural language)
    metadata_map.update(PROCESSING.get_description_map())
    # This will add the additional fields defined above to it
    metadata_map.update(DATASET_META)

    metadata_path = os.path.join(dataset_path, '.meta.yml')
    with open(metadata_path, mode='w') as file:
        yaml.dump(metadata_map, file)

    # This will generate the code for the standalone processing module, which will process input elements in
    # exactly the same manner as defined by the processing instance.
    module_code = create_processing_module(PROCESSING)
    module_path = os.path.join(dataset_path, 'process.py')
    with open(module_path, mode='w') as file:
        file.write(module_code)


with Skippable(), e.analysis:

    e.info('starting analysis of the dataset...')

    e.info('loading the dataset...')
    metadata_map, index_data_map = load_visual_graph_dataset(
        e['dataset_path'],
        logger=e.logger,
        log_step=100,
    )
    dataset = [data['metadata']['graph'] for index, data in index_data_map.items()]
    e.info(f'loaded dataset with {len(index_data_map)} elements')

    e.info('dataset .meta.yml file:')
    print(metadata_map)

    e.info('visualizing ground truth importances for a few example elements...')
    indices = e['indices']
    example_indices = random.sample(indices, k=100)

    graph_list = [index_data_map[index]['metadata']['graph'] for index in example_indices]
    image_path_list = [index_data_map[index]['image_path'] for index in example_indices]
    node_position_list = [g['node_positions'] for g in graph_list]
    output_path = os.path.join(e.path, 'examples.pdf')
    create_importances_pdf(
        graph_list=graph_list,
        image_path_list=image_path_list,
        node_positions_list=node_position_list,
        importances_map={
            'target': (
                [g['node_importances_2'] for g in graph_list],
                [g['edge_importances_2'] for g in graph_list]
            ),
            'adversarial': (
                [g['node_importances_2_adverse'] for g in graph_list],
                [g['edge_importances_2_adverse'] for g in graph_list]
            ),
            'random': (
                [g['node_importances_2_random'] for g in graph_list],
                [g['edge_importances_2_random'] for g in graph_list]
            )
        },
        output_path=output_path,
        logger=e.logger,
        log_step=100,
    )

    # We plot the distribution of the number of red nodes in each graph separately for the two possible
    # target classes.
    # The purpose of this is to make sure that the dataset does not have the most obvious exploit that a
    # ML model could take advantage of: The simple number of red nodes in a graph should not be indicative
    # of the class! We want to make sure that it is actually the motif which is responsible for this!
    e.info('plotting red node count distributions...')
    fig, rows = plt.subplots(ncols=1, nrows=2, figsize=(10, 20), squeeze=False, sharex='col')
    fig.suptitle('Distributions of the number of RED nodes for the two target classes')
    for c in range(2):
        ax = rows[c][0]
        red_node_counts = [len([att for att in g['node_attributes'] if np.allclose(att, RED)])
                           for g in dataset
                           if g['graph_labels'][c] == 1]
        ax.hist(
            red_node_counts,
            bins=NUM_BINS,
            color=(1, 0, 0, 0.5)
        )

        ax.set_title(f'number of RED nodes in graphs for class label: {c}')

    e.commit_fig('red_node_counts.pdf', fig)

    # The same thing applies for the number of blue nodes and the adversarial class.
    e.info('blue node count distributions...')
    fig, rows = plt.subplots(ncols=1, nrows=2, figsize=(10, 20), squeeze=False, sharex='col')
    fig.suptitle('Distributions of the number of BLUE nodes for the two adversarial classes')
    for c in range(2):
        ax = rows[c][0]
        red_node_counts = [len([att for att in g['node_attributes'] if np.allclose(att, BLUE)])
                           for g in dataset
                           if g['graph_labels_adverse'][c] == 1]
        ax.hist(
            red_node_counts,
            bins=NUM_BINS,
            color=(0, 0, 1, 0.5)
        )

        ax.set_title(f'number of BLUE nodes in graphs for class label: {c}')

    e.commit_fig('blue_node_counts.pdf', fig)

