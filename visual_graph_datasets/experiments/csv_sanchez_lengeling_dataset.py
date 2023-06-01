"""
This is the base experiment to convert a dataset from the Sanchez-Lengeling et al "graph-attributions"
repository into a single CSV file.

CHANGELOG

0.1.0 - 29.03.2023 - Initial version
"""
import os
import sys
import csv
import random
import pathlib
import json
import typing as t

import requests
import numpy as np
from pycomex.experiment import Experiment
from pycomex.util import Skippable
from visual_graph_datasets.util import edge_importances_from_node_importances
from visual_graph_datasets.data import load_visual_graph_dataset
from visual_graph_datasets.data import NumericJsonEncoder
from visual_graph_datasets.processing.molecules import MoleculeProcessing
from visual_graph_datasets.visualization.molecules import mol_from_smiles
from visual_graph_datasets.visualization.importances import create_importances_pdf

PATH = pathlib.Path(__file__).parent.absolute()
ASSETS_PATH = os.path.join(PATH, 'assets')

# == SOURCE PARAMETERS ==
REPO_URL: str = 'https://raw.githubusercontent.com/google-research/graph-attribution/main/data/'
SOURCE_NAME: str = 'benzene'

# == DESTINATION PARAMETERS ==
DATASET_NAME: str = 'sl_benzene_logic'

# == EVALUATION PARAMETERS ==
NUM_EXAMPLES: int = 100
WIDTH: int = 1000
HEIGHT: int = 1000

# == EXPERIMENT PARAMETERS ==
BASE_PATH = PATH
NAMESPACE = 'results/' + os.path.basename(__file__).strip('.py')
DEBUG = True
with Skippable(), (e := Experiment(BASE_PATH, NAMESPACE, globals())):

    e.info('Starting to convert from "graph-attribution" dataset into pure CSV')

    e.info('downloading the relevant files from the remote repo...')
    csv_url = REPO_URL + f'{SOURCE_NAME}/{SOURCE_NAME}_smiles.csv'
    SOURCE_CSV_PATH = os.path.join(e.path, 'source.csv')
    response = requests.get(csv_url)
    with open(SOURCE_CSV_PATH, mode='wb') as file:
        file.write(response.content)

    npz_url = REPO_URL + f'{SOURCE_NAME}/true_raw_attribution_datadicts.npz'
    SOURCE_ATTRIBUTIONS_PATH = os.path.join(e.path, 'datadicts.npz')
    response = requests.get(npz_url)
    with open(SOURCE_ATTRIBUTIONS_PATH, mode='wb') as file:
        file.write(response.content)

    e.info('loading the CSV data...')
    dataset_map: t.Dict[int, dict] = {}
    with open(SOURCE_CSV_PATH, mode='r') as file:
        dict_reader = csv.DictReader(file)
        for index, data in enumerate(dict_reader):
            dataset_map[index] = {
                'name':     data['mol_id'],
                'smiles':   data['smiles'],
                'label':    data['label']
            }

    dataset_indices = list(dataset_map.keys())
    dataset_length = len(dataset_map)
    e.info(f'loaded {dataset_length} elements from the CSV file')

    e.info('loading the ground truth attributions...')
    data = np.load(SOURCE_ATTRIBUTIONS_PATH, allow_pickle=True)
    datadict_list: t.List[t.List[dict]] = data['datadict_list']
    e.info(f'loaded {len(datadict_list)} elements from the CSV file')

    for index, l in enumerate(datadict_list):
        # For some reason, the values in this list are not the dictionaries with the information directly
        # but instead another list, which only contains one item, which is the actual dictionary.
        data = l[0]

        # Each of these dictionaries contains the full graph representation of the molecule, as it has been
        # derived from RDKit.
        # The "nodes" field contains the binary node attributional explanations for the graph.
        node_importances = np.array(data['nodes'])
        # The attributions are separated into each individual motif in the last array dimension. In this
        # case we don't want that which is why we make sure to aggregate that here if it is the case.
        node_importances = np.sum(node_importances, axis=-1, keepdims=True)

        # In this dict the edges are encoded as two separate lists instead of the single edge indices tuple
        # list as it is done in this package so we convert that
        edge_indices = np.array(list(zip(data['senders'], data['receivers'])))

        # The ground truth does not contain edge importances, which is why we have to create them from
        # the node importances here.
        edge_importances = edge_importances_from_node_importances(
            edge_indices=edge_indices,
            node_importances=node_importances
        )

        dataset_map[index]['node_importances_1'] = node_importances
        dataset_map[index]['edge_importances_1'] = edge_importances

    # Here we create visualizations of the ground truth importances for a few of the elements of the dataset
    # the purpose of this is to visually check if they make sense - if they were re-associated with the
    # correct graph elements.

    # The idea for the visualization is to create a small visual graph dataset folder for the selected
    # examples and then to load that VGD and do the usual pipeline of creating a visualization PDF.
    # Paradoxically, this is actually simpler (code-wise) than doing it manually.
    e.info('visualizing the ground truth importances...')

    example_indices = random.sample(dataset_indices, k=NUM_EXAMPLES)
    e['example_indices'] = example_indices
    e.info(f'selected {len(example_indices)} indices for visualization')

    # Now, we need PNG images of those elements to use the visual_graph_datasets visualization utils to
    # create the example visualizations. We simply create them "manually" here for the few selected
    # elements.
    e.info('creating mini VGD for the example elements...')
    vgd_path = os.path.join(e.path, 'vgd')
    os.mkdir(vgd_path)

    # We can use this Processing instance to really easily create visual graph dataset element files based
    # on molecules SMILE representations!
    processing = MoleculeProcessing()

    for c, index in enumerate(example_indices):
        smiles = dataset_map[index]['smiles']
        mol = mol_from_smiles(smiles)

        processing.create(
            smiles=dataset_map[index]['smiles'],
            index=str(index),
            name=dataset_map[index]['name'],
            additional_graph_data={
                'node_importances_1': dataset_map[index]['node_importances_1'],
                'edge_importances_1': dataset_map[index]['edge_importances_1'],
            },
            width=WIDTH,
            height=HEIGHT,
            output_path=vgd_path,
        )

        if c % 10 == 0:
            e.info(f' * created ({c}/{NUM_EXAMPLES})')

    e.info('loading the mini VGD...')
    metadata_map, index_data_map = load_visual_graph_dataset(
        vgd_path,
        metadata_contains_index=True
    )

    e.info('creating visualizations...')
    graph_list = [d['metadata']['graph'] for d in index_data_map.values()]
    image_path_list = [d['image_path'] for d in index_data_map.values()]
    node_positions_list = [g['node_positions'] for g in graph_list]
    pdf_path = os.path.join(e.path, 'examples.pdf')
    create_importances_pdf(
        graph_list=graph_list,
        image_path_list=image_path_list,
        node_positions_list=node_positions_list,
        importances_map={
            'ground truth': (
                [g['node_importances_1'] for g in graph_list],
                [g['edge_importances_1'] for g in graph_list]
            )
        },
        output_path=pdf_path
    )

    e.info('Converting dataset into single CSV file...')
    csv_path = os.path.join(e.path, f'{DATASET_NAME}.csv')
    with open(csv_path, mode='w') as file:
        dict_writer = csv.DictWriter(file, fieldnames=[
            'index',
            'smiles',
            'label',
            'node_importances_1',
            'edge_importances_1',
        ])
        dict_writer.writeheader()

        for index, data in dataset_map.items():
            dict_writer.writerow({
                'index':                index,
                'smiles':               data['smiles'],
                'label':                data['label'],
                'node_importances_1':   json.dumps(data['node_importances_1'], cls=NumericJsonEncoder),
                'edge_importances_1':   json.dumps(data['edge_importances_1'], cls=NumericJsonEncoder),
            })


