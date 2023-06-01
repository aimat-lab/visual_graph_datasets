import os
import pathlib
import csv
import random
import json
import itertools
import typing as t
from collections import defaultdict

import numpy as np
from pycomex.experiment import Experiment
from pycomex.util import Skippable

from visual_graph_datasets.util import edge_importances_from_node_importances
from visual_graph_datasets.processing.molecules import mol_from_smiles
from visual_graph_datasets.processing.molecules import MoleculeProcessing
from visual_graph_datasets.data import load_visual_graph_dataset
from visual_graph_datasets.visualization.importances import create_importances_pdf

PATH = pathlib.Path(__file__).parent.absolute()
ASSETS_PATH = os.path.join(PATH, 'assets')

# == DATASET PARAMETERS ==
CSV_PATH = os.path.join(ASSETS_PATH, 'Ames_Mutagenicity.csv')
DATASET_NAME = 'mutagenicity'

# == VISUALIZATION PARAMETERS ==
NUM_EXAMPLES = 100
WIDTH = 1000
HEIGHT = 1000

# == EXPERIMENT PARAMETERS ==
BASE_PATH = PATH
NAMESPACE = 'results/' + os.path.basename(__file__).strip('.py')
DEBUG = True
with Skippable(), (e := Experiment(BASE_PATH, NAMESPACE, globals())):
    processing = MoleculeProcessing()

    dataset_map = {}
    e.info('loading the original csv file...')
    with open(CSV_PATH, mode='r') as file:
        dict_reader = csv.DictReader(file)
        for index, data in enumerate(dict_reader):

            try:
                smiles = data['Canonical_Smiles']
                g = processing.process(smiles, double_edges_undirected=True)
                label = int(data['Activity'])
                dataset_map[index] = {
                    'index': index,
                    'smiles': smiles,
                    'name': data['WDI'],
                    'label': label,
                    'mutag': int(label == 1),
                    'non-mutag': int(label == 0),
                    'test': False,
                    'graph': g,
                }
            except:
                e.info(f' * error for: {smiles}')
                continue

    e.info(f'loaded {len(dataset_map)} elements from csv')

    # ~ Creating the processed csv file
    result_path = os.path.join(e.path, f'{DATASET_NAME}.csv')
    with open(result_path, mode='w') as file:
        fieldnames = ['index', 'name', 'smiles', 'label', 'mutag', 'non-mutag']
        dict_writer = csv.DictWriter(file, fieldnames)
        dict_writer.writeheader()
        for data in dataset_map.values():
            dict_writer.writerow({name: data[name] for name in fieldnames})

    e.info(f'create result csv file: {result_path}')

    # ~ The explainable sub dataset - Mutagenicity 0
    motif_smiles = 'c1ccc(cc1)[N+](=O)[O-]'
    motif_mol = mol_from_smiles(motif_smiles)

    exp_dataset_map = {}
    j = 0
    label_counts = defaultdict(int)
    for i, data in dataset_map.items():
        mol = mol_from_smiles(data['smiles'])
        if mol:
            has_motif = bool(mol.HasSubstructMatch(motif_mol))

            if (has_motif and data['label']) or (not has_motif and not data['label']):
                matches = mol.GetSubstructMatches(motif_mol)
                if len(matches) > 0:
                    match_indices = [index for match in matches for index in match]
                else:
                    match_indices = []

                node_importances = [int(index in match_indices) for index in data['graph']['node_indices']]
                node_importances = np.array(node_importances)
                node_importances = np.expand_dims(node_importances, axis=-1)

                edge_importances = edge_importances_from_node_importances(
                    edge_indices=data['graph']['edge_indices'],
                    node_importances=node_importances
                )

                data['node_importances_1'] = node_importances
                data['edge_importances_1'] = edge_importances

                node_importances_2 = np.zeros(shape=(len(node_importances), 2))
                node_importances_2[:, 1] = node_importances[:, 0]

                edge_importances_2 = np.zeros(shape=(len(edge_importances), 2))
                edge_importances_2[:, 1] = edge_importances[:, 0]

                data['node_importances_2'] = node_importances_2
                data['edge_importances_2'] = edge_importances_2

                exp_dataset_map[j] = data
                label_counts[data['label']] += 1
                j += 1

    e.info(f'identified {len(exp_dataset_map)} elements for the explainable sub dataset')
    e.info(f'label counts: {label_counts}')

    # ~ Visualizing the annotations
    # We can use this Processing instance to really easily create visual graph dataset element files based
    # on molecules SMILE representations!
    indices = list(exp_dataset_map.keys())
    example_indices = random.sample(indices, k=NUM_EXAMPLES)
    vgd_path = os.path.join(e.path, 'vgd')
    os.mkdir(vgd_path)
    for c, index in enumerate(example_indices):
        smiles = exp_dataset_map[index]['smiles']
        mol = mol_from_smiles(smiles)

        try:
            node_importances = exp_dataset_map[index]['node_importances_1']
            edge_importances = exp_dataset_map[index]['edge_importances_1']

            processing.create(
                smiles=exp_dataset_map[index]['smiles'],
                index=str(index),
                name=exp_dataset_map[index]['name'],
                additional_graph_data={
                    'node_importances_1': node_importances,
                    'edge_importances_1': edge_importances,
                },
                width=WIDTH,
                height=HEIGHT,
                output_path=vgd_path,
            )
        except:
            e.info(f' * error for: {exp_dataset_map[index]["smiles"]}')
            continue

        if c % 100 == 0:
            e.info(f' * created ({c}/{len(example_indices)})'
                   f' - num atoms: {len(mol.GetAtoms())} '
                   f' - num nis: {len(node_importances)}')

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
        output_path=pdf_path,
        logger=e.logger,
        log_step=100,
    )

    # ~ choosing the test indices
    e.info('choosing the canonical the test indices...')
    label_index_map = defaultdict(list)
    for index, data in exp_dataset_map.items():
        label = data['label']
        label_index_map[label].append(index)

    test_indices = []
    for label, indices in label_index_map.items():
        sample = random.sample(indices, k=250)
        test_indices += sample
        for index in indices:
            exp_dataset_map[index]['test'] = True

    file_path = os.path.join(e.path, 'test_indices.json')
    with open(file_path, mode='w') as file:
        content = json.dumps(test_indices)
        file.write(content)

    e.info(f'saved the test indices to: {file_path}')

    # ~ Writing the result csv file
    result_path = os.path.join(e.path, f'{DATASET_NAME}_exp.csv')
    with open(result_path, mode='w') as file:
        fieldnames = ['index', 'name', 'smiles', 'label', 'mutag', 'non-mutag',
                      'node_importances_1', 'edge_importances_1',
                      'node_importances_2', 'edge_importances_2']
        dict_writer = csv.DictWriter(file, fieldnames)
        dict_writer.writeheader()
        for data in exp_dataset_map.values():
            _data = {}
            for name in fieldnames:
                value = data[name]
                if isinstance(value, np.ndarray):
                    value = json.dumps(value.tolist())

                _data[name] = value

            dict_writer.writerow(_data)

    e.info(f'create result csv file: {result_path}')
