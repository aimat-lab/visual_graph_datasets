import os
import pathlib
import csv
import json
import itertools
import typing as t

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

# == VISUALIZATION PARAMETERS ==
WIDTH = 1000
HEIGHT = 1000

# == EXPERIMENT PARAMETERS ==
BASE_PATH = PATH
NAMESPACE = 'results/' + os.path.basename(__file__).strip('.py')
DEBUG = True
with Skippable(), (e := Experiment(BASE_PATH, NAMESPACE, globals())):
    processing = MoleculeProcessing()

    e.info('loading the BBBP base dataset...')
    dataset_map: t.Dict[int, dict] = {}
    csv_path = os.path.join(ASSETS_PATH, 'BBBP_clean.csv')
    annotated_indices = []
    with open(csv_path, mode='r') as file:
        dict_reader = csv.DictReader(file)
        for index, data in enumerate(dict_reader):
            index = int(data['index'])
            label = int(data['label'])
            dataset_map[index] = {
                'index': index,
                'smiles': data['smiles'],
                'name': data['name'],
                'label': data['label'],
                'pass': int(label == 1),
                'non-pass': int(label == 0),
                'node_importances_1_gao': None,
                'edge_importances_1_gao': None,
                'node_importances_2_gao': None,
                'edge_importances_2_gao': None,
                'split': 0,
            }

            if data['node_importances'] and data['adjacency_importances']:
                annotated_indices.append(index)

                g = processing.process(
                    smiles=data['smiles'],
                    double_edges_undirected=True,
                )

                # node importances
                node_importances = np.array(json.loads(data['node_importances']))
                node_importances = np.expand_dims(node_importances, axis=-1)

                # edge importances
                edge_importances = edge_importances_from_node_importances(
                    edge_indices=g['edge_indices'],
                    node_importances=node_importances,
                )

                dataset_map[index]['node_importances_1_gao'] = node_importances
                dataset_map[index]['edge_importances_1_gao'] = edge_importances

                label = int(data['label'])
                dataset_map[index]['node_importances_2_gao'] = np.zeros(shape=(len(g['node_indices']), 2))
                dataset_map[index]['node_importances_2_gao'][:, 0] = node_importances[:, 0]

                dataset_map[index]['edge_importances_2_gao'] = np.zeros(shape=(len(g['edge_indices']), 2))
                dataset_map[index]['edge_importances_2_gao'][:, 0] = edge_importances[:, 0]

    e.info(f'loaded dataset with {len(dataset_map)} elements - {len(annotated_indices)} annotations')

    # ~ Visualizing the annotations
    # We can use this Processing instance to really easily create visual graph dataset element files based
    # on molecules SMILE representations!

    vgd_path = os.path.join(e.path, 'vgd')
    os.mkdir(vgd_path)
    for c, index in enumerate(annotated_indices):
        smiles = dataset_map[index]['smiles']
        mol = mol_from_smiles(smiles)

        try:
            node_importances = dataset_map[index]['node_importances_1_gao']
            edge_importances = dataset_map[index]['edge_importances_1_gao']

            processing.create(
                smiles=dataset_map[index]['smiles'],
                index=str(index),
                name=dataset_map[index]['name'],
                additional_graph_data={
                    'node_importances_1': node_importances,
                    'edge_importances_1': edge_importances,
                },
                width=WIDTH,
                height=HEIGHT,
                output_path=vgd_path,
            )
        except:
            e.info(f' * error for: {dataset_map[index]["smiles"]}')
            continue

        if c % 100 == 0:
            e.info(f' * created ({c}/{len(annotated_indices)})'
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

    # ~ Writing it all into one overall CSV file...
    e.info('Writing the CSV file...')
    result_path = os.path.join(e.path, 'bbbp.csv')
    with open(result_path, mode='w') as file:
        fieldnames = [
            'index', 'smiles', 'name', 'label', 'split', 'pass', 'non-pass',
            'node_importances_1_gao', 'edge_importances_1_gao',
            'node_importances_2_gao', 'edge_importances_2_gao',
        ]
        dict_writer = csv.DictWriter(file, fieldnames=fieldnames)
        dict_writer.writeheader()
        for data in dataset_map.values():
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    data[key] = json.dumps(value.tolist())

            dict_writer.writerow(data)
