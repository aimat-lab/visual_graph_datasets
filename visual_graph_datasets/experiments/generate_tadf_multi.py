"""
Generates a single molecule-based "visual graph dataset" by merging multiple CSV files containing molecular
SMILES codes and target value annotations. More specifically, the outcome of this experiment is a visual
graph dataset whose elements and visualizations represent molecules. The target will be a multitask
regression problem, where every molecule is associated with a vector of multiple cont. regression targets.

**CHANGELOG**

0.1.0 - 16.12.2022 - Initial version

0.1.1 - 30.12.2022 - Fixed a bug, where smiles representing single atoms were able to enter the dataset.
These single node graphs would cause problems down the line for graph neural networks since they do not
include any edges at all. Such molecules are now filtered and NOT added to the final VGD.
"""
import os
import shutil
import sys
import json
import csv
import typing as t
from itertools import accumulate, chain
from pprint import pprint

import rdkit.Chem.Descriptors
import rdkit.Chem.Crippen
from rdkit import Chem
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pycomex.experiment import Experiment
from pycomex.util import Skippable
from rdkit import Chem

import visual_graph_datasets.typing as tc
from visual_graph_datasets.config import Config
from visual_graph_datasets.web import AbstractFileShare, get_file_share
from visual_graph_datasets.processing.base import identity, list_identity
from visual_graph_datasets.processing.molecules import chem_prop, chem_descriptor
from visual_graph_datasets.processing.molecules import apply_atom_callbacks, apply_bond_callbacks
from visual_graph_datasets.processing.molecules import mol_from_smiles
from visual_graph_datasets.processing.molecules import OneHotEncoder
from visual_graph_datasets.visualization.base import create_frameless_figure
from visual_graph_datasets.visualization.molecules import visualize_molecular_graph_from_mol
from visual_graph_datasets.data import NumericJsonEncoder
from visual_graph_datasets.data import load_visual_graph_dataset

VERSION = '0.1.1'
SHORT_DESCRIPTION = (
    'Generates a single molecule-based "visual graph dataset" by merging multiple CSV files containing '
    'molecular SMILES codes and target values annotations.'
)

# == SOURCE PARAMETERS ==
# This section contains the parameters which define the source CSV files which are to be used as the basis
# for the final dataset. At the bare minimum, these CSV files have to contain one column with the SMILES
# representation of the molecule and one column with the corresponding target value to be predicted.
# It is also possible to use multiple target value from a single CSV file.
# The CSV files can either be supplied as local files or them can first be downloaded from a remote VGD
# file share provider.

# This is the ID of the VGD file share provider, which will (optionally) be used to download the CSV source
# files
FILE_SHARE_PROVIDER: str = 'main'
# The keys of this dictionary should be unique keys which identify the theme of each CSV file to be used in
# the subsequent merge. The values should be paths to the CSV files. If local files are to be used,
# the absolute(!) paths have to be supplied. Alternatively, if the paths cannot be found on the local system,
# they will be interpreted as path relative to the remote file share provider and it is attempted to
# download those files from there.
CSV_FILE_NAME_MAP: t.Dict[str, str] = {
    'tadf': 'source/tadf.csv',
}
# The keys should be the same keys as defined above for each of the CSV files and the values should be the
# string names of the columns of each file, which contain the SMILES string.
SMILES_COLUMN_NAME_MAP: t.Dict[str, str] = {
    'tadf': 'smiles',
}
# THe keys should be the same as defined above for each of the CSV files and the values should be lists
# containing all the string column names which contain the target values of the corresponding dataset.
TARGET_COLUMN_NAME_MAP: t.Dict[str, t.List[str]] = {
    'tadf': [
        'splitting',   # singlet-triplet splitting energy gap E_st
        'strength',    # The oscillator strength f_osc
        'absorption',  # k_tadf
    ],
}
SOURCE_KEYS = list(CSV_FILE_NAME_MAP.keys())  # do not modify

# == PROCESSING PARAMETERS ==
# This section contains the parameters for the processing pipeline. These for example

# boolean flag whether to represent undirected edges as two directed edges in opposite directions.
# If dataset is to be used with KGCNN model, this needs to be True
UNDIRECTED_EDGES_AS_TWO: bool = True
# This dictionary defines which node attributes/features will be extracted for the graph representations of
# the dataset the keys should be descriptive names for the attributes and the values are callback functions
# which take the Mol object and the CSV row data dict as inputs and return the corresponding partial
# feature vector.
NODE_ATTRIBUTE_CALLBACKS = {
    'symbol':                   chem_prop('GetSymbol', OneHotEncoder(
        ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At'],
        add_unknown=True,
        dtype=str
    )),
    'hybridization':            chem_prop('GetHybridization', OneHotEncoder(
        [2, 3, 4, 5, 6],
        add_unknown=True,
        dtype=int
    )),
    'total_degree':             chem_prop('GetTotalDegree', OneHotEncoder(
        [0, 1, 2, 3, 4, 5],
        add_unknown=False,
        dtype=int
    )),
    'num_hydrogen_atoms':       chem_prop('GetTotalNumHs', OneHotEncoder(
        [0, 1, 2, 3, 4],
        add_unknown=False,
        dtype=int
    )),
    'num_radical_electrons':    chem_prop('GetNumRadicalElectrons', list_identity),
    'charge':                   chem_prop('GetFormalCharge', list_identity),
    'is_aromatic':              chem_prop('GetIsAromatic', list_identity),
    'is_in_ring':               chem_prop('IsInRing', list_identity),
}
# This dictionary is exactly the same thing but for the edge attributes.
EDGE_ATTRIBUTE_CALLBACKS = {
    'bond_type':                chem_prop('GetBondType', OneHotEncoder(
        [1, 2, 3, 12],
        add_unknown=False,
        dtype=int,
    )),
    'stereo':                   chem_prop('GetStereo', OneHotEncoder(
        [0, 1, 2, 3],
        add_unknown=False,
        dtype=int,
    )),
    'is_aromatic':              chem_prop('GetIsAromatic', list_identity),
    'is_in_ring':               chem_prop('IsInRing', list_identity),
    'is_conjugated':            chem_prop('GetIsConjugated', list_identity)
}
# This dictionary is used to define which global graph attributes/features are supposed to be generated for
# each element of the dataset. THe keys should be descriptive names and the values should be descriptive
# names and the values should be callback functions which take the Mol object and the CSV row data dict as
# input and return a vector with the partial graph features.
GRAPH_ATTRIBUTE_CALLBACKS = {
    'molecular_weight':         chem_descriptor(Chem.Descriptors.ExactMolWt, list_identity),
    'num_radical_electrons':    chem_descriptor(Chem.Descriptors.NumRadicalElectrons, list_identity),
    'num_valence_electrons':    chem_descriptor(Chem.Descriptors.NumValenceElectrons, list_identity),
    'fp_density_morgan_1':      chem_descriptor(Chem.Descriptors.FpDensityMorgan1, list_identity),
    'fp_density_morgan_2':      chem_descriptor(Chem.Descriptors.FpDensityMorgan2, list_identity),
    'fp_density_morgan_3':      chem_descriptor(Chem.Descriptors.FpDensityMorgan3, list_identity),
    'log_p':                    chem_descriptor(Chem.Crippen.MolLogP, list_identity),
}
GRAPH_METADATA_CALLBACKS = {  # Not in use at the moment
    #'name': lambda mol, data: data['smiles'],
    #'smiles': lambda mol, data: data['smiles'],
}

# == DATASET PARAMETERS ==
# This section defines parameters for the creation of the visual graph dataset.

# This will be the name of the finished visual graph dataset folder. This dataset folder can be found in the
# archive folder of the experiment run.
DATASET_NAME: str = 'tadf_multi'
# The width and height (in pixels) with which the visualizations will be created.
IMAGE_WIDTH: int = 1000
IMAGE_HEIGHT: int = 1000

# == EVALUATION PARAMETERS ==
EVAL_LOG_STEP = 100
NUM_BINS = 10
PLOT_COLOR = 'gray'

# == EXPERIMENT PARAMETERS ==
DEBUG = True
BASE_PATH = os.getcwd()
NAMESPACE = 'results/generate_molecule_multitask_dataset_from_csv/tadf_multi'
with Skippable(), (e := Experiment(base_path=BASE_PATH, namespace=NAMESPACE, glob=globals())):
    e.info('generating a molecule visual graph dataset from CSV source file...')
    config = Config()
    config.load()

    # -- get source datasets --
    # As for the sources there are two possibilities: Either we use local files or we download the files
    # from the VGD file share provider first. The distinction is made by checking if the provided paths are
    e.info('collecting source files...')
    file_path_map: t.Dict[str, str] = {}
    file_share: AbstractFileShare = get_file_share(config, FILE_SHARE_PROVIDER)
    for key, file_name in CSV_FILE_NAME_MAP.items():
        if os.path.exists(file_name):
            e.info(f' * copying {file_name}...')
            local_file_path = os.path.join(e.path, os.path.basename(file_name))
            shutil.copy(file_name, e.path)
        else:
            e.info(f' * downloading {file_name} ...')
            local_file_path = file_share.download_file(file_name, e.path)

        file_path_map[key] = local_file_path

    # -- merge datasets & convert dataset to molecules --
    # Next we need to read all the CSV files and then merge them based on the SMILES strings. This means
    # that if in multiple datasets we find the same molecule (same smile) we can assign multiple target
    # values to that smiles corresponding to the multiple target values assigned by the different csv
    # files. But most of the time, this will not be the case. In that case we will assign the value "None"
    # as the target value for all the target values from the other CSV files. For a machine learning method
    # this will then be converted to NaN - which the method will have to figure out how to handle.

    # After the data is merged, all the individual elements of the merged dataset have to be converted from
    # the SMILES string to Mol objects which are then ultimately converted into the graph representation
    # expected for the visual graph dataset format.
    e.info('merging datasets...')
    overlap_counter = 0
    smiles_data_map: t.Dict[int, t.Dict[str, t.Any]] = {}
    for key, csv_path in file_path_map.items():

        e.info(f' * reading csv for {key}...')
        with open(csv_path, mode='r') as file:
            reader = csv.DictReader(file)
            for i, row in enumerate(reader):
                smiles = row[SMILES_COLUMN_NAME_MAP[key]]

                # TODO: instead of checking for the exact same smile, check for reasonable edit distance.
                if smiles not in smiles_data_map:
                    mol = mol_from_smiles(smiles)
                    smiles_data_map[smiles] = {
                        'mol': mol,
                        'smiles': smiles,
                        'data_map': {
                            key: row,
                            **{k: None for k in CSV_FILE_NAME_MAP.keys() if k != key}
                        },
                    }
                else:
                    smiles_data_map[smiles]['data_map'][key] = row
                    overlap_counter += 1

    e.info(f'overlapping elements: {overlap_counter}')
    e.info('converting smiles to molecules...')
    index_data_map: t.Dict[int, t.Dict[str, t.Any]] = {}
    index = 0
    for i, (smiles, data) in enumerate(smiles_data_map.items()):

        data['data'] = {}
        for d in data['data_map'].values():
            if d is not None:
                data['data'].update(d)

        data['data']['target'] = [float(d[column_name]) if (d := data['data_map'][key]) is not None else None
                                  for key in SOURCE_KEYS
                                  for column_name in TARGET_COLUMN_NAME_MAP[key]]
        data['data']['smiles'] = smiles

        # 30.12.2022
        # This check here prevents "molecules" which consists of only a single atom to enter the dataset.
        # These would be single node graphs and due to not having any edges, they would cause problems
        # down the line for graph neural networks - hence they are simply excluded
        if data['mol'].GetAtoms() == 1 or len(data['mol'].GetBonds()) == 0:
            e.info(f' ! invalid: {smiles}')
            continue

        index_data_map[index] = data
        index += 1

    dataset_length = len(index_data_map)
    e.info(f'dataset with {dataset_length} elements')

    # -- Processing the dataset into visual graph dataset --
    e.info('creating the dataset folder...')
    dataset_path = os.path.join(e.path, DATASET_NAME)
    os.mkdir(dataset_path)
    e['dataset_path'] = dataset_path

    e.info('creating visual graph dataset...')
    for c, (index, d) in enumerate(index_data_map.items()):
        # ~ Convert the Mol object into a GraphDict
        data: dict = d['data']
        mol: Chem.Mol = d['mol']
        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()

        g = {}
        target = data['target']
        g['graph_labels'] = np.array(target)

        node_indices = []
        node_attributes = []
        for atom in atoms:
            node_indices.append(int(atom.GetIdx()))

            attributes = []
            for name, callback in NODE_ATTRIBUTE_CALLBACKS.items():
                value = callback(atom, data)
                attributes += value

            node_attributes.append(attributes)

        g['node_indices'] = np.array(node_indices)
        g['node_attributes'] = np.array(node_attributes)

        edge_indices = []
        edge_attributes = []
        for bond in bonds:
            i = int(bond.GetBeginAtomIdx())
            j = int(bond.GetEndAtomIdx())

            edge_indices.append([i, j])
            if UNDIRECTED_EDGES_AS_TWO:
                edge_indices.append([j, i])

            attributes = []
            for name, callback in EDGE_ATTRIBUTE_CALLBACKS.items():
                value = callback(bond, data)
                attributes += value

            edge_attributes.append(attributes)
            if UNDIRECTED_EDGES_AS_TWO:
                edge_attributes.append(attributes)

        g['edge_indices'] = np.array(edge_indices)
        g['edge_attributes'] = np.array(edge_attributes)

        graph_attributes = []
        for name, callback in GRAPH_ATTRIBUTE_CALLBACKS.items():
            value = callback(mol, data)
            graph_attributes += value

        g['graph_attributes'] = np.array(graph_attributes)

        # ~ creating a visualization
        # We need to do this *before* the metadata because one side result of the visualization process is
        # the node positions within that image, which have to be added as graph properties as well.
        fig, ax = create_frameless_figure(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
        node_positions, _ = visualize_molecular_graph_from_mol(
            ax=ax,
            mol=mol,
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
        )
        image_node_positions = [[int(v) for v in ax.transData.transform((x, y))]
                                for x, y in node_positions]
        g['node_positions'] = image_node_positions
        image_path = os.path.join(dataset_path, f'{index}.png')
        fig.savefig(image_path)
        plt.close(fig)

        # ~ constructing and saving metadata
        # First of all there are certain keys which always have to be present in the metadata dict, for
        # example the graph structure itself.
        metadata = {
            'index': index,
            'image_width': IMAGE_WIDTH,
            'image_height': IMAGE_HEIGHT,
            'target': target,
            'graph': g,
        }
        # But there can also be custom entries which are defined as callbacks in this dictionary. These
        # values will be associated with the same string keys, which are also used in the callbacks dict
        for name, callback in GRAPH_METADATA_CALLBACKS.items():
            metadata[name] = callback(mol, data)

        metadata_path = os.path.join(dataset_path, f'{index}.json')
        with open(metadata_path, mode='w') as file:
            content = json.dumps(metadata, cls=NumericJsonEncoder)
            file.write(content)

        if c % EVAL_LOG_STEP == 0:
            e.info(f' * {c}/{dataset_length} elements created')


with Skippable(), e.analysis:
    e.info('attempting to load visual graph dataset...')
    index_data_map, _ = load_visual_graph_dataset(
        e['dataset_path'],
        logger=e.logger,
        log_step=EVAL_LOG_STEP,
        metadata_contains_index=True
    )
    e.info(f'loaded visual graph dataset with {len(index_data_map)} elements')

    e.info(f'plotting dataset analyses...')
    pdf_path = os.path.join(e.path, 'dataset_info.pdf')
    with PdfPages(pdf_path) as pdf:
        e.info(f'target value distribution...')
        n_cols = len(SOURCE_KEYS)
        fig, rows = plt.subplots(ncols=n_cols, nrows=1, figsize=(12 * n_cols, 12), squeeze=False)
        for c, ax, key in zip(range(n_cols), rows[0], SOURCE_KEYS):
            ax.set_title(f'Target Value Distribution - {key}')
            targets = [v
                       for i, d in index_data_map.items()
                       if (v := d['metadata']['target'][c]) is not None]
            e.info(f' * number of targets: {len(targets)}')
            e.info(f' * min: {np.min(targets):.2f} - mean: {np.mean(targets)} - max: {np.max(targets):.2f}')
            n, bins, edges = ax.hist(
                targets,
                bins=NUM_BINS,
                color=PLOT_COLOR
            )
            ax.set_xticks(bins)
            ax.set_xticklabels([round(v, 2) for v in bins])

        pdf.savefig(fig)

        e.info('graph size distribution...')
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 12))
        ax.set_title('Graph Size Distribution')
        sizes = [len(d['metadata']['graph']['node_indices']) for d in index_data_map.values()]
        n, bins, edges = ax.hist(
            sizes,
            bins=NUM_BINS,
            color=PLOT_COLOR
        )
        ax.set_xticks(bins)
        ax.set_xticklabels([int(v) for v in bins])
        pdf.savefig(fig)

