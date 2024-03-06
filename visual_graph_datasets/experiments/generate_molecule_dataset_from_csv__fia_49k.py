"""
Generates the AqSolDB visual graph dataset from a CSV base file. This dataset consists of roughly 10k
molecular graphs, which are annotated with experimentally determined values of water solubility as target values.

CHANGELOG

0.1.0 - 29.01.2023 - Initial version

0.2.0 - 24.03.2023 - Changed the dataset generation to now include .meta.yml dataset metadata
file and process.py standalone pre-processing module in the dataset folder as well.

0.3.0 - 05.05.2023 - Switched to the pycomex.functional api
"""
import os
import pathlib
import typing as t

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from rdkit import Chem

from visual_graph_datasets.processing.base import identity, list_identity
from visual_graph_datasets.processing.base import create_processing_module
from visual_graph_datasets.processing.base import ProcessingError
from visual_graph_datasets.processing.molecules import chem_prop, chem_descriptor
from visual_graph_datasets.processing.molecules import apply_atom_callbacks, apply_bond_callbacks
from visual_graph_datasets.processing.molecules import mol_from_smiles
from visual_graph_datasets.processing.molecules import OneHotEncoder
from visual_graph_datasets.processing.molecules import MoleculeProcessing
from visual_graph_datasets.processing.molecules import crippen_contrib, lasa_contrib, tpsa_contrib
from visual_graph_datasets.processing.molecules import gasteiger_charges, estate_indices

PATH = pathlib.Path(__file__).parent.absolute()

# == SOURCE PARAMETERS ==
# These parameters determine how to handle the source CSV file of the dataset. There exists the possibility
# to define a file from the local system or to download a file from the VGD remote file share location.
# In this section one also has to determine, for example, the type of the source dataset (regression, 
# classification) and provide the names of the relevant columns in the CSV file.

# :param FILE_SHARE_PROVIDER:
#       The vgd file share provider from which to download the CSV file to be used as the source for the VGD
#       conversion. 
FILE_SHARE_PROVIDER: str = 'main'
# :param CSV_FILE_NAME:
#       The name of the CSV file to be used as the source for the dataset conversion.
#       This may be one of the following two things:
#       1. A valid absolute file path on the local system pointing to a CSV file to be used as the source for
#       the VGD conversion
#       2. A valid relative path to a CSV file stashed on the given vgd file share provider which will be
#       downloaded first and then processed.
CSV_FILE_NAME: str = os.path.join(PATH, 'assets', 'FIA49k.csv')
# :param SMILES_COLUMN_NAME:
#       This has to be the string name of the CSV column which contains the SMILES string representation of
#       the molecule.
INDEX_COLUMN_NAME: t.Optional[str] = None
# :param INDICES_BLACKLIST_PATH:
#       Optionally it is possible to define the path to a file which defines the blacklisted indices for the 
#       dataset. This file should contain a list of integers, where each integer represents the index of an
#       element which should be excluded from the final dataset. The file should be a normal TXT file where each 
#       integer is on a new line.
#       The indices listed in that file will be immediately skipped during processing without even loading the 
#       the molecule.
INDICES_BLACKLIST_PATH: t.Optional[str] = os.path.join(PATH, 'assets', 'FIA49k_blacklist.txt')
# :param TARGET_TYPE:
#       This has to be the string name of the type of dataset that the source file represents. The valid 
#       options here are "regression" and "classification"
SMILES_COLUMN_NAME: str = 'la_smiles'
# :param TARGET_COLUMN_NAMES:
#       This has to be a list of string column names within the source CSV file, where each name defines 
#       one column that contains a target value for each row. In the regression case, this may be multiple 
#       different regression targets for each element and in the classification case there has to be one 
#       column per class.
TARGET_COLUMN_NAMES: t.List[str] = ['fia_gas-DSDBLYP']
# :param SPLIT_COLUMN_NAMES:
#       The keys of this dictionary are integers which represent the indices of various train test splits. The
#       values are the string names of the columns which define those corresponding splits. It is expected that
#       these CSV columns contain a "1" if that corresponding element is considered as part of the training set
#       of that split and "0" if it is part of the test set.
#       This dictionary may be empty and then no information about splits will be added to the dataset at all.
SPLIT_COLUMN_NAMES: t.Dict[int, str] = {
    0: 'split',
    1: 'split_val',
}

# == DATASET PARAMETERS ==
# These parameters control aspects of the visual graph dataset creation process. This for example includes 
# the dimensions of the graph visualization images to be created or the name of the visual graph dataset 
# that should be given to the dataset folder.

# :param DATASET_CHUNK_SIZE:
#       This number will determine the chunking of the dataset. Dataset chunking will split the dataset
#       elements into multiple sub folders within the main VGD folder. Especially for larger datasets
#       this should increase the efficiency of subsequent IO operations.
#       If this is None then no chunking will be applied at all and everything will be placed into the
#       top level folder.
#       For this particular dataset we disable chunking (None) because with only 10k elements it is small 
#       enough such that this does not matter.
DATASET_CHUNK_SIZE: t.Optional[int] = None
# :param DATASET_NAME:
#       The name given to the visual graph dataset folder which will be created.
DATASET_NAME: str = 'fia_49k'
# :parm DATASET_META:
#       This dict will be converted into the .meta.yml file which will be added to the final visual graph dataset
#       folder. This is an optional file, which can add additional meta information about the entire dataset
#       itself. Such as documentation in the form of a description of the dataset etc.
DATASET_META: t.Optional[dict] = {
    'version': '0.1.0',
    'changelog': [
        '0.1.0 - 24.01.2024 - initial version',
    ],
    'description': (
        'Dataset consisting of roughly 44_000 molecular graphs annotated with calculated FIA values.'
    ),
    'references': [
        'Library used for the processing and visualization of molecules. https://www.rdkit.org/',
    ],
    'visualization_description': (
        'Molecular graphs generated by RDKit based on the SMILES representation of the molecule.'
    ),
    'target_descriptions': {
        0: 'FIA values'
    }
}

# == PROCESSING PARAMETERS ==
# These parameters control the processing of the raw SMILES into the molecule representations with RDKit
# and then finally the conversion into the graph dict representation.


class VgdMoleculeProcessing(MoleculeProcessing):

    node_attribute_map = {
        'symbol': {
            'callback': chem_prop('GetSymbol', OneHotEncoder(
                [
                    # core atoms
                    'B', 'Al', 'Ga', 'In', 'Si', 'Ge', 'Sn', 'Pb', 'P', 'As', 'Sb', 'Bi', 'Te',
                    # ligands
                    'H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br',
                ],
                add_unknown=True,
                dtype=str
            )),
            'description': 'one-hot encoding of atom type',
        },
        'hybridization': {
            'callback': chem_prop('GetHybridization', OneHotEncoder(
                [2, 3, 4, 5, 6],
                add_unknown=True,
                dtype=int,
            )),
            'description': 'one-hot encoding of atom hybridization',
        },
        'total_degree': {
            'callback': chem_prop('GetTotalDegree', OneHotEncoder(
                [0, 1, 2, 3, 4, 5],
                add_unknown=False,
                dtype=int
            )),
            'description': 'one-hot encoding of the degree of the atom'
        },
        'num_hydrogen_atoms': {
            'callback': chem_prop('GetTotalNumHs', OneHotEncoder(
                [0, 1, 2, 3, 4],
                add_unknown=False,
                dtype=int
            )),
            'description': 'one-hot encoding of the total number of attached hydrogen atoms'
        },
        'mass': {
            'callback': chem_prop('GetMass', list_identity),
            'description': 'The mass of the atom'
        },
        'charge': {
            'callback': chem_prop('GetFormalCharge', list_identity),
            'description': 'The charge of the atom',
        },
        'is_aromatic': {
            'callback': chem_prop('GetIsAromatic', list_identity),
            'description': 'Boolean flag of whether the atom is aromatic',
        },
        'is_in_ring': {
            'callback': chem_prop('IsInRing', list_identity),
            'description': 'Boolean flag of whether atom is part of a ring'
        },
        'crippen_contributions': {
            'callback': crippen_contrib(),
            'description': 'The crippen logP contributions of the atom as computed by RDKit'
        },
        'tpsa_contribution': {
            'callback': tpsa_contrib(),
            'description': 'Contribution to TPSA as computed by RDKit',
        },
        'lasa_contribution': {
            'callback': lasa_contrib(),
            'description': 'Contribution to ASA as computed by RDKit'
        },
        'gasteiger_charge': {
            'callback': gasteiger_charges(),
            'description': 'The partial gasteiger charge attributed to atom as computed by RDKit'
        },
        'estate_indices': {
            'callback': estate_indices(),
            'description': 'EState index as computed by RDKit'
        }
    }

    edge_attribute_map = {
        'bond_type': {
            'callback': chem_prop('GetBondType', OneHotEncoder(
                [1, 2, 3, 12],
                add_unknown=False,
                dtype=int,
            )),
            'description': 'one-hot encoding of the bond type',
        },
        'stereo': {
            'callback': chem_prop('GetStereo', OneHotEncoder(
                [0, 1, 2, 3],
                add_unknown=False,
                dtype=int,
            )),
            'description': 'one-hot encoding of the stereo property'
        },
        'is_aromatic': {
            'callback': chem_prop('GetIsAromatic', list_identity),
            'description': 'boolean flag of whether bond is aromatic',
        },
        'is_in_ring': {
            'callback': chem_prop('IsInRing', list_identity),
            'description': 'boolean flag of whether bond is part of ring',
        },
        'is_conjugated': {
            'callback': chem_prop('GetIsConjugated', list_identity),
            'description': 'boolean flag of whether bond is conjugated'
        }
    }

    graph_attribute_map = {
        'molecular_weight': {
            'callback': chem_descriptor(Chem.Descriptors.ExactMolWt, list_identity),
            'description': 'the exact molecular weight of the molecule',
        },
        'num_radical_electrons': {
            'callback': chem_descriptor(Chem.Descriptors.NumRadicalElectrons, list_identity),
            'description': 'the total number of radical electrons in the molecule',
        },
        'num_valence_electrons': {
            'callback': chem_descriptor(Chem.Descriptors.NumValenceElectrons, list_identity),
            'description': 'the total number of valence electrons in the molecule'
        }
    }


# :param PROCESSING:
#       A MoleculeProcessing instance which will be used to convert the molecule smiles representations 
#       into strings. 
PROCESSING = VgdMoleculeProcessing()
# :param UNDIRECTED_EDGES_AS_TWO:
#       If this flag is True, the undirected edges which make up molecular graph will be converted into two
#       opposing directed edges. Depends on the downstream ML framework to be used.
UNDIRECTED_EDGES_AS_TWO: bool = True
# :param USE_NODE_COORDINATES:
#       If this flag is True, the coordinates of each atom will be calculated for each molecule and the resulting
#       3D coordinate vector will be added as a separate property of the resulting graph dict.
USE_NODE_COORDINATES: bool = True
# :param GRAPH_METADATA_CALLBACKS:
#       This is a dictionary that can be use to define additional information that should be extracted from the 
#       the csv file and to be transferred to the metadata dictionary of the visual graph dataset elements.
#       The keys of this dict should be the string names that the properties will then have in the final metadata 
#       dictionary. The values should be callback functions with two parameters: "mol" is the rdkit molecule object 
#       representation of each dataset element and "data" is the corresponding dictionary containing all the 
#       values from the csv file indexed by the names of the columns. The function itself should return the actual 
#       data to be used for the corresponding custom property. 
GRAPH_METADATA_CALLBACKS = {
    'name':         lambda mol, data: data['Compound'],
    'smiles':       lambda mol, data: data['la_smiles'],
    'target_solv':  lambda mol, data: data['fia_solv-DSDBLYP'],
    'target_gas':   lambda mol, data: data['fia_gas-DSDBLYP'],
    'central_atom': lambda mol, data: data['ca']
}

# == EXPERIMENT PARAMETERS ==

experiment = Experiment.extend(
    'generate_molecule_dataset_from_csv.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

# ~ Adding filters to the dataset processing step
# By adding these specific filters to the pre-processing of the dataset we implement the same processing
# steps described in the original paper which introduces this dataset.


def is_charged(mol, data):
    smiles = data['smiles']
    return '+' in smiles or '-' in smiles


def is_adjoined_mixture(mol, data):
    smiles = data['smiles']
    return '.' in smiles


def no_carbon(mol, data):
    smiles = data['smiles']
    return 'C' not in smiles


def filter_moltype(mol, data):
    moltype = data['mol_type']
    core_atom = data['ca']
    return moltype != 'fia44k'


@experiment.hook('modify_filter_callbacks')
def add_filters(e: Experiment, filter_callbacks: t.List[t.Callable]):
    # filter_callbacks.append(is_charged)
    filter_callbacks.append(is_adjoined_mixture)
    filter_callbacks.append(filter_moltype)
    return filter_callbacks


experiment.run_if_main()
