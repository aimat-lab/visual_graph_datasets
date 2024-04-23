"""
This experiment processes the aggregators dataset. This is orignally a dataset consisting of about 300k molecules 
that are annotated with a binary classification label that identifies them as either an aggregator or a 
non-aggregator.

CHANGELOG

0.1.0 - 23.02.2023 - initial version
"""
import os
import pathlib
import typing as t

# from pycomex.experiment import SubExperiment
# from pycomex.util import Skippable
import rdkit.Chem as Chem
from visual_graph_datasets.processing.base import identity, list_identity
from visual_graph_datasets.processing.base import create_processing_module
from visual_graph_datasets.processing.base import ProcessingError
from visual_graph_datasets.processing.molecules import chem_prop, chem_descriptor
from visual_graph_datasets.processing.molecules import OneHotEncoder
from visual_graph_datasets.processing.molecules import MoleculeProcessing
from visual_graph_datasets.processing.molecules import crippen_contrib, lasa_contrib, tpsa_contrib
from visual_graph_datasets.processing.molecules import gasteiger_charges, estate_indices
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace


PATH = pathlib.Path(__file__).parent.absolute()
ASSETS_PATH = os.path.join(PATH, 'assets')

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
CSV_FILE_NAME: str = os.path.join(ASSETS_PATH, 'aggregators_binary.csv')
# :param INDEX_COLUMN_NAME:
#       (Optional) this may define the string name of the CSV column which contains the integer index
#       associated with each dataset element. If this is not given, then integer indices will be randomly
#       generated for each element in the final VGD
INDEX_COLUMN_NAME: t.Optional[str] = None
# :param SMILES_COLUMN_NAME:
#       This has to be the string name of the CSV column which contains the SMILES string representation of
#       the molecule.
SMILES_COLUMN_NAME: str = 'smiles'
# :param TARGET_TYPE:
#       This has to be the string name of the type of dataset that the source file represents. The valid 
#       options here are "regression" and "classification"
TARGET_TYPE: str = 'classification'
# :param TARGET_COLUMN_NAMES:
#       This has to be a list of string column names within the source CSV file, where each name defines 
#       one column that contains a target value for each row. In the regression case, this may be multiple 
#       different regression targets for each element and in the classification case there has to be one 
#       column per class.
TARGET_COLUMN_NAMES: t.List[str] = ['aggregator', 'nonaggregator']

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
DATASET_CHUNK_SIZE: t.Optional[int] = 10_000
# :param DATASET_NAME:
#       The name given to the visual graph dataset folder which will be created.
DATASET_NAME: str = 'aggregators_binary'
# :param IMAGE_WIDTH:
#       The width molecule visualization PNG image
IMAGE_WIDTH: int = 1000
# :param IMAGE_HEIGHT:
#       The height of the molecule visualization PNG image
IMAGE_HEIGHT: int = 1000
# :parm DATASET_META:
#       This dict will be converted into the .meta.yml file which will be added to the final visual graph dataset
#       folder. This is an optional file, which can add additional meta information about the entire dataset
#       itself. Such as documentation in the form of a description of the dataset etc.
DATASET_META: t.Optional[dict] = {
    'version': '0.1.0',
    # A list of strings where each element is a description about the changes introduced in a newer
    # version of the dataset.
    'changelog': [
        '0.1.0 - 29.01.2023 - initial version'
    ],
    # A general description about the dataset, which gives a general overview about where the data was
    # sampled from, what the input features look like, what the prediction target is etc...
    'description': (
        'large dataset consisting of organic compounds which are divided into two classes: aggregators '
        'and non-aggregators.'
    ),
    # A list of informative strings (best case containing URLS) which are used as references for the
    # dataset. This could for example be a reference to a paper where the dataset was first introduced
    # or a link to site where the raw data can be downloaded etc.
    'references': [
        'Library used for the processing and visualization of molecules. https://www.rdkit.org/',
    ],
    # A small description about how to interpret the visualizations which were created by this dataset.
    'visualization_description': (
        'Molecular graphs generated by RDKit based on the SMILES representation of the molecule.'
    ),
    # A dictionary, where the keys should be the integer indices of the target value vector for the dataset
    # and the values should be string descriptions of what the corresponding target value is about.
    'target_descriptions': {
        0: 'one-hot: aggregator class',
        1: 'one-hot: non-aggregator class'
    }
}
GRAPH_METADATA_CALLBACKS = {
    'name': lambda mol, data: data['name'],
    'label': lambda mol, data: data['label'],
    'smiles': lambda mol, data: data['smiles'],
}

class AggregatorMoleculeProcessing(MoleculeProcessing):

    node_attribute_map = {
        'symbol': {
            'callback': chem_prop('GetSymbol', OneHotEncoder(
                ['H', 'C', 'N', 'O', 'B', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At'],
                add_unknown=True,
                dtype=str
            )),
            'description': 'one-hot encoding of atom type',
            'is_type': True,
            'encodes_symbol': True, 
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
        # 'charge': {
        #     'callback': chem_prop('GetFormalCharge', list_identity),
        #     'description': 'The charge of the atom',
        # },
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
        # 'gasteiger_charge': {
        #     'callback': gasteiger_charges(),
        #     'description': 'The partial gasteiger charge attributed to atom as computed by RDKit'
        # },
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
            'is_type': True,
            'encodes_bond': True,
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
PROCESSING = AggregatorMoleculeProcessing()
# :param UNDIRECTED_EDGES_AS_TWO:
#       If this flag is True, the undirected edges which make up molecular graph will be converted into two
#       opposing directed edges. Depends on the downstream ML framework to be used.
UNDIRECTED_EDGES_AS_TWO: bool = True
# :param USE_NODE_COORDINATES:
#       If this flag is True, the coordinates of each atom will be calculated for each molecule and the resulting
#       3D coordinate vector will be added as a separate property of the resulting graph dict.
USE_NODE_COORDINATES: bool = False

# == EVALUATION PARAMETERS ==
# These parameters control the evaluation process which included the plotting of the dataset statistics 
# after the dataset has been completed for example.

# :param EVAL_LOG_STEP:
#       The number of iterations after which to print a log message
EVAL_LOG_STEP = 100
# :param NUM_BINS:
#       The number of bins to use for the histogram 
NUM_BINS = 10
# :param PLOT_COLOR:
#       The color to be used for the plots
PLOT_COLOR = 'gray'

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

def max_graph_size(mol, data):
    """
    We want to filter the very big molecules because we will likely not be able to properly visualize
    those for the explanations anyways.
    """
    return len(mol.GetAtoms()) >= 100


def is_charged(mol, data):
    smiles = data['smiles']
    return '+' in smiles or '-' in smiles


def is_adjoined_mixture(mol, data):
    smiles = data['smiles']
    return '.' in smiles


# Here we add our own custom filter to the list of filters that are applied on the dataset
@experiment.hook('modify_filter_callbacks')
def add_filters(e: Experiment, filter_callbacks: t.List[t.Callable]):
    # filter_callbacks.append(is_charged)
    filter_callbacks.append(is_adjoined_mixture)
    return filter_callbacks


experiment.run_if_main()
