"""
Generates the logP dataset
"""
import os
import pathlib
import typing as t

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from rdkit import Chem

from visual_graph_datasets.processing.molecules import chem_prop, chem_descriptor, list_identity
from visual_graph_datasets.processing.molecules import apply_atom_callbacks, apply_bond_callbacks
from visual_graph_datasets.processing.molecules import mol_from_smiles
from visual_graph_datasets.processing.molecules import OneHotEncoder
from visual_graph_datasets.processing.molecules import MoleculeProcessing
from visual_graph_datasets.processing.molecules import crippen_contrib, lasa_contrib, tpsa_contrib
from visual_graph_datasets.processing.molecules import gasteiger_charges, estate_indices

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
CSV_FILE_NAME: str = 'source/aqsoldb.csv'
# :param SMILES_COLUMN_NAME:
#       This has to be the string name of the CSV column which contains the SMILES string representation of
#       the molecule.
INDEX_COLUMN_NAME: t.Optional[str] = None
# :param TARGET_TYPE:
#       This has to be the string name of the type of dataset that the source file represents. The valid 
#       options here are "regression" and "classification"
SMILES_COLUMN_NAME: str = 'SMILES'
# :param TARGET_COLUMN_NAMES:
#       This has to be a list of string column names within the source CSV file, where each name defines 
#       one column that contains a target value for each row. In the regression case, this may be multiple 
#       different regression targets for each element and in the classification case there has to be one 
#       column per class.
TARGET_COLUMN_NAMES: t.List[str] = ['Solubility']
# :param SPLIT_COLUMN_NAMES:
#       The keys of this dictionary are integers which represent the indices of various train test splits. The
#       values are the string names of the columns which define those corresponding splits. It is expected that
#       these CSV columns contain a "1" if that corresponding element is considered as part of the training set
#       of that split and "0" if it is part of the test set.
#       This dictionary may be empty and then no information about splits will be added to the dataset at all.
SPLIT_COLUMN_NAMES: t.Dict[int, str] = {
    0: 'split'
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
DATASET_NAME: str = 'aqsoldb'
# :parm DATASET_META:
#       This dict will be converted into the .meta.yml file which will be added to the final visual graph dataset
#       folder. This is an optional file, which can add additional meta information about the entire dataset
#       itself. Such as documentation in the form of a description of the dataset etc.
DATASET_META: t.Optional[dict] = {
    'version': '0.2.0',
    'changelog': [
        '0.1.0 - 29.01.2023 - initial version',
        '0.2.0 - 24.03.2023 - Changed the dataset generation to now include .meta.yml dataset metadata '
        'file and process.py standalone pre-processing module in the dataset folder as well.'
    ],
    'description': (
        'Dataset consisting of roughly 10_000 molecular graphs annotated with measured values of their '
        'corresponding solubility (logS) value in water.'
    ),
    'references': [
        'Library used for the processing and visualization of molecules. https://www.rdkit.org/',
    ],
    'visualization_description': (
        'Molecular graphs generated by RDKit based on the SMILES representation of the molecule.'
    ),
    'target_descriptions': {
        0: 'measured logS values of the molecules solubility in Water. (unmodified)'
    }
}


# CUSTOM PROCRESSING
# Very importantly for this dataset we need to define a Processing class that is slightly different than the 
# default Processing class that is defined in the parent experiment. The reason for this is that the default 
# processing defined a node attribute "crippen_contribution" which are the linear contributions of each atom 
# to the crippen logp value.
# In the logp dataset we want to predict this crippen logp value. Therefore including these attributes would 
# leak the label.
class ReducedMoleculeProcessing(MoleculeProcessing):

    node_attribute_map = {
        'symbol': {
            'callback': chem_prop('GetSymbol', OneHotEncoder(
                ['H', 'C', 'N', 'O', 'B', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At'],
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
        # 'is_aromatic': {
        #     'callback': chem_prop('GetIsAromatic', list_identity),
        #     'description': 'Boolean flag of whether the atom is aromatic',
        # },
        # 'is_in_ring': {
        #     'callback': chem_prop('IsInRing', list_identity),
        #     'description': 'Boolean flag of whether atom is part of a ring'
        # },
        # 'crippen_contributions': {
        #     'callback': crippen_contrib(),
        #     'description': 'The crippen logP contributions of the atom as computed by RDKit'
        # },
        # 'tpsa_contribution': {
        #     'callback': tpsa_contrib(),
        #     'description': 'Contribution to TPSA as computed by RDKit',
        # },
        # 'lasa_contribution': {
        #     'callback': lasa_contrib(),
        #     'description': 'Contribution to ASA as computed by RDKit'
        # },
        # 'gasteiger_charge': {
        #     'callback': gasteiger_charges(),
        #     'description': 'The partial gasteiger charge attributed to atom as computed by RDKit'
        # },
        # 'estate_indices': {
        #     'callback': estate_indices(),
        #     'description': 'EState index as computed by RDKit'
        # }
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
        # 'is_aromatic': {
        #     'callback': chem_prop('GetIsAromatic', list_identity),
        #     'description': 'boolean flag of whether bond is aromatic',
        # },
        # 'is_in_ring': {
        #     'callback': chem_prop('IsInRing', list_identity),
        #     'description': 'boolean flag of whether bond is part of ring',
        # },
        # 'is_conjugated': {
        #     'callback': chem_prop('GetIsConjugated', list_identity),
        #     'description': 'boolean flag of whether bond is conjugated'
        # }
    }

    graph_attribute_map = {}

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


@experiment.hook('modify_filter_callbacks')
def add_filters(e: Experiment, filter_callbacks: t.List[t.Callable]):
    filter_callbacks.append(is_charged)
    filter_callbacks.append(is_adjoined_mixture)
    return filter_callbacks

# CUSTOM COMPUTED TARGET VALUES
# For this value we dont actually need the source CSV file to contain a target value column. The target value is
# a value that can be deterministically computed with RDKit. It is the crippen logp algorithm result.
# In this section we hijack the existing hooks for modifying graph data and metadata to inject these  
# computed values instead of the dataset's own target values

@experiment.hook('additional_graph_data')
def additional_graph_data(e: Experiment,
                          additional_graph_data: dict,
                          mol: t.Any,
                          data: dict):
    value = Chem.Crippen.MolLogP(mol)
    additional_graph_data['graph_labels'] = [value]
    return additional_graph_data


experiment.run_if_main()
