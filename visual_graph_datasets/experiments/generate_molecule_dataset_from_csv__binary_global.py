"""
This experiment processes the synthetic "binary_global" dataset.

This dataset is a synthetic dataset based on the molecular graphs found within the Zinc 250k dataset. However, 
instead of using the original target values, this dataset defines a synthetic binary classification task. This 
task is loosely inspired by drug-likeness estimation. In total, the task consists of 7 different criteria which 
check **global** molecule descriptors against certain threshold values. If the molecule meets all of the criteria 
it is classified as class "1" - otherwise class "0".

**CHANGELOG**

0.1.0 - 23.02.23 - initial version
"""
import os
import json
import pathlib
import typing as t
from typing import Dict, List, Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, QED
from pycomex.functional.experiment import Experiment
from pycomex.util import folder_path, file_namespace

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
CSV_FILE_NAME: str = os.path.join(ASSETS_PATH, 'zinc_250k.csv')
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
TARGET_TYPE: str = 'regression'
# :param TARGET_COLUMN_NAMES:
#       This has to be a list of string column names within the source CSV file, where each name defines 
#       one column that contains a target value for each row. In the regression case, this may be multiple 
#       different regression targets for each element and in the classification case there has to be one 
#       column per class.
TARGET_COLUMN_NAMES: t.List[str] = ['SAS']

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
DATASET_CHUNK_SIZE: t.Optional[int] = 50_000
# :param DATASET_NAME:
#       The name given to the visual graph dataset folder which will be created.
DATASET_NAME: str = 'binary_global'
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
        '0.1.0 - 24.04.2025 - initial version'
    ],
    # A general description about the dataset, which gives a general overview about where the data was
    # sampled from, what the input features look like, what the prediction target is etc...
    'description': (
        'A large dataset of molecular graphs which were generated from the SMILES representations of the '
        'Zinc 250k dataset. This dataset contains a total of 250k molecules annotated with synthetic '
        'binary classification labels.'
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
        0: 'If the molecule does not meet one of the criteria',
        1: 'If the molecule meets all of the synthetic criteria',
    },
    'target_names': {
        0: 'Negative Class',
        1: 'Positive Class',
    }
}

__DEBUG__ = False

experiment = Experiment.extend(
    'generate_molecule_dataset_from_csv.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

# This dict defines the 
DRUGLIKENESS_CRITERIA: Dict[str, Any] = {
    'MW':        ('<=', 500.0),
    'LogP':      ('<=', 5.0),
    'HBD':       ('<=', 5),
    'HBA':       ('<=', 10),
    'TPSA':      ('<=', 140.0),
    'RotB':      ('<=', 10),
    'QED':       ('>=', 0.50),
}

def compute_descriptors(mol: Chem.Mol) -> dict:
    """
    Compute the set of global descriptors used for druglikeness.
    """
    
    return {
        'MW':   Descriptors.MolWt(mol),
        'LogP': Crippen.MolLogP(mol),
        'HBD':  rdMolDescriptors.CalcNumHBD(mol),
        'HBA':  rdMolDescriptors.CalcNumHBA(mol),
        'TPSA': rdMolDescriptors.CalcTPSA(mol),
        'RotB': rdMolDescriptors.CalcNumRotatableBonds(mol),
        'QED':  QED.qed(mol),
    }

def druglikeness_label(mol: Chem.Mol, criteria: dict) -> int:
    """
    Return 1 if molecule meets *all* drug-likeness criteria, else 0.
    
      - MW ≤ 500
      - LogP ≤ 5
      - HBD ≤ 5
      - HBA ≤ 10
      - TPSA ≤ 140
      - RotB ≤ 10
      - QED ≥ 0.50
    """
    desc = compute_descriptors(mol)
    for key, (op, thresh) in criteria.items():
        val = desc[key]
        if   op == '<=' and not (val <=  thresh): return 0
        elif op == '>=' and not (val >=  thresh): return 0
    return 1


# We are going to hijack this function to modify the target value of the dataset elements
# to use the custom synthetic target value that we define above.
@experiment.hook('additional_graph_data', replace=True, default=False)
def additional_graph_data(e: Experiment,
                          additional_graph_data: Dict[str, Any],
                          mol: Chem.Mol,
                          data: Dict[str, Any],
                          **kwargs,
                          ) -> Dict[str, Any]:
    
    # Here we invoke the custom function that we've defined to obtain the binary target label.
    label: int = druglikeness_label(mol, e.DRUGLIKENESS_CRITERIA)
    # In the end we dont need the integer but we need a one-hot encoded vector to represent the 
    # target classification label, which we construct here.
    graph_labels = np.array([1 - label, label], dtype=np.float32)
    
    # we overwrite the target value of the original dataset with the our
    # synthetically calculated target value.
    additional_graph_data.update({
        'graph_labels': graph_labels,
    })
    return additional_graph_data

experiment.run_if_main()