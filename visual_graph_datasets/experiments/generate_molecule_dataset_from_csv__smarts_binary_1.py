"""
This experiment processes the TADF dataset. This is a dataset consisting of ~1M molecular graphs annotated 
with various properties related to their thermally activated delayed flourescence (TADF) behavior. The 
dataset was created during a large scale virtual screening experiment by `Gomez-Bombarelli et al.`_ .
The target value labels were created through DFT calculations.

.. _`Gomez-Bombarelli et al.`: https://www.nature.com/articles/nmat4717

**CHANGELOG**

0.1.0 - 23.02.23 - initial version

0.2.0 - 08.05.23 - moved to the pycomex functional API
"""
import os
import json
import pathlib
import typing as t
from typing import Dict, List, Any

import numpy as np
import rdkit.Chem as Chem
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
DATASET_NAME: str = 'smarts_binary_1'
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
        'A large dataset consisting of roughly half a million molecules annotated with properties '
        'relevant to their thermally activated delayed fluorescent (TADF) behavior. The dataset was '
        'created in a large scale virtual screening application using DFT calculations.'
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
        0: 'pass',
        1: 'pass',
    },
    'target_names': {
        0: 'Negative Class',
        1: 'Positive Class',
    }
}

experiment = Experiment.extend(
    'generate_molecule_dataset_from_csv.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


SMARTS_PATTERNS = {
    'benzene'           : 'c1ccccc1',
    'nitro'             : '[NX3](=O)=O',
    'halogen'           : '[F,Cl,Br,I]',
    'carboxylic_acid'   : 'C(=O)[OH]',
    'tertiary_amine'    : '[$([NX3]([#6])([#6])[#6])]',  # N with three C neighbors
    'sulfoxide'         : '[#16](=O)',
    'ether'             : 'C-O-C',
    'pyridine'          : 'n1ccccc1',
    'furan'             : 'o1cccc1',
    'ketone'            : 'C(=O)[#6]',
    'alcohol'           : '[OX2H]',
    'thiol'             : '[SX2H]'
}

# compile the SMARTS patterns
SMARTS_PATTERNS_COMPILED = {
    name: Chem.MolFromSmarts(smarts)
    for name, smarts in SMARTS_PATTERNS.items()
}


def extract_motifs(mol: Chem.Mol, smarts_patterns: dict = SMARTS_PATTERNS_COMPILED) -> dict:
    """Return a dict mapping motif name → bool for whether
       that SMARTS appears in the molecule."""
    return {
        name: bool(mol.HasSubstructMatch(pat))
        for name, pat in smarts_patterns.items()
    }


def motif_label(mol: Chem.Mol) -> int:
    """
    A rather complicated rule:
    
      Label = 1 if any of these three big clauses holds:
      
        Clause 1:
          benzene AND (nitro OR (halogen AND carboxylic_acid))
        
        Clause 2:
          tertiary_amine AND NOT sulfoxide
          AND (ether OR halogen)
        
        Clause 3:
          (pyridine OR furan) AND ketone AND NOT alcohol
        
      Otherwise 0.
    """
    m = extract_motifs(mol)
    
    c1 = m['benzene'] and (m['nitro'] or (m['halogen'] and m['carboxylic_acid']))
    c2 = m['tertiary_amine'] and not m['sulfoxide'] and (m['ether'] or m['halogen'])
    c3 = (m['pyridine'] or m['furan']) and m['ketone'] and not m['alcohol']
    
    return int(c1 or c2 or c3)


# We are going to hijack this function to modify the target value of the dataset elements
# to use the custom synthetic target value that we define above.
@experiment.hook('additional_graph_data', replace=True, default=False)
def additional_graph_data(e: Experiment,
                          additional_graph_data: Dict[str, Any],
                          mol: Chem.Mol,
                          data: Dict[str, Any],
                          **kwargs,
                          ) -> Dict[str, Any]:
    
    label: int = motif_label(mol)
    graph_labels = np.array([1 - label, label], dtype=np.float32)
    
    additional_graph_data.update({
        'graph_labels': graph_labels,
    })
    return additional_graph_data

experiment.run_if_main()