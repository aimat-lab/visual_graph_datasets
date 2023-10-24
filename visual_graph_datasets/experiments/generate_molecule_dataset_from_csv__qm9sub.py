"""
This experiment processes the QM9 dataset. This is a dataset of roughly 100k small organic molecules
(molecules with at most 9 atoms - hence the name). It is annotated with various generic atom properties
which were derived through quantum chemical simulations (DFT).

This experiment 

**CHANGELOG**

0.1.0 - 24.10.23 - initial version
"""
import os
import json
import pathlib
import typing as t

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
CSV_FILE_NAME: str = os.path.join(ASSETS_PATH, 'qm9.csv')
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
# A,B,C,mu,alpha,homo,lumo,gap,r2,zpve,u0,u298,h298,g298,cv,u0_atom,u298_atom,h298_atom,g298_atom
TARGET_COLUMN_NAMES: t.List[str] = [
    'mu',
    'alpha',
    'homo',
    'lumo',
    'gap',
    'r2',
    'zpve',
    'u0',
    'cv',
]
# :param SUBSET:
#       Optional. This can be used to set a number of elements after which to terminate the processing procedure. 
#       If this is None, the whole dataset will be processed. This feature can be useful if only a certain 
#       part of the datase should be processed or for testing reasons for example.
SUBSET: t.Optional[int] = 25_000

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
DATASET_NAME: str = 'qm9sub'
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
        '0.1.0 - 24.10.2023 - initial version'
    ],
    # A general description about the dataset, which gives a general overview about where the data was
    # sampled from, what the input features look like, what the prediction target is etc...
    'description': (
        'A large dataset consisting of roughly 100k small organic molecules. Molecules have at most '
        '9 molecules, hence the name of the dataset. All molecules are annotated with various general '
        'molecular properties which were derived from quantum chemical simulations (DFT).'
    ),
    # A list of informative strings (best case containing URLS) which are used as references for the
    # dataset. This could for example be a reference to a paper where the dataset was first introduced
    # or a link to site where the raw data can be downloaded etc.
    'references': [
        'Library used for the processing and visualization of molecules. https://www.rdkit.org/',
        'Dataset source: https://paperswithcode.com/dataset/qm9 - http://quantum-machine.org/datasets/'
    ],
    # A small description about how to interpret the visualizations which were created by this dataset.
    'visualization_description': (
        'Molecular graphs generated by RDKit based on the SMILES representation of the molecule.'
    ),
    # A dictionary, where the keys should be the integer indices of the target value vector for the dataset
    # and the values should be string descriptions of what the corresponding target value is about.
    'target_descriptions': {
        0: ('mu (D) - The electric dipole moment of the molecule '
            'representing the separation of positive and negative charges '
            'within the molecule.'),
        1: ('alpha (Angstrom^3) - The polarizability of the molecule, '
            'indicating its ability to undergo induced changes in electron '
            'distribution in response to an external electric field.'),
        2: ('homo (eV) - The highest occupied molecular orbit'),
        3: ('lumo (eV) - The lowest occupied moleculer orbit'),
        4: ('gap (eV) - The energy gap between the LUMO and the HOMO energy '
            'of the molecule.'),
        5: ('r2 (Bohr^2) - Electronic spatial extent. A measure of the '
            'spatial extent of the electronic distribution of the molecule'),
        6: ('zpve (Ha) - The zero point vibrational energy contribution to '
            'the vibrational modes of the molecule. It represents the lowest '
            'possible energy the molecule can have due to quantum motion.'),
        7: ('u0 (Ha) - Atomization energy. The energy required to completely '
             'separate all the atoms in a molecule into isolated gaseous atoms.'),
        8: ('cv (cal/mol*K) - The heat capacity of the molecule at 298K, '
             'indicating the amount of heat energy required to raise the temperature '
             'of the molecule by 1K.'),
    }
}

experiment = Experiment.extend(
    'generate_molecule_dataset_from_csv.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

experiment.run_if_main()