"""
Unittests for ``processing.molecules``
"""
import os
import sys
import subprocess

import click
import click.testing
import numpy as np
import matplotlib.pyplot as plt
import visual_graph_datasets.typing as tc
from visual_graph_datasets.processing.base import create_processing_module
from visual_graph_datasets.processing.base import EncoderBase
from visual_graph_datasets.processing.molecules import MoleculeProcessing
from visual_graph_datasets.processing.molecules import crippen_contrib
from visual_graph_datasets.processing.molecules import tpsa_contrib
from visual_graph_datasets.processing.molecules import lasa_contrib
from visual_graph_datasets.processing.molecules import gasteiger_charges
from visual_graph_datasets.processing.molecules import estate_indices
from visual_graph_datasets.visualization.molecules import visualize_molecular_graph_from_mol
from visual_graph_datasets.testing import clear_files
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdPartialCharges
from rdkit.Chem import EState

from .util import ARTIFACTS_PATH


SMILES = 'c1ccccc1'


def test_molecule_processing_extract_works():
    """
    11.06.23 - MoleculeProcessing now implements the ``extract`` method which can be used to 
    extract a sub structure from an existing graph dict representation of a molecule in the 
    form of a SMILES string and a graph dict of just the sub structure
    """
    smiles = 'Cn1c(=O)c2c(ncn2C)n(C)c1=O' # has 14 atoms
    
    processing = MoleculeProcessing()
    graph = processing.process(smiles)
    assert len(graph['node_indices']) == 14
    
    mask = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    smiles_sub, graph_sub = processing.extract(graph, mask)
    assert isinstance(smiles_sub, str)
    assert len(smiles_sub) != 0
    assert len(smiles_sub) != len(smiles)
    
    # Plotting the graph and the sub graph to visuall confirm that it works
    fig, (ax, ax_sub) = plt.subplots(ncols=2, nrows=1, figsize=(20, 10))
    ax.set_title('original molecule with extraction site marked')
    node_positions, _ = visualize_molecular_graph_from_mol(
        ax=ax,
        mol=Chem.MolFromSmiles(smiles),
        image_height=1000, image_width=1000,
    )
    ax_sub.set_title('extracted section')
    visualize_molecular_graph_from_mol(
        ax=ax_sub,
        mol=Chem.MolFromSmiles(smiles_sub),
        image_width=1000, image_height=1000,
    )
    for index in graph_sub['node_indices_original']:
        ax.scatter(*node_positions[index], color='lightgray', s=500, zorder=-1)
    fig_path = os.path.join(ARTIFACTS_PATH, 'molecule_processing_extract_works')
    fig.savefig(fig_path)


def test_molecule_processing_unprocess_works():
    """
    11.06.23 - MoleculeProcessing now implements the ``unprocess`` method, which is supposed to 
    take a graph dict representation as an argument and return it's corresponding SMILES 
    representation.
    """
    smiles = 'Cn1c(=O)c2c(ncn2C)n(C)c1=O'
    
    processing = MoleculeProcessing()
    graph = processing.process(smiles)
    unprocessed = processing.unprocess(graph, clear_aromaticity=False)
    # This is the most effective check we can perform, which effectively checks if the 
    # if the smiles representation stays the same through the process -> unprcess chain.
    assert smiles == unprocessed
    
    
def test_molecule_processing_unprocess_charged_atoms_works():
    """
    30.01.24 - Extended the molecule unprocess method so that it now should be able to handle charged 
    atoms as well
    """
    smiles = 'CC[N+](=O)[O-]'
    
    processing = MoleculeProcessing()
    graph = processing.process(smiles)
    unprocessed = processing.unprocess(graph, clear_aromaticity=False)

    assert smiles == unprocessed


def test_molecule_processing_symbol_encoder_works():
    """
    11.06.23 - MoleculeProcessing should now have an attribute symbol_encoder which is an EncoderBase 
    derived object that can be used to encode and decode between a atom symbol string and the one-hot 
    representation as a numeric feature vector.
    """
    processing = MoleculeProcessing()
    assert processing.symbol_encoder is not None
    assert isinstance(processing.symbol_encoder, EncoderBase)
    
    value = 'C'
    
    # Since the interface of this standard class can change we only make the most simple 
    # check here we ensure that the result is in fact a non-empty vector
    symbol_encoder = processing.symbol_encoder
    encoded = symbol_encoder.encode('C')
    assert isinstance(encoded, list)
    assert len(encoded) != 0
    
    # We only perform a brief individual check here to confirm that the encoding and 
    # decoding acutally results in the exact original value.
    decoded = symbol_encoder.decode(encoded)
    assert decoded == value
    

def test_molecule_processing_derived_atom_features_working():

    class CustomProcessing(MoleculeProcessing):

        node_attribute_map = {
            'crippen_contribution': {
                'callback': crippen_contrib(),
                'description': 'Contributions to the Crippen LogP value.'
            },
            'tpsa_contribution': {
                'callback': tpsa_contrib(),
                'description': 'Contributions to TPSA',
            },
            'lasa_contribution': {
                'callback': lasa_contrib(),
                'description': 'Contributions to LabuteASA',
            },
            'gasteiger_charges': {
                'callback': gasteiger_charges(),
                'description': 'partial gasteiger charge attributed to this atom.'
            },
            'estate_indices': {
                'callback': estate_indices(),
                'description': 'EState Indices',
            }
        }

    processing = CustomProcessing()
    graph = processing.process(SMILES)
    assert isinstance(graph, dict)


def test_derived_atom_features():
    """
    30.05.23 - This is not a unittest but rather me trying out additional methods which can be used to
    derive more atom features for the molecule processing.
    """
    smiles = 'C1=CC=CC=C1CCCC(CN)CCC=O'

    mol = Chem.MolFromSmiles(smiles)
    rdPartialCharges.ComputeGasteigerCharges(mol)
    for atom in mol.GetAtoms():
        print(atom.GetProp("_GasteigerCharge"))

    crippen_contribs = rdMolDescriptors._CalcCrippenContribs(mol)
    print(crippen_contribs)

    tpsa_contribs = rdMolDescriptors._CalcTPSAContribs(mol)
    print(tpsa_contribs)

    lasa_contribs = list(rdMolDescriptors._CalcLabuteASAContribs(mol)[0])
    print(lasa_contribs)

    estate_indices = EState.EStateIndices(mol)
    print(estate_indices)


def test_molecule_processing_process_basically_works():
    """
    21.03.2023 - ``MoleculeProcessing.process()`` should turn a smiles string into a GraphDict
    representation.
    """
    processing = MoleculeProcessing()

    # This should result in a valid graph dict representation!
    g = processing.process(SMILES)
    assert isinstance(g, dict)
    fields = ['node_indices', 'node_attributes', 'edge_indices', 'edge_attributes']
    for field in fields:
        assert field in g
        assert len(g[field]) != 0

    # This function will check the correctness of it actually being a valid graph dict more in-depth
    tc.assert_graph_dict(g)


def test_molecule_processing_visualize_basically_works():
    """
    21.03.2023 - ``MoleculeProcessing.visualize`` is supposed to turn a given smiles string into a numpy
    array which represents an rgb image and thus has the shape (width, height, 3).
    """
    processing = MoleculeProcessing()

    array = processing.visualize(SMILES, width=500, height=500)
    assert isinstance(array, np.ndarray)
    assert array.shape == (500, 500, 3)


def test_molecule_processing_create_basically_works():
    """
    21.03.2023 - ``MoleculeProcessing.create`` is supposed to create a JSON metadata file and a PNG
    visualization of a molecule given the smiles string.
    """
    processing = MoleculeProcessing()

    # This method should create 3 files in the test artifacts folder:
    files = ['1.png', '1.json']
    file_paths = [os.path.join(ARTIFACTS_PATH, file) for file in files]
    clear_files(file_paths)
    processing.create(
        value=SMILES,
        name='molecule',
        index='1',
        width=500,
        height=500,
        output_path=ARTIFACTS_PATH,
    )
    # For each file we check if it exists and we make sure it is not empty
    for file_path in file_paths:
        assert os.path.exists(file_path)
        with open(file_path, mode='rb') as file:
            content = file.read()
            assert not len(content) < 10


def test_molecule_processing_get_description_map():
    """
    21.03.2023 - ``MoleculeProcessing.get_description_map`` is supposed to return a dictionary with the
    three keys node_attributes, edge_attributes and graph_attributes. Each one of these is a dict as well
    which has integer keys that represent the indices of the corresponding vectors and the values are
    natural language descriptive strings for each vector element.
    """
    processing = MoleculeProcessing()

    description_map = processing.get_description_map()
    print(description_map)
    assert isinstance(description_map, dict)


def test_molecule_processing_command_line_basically_works():
    """
    21.03.2023 - ``MoleculeProcessing`` inherits from click.MultiCommand and should thus be directly usable
    as a command line interface.
    """
    runner = click.testing.CliRunner()
    processing = MoleculeProcessing()

    # The simplest test is to try and show the help message of the CLI
    result = runner.invoke(processing, ['--help'])
    print(result.output)
    assert result.exit_code == 0
    assert 'help' in result.output
    # These three commands absolutely have to be listed within the help text
    commands = ['create', 'process', 'visualize']
    for command in commands:
        assert command in result.output


def test_molecule_processing_create_module():
    """
    21.03.2023 - ``create_processing_module`` should be able to directly convert a ``MoleculeProcessing``
    instance into generated python code for a standalone python module, acting as a command line application,
    that implements the same processing functionality.
    """
    processing = MoleculeProcessing()

    module_string = create_processing_module(processing, )
    path = os.path.join(ARTIFACTS_PATH, 'process_molecules.py')
    with open(path, mode='w') as file:
        file.write(module_string)

    # It should be possible to invoke the module directly to access the command line functionality
    help_command = f'{sys.executable} {path} --help'
    proc = subprocess.run(
        help_command,
        shell=True,
        stdout=subprocess.PIPE,
        cwd=ARTIFACTS_PATH
    )
    output = proc.stdout.decode()
    assert 'help' in output
    # all the default commands also have to be listed in there!
    assert 'process' in output
    assert 'visualize' in output
    assert 'create' in output

