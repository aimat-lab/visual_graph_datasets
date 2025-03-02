"""
Unittests for ``processing.molecules``
"""
import os
import sys
import pytest
import subprocess

import click
import click.testing
import numpy as np
import matplotlib.pyplot as plt
import visual_graph_datasets.typing as tc
from rich import print as pprint
from visual_graph_datasets.graph import nx_from_graph
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


def test_nx_from_graph():
    
    smiles = 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
    
    processing = MoleculeProcessing()
    graph = processing.process(smiles)
    graph_nx = nx_from_graph(graph)
    
    for e, (i, j) in enumerate(graph['edge_indices']):
        assert np.isclose(graph['node_attributes'][i], graph_nx.nodes[i]['node_attributes']).all()
        assert np.isclose(graph['edge_attributes'][e], graph_nx[i][j]['edge_attributes']).all()
            

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
    
    
class TestMoleculeProcessing:
    
    @pytest.mark.parametrize('smiles_1,smiles_2,match', [
        ('CNC(C)=CCC', 'CNC=C', True),
        ('CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'CN1C=CN=C1', True),
        # Now since we care about the edges, this is not a match anymore because the wrong edge type
        ('CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'CNCO', False),
    ])
    def test_contains_works_with_edges(self, smiles_1, smiles_2, match):
        
        processing = MoleculeProcessing()
        
        graph_1 = processing.process(smiles_1)
        graph_2 = processing.process(smiles_2)
        
        result = processing.contains(
            graph=graph_1,
            subgraph=graph_2,
            check_edges=True,
        )
        print(smiles_1, smiles_2, result)
        assert result == match


    @pytest.mark.parametrize('smiles_1,smiles_2,match', [
        # Here we just randomly check some cases with the caffeine molecule.
        ('CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'CN1C=CN=C1', True),
        ('CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'CNC=O', True),
        # Technically this is the same as the previous one, but one edge is not correct, which should 
        # not be a problem since we are not checking edges
        ('CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'CNCO', True),
        # This structure is for sure not contained in the main graph so the result should be False
        ('CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'CNC(=O)NN', False),
    ])
    def test_contains_basically_works(self, smiles_1, smiles_2, match):

        processing = MoleculeProcessing()
        
        graph_1 = processing.process(smiles_1)
        graph_2 = processing.process(smiles_2)
        # I am not
        result = processing.contains(
            graph=graph_1, 
            subgraph=graph_2,
            check_edges=False,
        )
        print(smiles_1, smiles_2, result)
        assert result == match


    def test_extract_basically_works(self):
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


    def test_unprocess_basically_works(self):
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
        
        
    def test_unprocess_with_charged_atoms_works(self):
        """
        30.01.24 - Extended the molecule unprocess method so that it now should be able to handle charged 
        atoms as well
        """
        smiles = 'CC[N+](=O)[O-]'
        
        processing = MoleculeProcessing()
        graph = processing.process(smiles)
        unprocessed = processing.unprocess(graph, clear_aromaticity=False)

        assert smiles == unprocessed


    def test_symbol_encoder_works(self):
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
        

    def test_derived_atom_features_basically_working(self):
        """
        There are some more advanced atom features which actually require access to the Mol object on top 
        of the atom index at the time of creation, which is what is being tested here.
        """
        
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


    def test_derived_atom_features(self):
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


    def test_process_basically_works(self):
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


    def test_visualize_basically_works(self):
        """
        21.03.2023 - ``MoleculeProcessing.visualize`` is supposed to turn a given smiles string into a numpy
        array which represents an rgb image and thus has the shape (width, height, 3).
        """
        processing = MoleculeProcessing()

        array = processing.visualize(SMILES, width=500, height=500)
        assert isinstance(array, np.ndarray)
        assert array.shape == (500, 500, 3)


    def test_create_basically_works(self):
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
    
    def test_get_description_map(self):
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
    
    def test_get_num_node_attributes_basically_works(self):
        """
        The ``MoleculeProcessing.get_num_node_attributes`` method should return the integer number of node 
        attributes/features that the nodes of the processed graphs will have.
        """
        processing = MoleculeProcessing()
        
        # The default MoleculeProcessing class defines 10 node attributes
        assert hasattr(processing, 'get_num_node_attributes')
        assert processing.get_num_node_attributes() == 10


    def test_get_num_edge_attributes_basically_works(self):
        """
        The ``MoleculeProcessing.get_num_edge_attributes`` method should return the integer number of edge
        attributes/features that the edges of the processed graphs will have.
        """
        processing = MoleculeProcessing()
        
        # The default MoleculeProcessing class defines 4 edge attributes
        assert hasattr(processing, 'get_num_edge_attributes')
        assert processing.get_num_edge_attributes() == 4
        
    def test_node_atoms_and_bond_edges_attributes_work(self):
        """
        In the newest version, the result of the "process" method should also include the chemistry domain 
        specific optional attributes "node_atoms" and "bond_edges" which are lists of string representations
        of the atom types and bond types respectively.
        """
        processing = MoleculeProcessing()

        smiles = 'CCCNC(=O)'
        graph = processing.process(smiles)
        pprint(graph)
        
        assert 'node_atoms' in graph
        assert isinstance(graph['node_atoms'], np.ndarray) 
        assert len(graph['node_atoms']) == len(graph['node_indices'])
        # We know that there are only the following symbols in the given molecule
        assert set(graph['node_atoms']) == {'C', 'N', 'O'}
        
        assert 'edge_bonds' in graph
        assert isinstance(graph['edge_bonds'], np.ndarray) 
        assert len(graph['edge_bonds']) == len(graph['edge_bonds'])
        # We know that there are only the following edge types in the molecule
        assert set(graph['edge_bonds']) == {'S', 'D'}
        