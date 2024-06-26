import os
import time
import pathlib
import datetime
import typing as t

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Geometry import Point3D
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from visual_graph_datasets.visualization.molecules import visualize_molecular_graph_from_mol
from visual_graph_datasets.processing.base import create_processing_module
from visual_graph_datasets.processing.molecules import MoleculeProcessing
from visual_graph_datasets.processing.molecules import OneHotEncoder, list_identity, chem_prop
from visual_graph_datasets.processing.molecules import crippen_contrib, tpsa_contrib, lasa_contrib, gasteiger_charges, estate_indices
from visual_graph_datasets.data import VisualGraphDatasetWriter, VisualGraphDatasetReader
from visual_graph_datasets.util import dynamic_import

PATH = pathlib.Path(__file__).parent.absolute()
ASSETS_PATH = os.path.join(PATH, 'assets')

# == SOURCE PARAMETERS ==

NPZ_PATH: str = os.path.join(ASSETS_PATH, 'rmd17_aspirin.npz')
DISTANCE_THRESHOLD: float = 2.1
EDGE_INDICES = None
# EDGE_INDICES: np.ndarray = np.array([
#     [0, 2],
#     [0, 5],
#     [0, 14],
#     [1, 3],
#     [1, 6],
#     [1, 15],
#     [2, 3],
#     [2, 16],
#     [3, 17],
#     [4, 11],
#     [4, 18], 
#     [4, 19], 
#     [4, 20],
#     [5, 6], 
#     [5, 10],
#     [6, 12],
#     [7, 10],
#     [8, 11],
#     [9, 10],
#     [9, 13],
#     [11, 12], 
#     [12, 11]
# ])

# == PROCESSING PARAMETERS ==

CHUNK_SIZE: int = 10_000
NUM_ELEMENTS: int = 10_000

class VgdMoleculeProcessing(MoleculeProcessing):

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
        # 'hybridization': {
        #     'callback': chem_prop('GetHybridization', OneHotEncoder(
        #         [2, 3, 4, 5, 6],
        #         add_unknown=True,
        #         dtype=int,
        #     )),
        #     'description': 'one-hot encoding of atom hybridization',
        # },
        'mass': {
            'callback': chem_prop('GetMass', list_identity),
            'description': 'The mass of the atom'
        },
        'charge': {
            'callback': chem_prop('GetFormalCharge', list_identity),
            'description': 'The charge of the atom',
        },
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
    }

    graph_attribute_map = {}
    
    
PROCESSING = VgdMoleculeProcessing()


__DEBUG__ = True

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

def mol_from_coords(atom_types: np.ndarray,
                         coords: np.ndarray,
                         edge_indices: t.Optional[np.ndarray] = None,
                         threshold: float = 1.5,
                         ) -> Chem.Mol:
    mol = Chem.RWMol()
    
    for atom_number, coord in zip(atom_types, coords):
        atom = Chem.Atom(int(atom_number))
        mol.AddAtom(atom)
    
    conf = Chem.Conformer(len(atom_types))
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(x, y, z))
    mol.AddConformer(conf, assignId=True)
        
    if edge_indices is not None:
        for i, j in edge_indices.tolist():
            if mol.GetBondBetweenAtoms(i, j) is None:
                mol.AddBond(i, j, Chem.BondType.SINGLE)
                
    else:
        for i in range(mol.GetNumAtoms()):
            for j in range(i + 1, mol.GetNumAtoms()):
                dist = la.norm(coords[i] - coords[j])
                if dist < threshold:  # This threshold should be adjusted based on the element types
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
    
    return mol


@experiment
def experiment(e: Experiment):
    
    e.log('starting experiment...')
    
    e.log('loading dataset...')
    data = np.load(NPZ_PATH)
    e.log(f'loaded data with keys: {",".join(data.keys())}')

    num_elements = len(data['coords'])
    nuclear_charges = data['nuclear_charges']
    e.log(f'molecule with {len(nuclear_charges)} atoms')
    e.log(f'dataset with {num_elements} configurations')
    
    if e.NUM_ELEMENTS: 
        num_elements = min(num_elements, e.NUM_ELEMENTS)
        e.log(f'creating a dataset from {num_elements} elements')
    
    e.log('creating dataset path...')
    dataset_path = os.path.join(e.path, 'dataset')
    os.mkdir(dataset_path)
    
    e.log('exporting processing persistently to disk...')
    e.log(f'generating pre-processing python module for {PROCESSING.__class__}...')
    module_code = create_processing_module(PROCESSING)
    for path in [os.path.join(p, 'process.py') for p in [e.path, dataset_path]]:
        with open(path, mode='w') as file:
            file.write(module_code)
    
    e.log('creating dataset writer...')
    writer = VisualGraphDatasetWriter(
        path=dataset_path,
        chunk_size=e.CHUNK_SIZE,
    )
    
    e.log('starting dataset processing...')
    start_time = time.time()
    index = 0

    for coords, energy in zip(data['coords'], data['old_energies']):
        
        mol = mol_from_coords(
            nuclear_charges, 
            coords, 
            threshold=e.DISTANCE_THRESHOLD,
            edge_indices=e.EDGE_INDICES,
        )
        num_atoms = len(mol.GetAtoms())
        
        if not mol:
            e.log(' ! skipping configuration due to invalid molecule')
            
        smiles = Chem.MolToSmiles(mol)
        
        # edge_indices = []
        # for i in range(num_atoms):
        #     for j in range(i, num_atoms):
        #         if i != j:
        #             edge_indices.append([i, j])
                    
        # edge_indices = np.array(edge_indices, dtype=int)
        
        graph_labels = np.array([float(energy) + 406738])
        PROCESSING.create(
            value=mol,
            index=index,
            additional_graph_data={
                'graph_labels': graph_labels, 
            #    'edge_indices': edge_indices
            },
            additional_metadata={'target': graph_labels},
            node_coordinates=coords,
            writer=writer,
        )
    
        if index % 250 == 0:
            time_passed = time.time() - start_time
            num_remaining = num_elements - (index + 1)
            time_per_element = time_passed / (index + 1)
            time_remaining = time_per_element * num_remaining
            eta = datetime.datetime.now() + datetime.timedelta(seconds=time_remaining)
            e.log(f' * ({index}/{num_elements}) processed'
                  f' - time passed: {time_passed:.1f}s'
                  f' - time remaining: {time_remaining:.1f}s'
                  f' - eta: {eta:%Y-%m-%d %H:%M}'
                  f' - smiles: {smiles}'
                  f' - target: {graph_labels}')
    
        index += 1
        
        if index > num_elements:
            break

    e.log('finished dataset processing...')
    e.log('closing dataset writer...')


@experiment.analysis
def analysis(e: Experiment):
    
    e.log('starting analysis...')
    
    e.log('loading the processing instance...')
    module = dynamic_import(e.path, 'process.py')
    processing = module.processing
    
    e.log('loading the dataset...')
    reader = VisualGraphDatasetReader()
    dataset = reader.read()
    e.log(f'loaded dataset with {len(dataset)} elements')
    

experiment.run_if_main()