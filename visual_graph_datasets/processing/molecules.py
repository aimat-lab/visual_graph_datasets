"""
This module specifically deals with the processing of *molecular graphs*
"""
import os
from collections import OrderedDict
import typing as t

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import rdkit.Chem.Descriptors
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from rdkit import Chem
import rdkit.Chem.AllChem
from rdkit.Chem.Draw.rdMolDraw2D import MolDraw2DSVG
from rdkit.Chem import rdDepictor
from rdkit import RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdPartialCharges
from rdkit.Chem import EState
rdDepictor.SetPreferCoordGen(True)
RDLogger.DisableLog('rdApp.*')

import visual_graph_datasets.typing as tc
import visual_graph_datasets.typing as tv
from visual_graph_datasets.processing.base import ProcessingBase
from visual_graph_datasets.processing.base import ProcessingError
from visual_graph_datasets.processing.base import EncoderBase, OneHotEncoder
from visual_graph_datasets.processing.base import Scaler
from visual_graph_datasets.visualization.base import create_frameless_figure
from visual_graph_datasets.visualization.molecules import visualize_molecular_graph_from_mol
from visual_graph_datasets.data import VisualGraphDatasetWriter


def mol_from_smiles(smiles: str
                    ) -> Chem.Mol:
    return Chem.MolFromSmiles(smiles)


def list_identity(value: t.Any) -> t.List[t.Any]:
    return [value]


def estate_indices() -> t.Callable:
    def func(mol: Chem.Mol, atom: Chem.Atom, data: dict = {}):
        # First of all we need to calculate the crippen contributions with an external method IF it does
        # not already exist!
        if not hasattr(mol, 'estate_indices'):
            indices = EState.EStateIndices(mol)
            setattr(mol, 'estate_indices', indices)

        # At this point we can be certain, that every atom has been updated with the corresponding
        # attribute that contains it's crippen contribution values
        contributions = getattr(mol, 'estate_indices')
        return [contributions[atom.GetIdx()]]

    # Now this is important to set this flag here as a dynamic attribute. The MoleculeProcessing will later
    # on check for this attribute and only if it exists will it change the signature with which the
    # callback function is called to actually include the mol object. Otherwise it would just pass the local
    # atom!
    setattr(func, 'requires_molecule', True)

    return func


def gasteiger_charges():
    """
    Returns a node (atom) feature callback which computes the single value for the partial gasteiger
    charge associated with the given atom.

    This function is designed to be used as a standalone callback for the "node_attributes" section of
    MoleculeProcessing instances.

    :returns: a function with the signature [Chem.Mol, Chem.Atom] -> List[float]
    """
    def func(mol: Chem.Mol, atom: Chem.Atom, data: dict = {}):
        # First of all we need to calculate the crippen contributions with an external method IF it does
        # not already exist!
        if not hasattr(mol, 'gasteiger_charges'):
            rdPartialCharges.ComputeGasteigerCharges(mol, nIter=25)
            setattr(mol, 'gasteiger_charges', True)

        # We need to reload the atom here based on the molecule index because the charge computations
        # may not have been updated yet on the instance we have received as the argument.
        atom = mol.GetAtomWithIdx(atom.GetIdx())
        # We actually have to do it with GetProp as we can't access it otherwise...
        value0 = float(atom.GetProp('_GasteigerCharge'))
        value1 = float(atom.GetProp('_GasteigerHCharge'))
        # There is the chance that this could be NaN in which case we have to fix that
        value0 = value0 if not np.isnan(value0) else 0
        value1 = value1 if not np.isnan(value1) else 0

        return [value0, value1]

    # Now this is important to set this flag here as a dynamic attribute. The MoleculeProcessing will later
    # on check for this attribute and only if it exists will it change the signature with which the
    # callback function is called to actually include the mol object. Otherwise it would just pass the local
    # atom!
    setattr(func, 'requires_molecule', True)

    return func


def lasa_contrib() -> t.Callable:
    def func(mol: Chem.Mol, atom: Chem.Atom, data: dict = {}):
        # First of all we need to calculate the crippen contributions with an external method IF it does
        # not already exist!
        if not hasattr(mol, 'lasa_contributions'):
            contributions = list(rdMolDescriptors._CalcLabuteASAContribs(mol)[0])
            setattr(mol, 'lasa_contributions', contributions)

        # At this point we can be certain, that every atom has been updated with the corresponding
        # attribute that contains it's crippen contribution values
        contributions = getattr(mol, 'lasa_contributions')
        return [contributions[atom.GetIdx()]]

    # Now this is important to set this flag here as a dynamic attribute. The MoleculeProcessing will later
    # on check for this attribute and only if it exists will it change the signature with which the
    # callback function is called to actually include the mol object. Otherwise it would just pass the local
    # atom!
    setattr(func, 'requires_molecule', True)

    return func


def tpsa_contrib() -> t.Callable:
    def func(mol: Chem.Mol, atom: Chem.Atom, data: dict = {}):
        # First of all we need to calculate the crippen contributions with an external method IF it does
        # not already exist!
        if not hasattr(mol, 'tpsa_contributions'):
            contributions = rdMolDescriptors._CalcTPSAContribs(mol)
            setattr(mol, 'tpsa_contributions', contributions)

        # At this point we can be certain, that every atom has been updated with the corresponding
        # attribute that contains it's crippen contribution values
        contributions = getattr(mol, 'tpsa_contributions')
        return [contributions[atom.GetIdx()]]

    # Now this is important to set this flag here as a dynamic attribute. The MoleculeProcessing will later
    # on check for this attribute and only if it exists will it change the signature with which the
    # callback function is called to actually include the mol object. Otherwise it would just pass the local
    # atom!
    setattr(func, 'requires_molecule', True)

    return func


def crippen_contrib() -> t.Callable:
    """
    Returns a node (atom) feature callback which computes the 2 crippen logP contribution values given
    the Mol and the Atom object.

    This function is designed to be used as a standalone callback for the "node_attributes" section of
    MoleculeProcessing instances.

    :returns: a function with the signature [Chem.Mol, Chem.Atom] -> List[float]
    """
    def func(mol: Chem.Mol, atom: Chem.Atom, data: dict = {}):
        # First of all we need to calculate the crippen contributions with an external method IF it does
        # not already exist!
        if not hasattr(mol, 'crippen_contributions'):
            contributions = rdMolDescriptors._CalcCrippenContribs(mol)
            setattr(mol, 'crippen_contributions', contributions)

        # At this point we can be certain, that every atom has been updated with the corresponding
        # attribute that contains it's crippen contribution values
        contributions = getattr(mol, 'crippen_contributions')
        return list(contributions[atom.GetIdx()])

    # Now this is important to set this flag here as a dynamic attribute. The MoleculeProcessing will later
    # on check for this attribute and only if it exists will it change the signature with which the
    # callback function is called to actually include the mol object. Otherwise it would just pass the local
    # atom!
    setattr(func, 'requires_molecule', True)

    return func


def chem_prop(property_name: str,
              callback: t.Callable[[t.Any], t.Any],
              ) -> t.Callable:
    """
    This function can be used to construct a callback function to encode a property of either a Atom or a
    Bond object belonging to an RDKit Mol. The returned function will query the ``property_name`` from the
    atom or bond object and apply the additional ``callback`` function to it and return the result.

    :param property_name: The string name of the method(!) of the atom or bond object to use to get the
        property value
    :param callback: An additional function that can be used to encode the extracted property value into
        the correct format of a list of floats.

    :returns: A function with the signature [Union[Chem.Atom, Chem.Bond]] -> List[float]
    """
    def func(element: t.Union[Chem.Atom, Chem.Bond], data: dict = {}):
        method = getattr(element, property_name)
        value = method()
        value = callback(value)
        return value
    
    # 11.06.23 - We are attaching the callback object itself here as a property of the decorated function here. 
    # We do that because in some advanced functionality it will actually be necessary to retrieve that object 
    # from the decorated function again. Specifically, when using a OneHotEncoder object as the callback 
    # we want to be able to access that original encoder object to also be able to make use of it's "decode" 
    # method.
    setattr(func, 'callback', callback)

    return func


def chem_descriptor(descriptor_func: t.Callable[[Chem.Mol], t.Any],
                    callback: t.Callable[[t.Any], t.Any],
                    ) -> t.Any:

    def func(mol: Chem.Mol, data: dict = {}):
        value = descriptor_func(mol)
        value = callback(value)
        return value

    return func


def apply_mol_element_callbacks(mol: Chem.Mol,
                                data: dict,
                                callback_map: t.Dict[str, t.Callable[[Chem.Atom, dict], t.Any]],
                                element_property: str,
                                ) -> t.OrderedDict[str, t.Any]:
    element_method = getattr(mol, element_property)
    elements = element_method()

    values_map: t.OrderedDict[str, t.List[t.Any]] = OrderedDict()
    for name, callback in callback_map.items():
        values_map[name] = []
        for element in elements:
            value = callback(element, data)
            values_map[name].append(value)

    return values_map


def apply_atom_callbacks(mol: Chem.Mol,
                         data: dict,
                         callback_map: t.Dict[str, t.Callable[[Chem.Atom, dict], t.Any]]
                         ) -> t.OrderedDict[str, t.Any]:
    return apply_mol_element_callbacks(
        mol=mol,
        data=data,
        callback_map=callback_map,
        element_property='GetAtoms'
    )


def apply_bond_callbacks(mol: Chem.Mol,
                         data: dict,
                         callback_map: t.Dict[str, t.Callable[[Chem.Atom, dict], t.Any]]
                         ) -> t.OrderedDict[str, t.Any]:
    return apply_mol_element_callbacks(
        mol=mol,
        data=data,
        callback_map=callback_map,
        element_property='GetBonds'
    )


def apply_graph_callbacks(mol):
    pass



class MoleculeProcessing(ProcessingBase):
    """
    This class can be used as a
    """

    # This is the descriptive string which will be used for the --help option if the command line interface 
    # for this class is invoked.
    description = (
        'This module exposes commands, which can be used to process domain-specific input data into valid '
        'elements of a visual graph dataset\n\n'
        'In this case, a SMILES string is used as the domain-specific representation of a molecular graph. '
        'Using the commands provided by this module, this smiles string can be converted into a JSON '
        'graph representation or be visualized as a PNG.\n\n'
        'Use the --help options for the various individual commands for more information.'
    )

    node_attribute_map = {
        'symbol': {
            'callback': chem_prop('GetSymbol', OneHotEncoder(
                ['B', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl'],
                add_unknown=True,
                dtype=str
            )),
            'description': 'One hot encoding of the atom type',
            'is_type': True,
            'encodes_symbol': True,
        }
    }

    edge_attribute_map = {
        'bond_type': {
            'callback': chem_prop('GetBondType', OneHotEncoder(
                [1, 2, 3, 12],
                add_unknown=False,
                dtype=int,
            )),
            'description': 'One hot encoding of the bond type',
            'is_type': True,
            'encodes_bond': True 
        }
    }

    graph_attribute_map = {
        'molecular_weight': {
            'callback': chem_descriptor(Chem.Descriptors.ExactMolWt, list_identity),
            'description': 'The molecular weight of the entire molecule'
        }
    }

    # These are simply utility variables. These object will be needed to query the various callbacks which
    # are defined in the dictionaries above. They obviously won't result in a real value, but they are only
    # needed to get *any* result from the callback, because for the purpose of constructing the description
    # map we only need to know the length of the lists which are returned by them, not the content.
    MOCK_MOLECULE = mol_from_smiles('CC')
    MOCK_ATOM = MOCK_MOLECULE.GetAtoms()[0]
    MOCK_BOND = MOCK_MOLECULE.GetBonds()[0]
    
    def __init__(self, *args, ignore_issues: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_issues = ignore_issues
        
        # This will be an array of node attribute vector indices of all those elements in 
        # the node attribute vector which have been annotated with the special "is_type" flag. 
        # This flag determines that these elements are relevant to match node types.
        self.node_type_indices = np.array(self.get_attribute_indices(
            self.node_attribute_map,
            self.MOCK_ATOM,
            lambda data: 'is_type' in data and data['is_type']
        ), dtype=int)
        
        # This will be an array of edge attribute vector indices of all those elements in the 
        # edge attribute vector which have been annotated with the special "is_type" flag. 
        # This flag determines that these elements are relevant to match edge types.
        self.edge_type_indices = np.array(self.get_attribute_indices(
            self.edge_attribute_map,
            self.MOCK_BOND,
            lambda data: 'is_type' in data and data['is_type']
        ), dtype=int)
        
        # Here we search for the entry in the node attribute map which implements the "encodes_symbol" 
        # flag. This signals it being the Encoder which is responsible for the main atom symbol type 
        # encoding. 
        # Ultimately we want to extract that very Encoder object for future use where we want to 
        # use it for decoding numeric vectors back into symbols as well.
        data = self.get_attribute_data(
            self.node_attribute_map,
            lambda data: 'encodes_symbol' in data and data['encodes_symbol']
        )
        try:
            self.symbol_encoder: t.Optional[t.Any] = data['callback'].callback
        except TypeError:
            if not self.ignore_issues:
                raise AssertionError('None of elements defined in node_attribute_map implement the flag'
                                     '"encodes_symbol". Please make sure to add the flag to identify the '
                                     'the main symbol Encoder which will be required for the processing functions.')

        # This is an array which contains all the node attribute vector indices which can be used 
        # to extract the sub vector responsible for encoding the symbol.
        self.symbol_indices = np.array(self.get_attribute_indices(
            self.node_attribute_map,
            self.MOCK_ATOM,
            lambda data: 'encodes_symbol' in data and data['encodes_symbol']
        ), dtype=int)
        
        data = self.get_attribute_data(
            self.edge_attribute_map,
            lambda data: 'encodes_bond' in data and data['encodes_bond'],
        )
        try:
            self.bond_encoder: t.Optional[t.Any] = data['callback'].callback
        except TypeError:
            if not self.ignore_issues:
                raise AssertionError('None of the elements defined in edge_attribute_map implement the flag '
                                     '"encodes_bond". Please make sure to add the flag to identify the '
                                     'the main bond type Encoder which will be required for the processing functions')
                
        self.bond_indices = np.array(self.get_attribute_indices(
            self.edge_attribute_map,
            self.MOCK_BOND,
            lambda data: 'encodes_bond' in data and data['encodes_bond'],
        ))
        
        
    def get_attribute_data(self,
                           attribute_map: t.Dict[str, dict],
                           condition: t.Callable,
                           ) -> dict:
        for name, data in attribute_map.items():
            if condition(data):
                return data
        
    def get_attribute_indices(self,
                              attribute_map: t.Dict[str, dict],
                              element: t.Any,
                              condition: t.Callable
                              ) -> t.List[int]:
        indices = []
        index = 0
        for name, data in attribute_map.items():
            callback = data['callback']
            value: list = self.apply_callback(callback, self.MOCK_MOLECULE, element)
            for _ in value:
                if condition(data):
                    indices.append(index)
                index += 1
                
        return indices
    
    def node_match(self, node_attributes_1, node_attributes_2):
        return np.isclose(
            node_attributes_1[self.node_type_indices], 
            node_attributes_2[self.node_type_indices],
        ).all()

    def edge_match(self, edge_attributes_1, edge_attributes_2):
        return np.isclose(
            edge_attributes_1[self.edge_type_indices],
            edge_attributes_2[self.edge_type_indices],
        ).all()
    
    def extract(self,
                graph: tv.GraphDict,
                mask: np.ndarray,
                clear_aromaticity: bool = True,
                process_kwargs: dict = {},
                unprocess_kwargs: dict = {},
                ) -> t.Tuple[tv.DomainRepr, tv.GraphDict]:
        return super().extract(
            graph=graph,
            mask=mask,
            process_kwargs={
                **process_kwargs,
            },
            unprocess_kwargs={
                'clear_aromaticity': clear_aromaticity,
                **unprocess_kwargs,
            }
        )
    
    def unprocess(self,
                  graph: tv.GraphDict,
                  clear_aromaticity: bool = False,
                  **kwargs
                  ) -> tv.DomainRepr:
        """
        Given the ``graph`` dict representation of a molecular graph, this method will transform that graph 
        back into it's domain representation which in this case is a SMILES string.
        
        the aromaticity problem for fragments
        -------------------------------------
        
        This method should also work for molecular fragements, so graph dicts which don't necessarily describe 
        a complete molecule but rather only parts of it that were extracted from other larger molecules. This 
        can cause a problem when atoms were extracted from an aromatic ring but now in the extracted form they 
        are no longer part of a valid aromatic ring. These kinds of smiles cannot be turned back into a Mol 
        object successfully.
        In such cases the ``clear_aromaticity`` flag of this method has to be used, which will erase the 
        aromatic flags such that the resulting SMILES is still somewhat valid, although the molecule which 
        can then be reconstructed from that smiles is not in itself valid!
        
        :param graph: The graph dict to be converted to smiles
        :param clear_aromaticity: If set, this will forcefully clear the aromaticity flags of the molecule  
            which will result in a valid molecular representation as far as RDKit is concerned but not  
            in the chemical sense.
        
        :returns: The SMILES string corresponding to the graph dict
        """
        # TODO: Add support for charged atoms.
        
        # For molecular graphs the domain representation is SMILES strings. For SMILES strings we need 
        # to rely on the conversion functionality implemented in RDKit and that in turn can perform the 
        # conversion when provided with a Mol object. So what we have to do here is to iteratively 
        # construct such a Mol object from the given graph dict.
        
        mol = Chem.RWMol()
        for node_index in graph['node_indices']:
            node_attributes = graph['node_attributes'][node_index]
            symbol = self.symbol_encoder.decode(node_attributes[self.symbol_indices])
            atom = Chem.AtomFromSmarts(symbol)
            mol.AddAtom(atom)
        
        for edge_index, (i, j) in enumerate(graph['edge_indices']):
            i, j = int(i), int(j)
            edge_attributes = graph['edge_attributes'][edge_index]
            bond_type = self.bond_encoder.decode(edge_attributes[self.bond_indices])
            # The decoder here only returns the integer representation of the bond type, but the signature 
            # of the AddBond method REALLY wants that to be wrapped as a BondType object...
            bond_type = Chem.BondType(bond_type)
            if not mol.GetBondBetweenAtoms(i, j):
                mol.AddBond(i, j, bond_type)
            
        # If we only extract a sub graph, aromatic bond types will probably cause a problem since at that 
        # point they are no longer part of a valid ring. In that case we need to manually set all the aromatic 
        # flags to false here to fix that.
        # Note: This does not result in a canonical SMILES then in the end!
        if clear_aromaticity:
            for atom in mol.GetAtoms():
                atom.SetIsAromatic(False)

        # The previous RWMol object is a special kind of *mutable* mol object and here we need to 
        # convert that into the regular mol object.
        mol = mol.GetMol()
        return Chem.MolToSmiles(mol)
    
    def apply_callback(self, 
                       callback: t.Callable, 
                       mol: Chem.Mol, 
                       element: t.Any
                       ) -> t.List[float]:
        """
        Given a ``callback`` function, the base ``mol`` molecule object and the ``element`` - an atom or a bond 
        object - on which to apply that callback, this method will apply the callback in the correct manner and 
        return the return value of that callback. 

        The seemingly unnecessary wrapping of this functionality in it's own method here is necessary because the 
        application of the callback is not necessary due to some special rules whic should not be scattered 
        throughout the rest of the code in this class.

        :param callback: _description_
        :type callback: t.Callable
        :param mol: _description_
        :type mol: Chem.Mol
        :param element: _description_
        :type element: t.Any
        
        :return: _description_
        :rtype: t.List[float]
        """
        # Most callbacks are rather simple and only operate on the basis of the element (atom / bond) itself 
        # but there are other - more advanced - callbacks which need the context of the whole molecule for 
        # some more fancy computations. These can be detected by an additional property that was attached to 
        # them. In those cases the signature of the callback is different which is why we need to check this.
        # I know that this is essentially bad design, but the necessity crept up only later and at that 
        # point I needed to maintain back-comp and couldn't generally change the signature.
        if hasattr(callback, 'requires_molecule') and getattr(callback, 'requires_molecule'):
            value = callback(mol, element)
        else:
            value = callback(element)
        
        return value

    def process(self,
                value: str,
                double_edges_undirected: bool = True,
                use_node_coordinates: bool = False,
                graph_labels: list = [],
                ) -> dict:
        """
        Converts SMILES string into graph representation.

        This command processes the SMILES string and creates the full graph representation of the
        corresponding molecular graph. The node and edge feature vectors will be created in the exact
        same manner as the rest of the dataset, making the resulting representation compatible with
        any model trained on the original VGD.

        This command outputs the JSON representation of the graph dictionary representation to the console.

        :param value: The SMILES string of the molecule to be converted
        :param double_edges_undirected: A boolean flag of whether to represent an edge as two undirected
            edges
        :param use_node_coordinates: A boolean flag of whether to include 3D node_coordinates into the
            graph representation. These would be created by using an RDKit conformer. However, this
            process can fail.
        :param graph_labels: A list containing the various ground truth labels to be additionally associated
            with the element
        """
        # 01.06.23 - When working with the counterfactuals I have noticed that there is a problem where
        # it is not easily possible to maintain the original molecules atom indices through a SMILES
        # conversion, which seriously messes with the localization of which parts were modified.
        # So as a solution we introduce here the option to generate the graph based on a Mol object
        # directly instead of having to use the SMILES every time. This could potentially be slightly
        # more efficient as well.
        if isinstance(value, Chem.Mol):
            mol = value
            smiles = Chem.MolToSmiles(mol)
        else:
            smiles = value
            mol = Chem.MolFromSmiles(smiles)
            print(smiles, mol)

        atoms = mol.GetAtoms()
        # First of all we iterate over all the atoms in the molecule and apply all the callback
        # functions on the atom objects which then calculate the actual attribute values for the final
        # node attribute vector.
        node_indices = []
        node_attributes = []
        for atom in atoms:
            node_indices.append(atom.GetIdx())

            attributes = []
            # "node_attribute_callbacks" is a dictionary which specifies all the transformation functions
            # that are to be applied on each atom to calculate part of the node feature vector
            for name, data in self.node_attribute_map.items():
                callback: t.Callable[[Chem.Mol, Chem.Atom], list] = data['callback']
                value: list = self.apply_callback(callback, mol, atom)

                attributes += value

            node_attributes.append(attributes)

        bonds = mol.GetBonds()
        # Next up is the same with the bonds
        edge_indices = []
        edge_attributes = []
        for bond in bonds:
            i = int(bond.GetBeginAtomIdx())
            j = int(bond.GetEndAtomIdx())

            edge_indices.append([i, j])
            if double_edges_undirected:
                edge_indices.append([j, i])

            attributes = []
            for name, data in self.edge_attribute_map.items():
                callback: t.Callable[[Chem.Bond], list] = data['callback']
                value: list = self.apply_callback(callback, mol, bond)
                attributes += value

            # We have to be careful here to really insert the attributes as often as there are index
            # tuples.
            edge_attributes.append(attributes)
            if double_edges_undirected:
                edge_attributes.append(attributes)

        # Then there is also the option to add global graph attributes. The callbacks for this kind of
        # attribute take the entire molecule object as an argument rather than just atom or bond
        graph_attributes = []
        for name, data in self.graph_attribute_map.items():
            callback: t.Callable[[Chem.Mol], list] = data['callback']
            value: list = callback(mol)
            graph_attributes += value

        # Now we can construct the preliminary graph dict representation. All of these properties form the
        # core of the graph dict - they always have to be present. All the following code after that is
        # optional additions which can be added to support certain additional features
        graph: tc.GraphDict = {
            'node_indices':         node_indices,
            'node_attributes':      np.array(node_attributes, dtype=float),
            'edge_indices':         edge_indices,
            'edge_attributes':      np.array(edge_attributes, dtype=float),
            'graph_attributes':     np.array(graph_attributes, dtype=float),
            'graph_labels':         graph_labels,
        }

        # Optionally, if the flag is set, this will apply a conformer on the molecule which will
        # then calculate the 3D coordinates of each atom in space.
        if use_node_coordinates:
            try:
                # https://sourceforge.net/p/rdkit/mailman/message/33386856/
                try:
                    rdkit.Chem.AllChem.EmbedMolecule(mol)
                    rdkit.Chem.AllChem.MMFFOptimizeMolecule(mol)
                except:
                    rdkit.Chem.AllChem.EmbedMolecule(mol, useRandomCoords=True)
                    rdkit.Chem.AllChem.UFFOptimizeMolecule(mol)

                conformer = mol.GetConformers()[0]
                node_coordinates = np.array(conformer.GetPositions())
                graph['node_coordinates'] = node_coordinates

                # Now that we have the 3D coordinates for every atom, we can also calculate the length of all
                # the bonds from that!
                edge_lengths = []
                for i, j in graph['edge_indices']:
                    c_i = graph['node_coordinates'][i]
                    c_j = graph['node_coordinates'][j]
                    length = la.norm(c_i - c_j)
                    edge_lengths.append([length])

                graph['edge_lengths'] = np.array(edge_lengths)

            except Exception as exc:
                raise ProcessingError(f'Cannot calculate node_coordinates for the given '
                                      f'molecule with smiles code: {smiles}')

        return graph

    def visualize_as_figure(self,
                            value: str,
                            width: int,
                            height: int,
                            additional_returns: dict = {},
                            **kwargs,
                            ) -> t.Tuple[plt.Figure, np.ndarray]:
        """
        The normal "visualize" method has to return a numpy array representation of the image. While that
        is a decent choice for printing it to the console, it is not a good choice for direct API usage.
        This method will instead return the visualization as a matplotlib Figure object.

        This method should preferably used when directly interacting with the processing functionality
        with code.
        """
        smiles = value
        mol = mol_from_smiles(smiles)
        fig, ax = create_frameless_figure(width=width, height=height)
        node_positions, svg_string = visualize_molecular_graph_from_mol(
            ax=ax,
            mol=mol,
            image_width=width,
            image_height=height
        )
        # The "node_positions" which are returned by the above function are values within the axes object
        # coordinate system. Using the following piece of code we transform these into the actual pixel
        # coordinates of the figure image.
        node_positions = [[int(v) for v in ax.transData.transform((x, y))]
                          for x, y in node_positions]
        node_positions = np.array(node_positions)

        additional_returns['svg_string'] = svg_string

        return fig, node_positions

    def visualize(self,
                  value: str,
                  width: int = 1000,
                  height: int = 1000,
                  **kwargs,
                  ) -> np.ndarray:
        """
        Creates a visualization of the given SMILES molecule.

        This command used RDKit to generate a visual representation of the molecular graph corresponding
        to the given SMILES string. This will result in an colored image of the given dimensions WIDTH
        and HEIGHT.

        This command outputs the JSON representation of the RGB image array to the console. This array will
        have the shape (width, height, 3).
        """
        fig, _ = self.visualize_as_figure(
            value=value,
            width=width,
            height=height,
        )

        # This section turns the image which is currently a matplotlib figure object into a numpy array
        # https://stackoverflow.com/questions/35355930/matplotlib-figure-to-image-as-a-numpy-array
        canvas = FigureCanvas(fig)
        canvas.draw()
        array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        array = array.reshape((height, width, 3))

        plt.close(fig)

        return array

    def create(self,
               value: str,
               index: str = '0',
               name: str = 'molecule',
               double_edges_undirected: bool = True,
               use_node_coordinates: bool = False,
               additional_metadata: dict = {},
               additional_graph_data: dict = {},
               width: int = 1000,
               height: int = 1000,
               create_svg: bool = False,
               output_path: str = os.getcwd(),
               writer: t.Optional[VisualGraphDatasetWriter] = None,
               **kwargs,
               ):
        """
        Creates VGD element file representations (PNG & JSON).

        This command will create the VGD file representations corresponding to the molecular graph given
        by the SMILES string. This representation includes two files by default: (1) A JSON file which
        contains the full graph representation of the molecular graph and additional metadata. (2) A PNG
        file containing a visual representation of the molecular graph. Both file are the same

        This command will not output anything to the console. Instead, it will create new files within
        the given OUTPUT_PATH folder. The files will have the name which is simply a number that is
        specified with the INDEX option.
        """
        smiles = value
        g = self.process(
            smiles,
            double_edges_undirected=double_edges_undirected,
            use_node_coordinates=use_node_coordinates,
        )
        g.update({
            **additional_graph_data,
        })
        fig, node_positions = self.visualize_as_figure(
            value=value,
            width=width,
            height=height,
        )
        # This is important! we need to add the node positions to the graph representation to enable the
        # visualizations later on. This array contains the pixel coordinates within the image for each of
        # the nodes
        g['node_positions'] = node_positions

        metadata = {
            'index':    int(index),
            'name':     name,
            'smiles':   smiles,
            'repr':     smiles,
            'graph':    g,
            **additional_metadata,
        }

        # 02.06.2023 - I have added the option to specify a Writer instance with which the elements should
        # actually be saved to the disk. This is optional and if no writer instance is given it will work
        # just as before. But using a writer instance for dataset creation is preferable because it can
        # make use of optimizations while saving the dataset.
        if writer is not None:
            writer.write(
                name=int(index),
                metadata=metadata,
                figure=fig,
            )

        else:
            fig_path = os.path.join(output_path, f'{index}.png')
            self.save_figure(fig, fig_path)
            plt.close(fig)

            metadata_path = os.path.join(output_path, f'{index}.json')
            self.save_metadata(metadata, metadata_path)

        # if create_svg:
        #     svg_path = os.path.join(output_path, f'{index}.svg')
        #     self.save_svg(svg_string, svg_path)

        return metadata

    def get_description_map(self) -> dict:
        """
        This method returns a dictionary, which on the top level contains the three keys
        "node_attributes", "edge_attributes" and "graph_attributes". Each of these is a dictionary
        themselves, where the keys are the integer indices of the corresponding attribute vectors of an
        element and the values are strings containing natural language descriptions of the content of each
        index.
        """
        description_map = {
            'node_attributes': {},
            'edge_attributes': {},
            'graph_attributes': {},
        }

        # descriptions for the node attributes
        index = 0
        for name, data in self.node_attribute_map.items():
            description = data['description']
            callback = data['callback']
            value: list = callback(self.MOCK_ATOM)
            for _ in value:
                description_map['node_attributes'][index] = description
                index += 1

        # descriptions for the edge attributes
        index = 0
        for name, data in self.edge_attribute_map.items():
            description = data['description']
            callback = data['callback']
            value: list = callback(self.MOCK_BOND)
            for _ in value:
                description_map['edge_attributes'][index] = description
                index += 1

        # descriptions for the graph attributes
        index = 0
        for name, data in self.graph_attribute_map.items():
            description = data['description']
            callback = data['callback']
            value: list = callback(self.MOCK_MOLECULE)
            for _ in value:
                description_map['graph_attributes'][index] = description
                index += 1

        return description_map

    # -- utils --

    def save_svg(self, content: str, path: str):
        with open(path, mode='w') as file:
            file.write(content)
