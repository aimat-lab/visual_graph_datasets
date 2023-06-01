"""
This module defines custom typings which will be used throughout the package. "Custom typings" thereby
does NOT refer to custom data classes, but instead this means more descriptive names for special cases of
native datatypes.

This package has made the choice to represent certain core data instances as native datatypes such as
dictonaries and lists instead of creating custom classes. This choice was mainly made because the
corresponding data has to be (1) easily serializable and de-serializable and (2) have dynamic properties.
As such it has been decided that using native data types is simpler. Of course, these datatypes still
follow a certain internal structure which will be described in this module.

This module defines alternative names for these native datatypes, which are more descriptive and are used
as typing annotations throughout the package to identify whenever such a special format is used as a
parameter or return of a function.
"""
import typing as t

import numpy as np

"""
**Graph Dictionary Representation**

This is a special kind of dictionary representation which is used throughout the entire visual graph 
datasets package to represent the graphs. This dictionary contains AT LEAST the following keys:

- node_indices: (V, 1) array of node indices
- node_attributes: (V, N) array of node feature vectors
- edge_indices: (E, 2) array which is essentially an edge list that defines the graph structure. Is a 
  list of tuples of node indices.
- edge_attributes: (E, M) array of edge feature vectors

Some optional (!) keys which a GraphDict often also contains:

- graph_labels: (C, ) array containing the target value annotations for the graph
- node_positions: (V, 2) array of the (x,y) pixel coordinates of each node within the visualization 
  image
- node_importances: (V, K) array of node importance explanations along K channels
- edge_importances: (E, K) array of edge importance explanations along K channels

**why a dictionary?**

One might ask why this graph representation was chosen instead of a networkx.Graph for example. 
These are the following main reasons:

- Flexibility. It is very easy to attach new data to a dictionary dynamically on the fly if the application 
  requires it.
- Serialization. Dictionaries can be converted into JSON directly without any additional hassle. This is 
  basically THE major reason to use them, since the persistent representation of the dataset relies 
  heavily on JSON strings.
  
"""
GraphDict = t.Dict[str, t.Union[t.List[float], np.ndarray, dict]]

MetadataDict = t.Dict[str, t.Union[int, float, str, dict, list, GraphDict]]

VisGraphIndexDict = t.Dict[int, t.Union[str, MetadataDict]]

VisGraphNameDict = t.Dict[str, t.Union[str, MetadataDict]]

"""
**Visual Graph Element Dict**

In a visual graph dataset that was loaded from the memory, each element will be represented by a dictionary 
of the following format (having AT LEAST the following fields):

- image_path: The absolute path to the visualization image
- metadata: The dictionary containing all the metadata from the JSON file
    - graph: The GraphDict representation of the full graph.
"""
VgdElementDict = t.Dict[str, t.Union[str, dict, t.Any]]

"""
**Domain Specific Graph Representation**

In the context of this package, there are two primarily important representations of a graph. One of those 
is the GraphDict which is explained above, which is very easy to work with because it contains a lot of 
the information about a graph in a very explicit manner. But at the same time it is very long and thus 
not human readable.

Therefore the second important representation of a graph is the domain-specific representation. This is 
usually a very compressed representation of the graph (oftentimes just a single string). It's hard to work 
with directly because it only contains information implicitly.

For the purpose of this package this representation could be anything, but it is encouraged that it be a 
human readable string of some sorts. An example would be the "SMILES" representation of a molecular graph 
from chemistry.
"""
DomainRepr = t.Union[str, t.Any]

"""
**Visual Graph Dataset Metadata Dict**

This is a special kind of dictionary which is used to store the metadata information for an ENTIRE visual 
graph dataset in contrast to just one element. These metadata annotations can optionally be added to a 
dataset folder in the format of a YAML file. These metadata annotations will have some base fields which are 
usually present and some fields which are generated automatically. However, custom metadata may also be 
added to those files in the future to facilitate more advanced features for custom datasets.
"""
VisGraphMetaDict = t.Dict[str, t.Union[str, float, int]]

"""
This represents an RGB color definition using float values between 0 and 1 for each of the color aspects
"""
ColorList = t.Union[t.List[float], t.Tuple[float, float, float]]


# == DATA TYPE CHECKS ==

def assert_graph_dict(obj: t.Any) -> None:
    """
    Implements assertions to make sure that the given ``value`` is a valid GraphDict.

    :param obj: The obj to be checked
    :return: None
    """
    # Most importantly the value has to be a dict
    assert isinstance(obj, dict), ('The given object is not a dict and thus cannot be a GraphDict')

    # Then there are certain keys it must implement
    required_keys = [
        'node_indices',
        'node_attributes',
        'edge_indices',
        'edge_attributes',
    ]
    for key in required_keys:
        assert key in obj.keys(), (f'The given object is missing the key {key} to be a GraphDict')
        value = obj[key]
        assert isinstance(value, (list, np.ndarray)), (f'The value corresponding to the key {key} is not '
                                                       f'of the required type list or numpy array')

    # The shapes of the number of node indices has to be the same as the number of node attributes
    assert len(obj['node_indices']) == len(obj['node_attributes']), (
        'The number of node indices has to match the number of node attributes!'
    )

    # same for the edges, the number of edge attributes has to be the same as edge indices
    assert len(obj['edge_indices']) == len(obj['edge_attributes']), (
        'The number of edge indices has to match the number of edge attributes!'
    )
