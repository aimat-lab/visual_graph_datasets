"""
Base functionality and interfaces for processing domain-specific graph representation into the GraphDict
representations and visualizations which are used throughout this package.
"""
import re
import inspect
import json
import typing as t
import typing as typ

import click
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import networkx as nx
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import visual_graph_datasets.typing as tv
from visual_graph_datasets.data import NumericJsonEncoder
from visual_graph_datasets.util import TEMPLATE_ENV
from visual_graph_datasets.util import JSON_LIST, JSON_DICT, JSON_ARRAY
from visual_graph_datasets.util import sanitize_indents
from visual_graph_datasets.data import VisualGraphDatasetWriter
from visual_graph_datasets.data import extract_graph_mask, nx_from_graph


def identity(value: t.Any) -> t.Any:
    return value


def list_identity(value: t.Any, dtype: type = float) -> t.Any:
    return [dtype(value)]


class Scaler:

    def __init__(self,
                 min_value: float,
                 max_value: float,
                 dtype: type = float):
        self.min_value = min_value
        self.max_value = max_value
        self.dtype = dtype

    def __call__(self, value: t.Any, *args, **kwargs) -> t.List[float]:
        value_scaled = (value - self.min_value) / (self.max_value - self.min_value)
        return [value_scaled]


class EncoderBase: 
    """
    This is the abstract base class for the implementation of attribute processing encoder 
    objects.

    Such encoder objects have the function of effectively encoding some kind of non-numeric 
    property into a vector of numeric values, which unlike the orignal format is suitable as 
    part of the input for a machine learning model.
    
    Any encoder implementation has to implement the following two methods:
    
    - ``encode``: This method should take the original value to be encoded as the argument and 
      then return a list of float values, which represents the encoded vector.
    - ``decode``: This method is the exact inverse functionality. It should take the list 
      of numeric values as the input and return the equivalent original value whatever that 
      may be.
      
    Each encoder object is callable by default through an implementation of the __call__ method, 
    which internally uses the implementation of the ``encode`` method.
    """
    
    def __call__(self, value: t.Any, *args, **kwargs) -> t.List[float]:
        return self.encode(value, *args, **kwargs)
    
    def encode(self, value: t.Any, *args, **kwargs) -> t.List[float]:
        raise NotImplementedError()
    
    def decode(self, encoded: t.List[float]) -> t.Any:
        raise NotImplementedError()
    
    
class StringEncoderMixin:
    """
    This is an interface which can optionally be implemented by an EncoderBase subclass to provide the 
    additional functionality of encoding and decoding string representations of the domain values.
    
    Subclasses need to implement the following methods:
    - ``encode_string``: Given the domain value to be encoded, this method will return a human-readable
        string representation of that value.
    - ``decode_string``: Given the string representation of a domain value, this method will return the
        original domain value.
    """
    def encode_string(self, value: typ.Any) -> str:
        raise NotImplementedError()
    
    def decode_string(self, string: str) -> typ.Any:
        raise NotImplementedError()


class OneHotEncoder(EncoderBase, StringEncoderMixin):
    """
    This is the specific implementation of an attribute Encoder class for the process of 
    OneHotEncoding elements of different types.
    
    The one-hot encoder is constructed by supplying a list of elements which should be encoded. 
    This may be any data type or structure, which implements the __eq__ method, such as strings 
    for example. The encoded vector representation of a single element will have as many elements 
    as the provided list of elements, where all values are zero except for the position of the 
    that matches the given element through an equality check.
    
    :param values: A list of elements which each will be checked when encoding an element
    :param add_unknown: Boolean flag which determines whether an additional one-hot encoding 
        element is added to the end of the list. This element will be used as the encoding for
        any element which is not part of the original list of values. If this flag is False 
        and an unkown element is otherwise encountered, the encoder will silently ignore it 
        and return a vector of zeros.
    :param unknown: The value which will be used as the encoding for any element which is not part
        of the original list of values. This parameter is only relevant if the add_unknown flag is
        set to True.
    :param dtype: a type callable that defines the type of the elements in the ``values`` list
    :param string_values: Optionally a list of string which provide human-readable string representations 
        for each of the elements in the ``values`` parameter. Therefore, this list needs to be the same 
        length as the ``values`` list. This parameter can optionally be None, in which case simply the 
        str() transformation of the elements in the ``values`` list will be used as the string 
        representations.
    :param use_soft_decode: If this flag is set to True, instead of matching the given encoded vector 
        exactly, the decoder will return the value which has the highest value in the encoded vector. 
        This is useful when the encoded vector is not exactly one-hot encoded, but rather a probability 
        distribution over the possible values.
    """
    def __init__(self,
                 values: t.List[t.Any],
                 add_unknown: bool = False,
                 unknown: t.Any = 'H',
                 dtype: type = float,
                 string_values: typ.Optional[list[str]] = None,
                 use_soft_decode: bool = False,
                 ):
        EncoderBase.__init__(self)
        StringEncoderMixin.__init__(self)
        
        self.values = values
        self.add_unknown = add_unknown
        self.unknown = unknown
        self.dtype = dtype
        self.use_soft_decode = use_soft_decode
        
        # We want the "string_values" to always be a list of strings. If the parameter is None that means 
        # it is unnecessary to define a separate list and we can just use the "values" list as the string 
        # representation as well.
        if string_values is None:
            self.string_values: list[str] = [str(v) for v in values]
        else:
            self.string_values: list[str] = string_values

    def __call__(self, value: t.Any, *args, **kwargs) -> t.List[float]:
        return self.encode(value)
    
    # implement "EncoderBase"

    def encode(self, value: t.Any, *args, **kwargs) -> t.List[float]:
        """
        Given the domain ``value`` to be encoded, this method will return a list of float values that 
        represents the one-hot encoding corresponding to that exact value as defined by the list of possible 
        values given to the constructor.
        
        :param value: The domain value to be encoded. Must be part of the list of values given to the
            constructor - otherwise if add_unknown is True, the unknown one-hot encoding will be returned.
        
        :returns: list of float values which are either 1. or 0. (one-hot encoded)
        """
        one_hot = [1. if v == self.dtype(value) else 0. for v in self.values]
        if self.add_unknown:
            one_hot += [0. if 1 in one_hot else 1.]

        return one_hot
        
    def decode(self, encoded: t.List[float]) -> t.Any:
        """
        Given the one-hot encoded representation ``encoded`` of a domain value, this method will return the
        original domain value.
        
        Note that this method will try to do an exact match of the one-hot position. If the one-hot encoding
        is not exact, then the "unknown" value will be returned.
        
        :returns: The domain value which corresponds to the given one-hot encoding. This will have whatever 
            type the original domain values have.
        """
        if self.use_soft_decode:
            return self.decode_soft(encoded)
        else:
            return self.decode_hard(encoded)
    
    def decode_soft(self, encoded: t.List[float]) -> typ.Any:
        """
        Given the one-hot encoded representation ``encoded`` of a domain value, this method will return the
        original domain value. This method is a soft decoding method, which means that it will return the
        value which has the highest value in the encoded vector. This is useful when the encoded vector is
        not exactly one-hot encoded, but rather a probability distribution over the possible values.
        
        :param encoded: The one-hot encoded representation of the domain value
        
        :returns: The domain value which corresponds to the given one-hot encoding. This will have whatever
            type the original domain values have.
        """
        max_index = np.argmax(encoded)
        if max_index < len(self.values):
            return self.values[max_index]
        else:
            return self.unknown
    
    def decode_hard(self, encoded: t.List[float]) -> typ.Any:
        """
        Given the one-hot encoded representation ``encoded`` of a domain value, this method will return the
        original domain value. This method is a hard decoding method, which means that it will return the
        value which has the exact one-hot encoding as the given encoded vector.
        
        :param encoded: The one-hot encoded representation of the domain value
        
        :returns: The domain value which corresponds to the given one-hot encoding. This will have whatever
            type the original domain values have.
        """
        for one_hot, value in zip(encoded, self.values):
            if one_hot:
                return value
            
        # If the previous loop has failed to return anything then we can assume that the 
        # value is the unknown and we will instead return the "unkown" value provided in 
        # the constructor.
        return self.unknown

    # implement "StringEncoderMixin"

    def encode_string(self, value: typ.Any) -> str:
        """
        Given the domain ``value`` to be encoded, this method will return a human-readable string representation 
        of that value.
        
        :param value: The domain value to be encoded. Must be part of the list of values given to the
            constructor - otherwise if add_unknown is True, returns the string "unknown".
        
        :returns: A single string value which represents the given domain value
        """
        for v, s in zip(self.values, self.string_values):
            if v == self.dtype(value):
                return s
    
        return 'unknown'
    
    def decode_string(self, string: str) -> typ.Any:
        """
        Given the string representation ``string`` of a domain value, this method will return the original domain 
        value.
        
        Note that this method will try to do an exact match of the string. If the string is not exact, then the 
        "unknown" value will be returned.
        
        :returns: The domain value which corresponds to the given string representation. This will have whatever 
            type the original domain values have.
        """
        for v, s in zip(self.values, self.string_values):
            if s == string:
                return v
        
        return self.unknown


class ProcessingError(Exception):
    """
    An exceptions specifically to be used when there is a problem with the processing process.
    """
    pass


class ProcessingBase(click.MultiCommand):
    """
    Abstract base class for the implementation of the processing of a domain-specific graph representation
    into the generic GraphDict structure and visualizations which are used throughout this package.

    **Motivation**

    Most application domains have their own graph representation. An example for this would be SMILES strings
    which provide short and human-readable representations of molecular graphs or netlists in the case of
    electrical networks. The problem is that to apply these datasets to graph neural networks they always
    have to be preprocessed in some way to create a representation where nodes and edges are annotated with
    numeric feature vectors.

    A design choice of this package is that this pre-processing should be a property of the dataset itself.
    So essentially each dataset should be shipped in a format which contains all the GNN-ready graph
    representations already. This is done in an attempt to make results for the GNNs themselves more
    comparable by providing the exact same starting point for each of them.

    However, this opens another problem: If the preprocessing is done for the dataset a-priori, how do we
    format new and unknown elements into the same format? This would be necessary to really use the
    trained networks.

    That is the problem which this class aims to address: It provides a single place where the conversion of
    a domain specific graph representation to (1) a GNN-ready representation (2) a canonical representation
    can be created. More importantly this class implements a bit of magic: The implementation of the
    preprocessing functionality done in a subclass can automatically be converted into a standalone python
    module, which also implements the same functionality. That python module can be imported, but it can
    also directly be used as a command line application, making the processing functionality programming
    language independent as well.

    **Automatic CLI Generation**

    This class implements the functionality to generate a command line interface automatically
    from the simple python implementations of the processing procedure. At the core level, this process is
    done for the following three methods:

    - ``process``: Should turn a domain-specific graph representation into a GraphDict
    - ``visualize``: Should create a visualization of the graph given by the domain-specific represen.
    - ``create``: Should create the visual graph dataset file representation of a graph, which consists
      of the metadata JSON file and a PNG visualization.

    During the construction of a class instance, the corresponding click.Command objects are automatically
    derived from these methods, which will be available over the command line with the exact same names.

    To achieve this, the methods are wrapped in all the necessary click decorators and added as additional
    properties of the object instance, which are later detected as valid command implementation for the
    command line interface. The methods are parsed using the ``inspect`` module to extract and transfer as
    much information as possible to their command line counterparts.

    - Each positional argument of the method is converted into a position "argument" of the command as well
    - Each argument with a default value is instead converted into a command line "option" of the same name
    - The types annotations of each method argument are detected and converted into click.ParamType
      instance, if possible. This means that the string input values are automatically cast to the correct
      python types.
    - The docstring is parsed and the main description of the method is used as the --help text of the
      command
    - The individual :param: descriptions in the docstring are also detected and added as the descriptions
      of the corresponding command line "options"
    """

    # This string will be used as the main content of the description when using the --help text directly
    # on the top level command line interface, which is represented by this class.
    description: str = ''

    # This is a list of method names of this class.
    # This list determines which of the methods of this class will AUTOMATICALLY be constructed into CLI
    # commands upon the construction of an instance of this class.
    # For more information about how a method is dynamically turned into a Command see the method
    # "construct_command"
    DEFAULT_COMMANDS: t.List[str] = [
        'process',
        'visualize',
        'create',
    ]

    # This dictionary provides a mapping of "normal" python types to click.ParamTypes which are used in the
    # command line library click.
    # During the process of dynamically construction a click.Command from a method, the parameters of that
    # method and most importantly their type annotations are analyzed and those python type annotations are
    # then mapped into click parameter type instances using this dictionary.
    # This process is important because in the case of click, the type definitions are not just syntactic
    # sugar, but they also contain the functionality to automatically cast the string input values received
    # from the command line into the corresponding types!
    TYPE_CONVERSION_DICT = {
        int: click.INT,
        str: click.STRING,
        bool: click.BOOL,
        list: JSON_LIST,
        t.List: JSON_LIST,
        dict: JSON_DICT,
        t.Dict: JSON_DICT,
        np.ndarray: JSON_ARRAY,
    }

    def __init__(self, *args, **kwargs):
        super(ProcessingBase, self).__init__(*args, help=self.description, **kwargs)

        # This method implements the process of dynamic construction of the click.Commands from several of
        # the methods of this class
        self.scaffold()
        try:
            self.__code__ = inspect.getsource(self.__class__)
        except (AttributeError, TypeError):
            pass
        
    def extract(self,
                graph: tv.GraphDict,
                mask: np.ndarray,
                process_kwargs: dict = {},
                unprocess_kwargs: dict = {},
                **kwargs) -> t.Tuple[tv.DomainRepr, tv.GraphDict]:
        """
        Given a valid GraphDict representation ``graph`` and a binary node ``mask`` array, this method will 
        return the subgraph that is defined through that mask as a tuple of it's domain representation and 
        it's *canonical* graph dict representation.
        
        This method is implemented in a graph type agnostic manner and only assumes that specific implementations 
        for the ``process`` and ``unprocess`` methods exist. The basic procedure is to first extract the 
        the sub graph solely on the basis of the given graph dict and then ``unprocess`` that extracted sub graph 
        dict to obtain it's domain representation. This domain representation is then again subjected to ``process`` 
        to obtain the canonical graph dict representation of it. In the end a graph isomorphism matching is 
        performed to reassociate the extracted graph's node indices with the node indices of the original graph.

        :param graph: The graph from which a smaller subgraph should be extracted into it's own 
            individual representation
        :type graph: tv.GraphDict
        :param mask: A binary array (integer or boolean) of the shape (N, ) where N is the number of nodes in 
            the given graph
        :type mask: np.ndarray
        :param process_kwargs: A dict with additional options for the ``process`` method
        :param unprocess_kwargs: A dict with additional options for the ``unprocess`` method
        
        :return: _description_
        :rtype: t.Tuple[tv.DomainRepr, tv.GraphDict]
        """
        # Given a graph dict and a binary node mask, this function will return a graph dict which contains
        # just the sub graph section of the original graph that was defined by the mask.
        # Most importantly, this sub graph dict will contain an additional field "node_indices_original"
        # This sub graph will contain all the additional node and edge specific properties which the
        # original graph had as well for those masked nodes.
        # which is very important for us here. that dictionary will map the new graph indices to the graph
        # indices which the sub graph originally had in the original graph!
        sub_graph = extract_graph_mask(graph, mask, **process_kwargs)

        # Then we use the "unprocess" method to turn that sub graph into a domain representation and
        # immediately turn it back into graph dict using the "process" method. This should(!) result in
        # essentially exactly the same graph. The only important difference being that the indexing scheme
        # and order of the nodes is not different - it is *canonical* w.r.t. to the domain specific
        # graph encoding, which is important for the future handling of that graph.
        value = self.unprocess(sub_graph, **unprocess_kwargs)
        sub_graph_canon = self.process(value)

        # Now the problem we have is that with that newly processed graph dict we don't have the mapping
        # of the new indices to the indices within the original graph! To fix that we will perform an
        # isomorphism matching between the two sub graph representations.
        g_sub = nx_from_graph(sub_graph)
        g_canon = nx_from_graph(sub_graph_canon)
        matcher = nx.isomorphism.GraphMatcher(
            g_canon, g_sub,
            node_match=lambda a, b: self.node_match(a['node_attributes'], b['node_attributes']),
            edge_match=lambda a, b: self.edge_match(a['edge_attributes'], b['edge_attributes']),
        )
        matcher.initialize()
        matcher.is_isomorphic()

        sub_graph_canon['node_indices_original'] = np.array([
            sub_graph['node_indices_original'][matcher.mapping[i]] for i in sub_graph_canon['node_indices']
        ])

        node_keys = [key for key in graph.keys() if key.startswith('node') if key != 'node_indices']

        for key in node_keys:
            sub_graph_canon[key] = np.array([
                graph[key][sub_graph_canon['node_indices_original'][i]]
                for i in sub_graph_canon['node_indices']
            ])
            
        # TODO: Implement the transfer of the edge based properties as well.
            
        # All "graph" global attributes of the graph we will copy as they are. None of the graph 
        # restructuring we have done here will change anything for these values as they are supposed 
        # to be global attributes of the graph anyways.
        graph_keys = [key for key in graph.keys() if key.startswith('graph')]
        for key in graph_keys:
            sub_graph_canon[key] = graph[key]

        return value, sub_graph_canon

    # -- to be implemented --
    # All of these methods have to be implemented by subclasses. They need to contain the domain specific
    # implementations if how to actually process domain-specific graph representations into the graph
    # dict / visualizations.
    
    def node_match(self,
                   node_attributes_1: np.ndarray, 
                   node_attributes_2: np.ndarray
                   ) -> bool:
        """
        Given a two np arrays ``node_attributes_1`` and ``node_attributes_2``, this method should return 
        the boolean indicator of whether these two represent node attributes of the same *node type*.

        The default implementation of this method simply checks if the two arrays are exactly the same, but 
        this may not always make sense, since it is only about the node *type*. Two nodes can be of the same 
        type but still possess other additional parameters, in which case the the result of the default 
        implementation would be wrong. Thus it is encouraged to provide a custom implementation for this 
        method for every specific graph type.

        :param node_attributes_1: The array of numeric properties for the first node
        :type node_attributes_1: np.ndarray
        :param node_attributes_2: The array of numeric properties for the second node
        :type node_attributes_2: np.ndarray
        
        :return: bool
        :rtype: bool
        """
        return np.isclose(node_attributes_1, node_attributes_2).all()
    
    def edge_match(self,
                   edge_attributes_1: np.ndarray,
                   edge_attributes_2: np.ndarray,
                   ) -> bool:
        """
        Given a two np arrays ``edge_attributes_1`` and ``edge_attributes_2``, this method should return 
        the boolean indicator of whether these two represent edge attributes of the same *edge type*.

        The default implementation of this method simply checks if the two arrays are exactly the same, but 
        this may not always make sense, since it is only about the edge *type*. Two edges can be of the same 
        type but still possess other additional parameters, in which case the the result of the default 
        implementation would be wrong. Thus it is encouraged to provide a custom implementation for this 
        method for every specific graph type.

        :param edge_attributes_1: _description_
        :type edge_attributes_1: np.ndarray
        :param edge_attributes_2: _description_
        :type edge_attributes_2: np.ndarray
        :return: _description_
        :rtype: bool
        """
        return np.isclose(edge_attributes_1, edge_attributes_2).all()

    def process(self,
                value: tv.DomainRepr,
                *args,
                additional_graph_data: dict = {},
                **kwargs
                ) -> tv.GraphDict:
        """
        This method needs to implement the transformation of a domain-specific graph representation into a
        GraphDict representation, as it is used throughout this package. At the core, this means that
        the method should return a dict which contains the 4 keys:

        - node_indices: A list of integer indices for all the nodes
        - node_attributes: A list of node feature vectors (lists), where each one defines the dataset
            specific feature values for each node
        - edge_indices: A list of tuples. Each tuple defines a directed edge in the graph and thus consists
            of two node indices.
        - edge_attributes: A list of edge feature vectors (lists) for each edge defined in the edge indices
            list.

        The resulting graph dictionary representation may contain more custom fields to support custom
        features.

        But beware that it has to be possible to JSON encode the dictionary! This means that there cannot be
        any "fancy" additional fields such as callback functions or custom objects, only native data types!

        :param value: The domain-specific representation of the graph to be converted
        :param additional_graph_data: A dictionary with string keys and arbitrary (json encodable) values
            that will be added as additional attributes to the graph dict

        :returns: The graph dict representation of the given element
        """
        raise NotImplemented()

    def unprocess(self,
                  graph: tv.GraphDict,
                  **kwargs,
                  ) -> tv.DomainRepr:
        """
        This method is supposed to implement the inverse functionality to the "process" method, namely the 
        conversion of an existing graph dict representation back into a domain representation.

        :param graph: The graph to be converted into the domain representation
        :type graph: tv.GraphDict
        
        :return: The domain representation that is equivalent to the given graph dict.
        :rtype: tv.DomainRepr
        """
        raise NotImplemented()

    def visualize_as_figure(self,
                            value: tv.DomainRepr,
                            *args,
                            width: int,
                            height: int,
                            **kwargs
                            ) -> t.Tuple[plt.Figure, np.ndarray]:
        """
        This method needs to implement the transformation of a domain-specific graph representation into a
        visualization of the corresponding graph.
        This method should return a tuple consisting of two values:
        - The visualization as a plt.Figure object
        - A numpy array (V,2) containing the 2D coordinates of the nodes within that visualization

        :param value: The domain-specific representation of the graph to be converted
        :param width: The width of the image in pixels
        :param height: The height of the image in pixels

        :returns: Tuple (fig, node_positions)
        """
        raise NotImplemented()

    def visualize(self,
                  value: tv.DomainRepr,
                  width: int,
                  height: int,
                  **kwargs
                  ) -> np.ndarray:
        """
        This method needs to implement the transformation of a domain-specific graph representation into a
        visualization of the corresponding graph. This method in particular should return that visualization
        as a numpy rgb image array with the dimensions (width, height, 3).

        This choice of the array return type was to support it being possible to print the image to the
        command line as a string (JSON encoded).
        However, it is encouraged to also define an additional method which returns the visualization in a
        format that is better suited for direct code access, such as a matplotlib Figure object.

        :param value: The domain-specific representation of the graph to be converted
        :param width: The width of the image in pixels
        :param height: The height of the image in pixels

        :returns: A numpy array (width, height, 3) which represents the visualization RGB image
        """
        raise NotImplemented()

    def create(self,
               value: tv.DomainRepr,
               index: str,
               width: int,
               height: int,
               output_path: str,
               additional_graph_data: dict,
               additional_metadata: dict,
               writer: t.Optional[VisualGraphDatasetWriter] = None,
               *args,
               **kwargs,
               ) -> tv.MetadataDict:
        """
        This method needs to implement the creation of a visual graph dataset file representation of a graph,
        given its domain specific representation. This method is not supposed to return anything. Instead,
        it is supposed to directly create the necessary files.

        A visual graph dataset element is represented by two files:

        - A metadata JSON files, which contains the full GraphDict representation of the corresponding
          graph as well as all additional metadata, such as the index within the dataset, name etc.
        - A PNG image, which shows a visualization of the corresponding graph, which can then
          be used in the future to visualize explanations with.

        This method should return the metadata dictionary for that element.

        :param value: The domain-specific representation of the graph to be converted
        :param index: The index of the newly created elements in the larger dataset. This value will also
            be used as the file name for the two created files.
        :param width: The width of the image in pixels
        :param height: The height of the image in pixels
        :param output_path: The path of an existing FOLDER into which the files will be saved
        :param additional_graph_data: A dictionary with string keys and arbitrary (json encodable) values
            that will be added as additional attributes to the graph dict
        :param additional_metadata: A dictionary with string keys and arbitrary (json encodable) values
            that will be added as additional attributes to the metadata dict
        :param writer: There is the option to provide an instance of VisualGraphDatasetWriter to the
            create method. If that happens then the create method should use that writer instance to
            persistently save the files instead. This may be important because the writer instance is
            optimized to write large datasets more efficiently by implementing folder chunking under the
            hood.

        :returns: metadata dict
        """
        raise NotImplemented()

    def get_description_map(self) -> dict:
        """
        This method should return a dictionary, which contains descriptions of the results of the processing
        functionality implemented in the class.

        The dictionary should contain at least two keys:

        - node_attributes: A dictionary itself, whose keys are the indices of any node feature vector and
          the corresponding value is a natural language description of what kind of node property that value
          represents
        - edge_attributes: Also a dictionary, the same thing for the edge feature vectors.
        """
        raise NotImplemented()

    def __imports__(self):
        """
        One way in which this class will be used is by copying its entire source code into a separate
        python module, which will then be shipped with each visual graph dataset as a standalone input
        processing functionality.

        All the code of a class can easily be extracted and copied into a template using the "inspect"
        module, but it may need to use external imports which are not present in the template by default.
        This is the reason for this method.

        Within this method all necessary imports for the class to work properly should be defined. The code
        in this method will then be extracted and added to the top of the templated module in the imports
        section.
        """
        pass
    
    # 08.11.24 - The following two methods are necessary when constructing a new model from a pre-existing
    # processing instance only. To construct the model one needs to know the input shapes which are derived 
    # from the processing instance.
    
    def get_num_node_attributes(self) -> int:
        """
        This method is supposed to return the integer number of node features that each node of the 
        processed graph representation has.
        
        :returns int:
        """
        raise NotImplementedError()
    
    def get_num_edge_attributes(self) -> int:
        """
        This method is supposed to return the integer number of edge features that each edge of the
        processed graph representation has.
        
        :returns int:
        """
        raise NotImplementedError()

    # -- dynamic command generation magic --

    def scaffold(self):
        """
        This method turns all the methods of this class which are listed in the DEFAULT_COMMANDS list by
        name into corresponding click.Command instances automatically and attaches them as properties to
        this instance.

        :returns: None
        """
        for command_name in self.DEFAULT_COMMANDS:
            command_method_name = f'{command_name}_command'

            # ONLY if the class does not already have a method with the given name do we actually
            # dynamically create one!
            # This is because the users should be able to define their own custom implementation of the
            # cli method if they want some extra custom behavior. The automatic scaffolding based on the
            # main functions is just a convenience feature.
            if not hasattr(self, command_method_name):
                method = getattr(self, command_name)

                # Given a name and a callable object, this method will dynamically construct a click command
                # object which implements that method as a CLI command.
                command = self.construct_command(command_name, method)
                # This command object is then added as a class attribute, as that means that it can be
                # discovered as an actual CLI command later.
                setattr(self, command_method_name, command)

    @classmethod
    def construct_command(cls, name: str, method: t.Callable) -> click.Command:
        """
        This method constructs a click.Command instance, given the ``name`` for the command and the callable
        method instance ``method`` which is to act as the command implementation and base template.

        The resulting command will use the functionality implemented in the given method when it is executed.

        :param name: the string name which the resulting command is supposed to go by
        :param method: A method callable instance of this class, which should be turned into the command.

        :returns: A command instance
        """
        # This function here will serve as the base template for the command object. This will now
        # progressively be wrapped in decorators to create the click command object which resembles the
        # given method.
        def command(**kwargs):
            value = method(**kwargs)

            # At the end of the
            if isinstance(value, (dict, list, np.ndarray)):
                click.echo(json.dumps(value, cls=NumericJsonEncoder))

        # We also need to transfer the doc string, because that doc string will later be displayed
        # as the help text in the command line.
        body, param_description_map = cls.parse_docstring(method.__doc__)
        command.__doc__ = body

        # In the first step we will use "inspect" to dynamically derive as many properties from the given
        # method as possible. This for example includes what kinds of arguments the method accepts and
        # also what the help text is.

        # ~ arguments and options
        # First of all we iterate through all the arguments of the method and correspondingly add the click
        # argument decorators such that the command has the same arguments as the method.
        arg_spec = inspect.getfullargspec(method)
        signature = inspect.signature(method)

        # We reverse the order of this list here because stacking the decorators on top of each other will
        # already reverse the order so by reversing the order here again we actually get the correct order
        # of the arguments as they appear in the method as well.
        for arg_name in reversed(arg_spec.args):
            # Obviously "self" is not actually an argument of the command but just an artifact of the method
            # and thus we ignore it here.
            if arg_name not in ['self']:
                arg = signature.parameters[arg_name]

                # By default, every argument will be assumed to be string. But here we try to learn more
                # from the type annotation of the argument. If a type annotation exists and it is a
                # simple type that we know (like int for example) we can get the corresponding click Type
                # object from the conversion dict and use that instead.
                # This is actually quite important because click performs automatic type casting of the
                # string cli inputs based on those annotations!
                arg_type = click.STRING
                if arg.annotation in cls.TYPE_CONVERSION_DICT:
                    # 22.03.2023 - If the annotation is directly an instance of click.ParamType we
                    # obviously want to use that. Also fixed a bug, where I didn't check if the type is
                    # even defined in the map, which could cause KeyError for exotic annotations.
                    if isinstance(arg_type, click.ParamType):
                        arg_type = arg.annotation
                    elif arg_type in cls.TYPE_CONVERSION_DICT:
                        arg_type = cls.TYPE_CONVERSION_DICT[arg.annotation]
                    else:
                        arg_type = click.STRING

                # Only if the argument does not have a default parameter we actually consider it as a
                # positional argument in the CLI context! If the argument has a default value it rather
                # resembles a CLI option instead.
                if arg.default == inspect.Parameter.empty:
                    # Argument
                    command = click.argument(
                        arg_name,
                        type=arg_type
                    )(command)
                else:
                    # Option
                    help_string = (f'{param_description_map.get(arg_name, "")}'
                                   f'type: {str(arg.annotation)} '
                                   f'default: {arg.default}')
                    command = click.option(
                        f'--{arg_name}',
                        type=arg_type,
                        default=arg.default,
                        help=help_string
                    )(command)

        # Finally, in the last layer we need to wrap the function in the actual click Command decorator so
        # that it will be recognized as such.
        command = click.command(name)(command)

        return command

    def get_commands_dict(self) -> t.Dict[str, click.Command]:
        """
        Returns a dict which defines all the supported commands dynamically. The keys of the dict are the
        string names of the commands and the values are the actual click.Command object instance which
        implement the functionality of those commands.

        :returns: dict
        """
        command_dict = {}
        # Essentially we want to dynamically detect all the commands which are implemented for this class
        # We do this by iterating through all the properties and whenever we encounter a click.Command
        # instance we add that to the dict, because these are going to be the methods which were
        # decorated with the click.command() decorator.
        for name in dir(self):
            attribute = getattr(self, name)
            if isinstance(attribute, click.Command):
                # Each click.Command has a "name" attribute here which we use as the command name in the
                # dict as well and NOT the method name. The method name can be different
                name = attribute.name
                command_dict[name] = attribute

        return command_dict

    def list_commands(self, ctx):
        """
        This method returns a list of strings, where each string is the name of a command which is supported
        by the command line interface.

        This method is required implementation for the click.MultiCommand base class

        :param ctx: The click context object

        :returns: list of strings
        """
        # The "get_commands_dict" method returns a dictionary, whose keys are the string names of supported
        # commands and the values are the actual click.Command objects which wrap the corresponding method
        # objects that implement the command functionality.
        command_dict = self.get_commands_dict()
        return list(command_dict.keys())

    def get_command(self, ctx, cmd_name):
        """
        Given the string name of a command returns the corresponding click.Command object.

        This method is required implementation for the click.MultiCommand base class

        :param ctx: The click context object
        :param cmd_name: The string name of the command

        :returns: click.Command
        """
        # The "get_commands_dict" method returns a dictionary, whose keys are the string names of supported
        # commands and the values are the actual click.Command objects which wrap the corresponding method
        # objects that implement the command functionality.
        command_dict = self.get_commands_dict()
        if cmd_name in command_dict:
            return command_dict[cmd_name]

    # -- Utils --

    @classmethod
    def save_figure(cls, fig, path: str):
        fig.savefig(path)

    @classmethod
    def save_metadata(cls, data: dict, path: str):
        with open(path, mode='w') as file:
            content = json.dumps(data, cls=NumericJsonEncoder)
            file.write(content)

    def array_from_figure(self,
                          figure: plt.Figure,
                          width: int,
                          height: int
                          ) -> np.ndarray:
        """
        Given a matplotlib ``figure`` Figure instance, this method will turn that into a numpy rgb image
        array of the dimensions (``width``, ``height``, 3).

        This is a functionality which is often needed for the subclass implementations as it makes more
        sense to implement the visualization into a pyplot figure than to an array. This method can then
        be used to create the required array format for the "visualize" method from that figure.

        :returns: numpy array with 3 dimensions
        """
        # This section turns the image which is currently a matplotlib figure object into a numpy array
        # https://stackoverflow.com/questions/35355930/matplotlib-figure-to-image-as-a-numpy-array
        canvas = FigureCanvas(figure)
        canvas.draw()
        array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        array = array.reshape((height, width, 3))

        return array

    @classmethod
    def parse_docstring(cls,
                        docstring: str
                        ) -> t.Tuple[str, dict]:
        """
        This method parses the docstring of a method and separates it into two parts, which it returns as a
        tuple. The first element is the string of the main body of the description. The second element is a
        dictionary, whose keys are the names of the parameters of the method and the values are the
        descriptions for those parameters given in the docstring (:param name: description).

        If a docstring does not contain the description with the :param: syntax for one of the parameters
        it will not appear in the dictionary!

        Also beware that this method only works with the spinx rst style of parameter descriptions! All
        other syntax styles are not handled by this method.

        :param docstring: The full docstring of a method which describes it's functionality as well as
            the sphinx syntax for the parameter descriptions.

        :returns: (str, dict)
        """
        # 22.03.2023 - We catch an error in the trivial case here where there exists no docstring
        if docstring is None or docstring == '':
            return '', {}

        # This expression essentially matches the entire docstring UNTIL the first occurrence of
        # a parameter specification.
        expr_main = re.compile(r'(?P<main>(\n|\s|.)+?)(?=(:param)|$)')
        matches: dict = expr_main.match(docstring).groupdict()
        content = matches['main']

        # The keys of this dictionary are going to be the names of the parameters and the values will be
        # the string descriptions (ONLY the descriptions)
        param_map = {}
        # This expression matches a parameter specification (:param name: description). Also supports multi
        # line descriptions.
        # holy moly, getting this regex right took me half an hour...
        expr_param = re.compile(r'(?P<param>:param\s(?P<name>.*):(?P<description>(.|\n)*?))(?=(:param)|$)')
        for m in expr_param.finditer(docstring):
            name = m['name']
            description = m['description']
            param_map[name] = description

        return content, param_map


def create_processing_module(processing: ProcessingBase,
                             template_name: str = 'process.py.j2'):
    """
    Given an instance ``processing`` of a subclass if ProcessingBase, this method will template the code
    string for an independent python module based on the functionality provided by the given ProcessingBase
    subclass.

    :param ProcessingBase processing: An instance of a subclass of the ProcessingBase base class
    :param str template_name: The string name of the template to be used from the template folder of the
        package

    :return: The code STRING for the module. This will still have to be written into a file
    """
    template = TEMPLATE_ENV.get_template(template_name)

    # The obvious thing we need to extract from the given instance for this purpose is the source code of
    # the class.
    class_name = processing.__class__.__name__
    if hasattr(processing, '__code__'):
        code = processing.__code__
    else:
        code = inspect.getsource(processing.__class__)

    # 05.05.23 - This was a bug where I did not account for the possibility that the base Processing
    # class code might not be defined at the top-level indent, so here we make sure to catch that
    # possibility and normalize the indent
    code = sanitize_indents(code)

    imports_method = getattr(processing, '__imports__')
    raw_imports_code = inspect.getsource(imports_method)
    # The problem with the code which we get from that is that it is the code for the *entire* method which
    # also includes the name of the method and the arguments and the identation. But we only want the code
    # itself.
    # To achieve this we will apply regular expression magic. This regex will extract the method name
    # definition line and the actual content of the method into two named match groups
    expr = re.compile(r'(?P<def>.*def .*\((.|\n)*\):)(?P<content>(\n|.)*)')
    matches = expr.match(raw_imports_code).groupdict()
    # To get rid of the identation of the body, we first count the indent level of the definition line and
    # then for the content we assume it is one more indent level (+4 spaces). Then we remove that many
    # leading characters from each line of the content to then finally get only the content
    indent = len(matches['def']) - len(matches['def'].lstrip(' ')) + 4
    imports_code = '\n'.join([line[indent:] for line in matches['content'].split('\n')])

    context = {
        'class_name':   class_name,
        'code':         code,
        'imports_code': imports_code
    }
    return template.render(context)


def graph_count_motif(graph: tv.GraphDict,
                      motif: tv.GraphDict,
                      processing: ProcessingBase,
                      ) -> int:
    """
    This function counts how often the subgraph ``motif`` appears in the overall ``graph`` structure.
    the function also requires a graph Processing instance ``processing`` to be passed as a parameter. The 
    node_match and edge_match methods defined for that processing instance will determine what kinds of 
    node and edge attributes are semantically relevant for matching the motif. The function will return 
    the integer number of times the graph isomorphism engine has found the motif within the overall graph.
    
    :param graph: The graph dict representation of the larger graph
    :param motif: The graph dict representation of the motif that is to be found in the larger graph
    :param processing: An instance of a ProcessingBase subclass for a specific graph family (such as 
        color graphs or molecular graphs)
        
    :returns: The integer number of matches
    """
    graph_nx = nx_from_graph(graph)
    motif_nx = nx_from_graph(motif)
    
    matcher = nx.isomorphism.GraphMatcher(
        graph_nx, 
        motif_nx, 
        node_match=lambda a, b: processing.node_match(a['node_attributes'], b['node_attributes']),
        edge_match=lambda a, b: processing.edge_match(a['edge_attributes'], b['edge_attributes']),
    )
    count = sum([1 for _ in matcher.subgraph_isomorphisms_iter()])
    return count
