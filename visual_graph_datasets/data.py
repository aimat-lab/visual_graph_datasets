"""
This module contains all the functionality which is related to persistent data management of visual graph
datasets, meaning stuff like saving and loading of datasets from files / dataset folders.
"""
import os
import time
import yaml
import json
import logging
import shutil
import typing as t
from typing import Union, Optional

import orjson
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import visual_graph_datasets.typing as tc
import visual_graph_datasets.typing as tv
from visual_graph_datasets.util import NULL_LOGGER
from visual_graph_datasets.util import merge_optional_lists
from visual_graph_datasets.util import merge_nested_dicts
from visual_graph_datasets.util import dynamic_import


def str_presenter(dumper, data):
    """
    configures yaml for dumping multiline strings using the literal way "|" instead of ugly quotes

    https://stackoverflow.com/questions/8640959/how-can-i-control-what-scalar-form-pyyaml-uses-for-my-data
    https://github.com/yaml/pyyaml/issues/240
    """
    if data.count('\n') > 0:  # check for multiline string
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)


yaml.add_representer(str, str_presenter)


class NumericJsonEncoder(json.JSONEncoder):

    def default(self, o: t.Any) -> t.Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.generic):
            return o.item()
        else:
            return super(NumericJsonEncoder, self).default(o)


def nx_from_graph(graph: tv.GraphDict) -> nx.Graph:
    """
    Given a GraphDict ``graph``, this method will convert the graph dict representation into
    a networkx Graph object and return it.

    This networkx representation will also contain custom dynamically attached node and edge properties
    of the given graph dict.

    :param graph: The graph dict to be converted

    :return: nx.Graph
    """
    nx_graph = nx.Graph()

    node_keys = [key for key in graph.keys() if key.startswith('node')]
    edge_keys = [key for key in graph.keys() if key.startswith('edge')]

    for i in graph['node_indices']:
        nx_graph.add_node(i, **{key: graph[key][i] for key in node_keys})

    for e, (i, j) in enumerate(graph['edge_indices']):
        nx_graph.add_edge(i, j, **{key: graph[key][e] for key in edge_keys})

    return nx_graph


def extract_graph_mask(graph: tv.GraphDict, mask: np.ndarray) -> tv.GraphDict:
    """
    Given a GraphDict ``graph`` and a binary node masking array ``mask`` this function will extract the
    masked portion of the given graph and return a new GraphDict consisting exclusively of the masked
    areas.

    Additionally, the extracted graph will have an additional property called "node_indices_original" which
    is an array whose indices are the node indices of the extracted graph and the values are the integer
    node indices of the corresponding nodes in the original graph.

    - The masking may consist of disconnected areas of the original graph.
    - The masking will only include edges which are between two masked nodes. All edges which are only
      connected to a single masked node will be omitted.
    - The extraction procedure will also copy all node and edge related custom properties of the given
      graph dict, so long as they are canonically named by starting with either the prefix "node" or
      "edge". This specifically implies that the node_attributes and edge_attributes are correctly
      transferred as well.

    :param graph: A GraphDict from which to extract a sub structure
    :param mask: A binary numpy array (N, 1) mask defining which nodes should be extracted.

    :returns: GraphDict
    """
    node_indices = graph['node_indices']

    # with "node_adjacency" we construct an adj. matrix for the graph which is easier to work with in this
    # context than the edge list. It will be a very specific adj. matrix though. If there is no edge it
    # will be a negative value and if there is an edge the value will be the positive integer index of the
    # corresponding edge in the edge list of the graph dict! This will make the following procedure easier.
    node_adjacency = -1 * np.ones(shape=(len(node_indices), len(node_indices)))
    for e, (i, j) in enumerate(graph['edge_indices']):
        node_adjacency[i, j] = e

    # NOTE: All the variables in the following section with an underscore are the index variables in the
    # indexing-system of the new graph - the one which is the result of the masking process. All variables
    # without the underscore are index variables in the indexing-system of the original graph which is
    # provided as an input of this function.

    # here we ...
    node_mask_map = {}
    i_ = 0
    for i in node_indices:
        if mask[i]:
            node_mask_map[i] = i_
            i_ += 1

    # Then we also need the opposite mapping to re-associate it later on.
    mask_node_map = {i_: i for i, i_ in node_mask_map.items()}

    edge_mask_map = {}
    for i, i_ in node_mask_map.items():
        neighbor_indices = [(j, int(e)) for j, e in enumerate(node_adjacency[i]) if e >= 0]
        for (j, e) in neighbor_indices:
            if mask[j]:
                j_ = node_mask_map[j]
                edge_mask_map[e] = [i_, j_]

    mask_edge_map = {e_: e for e_, (e, _) in enumerate(edge_mask_map.items())}

    node_indices_ = list(range(len(node_mask_map)))
    edge_indices_ = list(edge_mask_map.values())
    masked_graph = {
        'node_indices':             np.array(node_indices_),
        # This is quite an important feature of the extraction process. Here we construct an array whose
        # indices are the node indices in the newly constructed extracted graph and the values in that
        # array are the integer node indices of the corresponding nodes in the ORIGINAL graph.
        # This information will be very important for further computations down the line.
        'node_indices_original':    np.array([mask_node_map[i_] for i_ in node_indices_]),
        'edge_indices':             np.array(edge_indices_),
        'edge_indices_original':    np.array([mask_edge_map[e_] for e_, _ in enumerate(edge_indices_)])
    }

    # ~ Transferring custom graph properties
    # Each graph dict consists of a set of absolutely required properties such as "node_indices" and
    # "edge_indices", but it is also possible that there are some dynamically attached additional properties
    # as well. Assuming that they follow a consistent naming scheme where node specific properties are
    # prefixed with "node" and the same for edges, we can transfer those to the new graph as well.

    node_keys = [key for key in graph.keys() if key.startswith('node') and key != 'node_indices']
    edge_keys = [key for key in graph.keys() if key.startswith('edge') and key != 'edge_indices']

    for key in node_keys:
        # For every node in the EXTRACTED graph we use the previously constructed mapping to get the
        # corresponding node index in the ORIGINAL graph to extract that node's value
        masked_graph[key] = [graph[key][mask_node_map[i_]] for i_ in range(len(node_indices_))]

    for key in edge_keys:
        masked_graph[key] = [graph[key][mask_edge_map[e_]] for e_ in range(len(edge_indices_))]

    return masked_graph


# This function will take a dataset and calculate all the metadata properties for the entire dataset which
# are possible to be determined automatically. At the same time this function will also check if a dataset
# is valid by checking if the extracted properties are consistent throughout the dataset.
def generate_visual_graph_dataset_metadata(dataset_map: tc.VisGraphIndexDict,
                                           validate: bool = True,
                                           ) -> dict:
    """
    Given a visual graph dataset dictionary, this function will create a partial metadata dict for that
    dataset as whole. The resulting metadata dict will contain all the information which is possible to
    be gathered automatically from the dataset's elements.

    If the ``validate`` flag is set, this function will raise an AssertionError in case any inconsistencies
    in the dataset are discovered while generating the metadata.

    :param dict dataset_map: The dictionary representing the VGD, whose keys are the element indices and the
        values are metadata dictionaries themselves describing each element.
    :param bool validate: Flag. If set, will raise assertion errors upon detecting issues with dataset.

    :returns: A dictionary containing the calculated metadata properties for the given dataset.
    """
    # Some very important metadata is about the dimensions of the data. This primarily means: How many
    # node/edge attributes are there? How many targets are there?
    # Finding this out could be done very efficiently in the dump way: by using the first element and simply
    # using it as a reference. We are taking the long route however and check every element in the dataset
    # this will simultaneously provide an opportunity to validate the correctness of the dataset in that
    # regard. I figured that custom dataset creations are a possibility it wouldn't be bad to include
    # automatic checks.
    num_node_attributes_set = set()
    num_edge_attributes_set = set()
    num_targets_set = set()
    for index, data in dataset_map.items():
        g = data['metadata']['graph']

        num_node_attributes_set.add(g['node_attributes'].shape[-1])
        num_edge_attributes_set.add(g['edge_attributes'].shape[-1])
        num_targets_set.add(g['graph_labels'].shape[-1])

    if validate:
        for num_set, desc in zip([num_node_attributes_set, num_edge_attributes_set, num_targets_set],
                                 ['node attributes', 'edge attributes', 'target values']):
            assert len(num_set) == 1, (f'The number of {desc} is expected to be consistent throughout the '
                                       f'dataset, however the following deviating values have been detected '
                                       f'{num_set}. Please fix your dataset in this regard!')

    # Another important, but simple to get information is about the total number of elements in the
    # dataset and the accumulated file size which that results in.
    total_file_size = 0
    num_elements = len(dataset_map)
    # TODO: Move the calculation of an elements file size contribution to the dataset loading part.
    for index, data in dataset_map.items():
        total_file_size += os.path.getsize(data['metadata_path'])
        total_file_size += os.path.getsize(data['image_path'])

    return {
        'num_node_attributes': list(num_node_attributes_set)[0],
        'num_edge_attributes': list(num_edge_attributes_set)[0],
        'num_targets': list(num_targets_set)[0],
        'num_elements': num_elements,
        'file_size': total_file_size,
    }


class DatasetReaderBase:
    """
    This is the base class for reading visual graph datasets from their persistent folder representation
    into memory.

    Reading the dataset
    -------------------

    Visual graph datasets are persistently represented as *folders* in the filesystem. Ultimately each
    element of the dataset is represented as *two* files - (1) a JSON file which contains all the
    metadata about the element as well as it's already fully pre-processed graph representation - (2) a
    PNG visualization of the graph.

    To read a dataset only the absolute ``path`` to that main folder is required as an argument for the
    constructor of a Writer instance. To read the dataset elements, the ``read`` function can be used. It
    returns a dictionary whose keys are the integer indices of the elements and whose elements are again
    dicts that contain all the necessary information about the element:

    - image_path: The absolute string path to the visualization image
    - metadata_path: The absolute string path to the metadata file.
    - metadata: the metadata dict including the graph representation

    A base class for different datasets
    -----------------------------------

    This class is only the base class for specific implementations of different datasets. This base class
    implements the generic processing of the dataset such as the discovery and enumeration of the all the
    different files within the file folder; including the handling of dataset chunking.

    The class method ``read_element``, however, has to be implemented for each subclass separately. This
    allows the extension towards slightly different variations of the VGD format. This mainly includes the
    the "visual graph dataset" format where each graph is visualized as a PNG image. In the future, however,
    it would be possible to extend this to a "text graph datasets" as well, where a natural language text
    is being treated as a graph. In this case the visualization would most likely not be a png but more
    likely a text or latex file.
    """
    def __init__(self,
                 path: str,
                 logger: logging.Logger = NULL_LOGGER,
                 log_step: int = 1000,
                 element_extensions: t.List[str] = ['.json', '.png']
                 ):
        self.path = path
        self.logger = logger
        self.log_step = log_step
        self.element_extensions = element_extensions

        # The name of the dataset is simply the name of the main folder
        self.name: str = os.path.basename(self.path)

        # This is a list of all the top-level contents of the main dataset folder.
        # For legacy datasets this is simply all the actual element files, for new datasets this is the
        # list of all the chunk folders.
        self.list: t.List[str] = list(sorted(os.listdir(self.path)))
        self.meta_path = os.path.join(self.path, '.meta.yml')
        self.process_path = os.path.join(self.path, 'process.py')

        # This will be the final reading result that represents the dataset as a whole. The keys will be
        # the int indices of the individual elements and the values will be dicts containing all the
        # element's information.
        self.index_data_map: t.Dict[int, dict] = {}

        # NOTE ABOUT CHUNKING: This class can read datasets with and without chunking. When chunking is
        # present, then all the chunks will have a unique index. Datasets without chunking will be handled
        # in the same way but by using "None" as the chunk index internally.

        # This dictionary will store the absolute paths to all the chunk folders. The keys will be the int
        # chunk indices and the values will be the absolute string paths to the chunk FOLDERS
        self.chunk_paths: t.Dict[t.Optional[int], str] = {
            None: self.path,
        }
        # This dictionary will contain the chunk contents. The keys are the int chunk indices and the values
        # are sets; these sets will contain the unique string names of all the dataset elements which are
        # part of that corresponding chunk.
        self.chunk_map: t.Dict[t.Optional[int], t.Set[str]] = {
            None: set()
        }
        # This method will actually populate the two previous dicts by iterating through the contents of
        # all the chunks in the main dataset folder. Only after this method will these dicts actually be
        # populated with the content
        self.resolve_chunking()

        self.num_files = sum(len(file_paths) for file_paths in self.chunk_map.values())

    # Reading the actual element data

    def resolve_chunking(self):
        """
        This method will iterate through all the contents of the main dataset folder and resolve any
        dataset chunking - if it exists - internally into a set of element names that can be processed
        directly.

        :returns None:
        """
        for name in sorted(self.list):
            element_path = os.path.join(self.path, name)
            if os.path.isdir(element_path) and 'chunk' in name:
                chunk_index, _, chunk_size = name.split('_')
                chunk_index = int(chunk_index)
                self.chunk_map[chunk_index] = set()
                self.chunk_paths[chunk_index] = element_path

                for sub_name in sorted(os.listdir(element_path)):
                    base, extension = os.path.splitext(sub_name)
                    self.chunk_map[chunk_index].add(base)

            # There are two conditions to consider a file as a valid file actually representing a dataset
            # element as opposed to for example a dataset-global metadata file.
            # The first one is that every file starting with a colon will be considered a "hidden" file
            # and thus be ignored; The second is that the file has to have one of the valid file extensions
            non_hidden_file = not name.startswith('.') and name != ''
            has_valid_extension = any([name.endswith(ext) for ext in self.element_extensions])
            if non_hidden_file and has_valid_extension:
                base, extension = os.path.splitext(name)
                self.chunk_map[None].add(base)

    def read(self, subset: t.Optional[int] = None) -> dict:
        """
        This method will actually read all the element files from the dataset folder and return the
        resulting ``self.index_data_map``.

        Optionally it is possible to only load a ``subset`` of the data by specifying the integer amount
        of data to be loaded from the dataset, which is smaller than the total number of elements. Note
        that this option will load the first N elements of the dataset and the dataset elements are sorted
        by their string filenames.

        :param subset: The integer amount of elements to load from the dataset which is smaller than the
            total size of the dataset.

        :returns: index_data_map dict.
        """
        c = 0
        _break = False
        # On the top level we iterate through all the chunks (which are sorted by chunk index)
        # self.chunk_map is a dict that maps chunk index to a set of string names, where each of these
        # string names belongs to one element that is part of that chunk
        for chunk_index, names in self.chunk_map.items():
            self.logger.info(f' * processing chunk {chunk_index}...')
            chunk_path = self.chunk_paths[chunk_index]

            # And then for each chunk we iterate through all the dataset elements in that folder (which are
            # sorted by name)
            for name in sorted(names):
                # How exactly the dataset element is loaded from that folder is implemented in the specific
                # Reader subclass which each implement slight variations of the basic VGD format.
                data = self.read_element(chunk_path, name)
                index = data['metadata']['index']
                self.index_data_map[index] = data

                c += 1
                if c % self.log_step == 0:
                    self.logger.info(f' * loaded ({c}/{self.num_files})')

                # Slightly complicates setup to terminate both layers of for-loops when the max subset of
                # elements from the dataset is reached.
                if subset is not None and c >= subset:
                    _break = True
                    break

            if _break:
                break

        return self.index_data_map
    
    def read_indices(self, indices: t.List[int]) -> t.List[dict]:
        """
        This method actually reads a subset of the dataset from the disk and returns a list containing the corresponding 
        data dictionaries for the read elements.

        The main advantage of this method is that it allows for index duplicates! So it is possible for any index 
        to appear multiple times in the index list which will also result in the corresponding element to be 
        added to the returned list of elements the same number of times.
        
        :returns: A list of dictionaries, where each dictionary is a data dictionary that describes one visual 
            graph element.
        """
        raise NotImplemented()

    def chunk_iterator(self) -> dict:
        for chunk_index, names in self.chunk_map.items():
            
            chunk_path = self.chunk_paths[chunk_index]
            index_data_map = {}
            
            for name in sorted(names):
                data = self.read_element(chunk_path, name)
                index = data['metadata']['index']
                index_data_map[index] = data
                
            yield index_data_map

    # Reading dataset metadata

    def read_process(self) -> t.Any:
        """
        This method will load the pre-processing module ``process.py`` which is part of the main dataset
        folder. The method will return the corresponding python module instance, from which the Processing
        instance can be loaded like this:

        .. code-block: python

            module = writer.read_process()
            processing = module.processing

        :returns: a python module instance
        """
        if not os.path.exists(self.process_path):
            raise ValueError(f'The given visual graph dataset folder "{self.path}" does not include the '
                             f'a process.py pre-processing file!')

        module = dynamic_import(self.process_path, f'process_{self.name}')
        return module

    def read_metadata(self) -> dict:
        """
        Reads the contents of the ``.meta.yml`` file which is part of the main dataset folder and returns
        the contents as a dictionary. This metadata dictionary will contain metadata about the dataset as
        a whole.

        :returns: metadata dict
        """
        if not os.path.exists(self.meta_path):
            raise ValueError(f'The given visual graph dataset folder "{self.path}" does not include the '
                             f'a .meta.yml metadata file!')

        with open(self.meta_path) as file:
            data = yaml.load(file, yaml.FullLoader)

        return data
    
    def __len__(self) -> int:
        length = 0
        for _, names in self.chunk_map.items():
            length += len(names)
            
        return length

    @classmethod
    def safe_list_remove(cls, lst: list, value: t.Any) -> None:
        try:
            lst.remove(value)
        except ValueError:
            pass

    @classmethod
    def read_graph(cls, graph: dict) -> dict:
        """
        Given the raw loaded GraphDict representation, this method returns that same graph dict where all
        the elements are converted into numpy arrays.

        :returns: dict
        """
        return {key: np.array(value) for key, value in graph.items()}

    @classmethod
    def read_element(cls, path: str, name: str) -> dict:
        """
        This method needs to be implemented by each specific implementation of the Reader class.

        This method takes the string absolute folder ``path`` and the string ``name`` of a dataset element
        and with that information it has to load the actual files that represent that dataset element in the
        given folder and return the correctly constructed ``data`` dict.

        :returns: dict
        """
        raise NotImplemented()


class VisualGraphDatasetReader(DatasetReaderBase):
    """
    The specific implementation to read a "visual graph dataset" format. This format represents one dataset
    element as exactly two files:
    - A JSON file containing the full graph representation and the metadata
    - A PNG file which is a visualization of the graph
    """
    @classmethod
    def read_element(cls, path: str, name: str) -> dict:
        data = {}

        metadata_path = os.path.join(path, f'{name}.json')
        with open(metadata_path, mode='r') as file:
            content = file.read()
            metadata = json.loads(content)
            # If the graph is part of the element, then we are going to convert all the values of that graph
            # dict to numpy arrays with that class method.
            if 'graph' in metadata:
                metadata['graph'] = cls.read_graph(metadata['graph'])

            data['metadata'] = metadata
            data['metadata_path'] = metadata_path

        # All the files belonging to the same dataset element always have the same actual file name but
        # simply different file extensions.
        image_path = os.path.join(path, f'{name}.png')
        data['image_path'] = image_path

        return data


class DatasetWriterBase:
    """
    This is the base class which implements the process of writing a dataset element from in-memory data
    structure representations into a persistent representation within a visual graph dataset folder.

    Writing dataset elements
    ------------------------

    To write a new graph dataset, the absolute string ``path`` of a folder has to be passed as an argument
    to the constructor of a Writer instance. ``chunk_size`` is an optional int argument which specifies how
    many elements to save into each "chunk" sub folder within the dataset.

    Writing dataset elements can be done by using the ``write`` method. The specific implementation of this
    method has to be done in each specific subclass. The first argument will always be the unique string
    ``name`` of the dataset element and the following arguments are determined by the specific
    implementations but generally have to consist of all the data structures that are necessary to fully
    represent one element of the dataset.
    """

    def __init__(self,
                 path: str,
                 chunk_size: t.Optional[int] = None):
        self.path = path
        self.chunk_size = chunk_size

        # This dictionary
        self.most_recent: t.Dict[str, t.Any] = {}

        self.current_chunk_names: t.List[str] = []
        self.current_chunk = 0
        if self.chunk_size is None:
            self.current_path = self.path
        else:
            self.current_path = self.create_chunk(self.current_chunk)

    def write(self,
              name: t.Union[int, str],
              *args,
              **kwargs
              ) -> None:
        """
        This function has to be implemented by each specific subclass. The method arguments should include
        all the specific data structures that are important to that specific format.

        At the end of this function the following code should be called to update the internal state of the
        writer: ``self.add_element(name)``!

        :returns: None
        """
        raise NotImplemented()

    def add_element(self, name: str):
        """
        Given the unique string ``name`` of the most recently written dataset ``element``, this method will
        have to be called to update the internal writer state - important for chunking.

        :returns None:
        """
        self.current_chunk_names.append(name)
        self.update_chunking()

    def update_chunking(self):
        """
        This method will update the chunking state of the writer. Chunking will split the dataset into
        several sub folders which will only contain a certain ``self.chunk_size`` amount of dataset elements
        to increase the IO efficiency of the dataset.

        This method will check if the number of elements saved into the current chunk has reached that max
        chunk size and if that is the case will create a new chunk folder and change the internal variables
        to point to that folder instead.
        """
        if self.chunk_size is not None and len(self.current_chunk_names) >= self.chunk_size:
            self.current_chunk_names = []
            self.current_chunk += 1
            self.current_path = self.create_chunk(self.current_chunk)

    def create_chunk(self, chunk_index: int) -> str:
        """
        This method will set up a new chunk folder with the given integer ``chunk_index``. This means that
        based on that that index a new folder will be created within the main dataset folder. Only after
        this method has been called can elements be saved into the new chunk.

        :param chunk_index: The integer index of the chunk folder to be newly created.

        :returns: The absolute string path to the newly created chunk folder
        """
        chunk_name = f'{chunk_index:06d}_chunk_{self.chunk_size}'
        chunk_path = os.path.join(self.path, chunk_name)
        os.mkdir(chunk_path)

        return chunk_path


class VisualGraphDatasetWriter(DatasetWriterBase):
    """
    The specific implementation of the Writer class for the "visual graph dataset" format. For this format
    the ``write`` method expects two arguments that represent each dataset element:

    - metadata: A dictionary containing the metadata that will be written in the JSON file of the element.
      This metadata dict should also contain the ``graph`` key which should itself be the full GraphDict
      representation of the graph.
    - figure: A matplotlib (frameless) Figure object which contains the visualization of the graph, that
      should then be saved as the elements corresponding PNG file.

    """
    def __init__(self,
                 path: str,
                 chunk_size: Optional[int] = None):
        super(VisualGraphDatasetWriter, self).__init__(path, chunk_size)

    def write(self,
              name: t.Union[int, str],
              metadata: Optional[dict],
              figure: Optional[Union[plt.Figure, str]],
              *args,
              **kwargs,
              ) -> None:

        if isinstance(name, str):
            file_name = name
        elif isinstance(name, int):
            file_name = f'{name:07d}'

        self.most_recent = {}

        if metadata is not None:
            metadata_path = os.path.join(self.current_path, f'{file_name}.json')
            self.most_recent['metadata_path'] = metadata_path
            with open(metadata_path, mode='w') as file:
                # content = orjson.dumps(
                #     metadata,
                #     option=orjson.OPT_SERIALIZE_NUMPY
                # )
                content = json.dumps(
                    metadata,
                    cls=NumericJsonEncoder,
                )
                file.write(content)

        if figure is not None:
            
            image_path = os.path.join(self.current_path, f'{file_name}.png')
            self.most_recent['image_path'] = image_path
            
            # In case the figure argument is a string we are goung to assume that it is the absolute path 
            # to an already existing graph visualization image and in that case we only need to copy 
            # that file!
            if isinstance(figure, str):
                shutil.copy(figure, image_path)
                
            # Otherwise the default case is that the graph visualization will be given as a plt figure object 
            # which which case we need to save that to the path.
            elif isinstance(figure, plt.Figure):
                figure.savefig(image_path)
                plt.close(figure)
                
            else:
                raise TypeError(f'cannot save figure of type: {type(figure)}!')

        # Calling this method is important to update the writers chunking state and should always be called
        # at the end of the write method!
        self.add_element(name)


# The previous function is not to be confused with this one: This function merely loads all the metadata
# which is already present as metadata files within a given dataset folder.
def load_visual_graph_dataset_metadata(path: str,
                                       logger: logging.Logger = NULL_LOGGER
                                       ) -> t.Tuple[dict, dict]:
    """
    Given the absolute string ``path`` to a visual graph dataset folder, this function will load all the
    additional metadata files from that folder and return a dictionary which contains all the metadata
    information which was contained within those files.

    :param str path: The absolute path to a valid visual graph dataset folder
    :param str logger: Optionally a Logger instance to log the progress
    :returns: A dictionary containing all the metadata about the dataset
    """
    # Inside a valid visual graph dataset folder, generally all files whose name start with a dot "." are
    # considered to be metadata files.
    # The keys of this dict are the file names (sans the dot at the front) and the values are the
    # corresponding absolute file paths.
    file_map: t.Dict[str, str] = {name: os.path.join(path, name)
                                  for name in os.listdir(path)
                                  if name.startswith('.')}

    data = {}
    # There is one special file name which is the most important: ".meta.yml". This file contains the bulk
    # of the metadata encoded as a yaml file, so we load that file here, if it exists
    if '.meta.yml' in file_map:
        logger.info('loading dataset metadata from .meta.yml file...')
        file_path = file_map['.meta.yml']
        with open(file_path, mode='r') as file:
            data.update(yaml.load(file, Loader=yaml.FullLoader))
    else:
        logger.info('no .meta.yaml file found for the dataset!')

    # 21.03.2023 - As a convenience feature we will add the absolute path towards the pre-processing module
    # process.py to the metadata as well.
    data['process_path'] = os.path.join(path, 'process.py')

    # TODO: Add support for arbitrary metadata files
    return data, {}


def load_visual_graph_element(path: str,
                              name: str,
                              ) -> dict:
    """
    A utility function to load a single visual graph element given the folder ``path`` for the folder in
    which its files are stored in and the ``name`` of the element. Since a visual graph element consists
    of multiple files, all files are expected to have the same name!.

    :param path: The string absolute path to the FOLDER which contains the element files
    :param name: The name, which all the element files share (without any file extensions!)

    :returns: A dictionary which represents the element. It has two fields 'metadata' which contains the
        full metadata dictionary of the element, and 'image_path' which contains the absolute path to the
        corresponding visualization image.
    """
    element_data = {}

    metadata_path = os.path.join(path, f'{name}.json')
    with open(metadata_path, mode='r') as file:
        content = file.read()
        metadata = json.loads(content)

        # 23.05.23 - This has been causing issues in a few places. The method load_visual_graph_dataset
        # which loads the whole dataset already pre-converts the graph properties to numpy arrays and this
        # function to load a single element did not do the same until now, which meant that it could not
        # be used interchangeably in a lot of places.
        if 'graph' in metadata:
            graph = metadata['graph']
            for key, value in graph.items():
                graph[key] = np.array(value)

    element_data['metadata'] = metadata

    image_path = os.path.join(path, f'{name}.png')
    element_data['image_path'] = image_path

    return element_data


def load_visual_graph_dataset(path: str,
                              logger: logging.Logger = NULL_LOGGER,
                              log_step: int = 100,
                              subset: t.Optional[int] = None,
                              # deprecated parameters
                              metadata_contains_index: bool = False,
                              image_extensions: t.Optional[t.List[str]] = ['png'],
                              text_extensions: t.Optional[t.List[str]] = None,
                              metadata_extensions: t.List[str] = ['json'],
                              ) -> t.Tuple[tc.VisGraphNameDict, tc.VisGraphIndexDict]:

    reader = VisualGraphDatasetReader(
        path=path,
        logger=logger,
        log_step=log_step,
    )
    index_data_map = reader.read(subset=subset)
    try:
        metadata_map = reader.read_metadata()
    except:
        metadata_map = {}

    return metadata_map, index_data_map


def _load_visual_graph_dataset(path: str,
                              logger: logging.Logger = NULL_LOGGER,
                              log_step: int = 100,
                              metadata_contains_index: bool = False,
                              subset: t.Optional[int] = None,
                              image_extensions: t.Optional[t.List[str]] = ['png'],
                              text_extensions: t.Optional[t.List[str]] = None,
                              metadata_extensions: t.List[str] = ['json'],
                              ) -> t.Tuple[tc.VisGraphNameDict, tc.VisGraphIndexDict]:
    """
    Loads a visual graph dataset, given the absolute path to the base folder that contains the dataset files.

    :param path: The absolute string path to the dataset folder
    :param logger: Optionally a Logger instance to log the progress of the loading process
    :param log_step: The number of files after which a log message should be printed
    :param metadata_contains_index: A boolean flag that determines if the canonical indices of the element
        can be found within the metadata of the element. If this is True, then the index of each element
        will be retrieved from the "index" field of the metadata. Otherwise, the index will be determined
        according to the order in which the files are loaded from the filesystem.
    :param image_extensions: -
    :param metadata_extensions: -
    :return: tuple where the first element is a dictionary containing all the extra metadata with which the
        dataset was (optionally) annotated. The second element is a dict which contains the dataset itself
        It's keys are the integer element indices and the values are dictionaries containing all the
        relevant information about the corresponding element of the dataset.
    """
    files = os.listdir(path)
    files = sorted(files)
    num_files = len(files)
    dataset_name = os.path.basename(path)

    valid_extensions = merge_optional_lists(image_extensions, text_extensions, metadata_extensions)

    logger.info(f'starting to load dataset "{dataset_name}" from {len(files)} files...')

    metadata_map, _ = load_visual_graph_dataset_metadata(path, logger=logger)

    start_time = time.time()

    dataset_name_map: t.Dict[str, dict] = {}
    dataset_index_map: t.Dict[int, dict] = {}
    current_index = 0
    inserted_names = set()
    for c, file_name in enumerate(files):
        # This will prevent errors for the case that some other kinds of files have polluted the folder
        # which do not have a file extension, which has caused exceptions in the past.
        if '.' not in file_name:
            continue

        # 30.12.2022
        # This is an important addition with the introduction of the dataset metadata feature which allows
        # to add additional files to dataset folders. All these metadata files start with a dot and are
        # obviously supposed to be ignored in terms of actual dataset elements here.
        if file_name.startswith('.'):
            continue

        name, extension = file_name.split('.')
        file_path = os.path.join(path, file_name)

        if extension in valid_extensions:
            # First of all there are some actions which need to performed regardless of the file type, mainly
            # to check if an entry for that name already exists in the dictionary and creating a new one if
            # that is not the case.
            # You might be thinking that this is a good use case for a defaultdict, but actually for very
            # large datasets a defaultdict will somehow consume way too much memory and should thus not be
            # used.
            if name not in inserted_names:
                dataset_name_map[name] = {}
                inserted_names.add(name)

            if image_extensions is not None and extension in image_extensions:
                dataset_name_map[name]['image_path'] = file_path

            # 16.11.2022
            # Some datasets may not be visual in the sense of providing pictures but rather natural
            # language datasets which were converted to text and thus the visualization basis will be the
            # the original text file.
            if text_extensions is not None and extension in text_extensions:
                dataset_name_map[name]['text_path'] = file_path

            if extension in metadata_extensions:
                dataset_name_map[name]['metadata_path'] = file_path
                # Now we actually load the metadata from that file
                # We use orjson here because it's faster than the standard library and we need all the speed
                # we can get for this.
                with open(file_path, mode='rb') as file:
                    content = file.read()
                    metadata = orjson.loads(content)
                    dataset_name_map[name]['metadata'] = metadata

                    # NOTE: This is incredibly important for memory management. If we do not convert the
                    # graph into numpy arrays we easily need almost 10 times as much memory per dataset
                    # which is absolutely not possible for the larger datasets...
                    dataset_name_map[name]['metadata']['graph'] = {k: np.array(v)
                                                                   for k, v in metadata['graph'].items()}

                del file, content

                # Either there is a canonical index saved in the metadata or we will assign index in the
                # order in which the files are loaded. NOTE this is a bad method because the order will be
                # OS dependent and should be avoided
                if metadata_contains_index:
                    index = metadata['index']
                else:
                    index = current_index
                    current_index += 1

                dataset_index_map[index] = dataset_name_map[name]

            if subset is not None and c > subset:
                break

            if c % log_step == 0:
                elapsed_time = time.time() - start_time
                remaining_time = (num_files - c) * (elapsed_time / (c + 1))
                logger.info(f' * ({c}/{num_files})'
                            f' - name: {name}'
                            f' - elapsed time: {elapsed_time:.1f} sec'
                            f' - remaining time: {remaining_time:.1f} sec')

    return metadata_map, dataset_index_map


def load_visual_graph_dataset_expansion(index_data_map: tc.VisGraphIndexDict,
                                        expansion_path: str,
                                        logger: logging.Logger = NULL_LOGGER,
                                        log_step: int = 100):

    assert os.path.exists(expansion_path), (
        f'The given visual graph dataset expansion path {expansion_path} does not exist and can thus '
        f'not be loaded! Please make sure the path exists and is spelled correctly.'
    )

    assert os.path.isdir(expansion_path), (
        f'The given visual graph dataset expansion path {expansion_path} is not a directory! A VGD '
        f'expansion has to be a directory much like a VGD dataset itself aswell.'
    )

    # We open the folder, iterate all the files and check if the described elements correlate with any of
    # the elements in the given visual graph dataset.
    base_path = expansion_path
    file_names = os.listdir(expansion_path)
    num_files = len(file_names)

    for c, file_name in enumerate(file_names):
        # As with the visual graph datasets themselves we also ignore files starting with a period
        if file_name.startswith('.'):
            continue

        # Technically at this point I think it makes sense to only accept json metadata files I dont see
        # what the use of other file types would be right now
        if file_name.endswith('.json'):

            file_path = os.path.join(base_path, file_name)
            with open(file_path) as file:
                content = file.read()
                metadata = json.loads(content)

            # Now we need to check if this data matches any of the existing elements in the given VGD
            # At this point we do this with the "index" information but in the future we can think about
            # if we need to support a different method as well
            index = metadata['index']
            if index in index_data_map:
                metadata_original = index_data_map[index]['metadata']
                # The two metadata dicts now need to be nested merged and then that version should be used
                # in the VGD
                metadata_merged = merge_nested_dicts(metadata_original, metadata)
                index_data_map[index]['metadata'] = metadata_merged

            else:
                logger.info(f' ! not found original entry for index {index}')

        if c % log_step == 0:
            logger.info(f' * ({c}/{num_files}) processed')


# This function is not to be confused with the functions above "generate_visual_graph_dataset_metadata"
# and "load_visual_graph_dataset_metadata". These two are rather simple ones which have singular and well
# defined purpose. This function is more complex: It will load a visual graph dataset given by its path,
# together with the metadata, then afterward will generate the metadata from the elements and update the
# originally loaded metadata dict with the newly calculated information.
def gather_visual_graph_dataset_metadata(path: str,
                                         logger: logging.Logger = NULL_LOGGER,
                                         validate: bool = True,
                                         **kwargs,
                                         ) -> tc.VisGraphMetaDict:
    # First we load the entire dataset from the disk
    metadata_map, index_data_map = load_visual_graph_dataset(path, logger=logger)

    # Then we need to generate the calculated metadata from the actual elements and use that to update
    # the original dict.
    generated_metadata_map = generate_visual_graph_dataset_metadata(index_data_map, validate=validate)
    metadata_map.update(generated_metadata_map)

    return metadata_map


class DatasetFolder:

    def __init__(self, path: str):
        self.path = path
        self.name = os.path.basename(self.path)

    def get_size(self) -> int:
        """
        Returns the size of the dataset in bytes

        :return:
        """
        total_size = 0

        files = os.listdir(self.path)
        for file_name in files:
            file_path = os.path.join(self.path, file_name)
            total_size += os.path.getsize(file_path)

        return total_size

    def __len__(self):
        files = os.listdir(self.path)
        counter = 0
        for file_name in files:
            if '.' in file_name:
                name, extension = file_name.split('.')
                if extension in ['json']:
                    counter += 1

        return counter

    def get_metadata(self) -> dict:
        return {
            'dataset_size': self.get_size(),
            'num_elements': len(self)
        }


def create_datasets_metadata(datasets_path: str) -> dict:
    """
    Given the absolute string path ``datasets_path`` to a folder, which itself contains multiple valid
    visual graph dataset folders, this function will create combined metadata dict for all these datasets.

    This is accomplished by merging the individual metadata information of each dataset, which is
    respectively represent by the ".meta.yaml" file within each dataset folder. If one such metadata file
    does not exist for a dataset, an empty dictionary will be used for that dataset.

    :param str datasets_path: The absolute string path towards a folder which itself contains multiple
        valid visual graph dataset folders.
    :returns: A dictionary containing the metadata of ALL the datasets in the given folder
    """
    members = os.listdir(datasets_path)

    dataset_metadata_map = {}
    for member in members:
        member_path = os.path.join(datasets_path, member)
        if os.path.isdir(member_path):
            dataset_metadata_map[member], _ = load_visual_graph_dataset_metadata(member_path)

    return {
        'datasets': dataset_metadata_map
    }
