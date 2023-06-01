"""
This module contains all the functionality which is related to persistent data management of visual graph
datasets, meaning stuff like saving and loading of datasets from files / dataset folders.
"""
import os
import time
import yaml
import json
import logging
import typing as t

import orjson
import numpy as np

import visual_graph_datasets.typing as tc
from visual_graph_datasets.util import NULL_LOGGER
from visual_graph_datasets.util import merge_optional_lists
from visual_graph_datasets.util import merge_nested_dicts


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
