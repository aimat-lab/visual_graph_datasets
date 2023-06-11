import os
import math
import tempfile

import yaml
import pytest
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import visual_graph_datasets.typing as tc
from visual_graph_datasets.typing import assert_graph_dict
from visual_graph_datasets.data import load_visual_graph_dataset
from visual_graph_datasets.data import generate_visual_graph_dataset_metadata
from visual_graph_datasets.data import load_visual_graph_dataset_metadata
from visual_graph_datasets.data import load_visual_graph_dataset_expansion
from visual_graph_datasets.data import load_visual_graph_element
from visual_graph_datasets.data import VisualGraphDatasetWriter
from visual_graph_datasets.data import VisualGraphDatasetReader
from visual_graph_datasets.data import extract_graph_mask

from .util import LOG
from .util import ASSETS_PATH, ARTIFACTS_PATH


def test_extract_graph_mask_basically_works():
    """
    05.06.23 - The ``extract_graph_mask`` function should take a graph dict and a node mask and then be able
    to create a NEW graph dict from that original one, which only contains the masked sections.
    """
    graph = {
        'node_indices': [0, 1, 2, 3, 4, 5],
        'node_attributes': [[0], [0], [1], [0], [1], [1]],
        'edge_indices': [
            # The first 4 nodes build a star shape around "1"
            [1, 0], [0, 1],
            [1, 2], [2, 1],
            [1, 3], [3, 1],
            # Then the other two nodes are just connected some random way
            [2, 4], [4, 2],
            [5, 3], [3, 5],
            [0, 4], [4, 0],
        ],
        'edge_attributes': [
            [1], [1],
            [1], [1],
            [1], [1],
            [1], [1],
            [1], [1],
            [1], [1],
        ]
    }
    # The first four nodes (the star pattern) we would like to extract from this graph as it's own graph
    # dict.
    mask = np.array([1, 1, 1, 1, 0, 0])
    sub_graph = extract_graph_mask(graph, mask)
    # most importantly the result must be a valid graph dict aside from that the lowest hanging tests are
    # just the correct shapes.
    assert_graph_dict(sub_graph)
    assert len(sub_graph['node_indices']) == 4
    assert len(sub_graph['node_attributes']) == 4
    assert len(sub_graph['edge_indices']) == 6


def test_visual_graph_dataset_reader_subset_works():
    """
    02.06.23 - One feature which is sometimes important is to be able to only read a subset of the dataset
    for example for testing purposes. this is being tested here
    """
    path = os.path.join(ASSETS_PATH, 'mock')

    reader = VisualGraphDatasetReader(
        path=path,
        logger=LOG,
        log_step=20,
    )
    index_data_map = reader.read(subset=50)
    assert len(index_data_map) == 50


def test_visual_graph_dataset_reader_works_with_chunking():
    """
    02.06.23 - This actually tests if the Reader instance can read a chunked dataset folder.
    """
    num_elements = 30
    chunk_size = 10

    with tempfile.TemporaryDirectory() as path:
        writer = VisualGraphDatasetWriter(
            path=path,
            chunk_size=chunk_size,
        )
        for index in range(num_elements):
            writer.write(
                name=index,
                metadata={'index': index},
                figure=None,
            )

        reader = VisualGraphDatasetReader(
            path=path,
            logger=LOG,
            log_step=100,
        )
        index_data_map = reader.read()
        # Since we have just created this, we know how many element
        assert len(index_data_map) != 0
        assert len(index_data_map) == num_elements


def test_visual_graph_dataset_reader_basically_works():
    """
    02.06.23 - The VisualGraphDatasetReader class is a new alternative to the load_visual_graph_dataset
    function, which has become necessary because that function would become very bloated when we now also
    have to add the chunking support to it...
    The read_all method of the writer class should essentially return the index_data_map as the function
    also does.
    """
    path = os.path.join(ASSETS_PATH, 'mock')

    reader = VisualGraphDatasetReader(
        path=path,
        logger=LOG,
        log_step=20,
    )
    index_data_map = reader.read()

    assert isinstance(index_data_map, dict)
    assert len(index_data_map) != 0
    # For this mock dataset we know that there are 100 elements in it, so we can test if it
    assert len(index_data_map) == 100
    # Also we can check if every individual element was loaded correctly.
    for index, data in index_data_map.items():
        assert 'metadata' in data
        assert 'metadata_path' in data
        assert 'image_path' in data
        tc.assert_graph_dict(data['metadata']['graph'])


def test_visual_graph_dataset_writer_chunking_works():
    """
    02.06.23 - The most important feature of why the VisualGraphDatasetWriter class exists is the automatic
    chunking of the dataset. Given a chunk size, the writer should automatically spread the dataset elements
    into sub folders which is supposed to make the dataset more efficient for IO operations.
    """
    num_elements = 29
    chunk_size = 10

    with tempfile.TemporaryDirectory() as path:
        writer = VisualGraphDatasetWriter(
            path=path,
            chunk_size=chunk_size,
        )
        for index in range(num_elements):
            writer.write(
                name=index,
                metadata={'index': index},
                figure=None,
            )

        elements = os.listdir(path)
        print(elements)
        # Since we are using chunking now, the actual dataset folder should on the top-level only contain
        # the vastly smaller amount of chunk folders...
        assert len(elements) == math.ceil(num_elements / chunk_size)


def test_visual_graph_dataset_writer_basically_works():
    """
    02.06.23 - The VisualGraphDatasetWriter is a class which is supposed to wrap the behavior of creating
    a new visual graph dataset folder. The write method can be used to commit a new element with a given
    index to the folder.
    """
    num_elements = 100

    with tempfile.TemporaryDirectory() as path:
        writer = VisualGraphDatasetWriter(path)
        for index in range(num_elements):
            writer.write(
                name=index,
                metadata={'index': index},
                figure=None,
            )

        files = os.listdir(path)

        # Since we are not saving figures as well there should now be exactly the given number of files
        # in that folder.
        assert len(files) == num_elements


def test_load_visual_graph_element_basically_works():
    """
    24.03.2023 - The ``load_visual_graph_element`` function is supposed to load only a single element from
    a visual graph dataset folder.
    """
    dataset_path = os.path.join(ASSETS_PATH, 'mock')
    data = load_visual_graph_element(dataset_path, '0')
    assert 'image_path' in data
    assert 'metadata' in data

    graph = data['metadata']['graph']
    tc.assert_graph_dict(graph)


def test_load_visual_graph_dataset_expansion_basically_works():
    """
    20.02.2023 - The ``load_visual_graph_dataset_expansion`` function is supposed to take an already
    existing index data map dictionary of a VGD and modify! it with the additional information found in a
    so-called VGD "expansion" folder. Such a folder may contain partial! metadata files which overwrite
    or extend the given dataset in specific fields.
    """
    dataset_path = os.path.join(ASSETS_PATH, 'mock')
    metadata_map, dataset_index_map = load_visual_graph_dataset(dataset_path)

    assert isinstance(dataset_index_map, dict)
    assert len(dataset_index_map) == 100

    # Loading the expansion
    # The "mock_expansion" modifies exactly the three elements with the indices 0, 1 and 5
    # Assigns them all the new metadata attribute "expansion" as True
    expansion_path = os.path.join(ASSETS_PATH, 'mock_expansion')
    load_visual_graph_dataset_expansion(dataset_index_map, expansion_path)

    for index in dataset_index_map.keys():
        metadata = dataset_index_map[index]['metadata']
        if index in [0, 1, 5]:
            assert 'expansion' in metadata.keys()
            assert len(metadata) > 1
            assert metadata['expansion'] is True
        else:
            assert 'expansion' not in metadata.keys()


def test_load_visual_graph_dataset_basically_works_mock():
    """
    If "load_visual_graph_datasets" generally works when loading the mock dataset without any errors and if
    all the elements are correctly loaded.
    """
    dataset_path = os.path.join(ASSETS_PATH, 'mock')
    metadata_map, dataset_index_map = load_visual_graph_dataset(dataset_path)

    assert isinstance(dataset_index_map, dict)
    assert len(dataset_index_map) == 100


def test_generate_visual_graph_datasets_metadata_basically_works_mock():
    """
    If "generate_visual_graph_datasets" works properly to generate the correct metadata on the "mock"
    dataset which is part of the test assets.
    """
    dataset_path = os.path.join(ASSETS_PATH, 'mock')
    _, dataset_index_map = load_visual_graph_dataset(dataset_path)

    metadata_map = generate_visual_graph_dataset_metadata(dataset_index_map)

    expected_values = {
        # The mock dataset is a very simple dataset with very few attributes and a binary one-hot encoded
        # classification target
        'num_node_attributes': 1,
        'num_edge_attributes': 1,
        'num_targets': 2
    }
    for key, value in expected_values.items():
        assert key in metadata_map
        assert metadata_map[key] == value, key


def test_load_visual_graph_dataset_metadata():
    dataset_path = os.path.join(ASSETS_PATH, 'mock')
    metadata_map, _ = load_visual_graph_dataset_metadata(path=dataset_path)
    # We just sporadically check that this is in fact a dict and not empty
    assert isinstance(metadata_map, dict)
    assert len(metadata_map) != 0
    assert 'version' in metadata_map


def test_yaml_loading_and_saving_multi_lines():
    """
    This is a simple test if pythons yaml functionality it capable of loading and saving multiline string
    values in yaml files appropriately, which is a pre-condition for using yaml files as dataset metadata
    files.
    """
    yaml_path = os.path.join(ASSETS_PATH, 'test.yaml')
    with open(yaml_path, mode='r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    assert isinstance(data, dict)
    assert 'multi_line' in data

    # The string has to have exactly 3 line breaks in it
    assert data['multi_line'].count('\n') == 3
    original = data['multi_line']

    # Now for the next step we will save this dict as a yaml file again and then load that new yaml file to
    # check if the multi-line string stays preserved in exactly the same way as it was.
    artifact_path = os.path.join(ARTIFACTS_PATH, 'test.yaml')
    with open(artifact_path, mode='w') as file:
        yaml.dump(data, file)
        assert os.path.exists(artifact_path)

    with open(artifact_path, mode='r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    assert 'multi_line' in data
    assert original == data['multi_line']


def test_bug_having_file_starting_with_period_breaks_loading():
    """
    27.01.2023: Files starting with a period break the loading process of VGDs
    """
    dataset_path = os.path.join(ASSETS_PATH, 'mock')
    _, dataset_index_map = load_visual_graph_dataset(dataset_path)

    # It doesn't even get here with the bug
    assert len(dataset_index_map) == 100
