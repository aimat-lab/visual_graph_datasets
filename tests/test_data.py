import os
import yaml
import pytest

import visual_graph_datasets.typing as tc
from visual_graph_datasets.data import load_visual_graph_dataset
from visual_graph_datasets.data import generate_visual_graph_dataset_metadata
from visual_graph_datasets.data import load_visual_graph_dataset_metadata
from visual_graph_datasets.data import load_visual_graph_dataset_expansion
from visual_graph_datasets.data import load_visual_graph_element

from .util import LOG
from .util import ASSETS_PATH, ARTIFACTS_PATH


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
