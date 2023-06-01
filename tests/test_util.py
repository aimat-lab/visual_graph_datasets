import os
import tempfile
import random

import pytest

import jinja2 as j2

from visual_graph_datasets.data import load_visual_graph_dataset
from visual_graph_datasets.util import TEMPLATE_ENV
from visual_graph_datasets.util import get_version
from visual_graph_datasets.util import get_dataset_path
from visual_graph_datasets.util import ensure_folder
from visual_graph_datasets.util import merge_nested_dicts
from visual_graph_datasets.util import sanitize_indents
from visual_graph_datasets.util import Batched


def test_batched_iterator_works():
    """
    ``Batched`` is a class that works as a custom iterator which will split a list into batches of a fixed
    size and then iterates those batches.
    """
    length = 1000
    batch_size = 100
    data = [random.random() for _ in range(length)]
    for batch in Batched(data, batch_size):
        assert len(batch) == 100

    # It should also work if the batch size does not directly match the number of elements
    num_total = 0
    batch_size = 32
    for batch in Batched(data, batch_size):
        assert len(batch) <= batch_size
        num_total += len(batch)

    # And in such cases it is important that overall still all of the elements will be visited
    assert num_total == length


def test_sanitize_indents():
    """
    The ``sanitize_indents`` function should remove all the additional indents from a string that consists
    of multiple lines such that the line with the min. indent does not have any indent at all.
    """
    # One problem with the triple quote string definition is that the method indent will be added to
    # the string as well.
    string = """
        This text should have some indents
            which we dont want
    """
    # If we use the brackets definition of the string however those additional indents won't be part of it
    expected = (
        'This text should have some indents\n'
        '    which we dont want'
    )
    assert expected != string
    assert expected == sanitize_indents(string)

def test_get_version():
    version = get_version()
    assert isinstance(version, str)
    assert len(version) != 0


def test_get_dataset_folder():
    # If we supply the correct name it should work without problem
    dataset_path = get_dataset_path('rb_dual_motifs')
    assert isinstance(dataset_path, str)
    assert os.path.exists(dataset_path)

    # If we supply a wrong name it should raise an error AND also provide a suggestion for a correction
    # in the error message!
    with pytest.raises(FileNotFoundError) as e:
        dataset_path = get_dataset_path('rb_motifs')

    assert 'rb_dual_motifs' in str(e.value)


def test_loading_jinja_templates_from_environment_works():
    # The config template should always exist
    template = TEMPLATE_ENV.get_template('config.yaml.j2')
    assert isinstance(template, j2.Template)


def test_ensure_folder_is_able_to_create_nested_folder_structures():
    with tempfile.TemporaryDirectory() as path:
        folder_path = os.path.join(path, 'nested', 'folder', 'structure')
        ensure_folder(folder_path)
        assert os.path.exists(folder_path)
        assert os.path.isdir(folder_path)


def test_merge_nested_dicts_basically_works():
    original = {
        'nested': {
            'value1': 10,
            'value2': 10
        },
        'replace': {'key1': 10}
    }
    update = {
        'nested': {
            'value1': 20,
            'missing': 20,
        },
        'missing': 20,
        'replace': [20, 20, 20]
    }
    expected = {
        'nested': {
            'value1': 20,
            'value2': 10,
            'missing': 20
        },
        'missing': 20,
        'replace': [20, 20, 20]
    }

    result = merge_nested_dicts(original, update)
    assert result == expected

