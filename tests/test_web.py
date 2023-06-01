import pytest

import os
import tempfile

from visual_graph_datasets.config import Config
from visual_graph_datasets.web import get_file_share
from visual_graph_datasets.web import AbstractFileShare, NextcloudFileShare
from visual_graph_datasets.testing import IsolatedConfig


# == UNITTESTS ==

def test_get_file_share_basically_works():
    """
    The function "get_file_share" is a convenience function which will automatically construct an appropriate
    file share object given only the provider_id that is defined in the config file.
    """
    with IsolatedConfig() as config:
        provider_id = 'main'
        file_share = get_file_share(config, provider_id)
        assert isinstance(file_share, AbstractFileShare)


def test_get_file_share_error_when_invalid_provider_id():
    """
    If the appropriate error message is triggered when an incorrect provider id is used for the
    get_file_share function
    """
    with IsolatedConfig() as config:
        with pytest.raises(KeyError):
            file_share = get_file_share(config, 'foobar')


def test_download_mock_dataset_works():
    """
    If downloading the mock dataset with the default file share works
    """
    with IsolatedConfig() as config:
        file_share = get_file_share(config, 'main')
        assert isinstance(file_share, NextcloudFileShare)
        file_share.download_dataset('mock', config.get_datasets_path())

        dataset_path = os.path.join(config.get_datasets_path(), 'mock')
        assert os.path.exists(dataset_path)
        assert os.path.isdir(dataset_path)
        assert len(os.listdir(dataset_path)) != 0


def test_download_source_dataset_file_works():
    """
    If downloading a nested file with the default file share works
    """
    with IsolatedConfig() as config:
        file_share = get_file_share(config, 'main')
        file_path = file_share.download_file('source/aqsoldb.csv', config.get_folder_path())
        assert os.path.exists(file_path)
        assert os.path.isfile(file_path)

        with open(file_path) as file:
            assert len(file.read()) > 100
