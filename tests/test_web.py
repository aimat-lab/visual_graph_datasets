import pytest

import os
import tempfile

from visual_graph_datasets.config import Config
from visual_graph_datasets.web import get_file_share
from visual_graph_datasets.web import ensure_dataset
from visual_graph_datasets.web import AbstractFileShare, NextcloudFileShare
from visual_graph_datasets.testing import IsolatedConfig

from .util import LOG

# Skip network tests by default - they can be run with VGD_RUN_NETWORK=1 pytest
pytestmark = pytest.mark.skipif(
    os.environ.get("VGD_RUN_NETWORK", "0") != "1",
    reason="Network tests skipped by default. Set VGD_RUN_NETWORK=1 to run them."
)


def test_ensure_dataset_dataset_not_found_error():
    """
    When the specified dataset does not exist in the remote location then there should be a ConnectionError and 
    this error should have an informative error message.
    """
    with IsolatedConfig() as config:
        with pytest.raises(ConnectionError) as e_info:
            ensure_dataset(
                dataset_name='does_not_exist',
                config=config,
                provider_id='main',
                logger=LOG,
            )
        
        print(e_info)


def test_ensure_dataset_basically_works():
    """
    calling "ensure_dataset" should make sure that the specified dataset exists on the system and if it does not 
    exist then the thing is that it should not be bothered. 
    """
    with IsolatedConfig() as config:
        # At the beginning there should be no datasets in this fresh config context
        files = os.listdir(config.get_datasets_path())
        assert len(files) == 0
        
        ensure_dataset(
            dataset_name='mock', 
            config=config, 
            provider_id='main', 
            logger=LOG
        )
         
        # only after we call that function, the dataset should be found in the local system
        files = os.listdir(config.get_datasets_path())
        assert len(files) == 1
        dataset_path = os.path.join(config.get_datasets_path(), 'mock')
        dataset_files = os.listdir(dataset_path)
        assert len(dataset_files) > 10
        

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
