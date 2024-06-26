import os
import json
import time
import zipfile
import tempfile
import logging

import fontTools.ttLib.woff2
import requests

from visual_graph_datasets.util import NULL_LOGGER
from visual_graph_datasets.config import Config


class AbstractFileShare:
    """
    Abstract base class for implementations of specific file share providers.

    This repository does not contain the raw datasets, because these can be large chunks of data (multiple
    GB), instead these have to be downloaded from a remote file share option. At this point we cannot be
    certain that this remote file sharing provider will always stay the same, it is possible that it will
    change in the future. Different providers will use different protocols for retrieving the data.

    This class defines the general interface which can be used from the outside perspective to retrieve
    the data in a generalized way. Different subclasses will have to implement the specific interactions
    with the different providers to implement this expected behavior.
    """
    def __init__(self, logger: logging.Logger = NULL_LOGGER, **kwargs):
        self.logger = logger

    def download_file(self,
                      file_name: str,
                      folder_path: str) -> str:
        raise NotImplementedError()

    def download_metadata(self) -> dict:
        raise NotImplementedError()

    def download_dataset(self,
                         dataset_name: str,
                         path: str) -> None:
        raise NotImplementedError()

    def check_dataset(self, dataset_name: str) -> None:
        metadata: dict = self.download_metadata()
        if dataset_name not in metadata['datasets'].keys():
            raise FileNotFoundError(f'It appears that no visual graph dataset by the name of '
                                    f'"{dataset_name}" exists on the remote file share '
                                    f'{str(self)}. Could it be a spelling mistake? Here is a list '
                                    f'of all available datasets:'
                                    f'{", ".join(metadata["datasets"].keys())}')

    def __str__(self) -> str:
        """
        Implementing the string representation for a file share is important so that it can be used inside
        an error or log message for example to provide more detailed information.

        :return str:
        """
        raise NotImplementedError()


# == SPECIFIC IMPLEMENTATIONS FOR DIFFERENT FILE SHARE OPTIONS ===


# https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests
class NextcloudFileShare(AbstractFileShare):
    """
    This class implements the possibility of retrieving datasets from a Nextcloud (https://nextcloud.com/)
    server.

    Specifically, the url which has to be provided to this class has to be the public "share" url for a
    *folder* which contains all the dataset ZIP files as well as the metadata JSON file.
    """

    def __init__(self, url, **kwargs):
        super(NextcloudFileShare, self).__init__(**kwargs)
        self.url: str = url.strip('/')

    def download_file(self,
                      file_name: str,
                      folder_path: str,
                      chunk_size: int = 8192,
                      log_step: int = 1 * 1024**2  # Log after every 10MB
                      ) -> str:
        """
        Given the ``file_name`` (which may be a path relative to the origin of the remote file share), this method 
        will download that file into the given ``folder_path``.
        
        :param file_name: The name of the file to be downloaded. This may also be a relative path to the file in question.
        :param folder_path: The absolute path to the local folder into which the file should be placed. The file will have 
            the same name inside this folder as it had on the remote file share server.
        :param chunk_size: The chunk size with which to download the file.
        :param log_step: The number of bytes after which to produce a log message about the progress of the download
        
        :returns: str
        """
        url = '/'.join([self.url, 'download'])
        params = {
            'path': '/',
            'files': file_name
        }

        # 15.12.22 - Introduced additional os.path.basename to fix a bug where the file_path would be
        # wrong when downloading nested files.
        file_path = os.path.join(folder_path, os.path.basename(file_name))
        with requests.get(url, params=params, stream=True) as r:
            
            # 20.10.23 - As the first thing we have to check for a potential error and then create a meaningful 
            # error message in case the download could not proceed correctly.
            if r.status_code != 200:
                raise ConnectionError(f'There was an error during the download of the file "{file_name}" from the '
                                      f'file share provider "{self.url}". This most likely means that there exists no '
                                      f'dataset with the given name on that file share. However, it could also indicate '
                                      f'connectivity issues in general - so please check your internet connection.')
            
            if 'Content-Length' in r.headers:
                total_bytes = int(r.headers['Content-Length'])

            fetched_bytes = 0
            logged_bytes = 0
            start_time = time.time()
            with open(file_path, mode='wb') as file:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    file.write(chunk)
                    fetched_bytes += chunk_size

                    if 'Content-Length' in r.headers and fetched_bytes > logged_bytes + log_step:
                        elapsed_time = time.time() - start_time
                        bytes_per_second = fetched_bytes / elapsed_time
                        remaining_time = (total_bytes - fetched_bytes) / bytes_per_second
                        self.logger.info(f'{fetched_bytes / 1024**2:.1f} / '
                                         f'{total_bytes / 1024**2:.1f} MB'
                                         f' - elapsed time: {elapsed_time/3600:.2f} hrs'
                                         f' - avg. speed: {bytes_per_second/1024**2:.1f} MB/s'
                                         f' - remaining time: {remaining_time/3600:.2f} hrs')
                        logged_bytes = fetched_bytes

        return file_path

    def download_dataset(self,
                         dataset_name: str,
                         folder_path: str) -> None:
        # All datasets are distributed as zip files so we download that file and extract it
        file_name = f'{dataset_name}.zip'
        file_path = self.download_file(file_name, folder_path)
        with zipfile.ZipFile(file_path) as zip_file:
            zip_file.extractall(folder_path)

        # Afterwards we delete the original zip file again
        os.remove(file_path)

    def download_metadata(self) -> dict:
        with tempfile.TemporaryDirectory() as folder_path:
            file_path = self.download_file('metadata.json', folder_path)
            with open(file_path, mode='r') as file:
                metadata = json.load(file)

        return metadata

    def __str__(self):
        return (
            f'NextcloudFileShare('
            f'url="{self.url}")'
        )


PROVIDER_CLASS_MAP = {
    'nextcloud': NextcloudFileShare
}


def get_file_share(config: Config = Config(),
                   provider_id: str = 'main') -> AbstractFileShare:
    """
    Given the ``config`` singleton and the unique string ``provider_id`` of the provider to be used, this
    function will create a corresponding FileShare instance and return it.

    :param config:
    :param provider_id: The unique string id which identifies the provider in the config file.

    :return: The constructed object instance of the appropriate AbstractFileShare subclass.
    """
    provider_map: dict = config.get_providers_map()
    # This might be a mistake which will happen quite often, that a provider id is used which is not
    # present in the config. In this case we should give a good error message.
    if provider_id not in provider_map:
        raise KeyError(f'The given provider_id "{provider_id}" is not specified within the currently '
                       f'loaded config file {str(config)}. The only valid options are: '
                       f'{", ".join(provider_map.keys())}. Please choose another provider or add one to '
                       f'the config by using the "config" command.')

    provider_kwargs = provider_map[provider_id]

    file_share_class = PROVIDER_CLASS_MAP[provider_kwargs['type']]
    file_share = file_share_class(**provider_kwargs)

    return file_share


def ensure_dataset(dataset_name: str, 
                   config: Config = Config(), 
                   provider_id: str = 'main',
                   logger: logging.Logger = NULL_LOGGER,
                   ) -> str:
    """
    Makes sure that the dataset with the given ``dataset_name`` exists on the local system. If that dataset does not 
    yet exist on the local system it will be downloaded from the file share identified by the given ``provider_id`` and 
    that is defined in the ``config``.
    
    Therefore, after this function has successfully terminated, one can be sure that the requested dataset does exist
    within the ``config.get_datasets_path`` folder!
    
    :returns: The string absolute path to the dataset folder on the local machine
    """
    dataset_path = os.path.join(config.get_datasets_path(), dataset_name)
    if not os.path.exists(dataset_path):
        logger.info(f'the dataset "{dataset_name}" not found, downloading from remote file share "{provider_id}"...')
        file_share = get_file_share(config=config, provider_id=provider_id)
        file_share.logger = logger
        
        # Here we actually download the dataset with the given name
        file_share.download_dataset(dataset_name, config.get_datasets_path())

    else:
        logger.info(f'the dataset "{dataset_name}" already exists locally! Skipping download.')
        
    return dataset_path