"""
Functionality to support unittests.
"""
import os
import tempfile
import typing as t

from visual_graph_datasets.util import TEMPLATE_ENV
from visual_graph_datasets.config import Config


def clear_files(file_paths: t.List[str]):
    """
    Small utility function, which will make sure that all the files with the given ``file_paths`` are
    definitely removed from the filesystem. Used for testing purposes.
    """
    for file_path in file_paths:
        if os.path.exists(file_path) and os.path.isfile(file_path):
            os.remove(file_path)


class IsolatedConfig:
    """
    This is a context manager which will create a fresh config file from the default template within
    it's context. Furthermore, this config file will be within a temporary folder, into which the
    default datasets path will be configured to as well.
    """
    def __init__(self):
        self.dir = tempfile.TemporaryDirectory()
        self.folder_path: t.Optional[str] = None
        self.datasets_folder: t.Optional[str] = None
        self.config_path: t.Optional[str] = None
        self.config = Config()

    def __enter__(self):
        # First of all we need to init the temporary directory
        self.folder_path = self.dir.__enter__()

        # Then we need to create the folder which will contain all the datasets that are potentially
        # downloaded:
        self.datasets_folder = os.path.join(self.folder_path, 'datasets')
        os.mkdir(self.datasets_folder)

        # Next we need to create the config file from the template. In this step it is important that we
        # pass in the custom dataset folder to be used in the config
        self.config_path = os.path.join(self.folder_path, 'config.yaml')
        template = TEMPLATE_ENV.get_template('config.yaml.j2')
        with open(self.config_path, mode='w') as file:
            content = template.render(datasets_path=self.datasets_folder)
            file.write(content)

        # And finally we load that config file into the config singleton
        self.config.load(self.config_path)

        return self.config

    def __exit__(self, *args, **kwargs):
        # In the end we need to make sure to destroy the temporary directory again
        self.dir.__exit__(*args, **kwargs)
