"""
Contains all the functionality to interact with the main config singleton as well as the config file.

**CHOICE - THIS MODULE IS THE BASE DEPENDENCY FOR EVERYTHING ELSE**
Some of the things that are done in this module would semantically fit better into a "utils" module, such
as the Singleton class. The problem is however, that almost all other parts of the program need access to
the config singleton, including the utils module. This means that this module here cannot have any other
project internal dependencies as that would lead to circular dependencies with all the other modules needing
to access the config...
"""
import os
import sys
import logging

import yaml
import pathlib
import platform
import typing as t

HOME_PATH = pathlib.Path.home()
FOLDER_PATH = os.path.join(HOME_PATH, '.visual_graph_datasets')
CONFIG_PATH = os.path.join(FOLDER_PATH, 'config.yaml')
DEFAULT_DATASETS_PATH = os.path.join(FOLDER_PATH, 'datasets')


def load_config(path: str = CONFIG_PATH) -> dict:
    if os.path.exists(path):
        # If the config file indeed exists, we will load the yml file as a dictionary and return that
        with open(path, mode='r') as file:
            data: dict = yaml.load(file, yaml.FullLoader)

        return data
    else:
        return {}


class Singleton(type):
    """
    This is metaclass definition, which implements the singleton pattern. The objective is that whatever
    class uses this as a metaclass does not work like a traditional class anymore, where upon calling the
    constructor a NEW instance is returned. This class overwrites the constructor behavior to return the
    same instance upon calling the constructor. This makes sure that always just a single instance
    exists in the runtime!

    **USAGE**
    To implement a class as a singleton it simply has to use this class as the metaclass.
    .. code-block:: python
        class MySingleton(metaclass=Singleton):
            def __init__(self):
                # The constructor still works the same, after all it needs to be called ONCE to create the
                # the first and only instance.
                pass
        # All of those actually return the same instance!
        a = MySingleton()
        b = MySingleton()
        c = MySingleton()
        print(a is b) # true
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Config(metaclass=Singleton):
    """
    This is the main config singleton for the program. It can be used to load the ``config.yaml`` file and
    provides methods that act as a facade to retrieve the config values.

    This class being a singleton means that invocations of the constructor will not create separate object
    instances, but instead return the same instance every time:

    .. code-block:: python

        from visual_graph_datasets.config import Config

        config1 = Config()
        config2 = Config()
        print(config1 == config2)  # True

    The data from the config file is not loaded by default, only after invoking ``load`` method. Each
    call to that method will freshly load the data from the file.
    """
    def __init__(self):
        self.data = {}
        self.path: t.Optional[str] = None
        self.logger = logging.Logger('MAIN')
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

    def load(self, path: str = CONFIG_PATH):
        self.path = path
        self.data = load_config(path=path)

    def get_folder_path(self) -> t.Optional[str]:
        return os.path.dirname(self.path)

    def get_platform(self) -> str:
        """
        A string identifier for the operating system of the current runtime

        Can be:
        - Darwin: Mac
        - Windows: Windows
        - otherwise Linux derivatives

        :return:
        """
        return platform.system()

    def get_datasets_path(self, default=DEFAULT_DATASETS_PATH) -> str:
        return self.retrieve_nested_with_default('base/datasets_path', default)

    def get_provider(self, default='nextcloud') -> str:
        return self.retrieve_nested_with_default('base/provider', default)

    def get_providers_map(self, default={}) -> t.Dict[str, dict]:
        return self.retrieve_nested_with_default('providers', default)

    def get_nextcloud_url(self, default='') -> str:
        return self.retrieve_nested_with_default('nextcloud/url', default)

    # -- utility methods --

    def retrieve_nested_with_default(self, query: str, default: t.Any):
        try:
            keys = query.split('/')
            current_value = self.data
            for key in keys:
                current_value = current_value[key]

            return current_value
        except (KeyError, TypeError):
            return default

    def __str__(self):
        return (f'Config(path="{self.path}")')

