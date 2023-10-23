import os
import tempfile
import yaml

import jinja2 as j2

from visual_graph_datasets.config import load_config
from visual_graph_datasets.config import Config
from visual_graph_datasets.util import TEMPLATE_ENV
from visual_graph_datasets.testing import IsolatedConfig


def test_testing_config():
    """
    IsolatedConfig is a utility context manager which can be used to create isolated environments for the
    testing purposes.
    """
    with IsolatedConfig() as config:
        assert isinstance(config, Config)
        assert isinstance(config.get_datasets_path(), str)
        assert os.path.exists(config.get_datasets_path())


def test_load_config_works_when_no_file_exists():
    """
    "load_config" should also work when the given path does not exist and in this case it should return a 
    an empty dict
    """
    config_dict = load_config('')
    assert isinstance(config_dict, dict)
    assert len(config_dict) == 0


def test_config_object_can_be_created_without_config_file():
    """
    A "Config" object should be able to be created without specifying a config file.
    """
    config = Config()
    assert isinstance(config, Config)
    assert isinstance(config.data, dict)


def test_config_singleton():
    """
    The config class is designed as a singleton, so multiple invocations of the constructor should all
    return the same instance
    """
    config_1 = Config()
    config_2 = Config()
    assert config_1 == config_2


def test_create_new_config_file_from_template():
    template = TEMPLATE_ENV.get_template('config.yaml.j2')
    content = template.render()

    with tempfile.TemporaryDirectory() as path:
        config_path = os.path.join(path, 'config.yaml')
        with open(config_path, mode='w') as file:
            file.write(content)

        assert os.path.exists(config_path)
        # Now we also need to make sure that this is a valid yaml file which can be loaded
        with open(config_path, mode='r') as file:
            data = yaml.load(file, yaml.FullLoader)

        assert isinstance(data, dict)


def test_config_can_be_loaded_from_file():
    with tempfile.TemporaryDirectory() as path:
        # First of all we need to create a new config file from the template
        template = TEMPLATE_ENV.get_template('config.yaml.j2')
        content = template.render()
        config_path = os.path.join(path, 'config.yaml')
        with open(config_path, mode='w') as file:
            file.write(content)

        config = Config()
        config.load(config_path)
        assert isinstance(config.data, dict)
        assert len(config.data) != 0


def test_config_retrieve_nested_value():
    with tempfile.TemporaryDirectory() as path:
        # First of all we need to create a new config file from the template
        template = TEMPLATE_ENV.get_template('config.yaml.j2')
        content = template.render()
        config_path = os.path.join(path, 'config.yaml')
        with open(config_path, mode='w') as file:
            file.write(content)

        config = Config()
        config.load(config_path)

        value = config.retrieve_nested_with_default('providers/main/type', '')
        assert value != ''
