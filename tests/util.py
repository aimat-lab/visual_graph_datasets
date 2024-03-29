import os
import sys
import json
import pathlib
import logging

import numpy as np

PATH = pathlib.Path(__file__).parent.absolute()
ASSETS_PATH = os.path.join(PATH, 'assets')
ARTIFACTS_PATH = os.path.join(PATH, 'artifacts')

LOG = logging.Logger('testing')
LOG.addHandler(logging.StreamHandler(sys.stdout))


def load_mock_color_graph() -> dict:
    """
    This function loads a mock color graph dict from the testing assets folder.
    
    :returns: graph dict representation of a color graph
    """
    json_path = os.path.join(ASSETS_PATH, 'g_color.json')
    with open(json_path) as file:
        graph = json.load(file)
        if 'node_adjacency' in graph:
            del graph['node_adjacency']
        
    # graph dicts should have numpy arrays as values!
    for key, value in graph.items():
        if isinstance(value, list):
            graph[key] = np.array(value)
        
    return graph
