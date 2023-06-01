import os
import json

import numpy as np

from visual_graph_datasets.visualization.base import create_frameless_figure
from visual_graph_datasets.visualization.base import layout_node_positions

from .util import ARTIFACTS_PATH, ASSETS_PATH


def test_create_frameless_ax_basically_works():
    fig, ax = create_frameless_figure(width=300, height=300, ratio=2)
    # This sets the background color of just the axes itself, which we can use to check if it indeed worked
    # if the resulting image is purely green we know that we effectively got rid of all the figure overhead
    fig.patch.set_facecolor('white')
    ax.patch.set_facecolor('xkcd:mint green')

    path = os.path.join(ARTIFACTS_PATH, 'frameless_ax.png')
    fig.savefig(path)


def test_layout_node_positions_basically_works():
    # ~ loading the example graph
    json_path = os.path.join(ASSETS_PATH, 'g_grayscale.json')
    with open(json_path) as file:
        g = json.load(file)

    # ~ create the positions
    node_positions = layout_node_positions(g)
    assert isinstance(node_positions, np.ndarray)
    assert node_positions.shape == (len(g['node_indices']), 2)
