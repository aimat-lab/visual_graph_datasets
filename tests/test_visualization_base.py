import os
import json
import pytest

import numpy as np

from imageio.v2 import imread
from visual_graph_datasets.visualization.base import create_frameless_figure
from visual_graph_datasets.visualization.base import layout_node_positions

from .util import ARTIFACTS_PATH, ASSETS_PATH


@pytest.mark.parametrize('width,height', [
    [300, 300],
    [2000, 2000],
])
def test_create_frameless_figure_3d_works(width: int, height: int):
    """
    "create_frameless_figure" now also accepts a optional argument "dim" which can be 3 to indicate
    that the returned axes is actually an Axes3D object.
    """
    fig, ax = create_frameless_figure(width=width, height=height, dim=3)
    
    fig_path = os.path.join(ARTIFACTS_PATH, 'test_create_frameless_figure_3d.png')
    fig.savefig(fig_path)

    # Now we can load the image from the disk and check if it actually has the correct dimensions
    image = imread(fig_path)
    assert isinstance(image, np.ndarray)
    assert image.shape[:2] == (height, width)


@pytest.mark.parametrize('width,height', [
    [300, 300],
    [500, 500],
])
def test_create_frameless_ax_basically_works(width: int, height: int):
    """
    "create_frameless_figure" should create a figure without any frame around it.
    
    This test only does visual verification
    """
    fig, ax = create_frameless_figure(width=width, height=height, ratio=2)
    # This sets the background color of just the axes itself, which we can use to check if it indeed worked
    # if the resulting image is purely green we know that we effectively got rid of all the figure overhead
    fig.patch.set_facecolor('white')
    ax.patch.set_facecolor('xkcd:mint green')

    path = os.path.join(ARTIFACTS_PATH, 'test_create_frameless_figure.png')
    fig.savefig(path)
    
    # Now we can load the image from the disk and check if it actually has the correct dimensions
    image = imread(path)
    assert isinstance(image, np.ndarray)
    assert image.shape[:2] == (height, width)

