import os

import numpy as np
import matplotlib.pyplot as plt
from imageio.v2 import imread

from visual_graph_datasets.visualization.base import create_frameless_figure
from visual_graph_datasets.visualization.molecules import mol_from_smiles
from visual_graph_datasets.visualization.molecules import visualize_molecular_graph_from_mol

from .util import ASSETS_PATH, ARTIFACTS_PATH


def test_visualize_molecular_graph_basically_works():
    np.set_printoptions(precision=0)

    smiles = 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
    mol = mol_from_smiles(smiles)
    num_atoms = len(mol.GetAtoms())

    image_width, image_height = 1000, 1000
    fig, ax = create_frameless_figure(image_width, image_height)
    node_positions, svg_string = visualize_molecular_graph_from_mol(
        ax=ax,
        mol=mol,
        image_width=image_width,
        image_height=image_height,
    )
    assert isinstance(node_positions, np.ndarray)
    assert node_positions.shape == (num_atoms, 2)

    assert isinstance(svg_string, str)
    assert len(svg_string) != 0

    # This is necessary...
    node_positions = [[v for i, v in enumerate(ax.transData.transform((x, y)))]
                      for x, y in node_positions]

    # Here we are going to save it as a PNG file and then afterwards load it again
    # using imread to emulate the process of visualization as best as possible
    vis_path = os.path.join(ARTIFACTS_PATH, 'molecule_visualization.png')
    fig.savefig(vis_path)
    plt.close(fig)

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
    image = imread(vis_path)
    ax.imshow(image, extent=[0, image_width, 0, image_height])

    # to make sure the positions are correct we are going to scatter a small dot to every position and can
    # then check in the image if they are at the correct atom positions.
    for i in range(num_atoms):
        ax.scatter(
            *node_positions[i],
            color='red',
        )

    img_path = os.path.join(ARTIFACTS_PATH, 'molecule_visualization_loaded.png')
    fig.savefig(img_path)
