import os
import re
import tempfile
import typing as t

import cairosvg
import numpy as np
import matplotlib.pyplot as plt
from imageio.v2 import imread
from rdkit import Chem
from rdkit.Chem.Draw.rdMolDraw2D import MolDraw2DSVG
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)


def mol_from_smiles(smiles: str
                    ) -> Chem.Mol:
    return Chem.MolFromSmiles(smiles)


def visualize_molecular_graph_from_mol(ax: plt.Axes,
                                       mol: Chem.Mol,
                                       image_width: 1000,
                                       image_height: 1000,
                                       line_width: int = 5,
                                       ) -> t.Tuple[np.ndarray, str]:
    """
    Creates a molecular graph visualization if given the RDKit Mol object ``mol`` and the matplotlib Axes
    ``ax`` to draw on. The image width and height have to be the same values as the final pixel values of
    the rendered PNG matplotlib figure.

    Returns a tuple, where the first value is the ``node_positions`` array of shape (V, 2) where V is the
    number of nodes in the graph (number of atoms in the molecule). This array is created alongside the
    visualization and for every atom it contains the

    :param ax:
    :param mol:
    :param image_width: Defines the line width used for the drawing of the bonds
    :param image_height:
    :param line_width:
    :return:
    """
    # To create the visualization of the molecule we are going to use the existing functionality of RDKit
    # which simply takes the Mol object and creates an SVG rendering of it.
    mol_drawer = MolDraw2DSVG(image_width, image_height)
    mol_drawer.SetLineWidth(line_width)
    mol_drawer.DrawMolecule(mol)
    mol_drawer.FinishDrawing()
    svg_string = mol_drawer.GetDrawingText()

    # Now the only problem we have with the SVG that has been created this way is that it still has a white
    # background, which we generally don't want for the graph visualizations and sadly there is no method
    # with which to control this directly for the drawer class. So we have to manually edit the svg string
    # to get rid of it...
    svg_string = re.sub(
        r'opacity:\d*\.\d*;fill:#FFFFFF',
        'opacity:0.0;fill:#FFFFFF',
        svg_string
    )

    # Now, we can't directly display SVG to a matplotlib canvas, which is why we first need to convert this
    # svg string into a PNG image file temporarily which we can then actually put onto the canvas.
    with tempfile.TemporaryDirectory() as path:
        svg_path = os.path.join(path, 'molecule.svg')
        with open(svg_path, mode='w') as file:
            file.write(svg_string)

        png_path = os.path.join(path, 'molecule.png')
        cairosvg.svg2png(
            bytestring=svg_string.encode(),
            write_to=png_path,
            output_width=image_width,
            output_height=image_height,
        )

        # The RDKit svg drawer class offers some nice functionality to figure out the coordinates of those
        # files within the drawer.
        node_coordinates = []
        for point in [mol_drawer.GetDrawCoords(i) for i, _ in enumerate(mol.GetAtoms())]:
            node_coordinates.append([
                point.x,
                point.y
            ])

        # We can actually paint this onto the canvas now...
        image = imread(png_path)
        ax.imshow(image)

    return np.array(node_coordinates), svg_string
