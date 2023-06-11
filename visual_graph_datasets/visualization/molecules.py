import os
import re
import io
import tempfile
import typing as t

import cairosvg
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
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
    visualization and for every atom it contains the (x, y) coordinates on the given Axes.

    NOTE: The node positions returned by this function are in the coordinates system of the given Axes
        object. When intending to save that into a persistent file it is important to convert these
        node coordinates into the Figure coordinate system first by using ax.transData.transform !

    05.06.23 - Previously, this function relied on the usage of a temp dir and created two temporary files
        as intermediates. This was now replaces such that no intermediate files are required anymore to
        improve the efficiency of the function.

    :param ax: The mpl Axes object onto which the visualization should be drawn
    :param mol: The Mol object which is to be visualized
    :param image_width: The pixel width of the resulting image
    :param image_height: The pixel height of the resulting image
    :param line_width: Defines the line width used for the drawing of the bonds

    :return: A tuple (node_positions, svg_string), where the first element is a numpy array (V, 2) of node
        mpl coordinates of each of the graphs nodes in the visualization on the given Axes and the second
        element is the SVG string from which that visualization was created.
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
    png_data = cairosvg.svg2png(
        bytestring=svg_string.encode(),
        parent_width=image_width,
        parent_height=image_height,
        output_width=image_width,
        output_height=image_height,
    )
    file_obj = io.BytesIO(png_data)

    image = Image.open(file_obj, formats=['png'])
    image = np.array(image)
    ax.imshow(image)

    # The RDKit svg drawer class offers some nice functionality to figure out the coordinates of those
    # files within the drawer.
    node_coordinates = []
    for point in [mol_drawer.GetDrawCoords(i) for i, _ in enumerate(mol.GetAtoms())]:
        node_coordinates.append([
            point.x,
            point.y
        ])

    node_coordinates = np.array(node_coordinates)

    return node_coordinates, svg_string
