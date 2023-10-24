"""
Module containing *base* functionality for the visualization of the various graphs types
"""
import types
import typing as t

import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from imageio.v2 import imread

import visual_graph_datasets.typing as tc


def close_fig(figure):
    plt.cla()
    plt.clf()
    plt.close(figure)
    plt.close()
    del figure.savefig
    del figure


def create_frameless_figure(width: int = 100,
                            height: int = 100,
                            ratio: int = 2) -> t.Tuple[plt.Figure, plt.Axes]:
    """
    Returns a tuple of a matplotlib Figure and Axes object, where the axes object is a complete blank slate
    that can act as the foundation of a matplotlib-based visualization of a graph.

    More specifically this means that upon saving the figure that is created by this function, there will
    be no splines for the axes, not any kind of labels, no background, nothing at all. When saving this
    figure it will be a completely transparent picture with the pixel size given by ``width`` and ``height``.

    :param int width: The width of the saved image in pixels
    :param int height: The height of the saved image in pixels
    :param float ratio: This ratio will change the internal matplotlib figure size but *not* the final size
        of the image. This will be important for example if there is text with a fixed font size within the
        axes. This value will affect the size of things like text, border widths etc. but not the actual
        size of the image.
    :return:
    """
    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(width / (100 * ratio), height / (100 * ratio)),
        frameon=False,
    )
    fig.set_dpi(100 * ratio)

    # https://stackoverflow.com/questions/14908576
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.axis('off')

    # Selecting the axis-X making the bottom and top axes False.
    plt.tick_params(axis='x', which='both', bottom=False,
                    top=False, labelbottom=False)

    # Selecting the axis-Y making the right and left axes False
    plt.tick_params(axis='y', which='both', right=False,
                    left=False, labelleft=False)

    # https://stackoverflow.com/questions/4581504
    fig.patch.set_facecolor((0, 0, 0, 0))
    fig.patch.set_visible(False)

    ax.patch.set_facecolor((0, 0, 0, 0))
    ax.patch.set_visible(False)

    # A big part of achieving the effect we desire here, that is having only the Axes show up in the final
    # file and none of the border or padding of the Figure, is which arguments are passed to the "savefig"
    # method of the figure object. Since the saving process will come later we make sure that the correct
    # parameters are used by overriding the default parameters for the savefig method here
    def savefig(this, *args, **kwargs):
        this._savefig(*args, dpi=100 * ratio, **kwargs)

    setattr(fig, '_savefig', fig.savefig)
    setattr(fig, 'savefig', types.MethodType(savefig, fig))

    return fig, ax


def layout_node_positions(g: tc.GraphDict,
                          layout_cb: t.Callable[[nx.Graph], dict] = nx.spring_layout
                          ) -> np.ndarray:
    """
    Given a GraphDict ``g`` this method will create 2D positions for each of the nodes of the graph using
    the networkx layout function given as ``layout_cb``.

    :param g: A GraphDict representation of a graph
    :param layout_cb: A callable which accepts a networkx.Graph instance and returns a position dictionary.
        This description fits for all networkx "layout" functions.

    :returns: A numpy array of graph node positions in a 2D plane with shape (num_nodes, 2)
    """
    # We will use "networkx" to actually execute the layouting algorithms for the graph.
    # To do this however we first need to convert the graph which is currently a GraphDict into a networkx
    # Graph object.
    graph = nx.Graph()
    for i in g['node_indices']:
        graph.add_node(i)

    for i, j in g['edge_indices']:
        graph.add_edge(i, j)

    # Now we can actually apply a layout function
    positions_dict: t.Dict[int, t.Tuple[float, float]] = layout_cb(graph)
    positions = [None for _ in range(len(positions_dict))]
    for i, pos in positions_dict.items():
        positions[i] = pos

    return np.array(positions)


def draw_image(ax: plt.Axes,
               image_path: str,
               remove_ticks: bool = True,
               ) -> None:
    """
    Given the path ``image_path`` of a suitable image file and a matplotlib axes canvas ``ax``, this
    function will draw the image onto the canvas.


    """
    image = imread(image_path)
    extent = [0, image.shape[0], 0, image.shape[1]]
    ax.imshow(image, extent=extent)

    if remove_ticks:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
