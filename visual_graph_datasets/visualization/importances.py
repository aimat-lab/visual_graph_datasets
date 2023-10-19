import os
import logging
import typing as t

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from imageio.v2 import imread

import visual_graph_datasets.typing as tc
from visual_graph_datasets.util import NULL_LOGGER


def plot_node_importances_background(ax: plt.Axes,
                                     g: tc.GraphDict,
                                     node_positions: np.ndarray,
                                     node_importances: np.ndarray,
                                     weight: float = 1.0,
                                     radius: float = 35.0,
                                     color_map: t.Optional[str | mcolors.Colormap] = None,
                                     color: str = 'lightgreen',
                                     v_min: float = 0.0,
                                     v_max: float = 1.0,
                                     zorder: int = -100,
                                     ):
    """
    Given an Axes ``ax`` to draw on and a graph dict ``g`` and the pixel ``node_positions``, this 
    function will plot the ``node importances`` as colored circles in the style of a heatmap in the 
    background behind each node. 
    
    The importance value determines the opacity of the element. It is also possible to pass 
    a valid matplotlib ``color_map`` and a ``weight`` value, which will then additionally change the 
    color of the elements depending on the weight. This can be used to encode a secondary property such 
    as an explanation fidelity value, for example, into the visualization.
    """

    normalize = mpl.colors.Normalize(vmin=v_min, vmax=v_max)
    
    # Only if the color_map argument is actually given, we actually use a dynamic color map
    # otherwise we are going to use the constant color that is provided.
    if color_map is None:
        cmap = lambda value: color
    elif isinstance(color_map, str):
        cmap: mpl.colors.Colormap = mpl.colormaps[color_map]
    elif isinstance(color_map, mcolors.Colormap):
        cmap = color_map

    # And then we iterate through all the nodes of the given graph and simply draw the circle 
    # at each position.
    for i in g['node_indices']:
        x, y = node_positions[i]
        importance = node_importances[i]

        value = normalize(importance)
        circle = plt.Circle(
            (x, y),
            radius=radius,
            lw=0,
            color=cmap(weight),
            fill=cmap(weight),
            alpha=value,
            zorder=zorder,
        )
        ax.add_artist(circle)


def plot_edge_importances_background(ax: plt.Axes,
                                     g: tc.GraphDict,
                                     node_positions: np.ndarray,
                                     edge_importances: np.ndarray,
                                     weight: float = 1.0,
                                     thickness: float = 20.0,
                                     color_map: t.Optional[str | mcolors.Colormap] = None,
                                     color='lightgreen',
                                     v_min: float = 0.0,
                                     v_max: float = 1.0,
                                     zorder: int = -200,
                                     ):
    """
    Given an Axes ``ax`` to draw on and a graph dict ``g`` and the pixel ``node_positions``, this 
    function will plot the ``edge_importances`` as colored lines in the style of a heatmap in the 
    background behind each each edge that connects two nodes.
    
    The importance value determines the opacity of the element. It is also possible to pass 
    a valid matplotlib ``color_map`` and a ``weight`` value, which will then additionally change the 
    color of the elements depending on the weight. This can be used to encode a secondary property such 
    as an explanation fidelity value, for example, into the visualization.
    """
    normalize = mpl.colors.Normalize(vmin=v_min, vmax=v_max)
    
    # Only if the color_map argument is actually given, we actually use a dynamic color map
    # otherwise we are going to use the constant color that is provided.
    if color_map is None:
        cmap = lambda value: color
    elif isinstance(color_map, str):
        cmap: mpl.colors.Colormap = mpl.colormaps[color_map]
    elif isinstance(color_map, mcolors.Colormap):
        cmap = color_map

    for (i, j), ei in zip(g['edge_indices'], edge_importances):
        coords_i = node_positions[i]
        coords_j = node_positions[j]

        x_i, y_i = coords_i
        x_j, y_j = coords_j

        value = normalize(ei)
        ax.plot(
            [x_i, x_j],
            [y_i, y_j],
            color=cmap(weight),
            lw=thickness,
            alpha=value,
            zorder=zorder,
        )


def plot_node_importances_border(ax: plt.Axes,
                                 g: tc.GraphDict,
                                 node_positions: np.ndarray,
                                 node_importances: np.ndarray,
                                 weight: float = 1.0,
                                 radius: float = 35.0,
                                 thickness: float = 5.0,
                                 color_map='Greys',
                                 color='black',
                                 v_min: float = 0.0,
                                 v_max: float = 1.0):

    normalize = mpl.colors.Normalize(vmin=v_min, vmax=v_max)
    cmap: mpl.colors.Colormap = mpl.colormaps[color_map]

    for i in g['node_indices']:
        x, y = node_positions[i]
        importance = node_importances[i]

        value = normalize(importance)
        circle = plt.Circle(
            (x, y),
            radius=radius,
            lw=thickness * max(weight, 0.1),
            color=cmap(weight),
            fill=False,
            alpha=value
        )
        ax.add_artist(circle)


def plot_edge_importances_border(ax: plt.Axes,
                                 g: tc.GraphDict,
                                 node_positions: np.ndarray,
                                 edge_importances: np.ndarray,
                                 weight: float = 1.0,
                                 radius: float = 35.0,
                                 thickness: float = 5.0,
                                 color_map='Greys',
                                 color='black',
                                 v_min: float = 0.0,
                                 v_max: float = 1.0
                                 ):
    normalize = mpl.colors.Normalize(vmin=v_min, vmax=v_max)
    cmap = mpl.colormaps[color_map]

    for (i, j), ei in zip(g['edge_indices'], edge_importances):
        coords_i = node_positions[i]
        coords_j = node_positions[j]
        # Here we determine the actual start and end points of the line to draw. Now we cannot simply use
        # the node coordinates, because that would look pretty bad. The lines have to start at the very
        # edge of the node importance circle which we have already drawn (presumably) at this point. This
        # circle is identified by it's radius. So what we do here is we reduce the length of the line on
        # either side of it by exactly this radius. We do this by first calculating the unit vector for that
        # line segment and then moving radius times this vector into the corresponding directions
        diff = (coords_j - coords_i)
        delta = (radius / np.linalg.norm(diff)) * diff
        x_i, y_i = coords_i + delta
        x_j, y_j = coords_j - delta

        value = normalize(ei)
        ax.plot(
            [x_i, x_j],
            [y_i, y_j],
            color=cmap(weight),
            lw=thickness * max(weight, 0.1),
            alpha=value
        )


def create_importances_pdf(graph_list: t.List[tc.GraphDict],
                           image_path_list: t.List[str],
                           node_positions_list: t.List[np.ndarray],
                           importances_map: t.Dict[str, t.Tuple[t.List[np.ndarray], t.List[np.ndarray]]],
                           output_path: str,
                           labels_list: t.Optional[t.List[str]] = None,
                           importance_channel_labels: t.Optional[t.List[str]] = None,
                           plot_node_importances_cb: t.Callable = plot_node_importances_border,
                           plot_edge_importances_cb: t.Callable = plot_edge_importances_border,
                           base_fig_size: float = 8,
                           normalize_importances: bool = True,
                           show_x_ticks: bool = False,
                           show_y_ticks: bool = False,
                           logger: logging.Logger = NULL_LOGGER,
                           log_step: int = 100,
                           ):
    """
    This function can be used to create a PDF file that shows the visualizations of multiple importance explanations.
    
    The pdf will consist of an individual page for each graph given in the ``graph_list``. For each element it is possible 
    to visualize multiple (columns) multi-channel (rows) explanations. This is mainly intended to provide a method to 
    directly compare the explanations created by multiple different explanation methods.
    
    :param graph_list: A list containing the graph dict representations of all the graphs to be visualized
    :param image_path_list: A list containing the absolute string paths to the visualization images of the 
        graphs. Must have the same order as graph_list
    :param node_positions_list: A list containing the node position arrays for the graphs to be visualized. 
        These arrays should determine the 2D pixel coordinates for each node of the graphs. Must have the 
        same order as graph_list.
    :param importances_map: A dictionary that defines the importance explanation masks to be visualized. The keys 
        of this dict are unique string identifiers that describe the different explanation sources from which the 
        explanations came from. This could for example be the names of different explainability methods. The values 
        are tuples of two elements. The first tuple element is a list of node importance arrays, which define the 
        actual node importance values for each node and each channel. The second element is a list of edge importance 
        arrays, which define the actual edge importance values for each edge and each channel.
        These lists must have the same order as the given graph_list.
    :param plot_node_importances_cb: The callable function that actually realizes the drawing of the node importances 
        onto the Axes canvas.
    :param plot_edge_importances_cb: The callable function that actually realizes the drawing of the edge importances 
        onto the Axes canvas.
    :param base_fig_size: The size of the figures. Modification of this value will influence the file size of the PDF 
        and the ratio of the size of the text and image elements within each page of the PDF.
    """
    # ~ ASSERTIONS ~
    # Some assertions in the beginning to avoid trouble later, because this function will be somewhat
    # computationally expensive.

    # First of all we will check if the channel numbers of all the provided importances line up, because
    # if that is not the case we will be doing a whole bunch of stuff for nothing
    num_channel_set: t.Set[int] = set()
    for node_importances_list, edge_importances_list in importances_map.values():
        for node_importances, edge_importances in zip(node_importances_list, edge_importances_list):
            assert isinstance(node_importances, np.ndarray), ('some of the node importances are not given '
                                                              'as numpy array!')
            assert isinstance(edge_importances, np.ndarray), ('some of the edge importances are not given '
                                                              'as numpy array!')
            num_channel_set.add(node_importances.shape[1])
            num_channel_set.add(edge_importances.shape[1])

    # If this set contains more than one element, this indicates that some (we don't know which with this
    # method) of the arrays have differing explanation channel shapes...
    assert len(num_channel_set) == 1, (f'Some of the given arrays of node and edge importances have '
                                       f'differing channel shapes. Please make sure that all of the '
                                       f'importance explanations you want to visualize have the same '
                                       f'dimension: {num_channel_set}')

    num_channels = list(num_channel_set)[0]

    # Now that we know the number of channels we also need to make sure that the number of importance channel labels,
    # if they are given, matches the number of channels
    if importance_channel_labels is not None:
        assert len(importance_channel_labels) == num_channels, (
            f'The number of labels given for the importance channels (current: {len(importance_channel_labels)}) has '
            f'to match the number of importance channels represented in the data.'
        )

    # ~ CREATING PDF ~

    with PdfPages(output_path) as pdf:
        for index, g in enumerate(graph_list):
            node_positions = node_positions_list[index]

            image_path = image_path_list[index]
            image = imread(image_path)
            extent = [0, image.shape[0], 0, image.shape[1]]

            # The number of columns in our multi plot is determined by how many different explanations
            # we want to plot. Each different set of explanations is supposed to be one entry in the
            # importances_map dict.
            num_cols = len(importances_map)
            # The number of rows is determined by the number of different explanation channels contained
            # within our explanations. Each row represents one explanation channel.
            num_rows = num_channels
            fig, rows = plt.subplots(
                ncols=num_cols,
                nrows=num_rows,
                figsize=(base_fig_size * num_cols, base_fig_size * num_rows),
                squeeze=False,
            )

            for r in range(num_rows):
                for c, (name, importances_list_tuple) in enumerate(importances_map.items()):
                    node_importances_list, edge_importances_list = importances_list_tuple

                    node_importances = node_importances_list[index]
                    edge_importances = edge_importances_list[index]

                    ax = rows[r][c]

                    # Only if we are currently in the first row we can use the axes title as a stand in for
                    # the entire column title, which is essentially what we do here: We set a string title
                    # for the column:
                    if r == 0:
                        title_string = f'key: "{name}"'
                        if labels_list is not None:
                            title_string += f'\n{labels_list[index]}'

                        ax.set_title(title_string)

                    # Likewise only for the first item in a row we can use the axes y-label as a kind of
                    # row title. By default, we just use the index of the channel as title
                    if c == 0:
                        label_string = f'Channel {r}'
                        if importance_channel_labels is not None:
                            label_string += f'\n{importance_channel_labels[r]}'

                        ax.set_ylabel(label_string)

                    ax.imshow(
                        image,
                        extent=extent
                    )
                    if not show_x_ticks:
                        ax.set_xticks([])
                    if not show_y_ticks:
                        ax.set_yticks([])

                    if normalize_importances:
                        node_v_max = np.max(node_importances)
                        edge_v_max = np.max(edge_importances)
                    else:
                        node_v_max = 1
                        edge_v_max = 1

                    # The actual functions to draw the importances are dependency injected so that the user
                    # can decide how the importances should actually be drawn.
                    plot_node_importances_cb(
                        g=g,
                        ax=ax,
                        node_positions=node_positions,
                        node_importances=node_importances[:, r],
                        v_max=node_v_max,
                        v_min=0,
                    )
                    plot_edge_importances_cb(
                        g=g,
                        ax=ax,
                        node_positions=node_positions,
                        edge_importances=edge_importances[:, r],
                        v_max=edge_v_max,
                        v_min=0,
                    )

            if index % log_step == 0:
                logger.info(f' * ({index} / {len(graph_list)}) visualizations created')

            pdf.savefig(fig)
            plt.close(fig)
