import os
import logging
import typing as t

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from imageio.v2 import imread

import visual_graph_datasets.typing as tc
from visual_graph_datasets.util import NULL_LOGGER


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
