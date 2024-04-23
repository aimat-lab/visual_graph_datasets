import os
import logging
import typing as t

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
from imageio.v2 import imread

import visual_graph_datasets.typing as tc
from visual_graph_datasets.util import DEFAULT_CHANNEL_INFOS
from visual_graph_datasets.util import NULL_LOGGER
from visual_graph_datasets.util import array_normalize
from visual_graph_datasets.util import binary_threshold
from visual_graph_datasets.visualization.base import draw_image


# == IMPORTANCE PLOTTING PRIMITIVES ==
# The following section contains functions which implement the actual plotting of one set of importance 
# values into one plot. These following functions are essentially alternatives for the same operation - where 
# the different functions just implement different styles for the visualization of the importances.

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
                                     **kwargs,
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

        # We first add a non-transparent white circle into the background of the node with basically the same 
        # radius as the importance highlighting circle. This is done to make the importance circle stand out
        # and most importantly to block the edge importance markings from overlapping with the node importance 
        # circle. Since both of these are drawn with some transparency they would cause weird looking artifacts
        # at their intersections.
        circle_background = plt.Circle(
            (x, y),
            radius=radius * 0.99,
            lw=0,
            color='white',
            fill='white',
            zorder=zorder - 10,
        )
        ax.add_artist(circle_background)

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
                                     **kwargs,
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
        
# These global dictionaries can be used to simplify the selection of the alternative plotting methods for 
# high level functions. Instead of having to pass the actual callable function instance as a parameter 
# it is possible to use the representative string and then obtain the callable from this dictionary.
        
PLOT_NODE_IMPORTANCES_OPTIONS: t.Dict[str, t.Callable] = {
    'border':       plot_node_importances_border,
    'background':   plot_node_importances_background,
}

PLOT_EDGE_IMPORTANCES_OPTIONS: t.Dict[str, t.Callable] = {
    'border':       plot_edge_importances_border,
    'background':   plot_edge_importances_background
}


# == ADVANCED IMPORTANCE PLOTTING ==
# The next section defines more advanced functions for plotting importance/explanation related stuff.

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


def create_combined_importances_pdf(graph_list: t.List[tc.GraphDict],
                                    image_path_list: t.List[str],
                                    node_positions_list: t.List[np.ndarray],
                                    graph_fidelity_list: t.List[np.array],
                                    node_importances_list: t.List[np.ndarray],
                                    edge_importances_list: t.List[np.ndarray],
                                    output_path: str,
                                    channel_colors_map: t.Optional[t.List[str]],
                                    channel_infos: list[dict] = DEFAULT_CHANNEL_INFOS,
                                    label_list: t.Optional[t.List[str]] = None,
                                    importance_threshold: t.Optional[float] = None,
                                    channel_limit_map: t.Optional[dict[int, float]] = None,
                                    base_fig_size: float = 8,
                                    show_ticks: bool = False,
                                    radius: float = 50.0,
                                    thickness: float = 20.0,
                                    logger: logging.Logger = NULL_LOGGER,
                                    log_step: int = 100,
                                    cache_path: t.Optional[str] = None,
                                    ):
    """
    This function will create a PDF file which contains the visualizations of multiple graph elements and their 
    corresponding explanation masks. This specific function will visualize the explanations from all explanation
    channels on top of one single graph visualization. The different explanation channels will thereby be encoded 
    by different colormaps. The importance value will be encoded as the alpha value of the color while the channel 
    fidelity will be encoded in the color map gradient.
    
    The PDF will have one page for each element to be visualized. The visualizations will be ordered in the same
    order as the given graph_list.
    
    :param graph_list: A list containing the graph dict representations of all the graphs to be visualized
    :param image_path_list: A list containing the absolute string paths to the visualization images of the
        graphs. Must have the same order as graph_list.
    :param node_positions_list: A list containing the node position arrays for the graphs to be visualized.
        These arrays should determine the 2D pixel coordinates for each node of the graphs. Must have the
        same order as graph_list.
    :param graph_fidelity_list: A list of numpy arrays that contain the fidelity values for each channel of the
        model for each graph. The fidelity value is a float value that describes how much the explanation channel
        actually contributes to the prediction outcome. The list must have the same order as the given graph_list.
    :param node_importances_list: A list of numpy arrays that contain the node importance values for each node and
        each channel of the model. The list must have the same order as the given graph_list.
    :param edge_importances_list: A list of numpy arrays that contain the edge importance values for each edge and
        each channel of the model. The list must have the same order as the given graph_list.
    :param output_path: The absolute string path to the output PDF file.
    :param channel_colors_map: A dictionary that maps the channel index to a valid matplotlib colormap. The colormap
        will be used to encode the channel fidelity values. The keys of this dict are the channel indices and the
        values are the colormaps. The number of channels must match the number of channels in the given node and edge
        importances arrays.
    :param channel_infos: This is a dictionary whose keys are the integer indices of the explanation channels that 
        are being used for the visualization. The values of this dictionary are themselves dictionaries that contain
        additional information about the explanation channels. The keys of these inner dictionaries are the string
        names of these additional properties. The following properties are required: 'name' (the name of the channel),
    :param label_list: A list of string labels that will be attached to the visualizations in the format of a title
        for the page.
    :param importance_threshold: A float value that will be used to threshold the importance values. If a threshold
        is given, all importance values that are below this threshold will be set to zero.
    :param channel_limit_map: A dictionary that maps the channel index to a float value. This float value will be used
        as the upper limit for the color map normalization. If no channel limit map is given, the limits will be
        defined as the 90th percentiles of the fidelity values across all the given graphs.
    :param base_fig_size: The size of the figures. Modification of this value will influence the file size of the PDF
        and the ratio of the size of the text and image elements within each page of the PDF.
    :param radius: The radius of the the node importance circles that will be drawn on the graph visualization. The 
        bigger this value, the larger the circles around each node will be.
    :param thickness: The thickness of the edge importance lines that will be drawn on the graph visualization. The
        bigger this value, the thicker the lines will be.
    :param show_ticks: A boolean value that determines if the ticks of the axis should be shown or not.
    :param logger: A logger instance that will be used to log the progress of the function.
    :param log_step: An integer value that determines how often the function should log the progress.
    :param cache_path: A string path to a cache directory. If this path is given, the function will try to cache the
        graph visualizations in this directory. This can be useful if the visualizations are expensive to compute.
    
    :returns: None
    """
    num_channels = len(channel_colors_map)
    
    # If no channel limit map is explicitly given we are going to define the limits ourselves as the 90th percentiles
    # of the fidelity values across all the given graphs.
    if not channel_limit_map:
        
        channel_limit_map = {}
        for channel_index in range(num_channels):
            value = np.percentile([fid[channel_index] for fid in graph_fidelity_list], 90)
            channel_limit_map[channel_index] = value
    
    with PdfPages(output_path) as pdf:
        
        for index, graph in enumerate(graph_list):
            node_positions = node_positions_list[index]

            # For this specific implementation of the visualization we only create one single figure for each page of 
            # the PDF. This figure shall then contain all the individual explanation channels at once, but just encoded 
            # with different color schemes.
            fig, rows = plt.subplots(
                ncols=1,
                nrows=1,
                figsize=(base_fig_size, base_fig_size),
                squeeze=False,
            )
            ax = rows[0][0]
            
            # This function will correctly draw the graph image from the file system to the axis in such a way that 
            # the node positions array matches the pixel coordinates in the axis.
            # Here we only need to draw the visualization once and can then draw all the explanations on top of 
            # that same graph visualization.
            image_path = image_path_list[index]
            draw_image(ax, image_path, remove_ticks=not show_ticks)
            
            # node_importance: (num_nodes, num_channels)
            node_importances: np.ndarray = array_normalize(node_importances_list[index])
            # edge_importance: (num_nodes, num_channels)
            edge_importances: np.ndarray = array_normalize(edge_importances_list[index])
            
            if importance_threshold is not None:
                node_importances = binary_threshold(node_importances, importance_threshold)
                edge_importances = binary_threshold(edge_importances, importance_threshold)
            
            # This is a numpy array that contains one float value for every channel of the model, where the 
            # value describes the fidelity value of that channel (== the value of how much that particular explanation 
            # channel actually contributes to the prediction outcome).
            # graph_fidelity: (num_channels, )
            graph_fidelity: np.ndarray = graph_fidelity_list[index]
            
            # Then we draw the same importance for every 
            for channel_index, cmap in channel_colors_map.items():
                
                # 24.04.24
                # This dict contains additional information about the explanation channels that are 
                # being used for the visualization. The keys of this dict are the string names of these 
                # additional properties.
                info: dict = channel_infos[channel_index]
                
                fidelity_channel: float = graph_fidelity[channel_index]
                
                norm = mcolors.Normalize(vmin=0, vmax=channel_limit_map[channel_index])
                color = cmap(norm(fidelity_channel))
                
                scalar_mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
                colorbar = fig.colorbar(
                    scalar_mappable, 
                    ax=ax, 
                    orientation='vertical', 
                    pad=0.05, 
                    shrink=0.6,
                )
                # We want to use the name of the explanation channel as the label of the color bar!
                colorbar.set_label(info['name'], rotation=270, labelpad=20)
                
                plot_node_importances_background(
                    ax=ax,
                    g=graph,
                    node_positions=node_positions,
                    node_importances=node_importances[:, channel_index],
                    color=color,
                    radius=radius,
                )
                plot_edge_importances_background(
                    ax=ax,
                    g=graph,
                    node_positions=node_positions,
                    edge_importances=edge_importances[:, channel_index],
                    color=color,
                    thickness=thickness,
                    radius=radius,
                )
            
            # There is also the option to attach a string label to every visualization in the format of 
            # a label / title of the page.
            if label_list:
                label = label_list[index]
                fig.suptitle(label)
                
            if index % log_step == 0:
                logger.info(f' * ({index} / {len(graph_list)}) visualizations created')

            pdf.savefig(fig)
            plt.close(fig)