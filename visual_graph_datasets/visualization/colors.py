import random
import typing as t

import networkx as nx
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import visual_graph_datasets.typing as tc


def colors_layout(G: nx.Graph, 
                  k: float = 1, 
                  scale: float = 10.0,
                  dim: int = 2,
                  node_positions: t.Optional[list] = None
                  ) -> t.Dict[int, tuple]:
    """
    This function implements a networkx layout function, which can be used to create a ``pos`` dictionary 
    for the given networkx graph ``G``. The layout is computed using the "spring" layout algorithm, which
    is akin to a phyiscal simulation of the graph, where the nodes are repelling each other and the edges
    are like springs.
    
    Additionally, for this specific layout it is possible to pass the ``node_positions`` argument, which is 
    a list with the same length as the number of nodes in the graph. The elements of this list are either
    None or a tuple of floats. For all the elements that are not None, those node positions will be fixed 
    during the layouting - meaning that the nodes will have those exact coordinates at the end of the layout.
    
    :param G: The networkx graph for which the layout should be computed. If the graph is given in the 
        graph dict representation it has to be converted to an nx Graph first using the ``nx_from_graph`` 
        utility function.
    :param k: The optimal distance between nodes (default: 1). This parameter can be changed to change the 
        scale of the resulting graph layout. The larger this value, the more the graph will "spread out".
    :param dim: The dimensionality of the layout to be created (default: 2).
    :node_positions: Optional list with the same length as the number of nodes in the graph. The indices of 
        this list are the node indices and the values can be None or a coordinate tuple. If the value is None 
        then the coordinate of that node will be computed by the layout. Otherwise that coordinate will be
        fixed.
    
    :returns: A dictioanry with the node indices as keys and the n-dimensional coordinates as values.
    """
    layout_kwargs = {
        'k': k,
        'scale': scale,
        'dim': dim,
        'center': np.zeros(shape=(dim, )),
    }
    
    if node_positions is not None:

        pos = {}
        fixed = []
        for index, coords in enumerate(node_positions):
            if coords is not None:
                pos[index] = coords
                fixed.append(index)
            else:
                pos[index] = np.random.uniform(-scale, scale, size=(dim, ))
        
        layout_kwargs.update({
            'pos': pos,
            'fixed': fixed,
        })
    
    # This function will then actually compute the layout using the "spring" layout algorithm. This algorithm 
    # is akin to a phyiscal simulation of the graph, where the nodes are repelling each other and the edges
    # are like springs. This will then result in a layout of the nodes in the chosen dimensional space.
    pos = nx.spring_layout(
        G,
        **layout_kwargs,
    )
    
    # 03.03.24 - Remove the additional kamada kawai layout since I felt like it does not change anything of the 
    # spring layout anyways.
    # pos = nx.kamada_kawai_layout(
    #     G,
    #     pos=pos,
    #     scale=scale,
    # )
    
    return pos


def visualize_grayscale_graph(ax: plt.Axes,
                              g: tc.GraphDict,
                              node_positions: np.ndarray,
                              color_map: t.Union[str, mpl.colors.Colormap] = 'gist_yarg',
                              node_border_width: float = 1.0,
                              node_size: float = 100.0) -> None:
    """
    Creates a grayscale color graph visualization for the given graph ``g`` drawn on the canvas of ``ax``
    using the node positions ``node_positions``. The node positions have to be in the figure coordinate
    system!

    :param plt.Axes ax: The axes onto which the visualization should be drawn
    :param tc.GraphDict g: The graph which is to be drawn
    :param node_positions: An array with the shape (V, 2) where V is the number of nodes in the graph. The
        values of this array represent the x, y coordinates of the corresponding nodes in the figure
    :param color_map: A color map to determine the color of each node visualization based on the node
        attribute. Currently grayscale, but could be any matplotlib color map
    :param node_border_width:
    :param node_size:
    :return: None
    """

    if isinstance(color_map, str):
        color_map = mpl.colormaps[color_map]

    # ~ drawing nodes
    for i in g['node_indices']:
        x, y = node_positions[i]

        value = g['node_attributes'][i][0]
        color = color_map(value)
        ax.scatter(
            x,
            y,
            color=color,
            edgecolors='black',
            linewidths=node_border_width,
            s=node_size,
            zorder=2,
        )

    # ~ drawing edges
    for e, (i, j) in enumerate(g['edge_indices']):
        x_i, y_i = node_positions[i]
        x_j, y_j = node_positions[j]

        ax.plot(
            (x_i, x_j),
            (y_i, y_j),
            color='black',
            zorder=1,
        )


def visualize_color_graph(ax: plt.Axes,
                          g: tc.GraphDict,
                          node_positions: np.ndarray,
                          node_border_width: float = 1.0,
                          edge_width: float = 1.0,
                          alpha: float = 1.0,
                          node_size: float = 100.0) -> None:
    """
    Creates a colored graph visualization for the given graph ``g`` drawn on the canvas of ``ax``
    using the node positions ``node_positions``. The node positions have to be in the figure coordinate
    system!

    For a "color" graph it is assumed that the first 3 node attributes of every node are values between
    0 and 1 which respectively define the red, green and blue (RGB) color value associated with that node.
    This color value is then used to visualize the graph.

    :param ax: The Axes onto which the visualization is drawn
    :param g: The graph to be drawn
    :param node_positions: An array with the shape (V, 2) where V is the number of nodes in the given graph.
        This array is supposed to contain the 2D coordinates for every node, defining where that node
        is to be drawn to the canvas.
    :param node_border_width: The line width of the black border around the circular node visualizations
    :param edge_width: The line width of the black edges between the nodes
    :param alpha: The alpha value for all the visualization elements
    :param node_size: The size of the node visualizations.

    :return: None
    """
    # ~ drawing nodes
    for i in g['node_indices']:
        x, y = node_positions[i]

        color = g['node_attributes'][i][:3]
        ax.scatter(
            x,
            y,
            color=(*color, alpha),
            edgecolors='black',
            linewidths=node_border_width,
            s=node_size,
            zorder=2,
        )

    # ~ drawing edges
    # The edges are simple: They are just black lines between the nodes.
    for e, (i, j) in enumerate(g['edge_indices']):
        x_i, y_i = node_positions[i]
        x_j, y_j = node_positions[j]

        ax.plot(
            (x_i, x_j),
            (y_i, y_j),
            color='black',
            lw=edge_width,
            zorder=1,
        )

    # 23.03.2023 - Now this is wild...
    # What we do here is we plot a completely transparent marker on top of every node, which functionally
    # has absolutely no effect because it's completely transparent. BUT, this is actually necessary for some
    # reason to prevent the position-drifting bug!
    for (x, y) in node_positions:
        ax.scatter(x, y, color=(1, 1, 1, 0), marker='x', s=100, zorder=3)
