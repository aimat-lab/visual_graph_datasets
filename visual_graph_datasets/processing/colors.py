"""
Functionality for processing color graphs.
"""
import os
import typing as t

import click
import networkx as nx
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d

import visual_graph_datasets.typing as tv
from visual_graph_datasets.data import DatasetWriterBase
from visual_graph_datasets.data import extract_graph_mask
from visual_graph_datasets.data import nx_from_graph
from visual_graph_datasets.processing.base import ProcessingBase
from visual_graph_datasets.processing.base import StringEncoderMixin
from visual_graph_datasets.processing.base import OneHotEncoder
from visual_graph_datasets.visualization.base import layout_node_positions
from visual_graph_datasets.visualization.base import create_frameless_figure
from visual_graph_datasets.visualization.colors import visualize_color_graph
from visual_graph_datasets.visualization.colors import colors_layout
from visual_graph_datasets.generation.colors import graph_from_cogiles
from visual_graph_datasets.generation.colors import graph_to_cogiles


class ColorProcessing(ProcessingBase):
    """
    This class is used to for the processing of the special "Color Graph" graph type. A normal color graph consists 
    of colored nodes (RGB color code) and edges without any attributes. The class provides methods to process the
    color graph into a graph dict representation and to visualize the color graph. The class also provides methods
    to create a new visual graph dataset element based on the given color graph and to get a description map of the
    color graph.
    
    **Domain Representation**
    
    The domain specific representation of a color graph is called COGILES. It is a string representation of the 
    various colored nodes and their interconnections. Each letter in the string identifies one node of the graph 
    and the specific letter identifies the color of the node. Nodes are connected in the order in which they appear 
    in the string. Additionally, a string may contain a numbered "anchors" which signify non-sequential connections
    between nodes. The anchors are represented by dash character and a number. Also, brackets are possible in the 
    string which determine a branching location.
    
    Examples:
    
    The string "R-1RRRR-1" represents a cycle of 5 red nodes (the first and the last node in a sequence 
    additionally being connected to each other)
    
    The string "BB(GG)CC" represents a sequence of 2 blue nodes which then branch off into 2 paths, where 
    one path is 2 green nodes and the other path is 2 cyan nodes.
    
    The string "Y(G)(G)(G)" represents a star pattern with a yellow node in the center and 3 green nodes 
    at the edges.
    
    **Graph Dict Representation**
    
    A domain-specific COGILES representation of a color graph can be processed into a graph dict representation 
    by using the ``process`` method. The graph dict representation is a dictionary which contains the full graph 
    data of the color graph.
    
    .. code-block:: python
    
        cogiles = "R-1RRRR-1"
        processing = ColorProcessing()
        graph = processing.process(cogiles)
        print(graph)
    
    **Visualization**
    
    A color graph can be visualized using the ``visualize`` method (numpy array) or alternatively the 
    ``visualize_as_figure`` method (matplotlib figure). The visualization will result in an image with the given 
    width and height in pixels. The layout strategy can be chosen from the available layout strategies.
    
    .. code-block:: python
    
        cogiles = "R-1RRRR-1"
        processing = ColorProcessing()
        fig = processing.visualize_as_figure(cogiles, width=1000, height=1000)
        plt.show()
    
    """
    LAYOUT_STRATEGY_MAP = {
        'spring': nx.spring_layout,
        'colors': colors_layout,
    }
    LAYOUT_STRATEGIES = list(LAYOUT_STRATEGY_MAP.keys())

    DEFAULT_STRATEGY = 'colors'

    # This encoder can be overwritten through the creation of a subclass of ColorProcessing. The only 
    # constraint is that the encoder has to implement the StringEncoderMixin interface so that we can
    # use the "encode_string" method. 
    encoder: StringEncoderMixin = OneHotEncoder(
        values = [
            [1.0, 0.0, 0.0],  # RED
            [0.0, 1.0, 0.0],  # GREEN
            [0.0, 0.0, 1.0],  # BLUE
            [1.0, 1.0, 0.0],  # YELLOW
            [1.0, 0.0, 1.0],  # MAGENTA
            [0.0, 1.0, 1.0],  # CYAN
            [0.8, 0.8, 0.8],  # GREY
        ],
        dtype=list,
        string_values = [
            'R',  # RED
            'G',  # GREEN
            'B',  # BLUE
            'Y',  # YELLOW
            'M',  # MAGENTA
            'C',  # CYAN
            'H',  # GREY
        ],
        use_soft_decode=True,
    )

    def process(self,
                value: tv.DomainRepr,
                *args,
                additional_graph_data: dict = {},
                **kwargs
                ) -> tv.GraphDict:
        """
        Given the ``value`` cogiles domain representation of a color graph, this function returns the graph dict 
        representation of the color graph. The ``additional_graph_data`` parameter can be used to provide 
        additional graph data which is added to the graph dict representation. The method returns the graph
        dict representation of the color graph.
        
        :param value: The cogiles domain representation of the color graph.
        :param additional_graph_data: Additional graph data which is added to the graph dict representation.
        
        :returns: The graph dict representation of the color graph.
        """
        graph = graph_from_cogiles(value)
        graph = {
            **graph,
            **additional_graph_data,
        }
        
        # 28.10.24 - Now it is possible to add a class property "encoder" to the color processing. This has to 
        # be a class that implements the StringEncoderMixin interface so that we can use the "encode_string" method 
        # here that encodes a node value into a human-readable string representation. We then attach this to the
        # graph dict representation.
        if self.encoder:
            
            node_colors: t.List[str] = []
            for attr in graph['node_attributes']:
                node_color: str = self.encoder.encode_string(attr)
                node_colors.append(node_color)
                
            graph['node_colors'] = node_colors
        
        return graph

    def unprocess(self,
                  graph: tv.GraphDict,
                  *args,
                  **kwargs) -> tv.DomainRepr:
        """
        Given the ``graph`` dict representation of a color graph, this function returns corresponding 
        the cogiles domain representation
        
        :param graph: The graph dict representation of the color graph.
        
        :returns: The cogiles domain representation of the color graph as a string
        """
        value = graph_to_cogiles(graph)
        return value

    def visualize_as_figure(self,
                            value: tv.DomainRepr,
                            width: int = 1000,
                            height: int = 1000,
                            layout_strategy: str = DEFAULT_STRATEGY,  # deprecated
                            graph: t.Optional[tv.GraphDict] = None,
                            node_positions: t.Optional[np.ndarray] = None,
                            dim: int = 2,
                            k: float = 1.0,
                            visualize_func: t.Callable = visualize_color_graph,
                            **kwargs,
                            ) -> t.Tuple[plt.Figure, np.ndarray]:
        """
        Visualizes the given color graph that is represented by the ``value`` cogiles domain representation. The 
        visualization will result in an image with the given ``width`` and ``height`` in pixels. The layout 
        strategy can be chosen from the available layout strategies. The ``graph`` parameter can be used to
        directly provide the graph dict representation instead of the domain specific representation. The
        ``node_positions`` parameter can be used to directly provide the node positions instead of calculating
        them using the layout strategy. The function returns the figure and the node positions array.
        
        :param value: The cogiles domain representation of the color graph.
        :param width: The width of the resulting image in pixels.
        :param height: The height of the resulting image in pixels.
        :param graph: The graph dict representation of the color graph. This is optional. If this is given 
            the ``value`` parameter is ignored and the graph is used directly for the visualization.
        :param node_positions: The node positions array of shape (V, 2). This is optional. If this is given
            the ``layout_strategy`` parameter is ignored and the node positions are used directly for the 
            visualization.
        :param dim: The dimension of the visualization. This can be either "2" or "3". If this is set to "3"
            the visualization will be a 3D visualization using an isometric view of the coordinate system 
            as the backdrop.
        :param k: A float number that controls the attractive force between the nodes in the spring layouting 
            algorithm that is used to arrange the nodes in space.
        :param t: The visualization function that is used to visualize the graph. This function 
            should accept three arguments: ``ax`` - the Axes object on which to draw the visualization. 
            ``g`` - the graph dict representation of the graph to be drawn (this for example contains the 
            information about the colors of the nodes). ``node_positions`` - the node positions array of shape
            (V, 2) which contains the pixel coordinates of the nodes in the visualization.
            If the ``dim`` argument is set to "3", the ``ax`` argument will be an Axes3D object and the 
            ``node_positions`` argument will be a 3D array of shape (V, 3).
        
        :returns: The figure and the node positions array of shape (V, 2). The node positions are coordinates 
            in the pixel coordinate system of the image that is exported from the figure.
        """
        # 15.11.23
        # Changed this section to provide the option to directly provide the graph dict representation 
        # instead of the domain specific representation. This actually makes more sense when generating 
        # color datasets because then we don't have to do a redundant conversion to and from COGILES. 
        if graph is None:
            graph = self.process(value)
        
        # ~ creating the node positions
        # node_positions is a 2D array which is used to store the pixel coordinates of the nodes in the
        # visualization. This is used to determine the position of the nodes in the visualization. If the 
        # node_positions are not given, we use the layout_node_positions function to calculate them which 
        # internally uses a networkx layouting function with the given layout strategy.
        
        # node_positions: (V, 2) or (V, 3) depending on "dim"
        graph_nx = nx_from_graph(graph)
        # 28.03.24
        # Previously, this was using the "layout_strategy" parameter to determine the layouting function 
        # dynamically, but with the extension towards optional 3D visualizations, dragging this along 
        # was too much hassle - therefore now fixed to be colors_layout, which internally uses
        # nx.spring_layout to determine the node positions.
        node_positions = colors_layout(
            graph_nx,
            node_positions=node_positions,
            dim=dim,
            k=k,
        )
        node_positions = np.array(list(node_positions.values()))

        # ~ creating the actual figure
        # create_frameless_figure is a helper function which creates a figure with the given width and height
        # without any axis or frame. This is useful for the visualization of the graph because we don't want
        # any axis or frame to be visible in the visualization.
        fig, ax = create_frameless_figure(width=width, height=height, dim=dim)
        
        # 27.03.24
        # Previously this was hardcoded to "visualize_color_graph" but that function only visualizes the 
        # graph on a euclidean 2D plane. When extending to different modes of visualization, such as 3D, it 
        # will be necessary to supply a custom visualization function. 
        visualize_func(
            ax=ax,
            g=graph,
            node_positions=node_positions,
        )

        # The "node_positions" which are returned by the above function are values within the axes object
        # coordinate system. Using the following piece of code we transform these into the actual pixel
        # coordinates of the exported image.
        
        if dim == 2:
            # ax.transData can be used for exactly that purpose with transforms from and to between the 
            # internal axes coordinates and the overall figure coordinates. 
            node_positions = [[int(v) for v in ax.transData.transform(pos)]
                            for pos in node_positions]
            node_positions = np.array(node_positions)
            
        elif dim == 3:
            # in the 3D case this process is a bit more complicated. First we need to use the proj3d utility 
            # function to project the 3D internal coordinates into a 2D coordinates of the axes object.
            # Only then can we apply the same transform as in the 2D case.
            node_positions_mapped = []
            for x, y, z in node_positions:
                x2d, y2d, _ = proj3d.proj_transform(x, y, z, ax.get_proj())

                # Now we transform the 2D display coordinates to pixel coordinates
                xpix, ypix = ax.transData.transform((x2d, y2d))
                node_positions_mapped.append([xpix, ypix])
                
            node_positions = np.array(node_positions_mapped)

        # We return the figure object itself and the node_positions array that has been calculated 
        # to contain the 2D pixel coordinates of all the nodes in the pixel coordinate system of 
        # the figure.
        return fig, node_positions

    def visualize(self,
                  value: tv.DomainRepr,
                  width: int = 1000,
                  height: int = 1000,
                  layout_strategy: click.Choice(LAYOUT_STRATEGIES) = DEFAULT_STRATEGY,
                  **kwargs,
                  ) -> np.ndarray:
        """
        This method visualizes the given color graph that is represented by the ``value`` cogiles domain 
        representation. The visualization will result in an image with the given ``width`` and ``height`` in
        pixels. The layout strategy can be chosen from the available layout strategies. The function returns
        the image as a numpy array. 
        
        :param value: The cogiles domain representation of the color graph.
        :param width: The width of the resulting image in pixels.
        :param height: The height of the resulting image in pixels.
        :param layout_strategy: The layout strategy to use for the visualization.
        
        :returns: the image as a numpy array of the shape (height, width, 3).
        """
        
        fig, _ = self.visualize_as_figure(
            value=value,
            width=width,
            height=height,
            layout_strategy=layout_strategy,
            **kwargs,
        )

        # This is a helper function which converts the figure into a numpy array. This is useful because 
        # we want to return the visualization as a numpy array.
        array = self.array_from_figure(
            figure=fig,
            width=width,
            height=height
        )
        plt.close(fig)

        return array

    def create(self,
               value: tv.DomainRepr,
               index: int = 0,
               output_path: str = os.getcwd(),
               graph_labels: list = [],
               additional_metadata: dict = {},
               additional_graph_data: dict = {},
               width: int = 1000,
               height: int = 1000,
               graph: t.Optional[tv.GraphDict] = None,
               writer: t.Optional[DatasetWriterBase] = None,
               *args,
               **kwargs,
               ) -> None:
        """
        This method creates a new visual graph dataset element based on the given ``value`` cogiles domain 
        representation. The resulting visualization file and the metadata file will be saved to the 
        given ``output_path`` folder path. 
        The ``index`` parameter is used to determine the name of the file. The ``graph_labels`` parameter can be used to 
        provide additional information about the graph. The ``additional_metadata`` parameter can be used to 
        provide additional metadata which is added to the metadata file. The ``additional_graph_data`` parameter 
        can be used to provide additional graph data which is added to the graph dict representation. The 
        ``width`` and ``height`` parameters are used to determine the size of the resulting visualization.
        
        :param value: The cogiles domain representation of the color graph.
        :param index: The index of the visual graph dataset element, this will also be used as the name of both the 
            visualization and the metadata file (both with different file extensions).
        :param output_path: The folder path where the visualization and the metadata file will be saved.
        :param graph_labels: A list of labels which provide additional information about the graph.
        :param additional_metadata: Additional metadata which is added to the metadata file.
        :param additional_graph_data: Additional graph data which is added to the graph dict representation.
        :param width: The width of the resulting visualization in pixels.
        :param height: The height of the resulting visualization in pixels.
        :param writer: The dataset writer which is used to write the visualization and the metadata file. If this is 
            given the files will not be saved to the output path but instead written using the dataset writer.
        
        :returns: None
        """
        if graph is None:
            graph = self.process(value)
        
        graph.update(additional_graph_data)
        fig, node_positions = self.visualize_as_figure(
            value=value,
            width=width,
            height=height,
            graph=graph,
            **kwargs,
        )
        graph['node_positions'] = node_positions

        metadata = {
            **additional_metadata,
            'index': int(index),
            'name': value,
            'value': value,
            'target': graph_labels,
            'image_width': width,
            'image_height': height,
            'graph': graph,
        }

        # Technically, NOT using a writer is deprecated, but we still support it for backwards compatibility.
        if writer is None:
            fig_path = os.path.join(output_path, f'{index}.png')
            self.save_figure(fig, fig_path)
            plt.close(fig)

            metadata_path = os.path.join(output_path, f'{index}.json')
            self.save_metadata(metadata, metadata_path)
        else:
            # The writer instance simply receives the metadata and the figure and writes them to the dataset.
            writer.write(
                name=int(index),
                metadata=metadata,
                figure=fig,
            )
            
    def get_num_node_attributes(self) -> int:
        """
        Returns the number of node attributes of the color graph. A color graph always has 3 
        node attributes which are the float RGB values.
        """
        return 3 
    
    def get_num_edge_attributes(self) -> int:
        """
        Returns the number of edge attributes of the color graph. A color graph always has 1
        edge attribute which is a constant edge weight of 1.
        """
        return 1

    def get_description_map(self) -> dict:
        """
        This method returns a description map of the color graph. The description map contains information about
        the node and edge attributes of the color graph. The description map is used to provide information about
        the graph to the user.
        
        :returns: A description map of the color graph.
        """
        return {
            'node_attributes': {
                0: 'A float value 0 to 1 representing the RED portion of the color',
                1: 'A float value 0 to 1 representing the GREEN portion of the color',
                2: 'A float value 0 to 1 representing the BLUE portion of the color'
            },
            'edge_attributes': {
                0: 'A constant edge weight of 1'
            }
        }
