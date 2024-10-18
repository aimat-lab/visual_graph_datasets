import os
import json
import typing as t

import networkx as nx
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import visual_graph_datasets.typing as tv
from visual_graph_datasets.data import VisualGraphDatasetWriter
from visual_graph_datasets.processing.base import NumericJsonEncoder
from visual_graph_datasets.processing.base import ProcessingBase
from visual_graph_datasets.visualization.base import draw_image
from visual_graph_datasets.visualization.base import layout_node_positions
from visual_graph_datasets.visualization.base import visualize_graph
from visual_graph_datasets.visualization.base import create_frameless_figure


class GenericProcessing(ProcessingBase):
    """
    This class implements the visual graph dataset ProcessingBase interface for "generic" graphs that 
    cannot be more closely described by any other existing domain-specific graph type.
    
    Therefore, this class represents some kind of a fallback solution for dealing with custom graphs. However, this 
    should only be used when absolutely necessary! Whenever possible, a domain-specific processing class should be 
    used to handle the graph instances, since these domain-specific implementations will be able to provide much 
    better graph processing as well as visualization capabilities.
    
    **DOMAIN SPECIFIC REPRESENTATION**

    In the case of of this particular class the representation is more "domain agnostic" rather than "domain specific" 
    since this class should be the fallback solution for classes that cannot be assigned to a specific domain. Therefore, 
    the string representation for this class is simply the JSON encoded string of the entire graph dict!
    
    **VISUALIZATION**
    
    The graphs will be visualized such that all nodes and edges have a uniform color and use a standard networkx 
    layouting strategy to align in the 2D images.
    """
    
    def process(self,
                value: str,
                additional_graph_data: t.Optional[dict] = None,
                ) -> tv.GraphDict:
        """
        This method is responsible for converting the string representation of the graph into the graph dict
        representation.
        
        :param value: The string representation of the graph. In the case of this class the string representation 
            is simply the JSON encoded string of the entire graph dict.
        :param additional_graph_data: Additional graph data that should be added to the graph dict.
        
        :returns: The graph dict representation of the graph.
        """
        graph = json.loads(value)
        
        if additional_graph_data:
            graph.update(additional_graph_data)
            
        # All subsequent processing operations expect the graph dict fields to be numpy arrays. Therefore, we
        # convert all lists to numpy arrays here.
        for key, value in graph.items():
            if isinstance(value, list):
                graph[key] = np.array(value)
        
        return graph
    
    def unprocess(self,
                  graph: tv.GraphDict,
                  ) -> str:
        """
        This method is responsible for converting the graph dict representation into the string representation of the 
        graph. This is the inverse operation of the "process" method.
        
        :param value: The graph dict representation of the graph.
        
        :returns: The string representation of the graph.
        """
        value = json.dumps(graph, cls=NumericJsonEncoder)
        return value
    
    def visualize_as_figure(self,
                            value: str,
                            graph: t.Optional[tv.GraphDict] = None,
                            node_positions: t.Optional[np.ndarray] = None,
                            width: int = 1000,
                            height: int = 1000,
                            layout_cb: t.Callable = nx.spring_layout,
                            ) -> tuple[plt.Figure, np.ndarray]:
        """
        This method will visualize the given ``value`` graph representation as a matplotlib figure with a 
        ``width`` and ``height`` in pixels. The graph will be visualized using the networkx library and the layouting
        strategy specified by the ``layout_cb`` callable.
        
        :param value: The string representation of the graph.
        :param graph: The graph dict representation of the graph. If not given, the graph will be processed from the 
            value. If this is given, the "value" will be ignored.
        :param node_positions: The node positions of the graph. If not given, the node positions will be calculated
            using the layouting strategy specified by the ``layout_cb`` callable. If this is given, the "layout_cb"
            will be ignored.
        :param width: The width of the visualization in pixels.
        :param height: The height of the visualization in pixels.
        :param layout_cb: The layouting strategy that should be used to calculate the node positions. This callable
            should take a graph and return a numpy array of shape (num_nodes, 2) where each row represents the x and y
            coordinates of a node.
        
        :returns: A tuple containing the matplotlib figure and the node positions.
        """
        if not graph:
            graph = self.process(value)
            
        # Now we visualize the graph as a normal kind of image where 
        fig, ax = create_frameless_figure(
            width=width,
            height=height,
        )
        
        if not node_positions:
            node_positions = layout_node_positions(
                g=graph,
                layout_cb=layout_cb,
            )
            
        visualize_graph(
            ax=ax,
            graph=graph,
            node_positions=node_positions,
        )
            
        # The "node_positions" which are returned by the above function are values within the axes object
        # coordinate system. Using the following piece of code we transform these into the actual pixel
        # coordinates of the figure image.
        node_positions = [[int(v) for v in ax.transData.transform((x, y))]
                          for x, y in node_positions]
        node_positions = np.array(node_positions)
        
        return fig, node_positions
    
    def visualize(self,
                  value: str,
                  graph: t.Optional[tv.GraphDict] = None,
                  width: int = 1000,
                  height: int = 1000,
                  ) -> np.ndarray:
        """
        This method visualizes the given ``value`` graph representation as a numpy array image with a ``width`` and
        ``height`` in pixels. Returns a numpy array that represents the image of the visualization.
        
        :returns: The numpy array image of the visualization.
        """
        # The "visualize_as_figure" method will do all of the heavy lifting of actually creating the 
        # visualization as a mpl Figure object.
        fig, node_positions = self.visualize_as_figure(
            value=value,
            graph=graph,
            width=width,
            height=height,
        )
            
        # Here we only need to convert that Figure to an image numpy array
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
               index: int,
               value: str,
               writer: VisualGraphDatasetWriter,
               graph: t.Optional[tv.GraphDict] = None,
               width: int = 1000,
               height: int = 1000,
               additional_graph_data: dict = {},
               additional_metadata: dict = {},
               **kwargs,
               ) -> None:
        """
        This method will create the visualization of the given ``value`` string representation and use the 
        VisualGraphDatasetWriter instance to write the visualization and the metadata to the disk.
        
        :param index: The index of the graph in the dataset. This will be used as the name of the element 
            files as well.
        :param value: The string representation of the graph.
        :param writer: The VisualGraphDatasetWriter instance that should be used to write the visualization and
            the metadata to the disk.
        
        :returns: None
        """
        if not graph:
            graph = self.process(value)

        graph.update(additional_graph_data)
        fig, node_positions = self.visualize_as_figure(
            value=value,
            width=width,
            height=height,
            graph=graph,
        )
        graph['node_positions'] = node_positions

        metadata = {
            **additional_metadata,
            'index': int(index),
            'name': value,
            'value': value,
            'image_width': width,
            'image_height': height,
            'graph': graph,
        }

        writer.write(
            name=int(index),
            metadata=metadata,
            figure=fig,
        )
        
    def get_num_node_attributes(self) -> int:
        """
        This method returns the number of node attributes that are present in the graph representation.
        A generic graph always has 1 node attribute which is a constant value.
        """
        return 1
    
    def get_num_edge_attributes(self) -> int:
        """
        This method returns the number of edge attributes that are present in the graph representation.
        A generic graph always has 1 edge attribute which is a constant value.
        """
        return 1