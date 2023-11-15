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

import visual_graph_datasets.typing as tv
from visual_graph_datasets.data import DatasetWriterBase
from visual_graph_datasets.data import extract_graph_mask
from visual_graph_datasets.data import nx_from_graph
from visual_graph_datasets.processing.base import ProcessingBase
from visual_graph_datasets.visualization.base import layout_node_positions
from visual_graph_datasets.visualization.base import create_frameless_figure
from visual_graph_datasets.visualization.colors import visualize_color_graph
from visual_graph_datasets.visualization.colors import colors_layout
from visual_graph_datasets.generation.colors import graph_from_cogiles
from visual_graph_datasets.generation.colors import graph_to_cogiles


class ColorProcessing(ProcessingBase):
    """
    This processing class can be used for colored graphs. These are graphs whose node attributes consist of 3 
    values which represent an RGB color code for the color associated with each node. These graphs are undirected 
    and do not have edge attributes.
    """
    LAYOUT_STRATEGY_MAP = {
        'spring': nx.spring_layout,
        'colors': colors_layout,
    }
    LAYOUT_STRATEGIES = list(LAYOUT_STRATEGY_MAP.keys())

    DEFAULT_STRATEGY = 'colors'

    def process(self,
                value: tv.DomainRepr,
                *args,
                additional_graph_data: dict = {},
                **kwargs
                ) -> tv.GraphDict:
        graph = graph_from_cogiles(value)
        graph = {
            **graph,
            **additional_graph_data
        }
        return graph

    def unprocess(self,
                  graph: tv.GraphDict,
                  *args,
                  **kwargs) -> tv.DomainRepr:
        value = graph_to_cogiles(graph)
        return value

    def visualize_as_figure(self,
                            value: tv.DomainRepr,
                            width: int = 1000,
                            height: int = 1000,
                            layout_strategy: str = DEFAULT_STRATEGY,
                            *args,
                            **kwargs,
                            ) -> t.Tuple[plt.Figure, np.ndarray]:
        # 15.11.23
        # Changed this section to provide the option to directly provide the graph dict representation 
        # instead of the domain specific representation. This actually makes more sense when generating 
        # color datasets because then we don't have to do a redundant conversion to and from COGILES. 
        if 'graph' in kwargs:
            g = kwargs['graph']
        # The default case, if the graph is not given, of course is still to use the given cogiles 
        # domain representation.
        else: 
            g = self.process(value)
        
        node_positions = layout_node_positions(
            g=g,
            layout_cb=self.LAYOUT_STRATEGY_MAP[layout_strategy],
        )

        fig, ax = create_frameless_figure(width=width, height=height)
        visualize_color_graph(
            ax=ax,
            g=g,
            node_positions=node_positions
        )

        # The "node_positions" which are returned by the above function are values within the axes object
        # coordinate system. Using the following piece of code we transform these into the actual pixel
        # coordinates of the figure image.
        node_positions = [[int(v) for v in ax.transData.transform((x, y))]
                          for x, y in node_positions]
        node_positions = np.array(node_positions)

        return fig, node_positions

    def visualize(self,
                  node_attributes: np.ndarray,
                  edge_indices: np.ndarray,
                  width: int = 1000,
                  height: int = 1000,
                  layout_strategy: click.Choice(LAYOUT_STRATEGIES) = DEFAULT_STRATEGY,
                  ) -> np.ndarray:
        fig, _ = self.visualize_as_figure(
            node_attributes=node_attributes,
            edge_indices=edge_indices,
            width=width,
            height=height,
            layout_strategy=layout_strategy,
        )

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
               writer: t.Optional[DatasetWriterBase] = None,
               *args,
               **kwargs,
               ) -> None:
        g = self.process(value)
        g.update(additional_graph_data)
        fig, node_positions = self.visualize_as_figure(
            value=value,
            width=width,
            height=height,
        )
        g['node_positions'] = node_positions

        metadata = {
            **additional_metadata,
            'index': int(index),
            'name': value,
            'value': value,
            'target': graph_labels,
            'image_width': width,
            'image_height': height,
            'graph': g,
        }

        if writer is None:
            fig_path = os.path.join(output_path, f'{index}.png')
            self.save_figure(fig, fig_path)
            plt.close(fig)

            metadata_path = os.path.join(output_path, f'{index}.json')
            self.save_metadata(metadata, metadata_path)
        else:
            writer.write(
                name=int(index),
                metadata=metadata,
                figure=fig,
            )

    def get_description_map(self) -> dict:
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
