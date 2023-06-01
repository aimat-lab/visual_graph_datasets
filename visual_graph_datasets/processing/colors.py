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

import visual_graph_datasets.typing as tc
from visual_graph_datasets.processing.base import ProcessingBase
from visual_graph_datasets.visualization.base import layout_node_positions
from visual_graph_datasets.visualization.base import create_frameless_figure
from visual_graph_datasets.visualization.colors import visualize_color_graph
from visual_graph_datasets.visualization.colors import colors_layout


# TODO: Invent a string representation of color graphs...
# TODO: Extend this class to have extra methods deal with the processing of that format then...
class ColorProcessing(ProcessingBase):

    LAYOUT_STRATEGY_MAP = {
        'spring': nx.spring_layout,
        'colors': colors_layout,
    }
    LAYOUT_STRATEGIES = list(LAYOUT_STRATEGY_MAP.keys())

    DEFAULT_STRATEGY = 'colors'

    def process(self,
                node_attributes: np.ndarray,
                edge_indices: np.ndarray,
                ) -> tc.GraphDict:
        # This is an incredibly
        graph = {
            'node_indices': list(range(len(node_attributes))),
            'node_attributes': node_attributes,
            'edge_indices': edge_indices,
            'edge_attributes': [[1] for _ in edge_indices]
        }

        return graph

    def visualize_as_figure(self,
                            node_attributes: np.ndarray,
                            edge_indices: np.ndarray,
                            width: int = 1000,
                            height: int = 1000,
                            layout_strategy: str = DEFAULT_STRATEGY
                            ) -> t.Tuple[plt.Figure, np.ndarray]:
        g = self.process(node_attributes, edge_indices)
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
               node_attributes: np.ndarray,
               edge_indices: np.ndarray,
               index: int = 0,
               output_path: str = os.getcwd(),
               graph_labels: list = [],
               additional_metadata: dict = {},
               additional_graph_data: dict = {},
               width: int = 1000,
               height: int = 1000,
               ) -> None:
        g = self.process(node_attributes, edge_indices)
        g.update(additional_graph_data)
        fig, node_positions = self.visualize_as_figure(
            node_attributes,
            edge_indices,
            width=width,
            height=height,
        )
        g['node_positions'] = node_positions

        fig_path = os.path.join(output_path, f'{index}.png')
        self.save_figure(fig, fig_path)
        plt.close(fig)

        metadata = {
            **additional_metadata,
            'index': int(index),
            'target': graph_labels,
            'image_width': width,
            'image_height': height,
            'graph': g,
        }
        metadata_path = os.path.join(output_path, f'{index}.json')
        self.save_metadata(metadata, metadata_path)

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
