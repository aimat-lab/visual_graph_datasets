"""
This experiment generates the "rb_dual_motifs" (red and blue dual motifs) synthetic graph regression 
dataset. 

the dataset
-----------

This dataset is a "color graph" dataset, which means that it consists graphs where each node is associated 
with 3 node features representative of an RGB color code. Edges are undirected and unweighted. 
This specific dataset is a graph regression dataset where each graph is associated with a single numeric 
value between -4 and +4. This value is entirely determined by the subgraph motifs present in the graph. 
There are two mainly red-dominated motifs that contribute positive values to the overall graph value and 
two mainly blue-dominated motifs which contribute negative values. Each graph can have any combination of 
0-2 of these motifs. Additionally, the graph value is also affected by a small random numeric value to 
make the underlying task slightly more difficult for a model to solve.

The primary purpose of this dataset is to act as a benchmark dataset for XAI applications. All of the 
above motifs can be considered as the perfect ground truth structural explanations for the corresponding 
target values of the graphs. The masks which identify those motifs within the graphs are also saved as 
part of the dataset to act as ground truth attributional explanations for the XAI benchmarking.

the procedure
-------------

This section describes briefly the idea behind the generation process of the dataset.

At first a stochastic graph generator is used to generate a lot of random color graphs. These graphs 
randomly vary in several parameters such as the number of nodes and edges within certain limits, but also 
the number and combinations of special subgraph motifs they are seeded with. 

From this large number of graphs a much smaller number of graphs is then randomly sampled, such that 
the final dataset is uniformly distributed in the target regression value.

This dataset is then ultimately saved into a "visual graph dataset" format onto the disk as an 
artifact of the experiment execution.
"""
import os
import random
import typing as t

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import visual_graph_datasets.typing as tv
from matplotlib.backends.backend_pdf import PdfPages
from scipy.special import softmax
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from visual_graph_datasets.data import nx_from_graph
from visual_graph_datasets.generation.graph import GraphGenerator
from visual_graph_datasets.generation.colors import *
from visual_graph_datasets.util import edge_importances_from_node_importances
from visual_graph_datasets.data import VisualGraphDatasetWriter
from visual_graph_datasets.processing.base import create_processing_module
from visual_graph_datasets.processing.colors import ColorProcessing
from visual_graph_datasets.visualization.base import draw_image
from visual_graph_datasets.visualization.colors import colors_layout
from visual_graph_datasets.visualization.importances import plot_node_importances_border
from visual_graph_datasets.visualization.importances import plot_edge_importances_border

# == DATASET PARAMETERS ==
DATASET_NAME: str = 'triangle_areas'
NUM_ELEMENTS_RAW: int = 10_000
NUM_ELEMENTS: int = 10_000
COLORS: t.List[list] = [
    GRAY, GRAY,
    RED,
    GREEN,
    BLUE,
    MAGENTA,
    CYAN,
    YELLOW,
]
VALUE = [0.5, 0, 0]
NUM_NODES_RANGE = (15, 40)
NUM_ADDITIONAL_EDGES_RANGE = (3, 10)
VALUE_MOTIF_MAP = {
    -2: make_star_motif(YELLOW, BLUE, k=3),
    -1: make_ring_motif(GREEN, BLUE, k=2),
    +1: make_ring_motif(GREEN, RED, k=2),
    +2: make_star_motif(YELLOW, RED, k=3),
}
VALUE_MOTIF_NX_MAP = {
    key: nx_from_graph(value)
    for key, value in VALUE_MOTIF_MAP.items()
}
NOISE_MAGNITUDE: float = 0.25

# == EVAL PARAMETERS ==
NUM_BINS: int = 100
LOG_STEP: int = 100
__DEBUG__ = True
__TESTING__ = False

def triangle_area(coords1, coords2, coords3):
    x1, y1 = coords1
    x2, y2 = coords2
    x3, y3 = coords3
    
    # Calculate the area using the Shoelace formula
    area = abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)) / 2
    return area


@Experiment(base_path=folder_path(__file__),
            namespace=file_namespace(__file__),
            glob=globals())
def experiment(e: Experiment):
    e.log('generating rb dual motifs regression dataset...')

    if __TESTING__:
        e.NUM_ELEMENTS_RAW = 1000
        e.NUM_ELEMENTS = 100

    # First step is to generate a bunch of graphs
    graphs: t.List[tv.GraphDict] = []
    c: int = 0
    while len(graphs) < NUM_ELEMENTS_RAW:

        num_nodes = random.randint(*NUM_NODES_RANGE)
        num_additional_edges = random.randint(*NUM_ADDITIONAL_EDGES_RANGE)

        # here we are going to create a triangle motif which does not contain
        motif = {
            'node_indices': [0, 1, 2],
            'node_attributes': [
                VALUE,
                VALUE,
                VALUE,
            ],
            'edge_indices': [
                [0, 1],
                [1, 0],
                [1, 2],
                [2, 1],
                [2, 0],
                [0, 2],
            ],
            'edge_attributes': [[1], [1], [1], [1], [1], [1]],
        }

        generator = GraphGenerator(
            num_nodes=num_nodes,
            num_additional_edges=num_additional_edges,
            node_attributes_cb=lambda *args: random.choice(COLORS),
            edge_attributes_cb=lambda *args: [1],
            seed_graphs=[motif],
            is_directed=False,
            prevent_edges_in_seed_graphs=True,
        )

        is_valid = False
        while not is_valid:
            try:
                generator.reset()
                graph = generator.generate()
                tc.assert_graph_dict(graph)
                is_valid = True
            except (AssertionError, IndexError, KeyError) as exc:
                e.log(f' * error: {str(exc)}')
        
        node_coordinates = []
        # generate the triangle corners coordinates
        for index in generator.seed_graph_indices:
            if index == -1:
                node_coordinates.append(None)
            else:
                size = random.uniform(0.0, 5.0)
                node_coordinates.append([
                    random.uniform(-1, 1) * size,
                    random.uniform(-1, 1) * size,
                ])
        
        area = triangle_area(*[coords for coords in node_coordinates if coords is not None])
        graph['_node_coordinates'] = node_coordinates
        graph_value = area
        print('area', area)
            
        graph['graph_labels'] = np.array([graph_value])
        graph['graph_labels_raw'] = np.array([graph_value])
        graphs.append(graph)

        if c % LOG_STEP == 0:
            e.log(f' * {c}/{NUM_ELEMENTS_RAW} generated'
                  f' - num_motifs: {1}'
                  f' - graph_value: {graph_value:.2f}')

        c += 1

    indices = list(range(len(graphs)))
    values = [graph['graph_labels'][0] for graph in graphs]
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
    hist, bins = np.histogram(values, bins=NUM_BINS)
    ax.hist(values, bins=NUM_BINS, color='lightgray')
    fig.savefig(os.path.join(e.path, 'distribution_raw.pdf'))
    
    # And finally after having obtained a list of graphs which show the desired uniform distribution 
    # of the target value, we can now create the visual graph dataset using those graphs.
    
    dataset_path = os.path.join(e.path, DATASET_NAME)
    os.mkdir(dataset_path)
    
    processing = ColorProcessing()
    processing_code = create_processing_module(
        processing=processing
    )
    processing_path = os.path.join(dataset_path, 'process.py')
    with open(processing_path, mode='w') as file:
        file.write(processing_code)
    
    writer = VisualGraphDatasetWriter(dataset_path)
    
    examples_path = os.path.join(e.path, 'examples.pdf')
    with PdfPages(examples_path) as pdf:
        for index, graph in enumerate(graphs):
            
            #print(list(graph.keys()))
            target = graph['graph_labels']
            #cogiles, _ = processing.extract(graph, mask=np.ones_like(graph['node_indices']))
            cogiles = ''
            node_positions = colors_layout(
                nx_from_graph(graph),
                k=0.8,
                dim=2,
                node_positions=graph['_node_coordinates']
            )
            node_positions = np.array([v.tolist() for v in node_positions.values()])
            # print('node_positions', node_positions)
            fig, node_positions = processing.visualize_as_figure(None, graph=graph, node_positions=node_positions)
            
            #del graph['_node_coordinates']
            graph['graph_labels'] = target
            graph['node_positions'] = node_positions
            graph['node_coordinates'] = node_positions
            if 'node_adjacency' in graph:
                del graph['node_adjacency']
            
            metadata = {
                'index':    index,
                'name':     cogiles,
                'value':    cogiles,
                'target':   target,
                'graph':    graph,
            }
            
            writer.write(
                name=index,
                metadata=metadata,
                figure=fig,
            )
            
            if index % LOG_STEP == 0:
                e.log(f' * ({index}/{NUM_ELEMENTS}) created')
                
                fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
                ax.set_title(f'index {index}')
                draw_image(
                    ax=ax,
                    image_path=writer.most_recent['image_path']
                )
                
                pdf.savefig(fig)
                plt.close(fig)

    

experiment.run_if_main()