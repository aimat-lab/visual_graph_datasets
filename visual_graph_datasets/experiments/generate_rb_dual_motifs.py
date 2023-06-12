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
from visual_graph_datasets.visualization.importances import plot_node_importances_border
from visual_graph_datasets.visualization.importances import plot_edge_importances_border

# == DATASET PARAMETERS ==
DATASET_NAME: str = 'rb_dual_motifs'
NUM_ELEMENTS_RAW: int = 50_000
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

        k_motifs = random.randint(0, 2)
        motifs = random.sample(list(VALUE_MOTIF_MAP.values()), k=k_motifs)

        generator = GraphGenerator(
            num_nodes=num_nodes,
            num_additional_edges=num_additional_edges,
            node_attributes_cb=lambda *args: random.choice(COLORS),
            edge_attributes_cb=lambda *args: [1],
            seed_graphs=motifs,
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

        # Calculating the target value
        graph['node_importances_1'] = np.zeros(shape=(num_nodes, 1)).astype(float)
        graph['node_importances_2'] = np.zeros(shape=(num_nodes, 2)).astype(float)

        graph_value = 0
        graph_value_noisy = 0
        g_nx = nx_from_graph(graph)
        for value, motif_nx in VALUE_MOTIF_NX_MAP.items():
            matcher = nx.isomorphism.GraphMatcher(
                g_nx, motif_nx,
                node_match=lambda a, b: np.isclose(a['node_attributes'], b['node_attributes']).all(),
                edge_match=lambda a, b: np.isclose(a['edge_attributes'], b['edge_attributes']).all(),
            )
            matcher.initialize()
            result = matcher.subgraph_is_isomorphic()
            if result:
                graph_value += value
                match_indices = np.array(list(matcher.mapping.keys()))
                graph['node_importances_1'][match_indices] = 1
                graph['node_importances_2'][match_indices, int(value > 0)] = 1
        
        # After the actual true value has been determined through the subgraph search, we apply an 
        # additional random noise to that value here. This will make the dataset slightly more 
        # difficult for a model to understand.
        graph_value_noisy = graph_value + random.uniform(-NOISE_MAGNITUDE, NOISE_MAGNITUDE)
            
        graph['graph_labels'] = np.array([graph_value_noisy])
        graph['graph_labels_raw'] = np.array([graph_value])
        graphs.append(graph)

        if c % LOG_STEP == 0:
            e.log(f' * {c}/{NUM_ELEMENTS_RAW} generated'
                  f' - num_motifs: {len(motifs)}'
                  f' - graph_value: {graph_value:.2f}')

        c += 1

    indices = list(range(len(graphs)))
    values = [graph['graph_labels'][0] for graph in graphs]
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
    hist, bins = np.histogram(values, bins=NUM_BINS)
    ax.hist(values, bins=NUM_BINS, color='lightgray')
    fig.savefig(os.path.join(e.path, 'distribution_raw.pdf'))

    # Now that we have generated a sufficient number of graphs we can sample a much smaller 
    # number of those to achieve the desired target value distribution
    #bin_indices = np.digitize(values, bins[:-2], right=True)
    bin_indices = np.searchsorted(bins[:-2], values, side='left')
    # print(len(bins), len(bins[:-1]), len(bins[:-2]))
    # print(max(bin_indices), min(bin_indices))
    weights = np.array([1/(hist[index-1]) if hist[index] >= 10 else 0 
                        for graph, index in zip(graphs, bin_indices)])
    weights = weights / np.sum(weights)
    indices_sampled = np.random.choice(indices, size=NUM_ELEMENTS, replace=True, p=weights)
    graphs_sampled = [graphs[i] for i in indices_sampled]
    
    values_sampled = [graph['graph_labels'][0] for graph in graphs_sampled]
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
    ax.hist(values_sampled, bins=2*NUM_BINS, color='lightgray')
    fig.savefig(os.path.join(e.path, 'distribution.pdf'))
    
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
        for index, graph in enumerate(graphs_sampled):
            
            target = graph['graph_labels']
            cogiles, graph = processing.extract(graph, mask=np.ones_like(graph['node_indices']))
            fig, node_positions = processing.visualize_as_figure(cogiles)
            
            graph['graph_labels'] = target
            graph['edge_importances_1'] = edge_importances_from_node_importances(
                edge_indices=graph['edge_indices'],
                node_importances=graph['node_importances_1'],
                calc_cb=lambda v1, v2: np.logical_and(v1, v2).astype(float)
            )
            graph['edge_importances_2'] = edge_importances_from_node_importances(
                edge_indices=graph['edge_indices'],
                node_importances=graph['node_importances_2'],
                calc_cb=lambda v1, v2: np.logical_and(v1, v2).astype(float)
            )
            graph['node_positions'] = node_positions
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
                plot_node_importances_border(
                    ax=ax,
                    g=graph,
                    node_positions=node_positions,
                    node_importances=graph['node_importances_1'][:, 0],
                )
                plot_edge_importances_border(
                    ax=ax,
                    g=graph,
                    node_positions=node_positions,
                    edge_importances=graph['edge_importances_1'][:, 0],
                )
                
                pdf.savefig(fig)
                plt.close(fig)

    

experiment.run_if_main()
