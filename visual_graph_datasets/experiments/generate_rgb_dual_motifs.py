"""
This experiment is used to generate the rgb_dual_motifs dataset. In this dataset the goal is to generate 
mostly random color graphs which are seeded with 3 different families of subgraph motifs that in turn 
determine the contribution to the overall regression target value of each graph. The three families of 
motifs are based on the three primary colors red, green and blue. There are two kinds of motifs considered: 
cycles and stars. Additionally each motif family has motif variants that cause either a positive or a negative 
contribution (dual nature).
"""
import os
import time
import random
import pathlib
import typing as t

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from visual_graph_datasets.generation.graph import GraphGenerator
from visual_graph_datasets.generation.colors import *
from visual_graph_datasets.data import VisualGraphDatasetWriter
from visual_graph_datasets.data import VisualGraphDatasetReader
from visual_graph_datasets.processing.colors import ColorProcessing
from visual_graph_datasets.visualization.base import draw_image

mpl.use('Agg')


# == DATASET PARAMETERS ==
# These parameters determine the dataset creation process including the graph properties of the elements.

# :param NUM_ELEMENTS:
#       The number of elements to be created for the dataset
NUM_ELEMENTS: int = 1_000
# :param NUM_NODES_RANGE:
#       A tuple that defines the range for the possible number of nodes in each graph element. Inside this 
#       range the actual number will be sampled uniformly for each element.
NUM_NODES_RANGE: tuple = (40, 80)
# :param NUM_EDGES_RANGE:
#       A tuple that defines the range for the possible numbers of additional edges in each graph element.
#       Inside this range the actual number will be sampled uniformly for each element.
NUM_EDGES_RANGE: tuple = (5, 10)
# :param NUM_MOTIFS:
#       The number of distinct motifs to seed every graph with.
NUM_MOTIFS: int = 4
# :param COLORS:
#       This is a list of colors which will be sampled randomly for the random generation of the color graphs.
#       Each random node attributes vector in the generated graphs will be sampled uniformly from this list.
COLORS: t.List[list] = [
    # This makes gray nodes twice as probable, which in turn just makes all the tasks a little bit easier
    # since there are now relatively less actualy colorful nodes that could interfere with one of the tasks.
    GRAY, GRAY, GRAY,
    RED,
    GREEN,
    BLUE,
    MAGENTA,
    CYAN,
    YELLOW,
    PURPLE,
    ORANGE,
]
COLOR_POS: list = ORANGE
COLOR_NEG: list = PURPLE
# :param MOTIF_MAP:
#       This is a dictionary that determines the possible motifs with which the elements of the dataset can be 
#       seeded. The keys of this dict should be unique string names for the motifs and the values are again 
#       dictionaries which contain the relevant information about those graph motifs.
MOTIF_MAP: dict = {
    # the red motifs
    'red_pos_ring': {
        'suffix': 'r', 
        'name': 'red_pos_ring',
        'graph': make_ring_motif(COLOR_POS, RED, 3),
        'value': 1.3, 
    },
    'red_pos_star': {
        'suffix': 'r', 
        'name': 'red_pos_star',
        'graph': make_star_motif(COLOR_POS, RED, 3),
        'value': 3.6,
    },
    'red_neg_ring': {
        'suffix': 'r', 
        'name': 'red_neg_ring',
        'graph': make_ring_motif(COLOR_NEG, RED, 3),
        'value': -1.3,
    },
    'red_neg_star': {
        'suffix': 'r', 
        'name': 'red_neg_star',
        'graph': make_star_motif(COLOR_NEG, RED, 3),
        'value': -3.6,
    },
    # the blue motifs
    'blue_pos_ring': {
        'suffix': 'b', 
        'name': 'blue_pos_ring',
        'graph': make_ring_motif(COLOR_POS, BLUE, 3),
        'value': 0.7, 
    },
    'blue_pos_star': {
        'suffix': 'b', 
        'name': 'blue_pos_star',
        'graph': make_star_motif(COLOR_POS, BLUE, 3),
        'value': 2.8, 
    },
    'blue_neg_ring': {
        'suffix': 'b', 
        'name': 'blue_neg_ring',
        'graph': make_ring_motif(COLOR_NEG, BLUE, 3),
        'value': -0.7, 
    },
    'blue_neg_star': {
        'suffix': 'b',
        'name': 'blue_neg_star',
        'graph': make_star_motif(COLOR_NEG, BLUE, 3),
        'value': -2.8
    },
    # the green motifs
    'green_pos_ring': {
        'suffix': 'g', 
        'name': 'green_pos_ring',
        'graph': make_ring_motif(COLOR_POS, GREEN, 3),
        'value': 1.0, 
    },
    'green_pos_star': {
        'suffix': 'g', 
        'name': 'green_pos_star',
        'graph': make_star_motif(COLOR_POS, GREEN, 3),
        'value': 2.0, 
    },
    'green_neg_ring': {
        'suffix': 'g', 
        'name': 'green_neg_ring',
        'graph': make_ring_motif(COLOR_NEG, GREEN, 3),
        'value': -1.0, 
    },
    'green_neg_star': {
        'suffix': 'g',
        'name': 'green_neg_star',
        'graph': make_star_motif(COLOR_NEG, GREEN, 3),
        'value': -2.0,
    },
}
# :param IMAGE_WIDHT:
#       The pixel width of the images created in the visual graph dataset
IMAGE_WIDTH: int = 1000
# :param IMAGE_HEIGHT:
#       The pixel height of the images created in the visual graph dataset
IMAGE_HEIGHT: int = 1000

# == EVALUATION PARAMETERS ==
# These are the parameters that determine the evaluation process.

# :param LOG_STEP:
#       The number of iterations after which to print a new experiment log message
LOG_STEP: int = 1000
# :param NUM_EXAMPLES:
#       The number of examples to print into the PDF file
NUM_EXAMPLES: int = 100


__DEBUG__ = True
__TESTING__ = False

@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)
def experiment(e: Experiment):
    e.log('starting experiment...')
    
    @e.testing
    def test(e: Experiment):
        e.NUM_ELEMENTS = 50
        e.NUM_EXAMPLES = 10
        e.LOG_STEP = 10
    
    # ~ Generating the graphs
    
    e.log('starting the generation loop...')
    # In this list we are goint to collect the metadata dictionaries for each of the graph elements
    # that are generated for the dataset.
    metadatas: t.List[dict] = []
    c = 0
    while len(metadatas) < e.NUM_ELEMENTS:
        
        # First of all we need to randomly sample the actual number of nodes and additional 
        # edges to use for the generation of this specific graph
        num_nodes = random.randint(*e.NUM_NODES_RANGE)
        num_additional_edges = random.randint(*e.NUM_EDGES_RANGE)
        # We want to select three motifs to be included in each graph
        motifs = random.sample(list(e.MOTIF_MAP.values()), k=e.NUM_MOTIFS)
    
        generator = GraphGenerator(
            num_nodes=num_nodes,
            num_additional_edges=num_additional_edges,
            node_attributes_cb=lambda *args: random.choice(e.COLORS),
            edge_attributes_cb=lambda *args: [1],
            seed_graphs=[info['graph'] for info in motifs],
            is_directed=False,
            # We dont want additional edges to be inserted between two nodes of a motif because
            # that might change the motif itself.
            prevent_edges_in_seed_graphs=True,
        )
        
        # The problem is that the generation process fails in some rare cases, which is why we 
        # need to do this loop here.
        is_valid = False
        while not is_valid:
            try:
                generator.reset()
                graph = generator.generate()
                tc.assert_graph_dict(graph)
                is_valid = True
            except (AssertionError, IndexError, KeyError) as exc:
                # e.log(f' * error: {str(exc)}')
                pass
            
        # ~ calculating target values
        # In this dataset we dont just want a single target value for each graph but three separate sets 
        # of target values for each of the primary colors - because technically we will regard those as 
        # three distinct tasks.
        graph_targets = {'r': 0, 'g': 0, 'b': 0}
        for motif in motifs:
            suffix = motif['suffix']
            graph_targets[suffix] += motif['value']
            
        # In this dict we want to store the name of EVERY possible motif as a key and the corresponding value 
        # is the integer number of how many times that particular motif appears in the current graph.
        motif_counts = {name: 0 for name in e.MOTIF_MAP.keys()}
        for motif in motifs:
            motif_counts[motif['name']] += 1
        
        # ~ assembling the metadata
        # We want to represent each generated element of the dataset as a metadata dictionary which we can 
        # later on directly use to write the visual graph dataset. So here we need to assemble that metadata 
        # dict.
        # Most importantly, this metadata dict has to contain the generated graph representation itself in 
        # the "graph" property. But we also need to attach the target values for the three separate tasks.
        metadata = {
            **{f'targets_{key}': [value] for key, value in graph_targets.items()},
            'motifs': motif_counts,
            'graph': graph,
        }
        metadatas.append(metadata)
        
        if c % e.LOG_STEP == 0:
            e.log(f' * ({c}/{e.NUM_ELEMENTS}) elements created')
        
        # incrementing the counter
        c += 1
        
    num_elements = len(metadatas)
    e.log(f'finished the generation with {num_elements} elements')
    
    # ~ writing visual graph dataset 
    
    e.log('setting up the dataset writer instance...')
    dataset_path = os.path.join(e.path, 'dataset')
    os.mkdir(dataset_path)
    e['dataset_path'] = dataset_path
    writer = VisualGraphDatasetWriter(
        path=dataset_path,
        chunk_size=10_000,
    )
    
    e.log('setting up the processing instance...')
    processing = ColorProcessing()
    
    e.log(f'writing visual graph dataset...')
    time_start = time.time()
    for index, metadata in enumerate(metadatas):
        metadata['index'] = index
        fig, _ = processing.visualize_as_figure(
            '',
            width=e.IMAGE_WIDTH,
            height=e.IMAGE_HEIGHT,
            graph=metadata['graph'],
        )
        writer.write(
            name=index,
            metadata=metadata,
            figure=fig,
        )
        plt.close(fig)
        
        if index % e.LOG_STEP == 0:
            time_elapsed = time.time() - time_start
            time_per_element = time_elapsed / (index+1)
            time_remaining = time_per_element * (num_elements - index)
            e.log(f' * ({index}/{num_elements}) written'
                  f' - time elapsed: {time_elapsed/60:.2f} min'
                  f' - time remaining: {time_remaining/60:.2f} min')
        
    e.log(f'finished writing of the dataset @ {dataset_path}')
    

@experiment.analysis
def analysis(e: Experiment):
    e.log('starting analysis...')
    
    e.log('loading the dataset...')
    dataset_path = e['dataset_path']
    reader = VisualGraphDatasetReader(
        path=dataset_path,
        logger=e.logger,
        log_step=1000,
    )
    index_data_map = reader.read()
    e.log(f'loaded dataset with {len(index_data_map)} values')

    # ~ creating some visualizations of example elements from the dataset
    e.log(f'selecting {e.NUM_EXAMPLES} example elements...')
    indices = list(index_data_map.keys())
    example_indices = random.sample(indices, k=e.NUM_EXAMPLES)
    
    e.log('visualizing example elements...')
    pdf_path = os.path.join(e.path, 'examples.pdf')
    with PdfPages(pdf_path) as pdf:
        for index in example_indices:
            data = index_data_map[index]
            motifs = [key for key, value in data['metadata']['motifs'].items() if value > 0]
            
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
            draw_image(ax, data['image_path'])
            ax.set_title(f'index: {data["metadata"]["index"]}\n'
                         f'motifs: {", ".join(motifs)}')
            
            pdf.savefig(fig)
            plt.close(fig)
            
    # ~ plotting target distributions
    e.log('plotting the target value distributions...')
    for suffix in ['r', 'g', 'b']:
        key = f'targets_{suffix}'
        values = np.array([index_data_map[index]['metadata'][key] for index in indices])
        
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
        ax.hist(values, bins=30)
        ax.set_title(f'target value distribution - {key}')
        ax.set_xlabel('target value')
        ax.set_ylabel('number of elements')

        fig_path = os.path.join(e.path, f'dist__{key}.pdf')
        fig.savefig(fig_path)
        plt.close(fig)


experiment.run_if_main()