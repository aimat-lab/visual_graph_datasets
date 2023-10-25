"""
The purpose of this experiment is to find certain subgraph motifs within the elements of a given visual 
graph dataset.

For this purpose the user can define a dictionary of graph motifs and a dataset. Additionally an appropriate 
ProcessingBase subclass instance for the given type of graph has to be provided. The experiment will iterate 
over all the elements of the dataset and perform a subgraph isomorphism check for each one of the defined 
motifs. The information about which motifs is contained in any given graph is then permanently added to the 
metadata file of each element under the "motifs" dictionary.
"""
import os
import pathlib 
import typing as t

import numpy as np
import networkx as nx
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from visual_graph_datasets.data import VisualGraphDatasetReader
from visual_graph_datasets.processing.base import ProcessingBase
from visual_graph_datasets.processing.colors import ColorProcessing
from visual_graph_datasets.graph import nx_from_graph

PATH = pathlib.Path(__file__).parent.absolute()

# == DATASET PARAMETERS ==
VISUAL_GRAPH_DATASET_PATH: str = '/media/ssd/.visual_graph_datasets/datasets/rb_dual_motifs'
MODIFY_DATASET: bool = True
SUBSET: t.Optional[int] = None

# == MOTIF PARAMETERS ==
MOTIFS = {
    'red_triangle': {
        'value': 'R-1RG-1'
    },
    'red_star': {
        'value': 'Y(R)(R)(R)'
    },
    'blue_triangle': {
        'value': 'B-1BG-1',
    },
    'blue_star': {
        'value': 'Y(B)(B)(B)'
    }
}

__DEBUG__ = True
__TESTING__ = False

@Experiment(namespace=file_namespace(__file__),
            base_path=folder_path(__file__),
            glob=globals())
def experiment(e: Experiment):
    
    if __TESTING__:
        e.SUBSET = 200

    @e.hook('get_processing', default=True)
    def get_processing(e):
        
        processing = ColorProcessing()
        return processing

    e.log('creating the processing instance...')
    # :hook get_processing:
    #       This hook is supposed to return an instance of a ProcessingBase subclass which can be used to decode 
    #       the domain specific representations used in this experiment into graph representations. Additionally 
    #       this processing instance has to implement appropriate "node_match" and "edge_match" methods for the 
    #       the given type of graph.
    processing: ProcessingBase = e.apply_hook(
        'get_processing',
    )

    # Now at first we have to do some pre-processing for the given motifs. On the one hand we would like to 
    # visualize these motifs and save the visualizations as artifacts such that the user can look at them 
    # later on and visually verify that they are indeed the correct motifs.
    # On the other hand it may be the case that the motifs are only provided in the format of the domain specific 
    # representation and not in fact as the graph representation, in which case we need to use the processing 
    # instance to create the graph representation.
    e.log('pre-process the motifs...')
    for motif_name, motif_data in MOTIFS.items():
        
        # visualization
        motif_path = os.path.join(e.path, motif_name)
        os.mkdir(motif_path)
        processing.create(
            value=motif_data['value'],
            index=0,
            width=1000,
            height=1000,
            output_path=motif_path,
        )

        # creating graph
        if 'graph' not in motif_data:
            motif_data['graph'] = processing.process(motif_data['value'])
            
        e.log(f' * motif "{motif_name}" done')

    # ~ loading the dataset
    dataset_name = os.path.basename(VISUAL_GRAPH_DATASET_PATH)
    e.log(f'loading the dataset "{dataset_name}"...')
    reader = VisualGraphDatasetReader(
        path=VISUAL_GRAPH_DATASET_PATH,
        logger=e.logger,
        log_step=1000,
    )
    # index_data_map = reader.read(subset=SUBSET)
    index_data_map = reader.read(subset=SUBSET)
    dataset_length = len(index_data_map)
    e.log(f'loaded dataset with {dataset_length} elements')

    # Now we have to iterate through every element in this dataset and for each graph we have to check 
    # whether it contains the given subgraph or not.
    for c, (index, data) in enumerate(index_data_map.items()):
        metadata = data['metadata']
        graph = data['metadata']['graph']
        
        matches = {}
        for motif_name, motif_data in MOTIFS.items():
            nx_base = nx_from_graph(graph)
            nx_motif = nx_from_graph(motif_data['graph'])
            
            graph_matcher = nx.isomorphism.GraphMatcher(
                nx_base, 
                nx_motif,
                node_match=lambda a, b: processing.node_match(a['node_attributes'], b['node_attributes']),
                edge_match=lambda a, b: processing.edge_match(a['edge_attributes'], b['edge_attributes']),
            )
            match: bool = graph_matcher.subgraph_is_isomorphic()
            matches[motif_name] = int(match)
        
        # If we are not currently in testing mode we will actually modify the given visual graph dataset 
        # with this new information about the containing motifs.
        if not __TESTING__:
            metadata['motifs'] = matches
            processing.save_metadata(metadata, data['metadata_path'])
            
        if c % 100 == 0:
            e.log(f' * ({c}/{dataset_length}) - {matches}')
            

experiment.run_if_main()