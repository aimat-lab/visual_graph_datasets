"""
This experiment is used to analyze visual graph datasets that consist of COLOR GRAPHS. Color graphs are 
one of the special types of graphs supported by this package. Color graphs consist of nodes which have 3 
numeric features that determine the RGB color code of that node. Color graphs edges do not have features.
"""
import os
import typing as t

import numpy as np
import matplotlib.pyplot as plt
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from networkx.algorithms import isomorphism

from visual_graph_datasets.config import Config
from visual_graph_datasets.web import ensure_dataset
from visual_graph_datasets.data import VisualGraphDatasetReader
from visual_graph_datasets.data import nx_from_graph
from visual_graph_datasets.processing.base import graph_count_motif
from visual_graph_datasets.processing.colors import ColorProcessing
from visual_graph_datasets.processing.colors import graph_from_cogiles

# == DATASET PARAMETERS ==
# These parameters determine the dataset that is to be loaded for the analysis. The 
# important part to consider here is that this dataset needs to be a VGD consisting of 
# COLOR GRAPHS.

# :param VISUAL_GRAPH_DATASET: 
#       This parameter identifies the dataset to be loaded for the analysis. This may optionally 
#       be the string absolute path to a valid visual graph dataset folder on the local system, 
#       which would then be loaded. Alternatively, this may be a valid string name for a dataset 
#       that can be downloaded from the main dataset file share provider.
VISUAL_GRAPH_DATASET: str = 'rb_dual_motifs'



# == EVALUATION PARAMETERS ==
# These parameters control the evaluation behavior

NUM_BINS: int = 20

# :param MOTIF_GROUPS:
#       This dictionary defines the analyses that should be performed in regards to motif frequencies.
#       On the top level this dictionary structure defines different motif groups. The keys are the unique 
#       string names of the motif group and the value is again a dictionary structure that defines several 
#       motifs to be anaylized as part of that same group.
MOTIF_GROUPS: t.Dict[str, dict] = {
    '1-node': {
        'R': {
            'graph': graph_from_cogiles('R'),
            'color': 'red',
        },
        'G': {
            'graph': graph_from_cogiles('G'),
            'color': 'green',
        },
        'B': {
            'graph': graph_from_cogiles('B'),
            'color': 'blue',
        },
        'H': {
            'graph': graph_from_cogiles('H'),
            'color': 'gray',
        }
    },
    '2-node': {
        'RY': {
            'graph': graph_from_cogiles('RY'),
            'color': 'red'
        },
        'GY': {
            'graph': graph_from_cogiles('GY'),
            'color': 'green',
        },
        'BY': {
            'graph': graph_from_cogiles('BY'),
            'color': 'blue',
        },
        'RG': {
            'graph': graph_from_cogiles('RG'),
            'color': 'yellow', 
        }
    }
}

__DEBUG__ = True

@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)
def experiment(e: Experiment):
    e.log('starting experiment...')
    
    # -- loading the dataset --
    
    e.log('loading the dataset...')
    config = Config()
    config.load()
    
    if os.path.exists(e.VISUAL_GRAPH_DATASET):
        dataset_path = e.VISUAL_GRAPH_DATASET
    else:
        dataset_path = ensure_dataset(e.VISUAL_GRAPH_DATASET)
    
    reader = VisualGraphDatasetReader(
        path=dataset_path,
        logger=e.logger,
        log_step=1000
    )
    index_data_map: dict = reader.read()
    e.log(f'loaded dataset with {len(index_data_map)} elements')
    
    processing = ColorProcessing()
    e.log(f'created processing instance - {processing.__class__.__name__}')
    
    # -- analyzing the dataset --
    
    # ~ node counts
    e.log('analyzing the graph size distribution...')
    node_counts = [len(data['metadata']['graph']['node_indices']) for data in index_data_map.values()]
    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
    ax.hist(
        node_counts,
        bins=e.NUM_BINS,
        color='gray',
        alpha=0.5,
        histtype='stepfilled',
    )
    fig.savefig(os.path.join(e.path, 'node_counts.pdf'))
    
    # ~ 1-node frequencies
    # In this section we are going to calculate the frequency of 1-node motifs (single nodes of different) 
    # colors. This is calculated for every graph and then the distribution of these values is plotted
    
    e.log('analyzing 1-node motif frequencies...')
    
    for group_name, motif_group in e.MOTIF_GROUPS.items():
        
        e.log(f'[#] motif group "{group_name}"')
        # In this data structure we are going to save the counts of the corresponding motifs for each of the motifs
        # that is specified in the current group. The keys of this dict will be the name of motifs and the values 
        # will be lists of integers where each element corresponds to one graph in the dataset and the integer value 
        # is the number of occurences of the corresponding motif.
        motif_counts: t.Dict[str, t.List[int]] = {}
        for motif_name, motif_info in motif_group.items():
            
            e.log(f' * motif "{motif_name}"')
            counts = []
            for graph in [data['metadata']['graph'] for data in index_data_map.values()]:
                
                count = graph_count_motif(
                    graph=graph, 
                    motif=motif_info['graph'],
                    processing=processing,
                )
                counts.append(count)
    
            motif_counts[motif_name] = counts
        
        motif_names = list(motif_counts.items())
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
        for motif_name, counts in motif_counts.items():
            info = motif_group[motif_name]
            
            ax.hist(
                counts,
                bins=e.NUM_BINS,
                color=info['color'],
                alpha=0.4,
                histtype='stepfilled',
                label=motif_name
            )
            
        ax.legend()
        fig.savefig(os.path.join(e.path, f'motif_group__{group_name}__histograms.pdf'))
        
        if len(motif_counts) > 1:
            # for each motif group we also want to plot the linear correlation between all the motifs that 
            # are involved in it.
            data = np.array([counts for counts in motif_counts.values()])
            corr = np.corrcoef(data)
            
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
            cax = ax.matshow(corr)
            fig.colorbar(cax)
            ticks = [i for i, _ in enumerate(motif_names)]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(motif_names)
            ax.set_yticklabels(motif_names)
            ax.set_title('Motif Frequency Correlation')
            
            fig.savefig(os.path.join(e.path, f'motif_group__{group_name}__correlation.pdf'))

    # :hook custom_analysis:
    #       This hook can be used to add custom, dataset-specific analysis steps to the experiment.
    #       It will be invoked as the end of the experiment, after all the other analyses have been 
    #       concluded. This hook receives the index_data_map of the dataset as the sole argument.
    e.apply_hook(
        'custom_analysis',
        index_data_map=index_data_map,
    )
    
experiment.run_if_main()