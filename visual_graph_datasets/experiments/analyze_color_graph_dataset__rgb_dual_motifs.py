import os
import typing as t

import numpy as np
import matplotlib.pyplot as plt
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path


# == DATASET PARAMETERS ==
# These parameters determine the dataset that is to be loaded for the analysis. The 
# important part to consider here is that this dataset needs to be a VGD consisting of 
# COLOR GRAPHS.

# :param VISUAL_GRAPH_DATASET: 
#       This parameter identifies the dataset to be loaded for the analysis. This may optionally 
#       be the string absolute path to a valid visual graph dataset folder on the local system, 
#       which would then be loaded. Alternatively, this may be a valid string name for a dataset 
#       that can be downloaded from the main dataset file share provider.
VISUAL_GRAPH_DATASET: str = 'rgb_dual_motifs_10k'

experiment = Experiment.extend(
    'analyze_color_graph_dataset.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

@experiment.hook('custom_analysis')
def custom_analysis(e: Experiment,
                    index_data_map: dict
                    ) -> None:
    e.log('starting custom analysis for dataset: rgb_dual_motifs')

    # ~ the 3 target values
    # This specific dataset has 3 independent target values for each of the graphs (each graph can have 
    # a series of different motifs which is associated with one of the 3 tasks)
    
    targets_r = [data['metadata']['targets_r'][0] for data in index_data_map.values()]
    targets_g = [data['metadata']['targets_g'][0] for data in index_data_map.values()]
    targets_b = [data['metadata']['targets_b'][0] for data in index_data_map.values()]
    
    fig, (ax_r, ax_g, ax_b) = plt.subplots(ncols=3, nrows=1, figsize=(30, 10))
    ax_r.hist(
        targets_r,
        bins=e.NUM_BINS,
        color='red',
        alpha=0.5,
        histtype='stepfilled'
    )
    ax_g.hist(
        targets_g,
        bins=e.NUM_BINS,
        color='green',
        alpha=0.5,
        histtype='stepfilled'
    )
    ax_b.hist(
        targets_b,
        bins=e.NUM_BINS,
        color='blue',
        alpha=0.5,
        histtype='stepfilled'
    )
    
    fig.savefig(os.path.join(e.path, 'task_targets.pdf'))
    
    # ~ correlation between the different targets
    # Then we can also check if there is a correlation between the different tasks that make up 
    # this dataset.
    
    data = np.array([targets_r, targets_g, targets_b])
    corr = np.corrcoef(data)
    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
    cax = ax.matshow(corr)
    fig.colorbar(cax)
    ticks = [0, 1, 2]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    tick_labels = ['red', 'green', 'blue']
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    ax.set_title('Task Correlation Matrix')

    fig.savefig(os.path.join(e.path, 'task_correlation.pdf'))
    

experiment.run_if_main()