"""
Updates an existing version of the "RbMotifs" dataset for the new visual graph dataset format
specifications.
"""
import os
import itertools
import json

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pycomex.functional.experiment import Experiment
from pycomex.util import folder_path, file_namespace

from visual_graph_datasets.util import get_dataset_path
from visual_graph_datasets.data import load_visual_graph_dataset
from visual_graph_datasets.data import NumericJsonEncoder
from visual_graph_datasets.data import VisualGraphDatasetReader, VisualGraphDatasetWriter
from visual_graph_datasets.visualization.base import create_frameless_figure
from visual_graph_datasets.visualization.base import layout_node_positions
from visual_graph_datasets.visualization.colors import visualize_color_graph
from visual_graph_datasets.processing.colors import ColorProcessing

# == DATASET PARAMETERS ==
VISUAL_GRAPH_DATASET_PATH = '/media/ssd/.visual_graph_datasets/datasets/rb_dual_motifs'

# == EVALUATION PARAMETERS ==
LOG_STEP = 100

# == EXPERIMENT PARAMETERS ==
__DEBUG__ = True


@Experiment(base_path=folder_path(__file__),
            namespace=file_namespace(__file__),
            glob=globals())
def experiment(e: Experiment):

    e.log('loading existing dataset...')
    reader = VisualGraphDatasetReader(
        VISUAL_GRAPH_DATASET_PATH,
        logger=e.logger,
        log_step=LOG_STEP
    )
    index_data_map = reader.read()
    dataset_length = len(index_data_map)
    e.log(f'loaded dataset with {dataset_length} elements')

    dataset_name = os.path.basename(VISUAL_GRAPH_DATASET_PATH)
    dataset_path = os.path.join(e.path, dataset_name)
    os.mkdir(dataset_path)

    processing = ColorProcessing()


experiment.run_if_main()