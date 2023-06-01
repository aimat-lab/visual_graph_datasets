"""
This example shows how to automatically visualize many elements from a visual graph dataset and their
corresponding explanations using the ``create_importances_pdf`` utility function. This PDF file will contain
one page per element and each page will showcase the various explanation channels for that corresponding
element of the dataset.
"""
import os
import random
import typing as t
import matplotlib.pyplot as plt
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

import visual_graph_datasets.typing as tv
from visual_graph_datasets.config import Config
from visual_graph_datasets.web import ensure_dataset
from visual_graph_datasets.data import load_visual_graph_dataset
from visual_graph_datasets.visualization.importances import create_importances_pdf

# The VGD configuration
CONFIG = Config()
CONFIG.load()
# The name of the dataset to be loaded and visualized
DATASET_NAME = 'rb_dual_motifs'
# The number of elements to be visualized in the PDF file.
NUM_EXAMPLES = 100


@Experiment(base_path=folder_path(__file__),
            namespace=file_namespace(__file__),
            glob=globals())
def experiment(e: Experiment):

    # First of all again we need to load the dataset again
    e.log('loading the dataset...')
    ensure_dataset('rb_dual_motifs', e.CONFIG)

    dataset_path = os.path.join(e.CONFIG.get_datasets_path(), 'rb_dual_motifs')
    metadata_map, index_data_map = load_visual_graph_dataset(
        dataset_path,
        # It is also possible to keep track of the loading process by passing a Logger instance, which
        # will print a log message at every "log_step"-th file that is being loaded from the disk.
        logger=e.logger,
        log_step=1000,
        metadata_contains_index=True,
    )

    e.log('choosing examples to visualize...')
    dataset_indices = list(index_data_map.keys())
    example_indices = random.sample(dataset_indices, k=e.NUM_EXAMPLES)

    e.log('visualizing the examples...')
    graph_list: t.List[tv.GraphDict] = [index_data_map[i]['metadata']['graph'] for i in example_indices]
    output_path = os.path.join(e.path, 'explanations.pdf')
    create_importances_pdf(
        # A list of all the GraphDicts to be visualized
        graph_list=graph_list,
        # The absolute string paths to the visualization PNG images for each of those graphs.
        image_path_list=[index_data_map[i]['image_path'] for i in example_indices],
        # A list of the corresponding "node_positions" arrays. These are arrays which contain the
        # coordinates of each node in the image as (x, y) pixel values. This information is crucial to be
        # able to draw the explanations to the correct locations in the image!
        node_positions_list=[g['node_positions'] for g in graph_list],
        # This is a dictionary which may contain as many explanations as desired. The key should be a
        # descriptive string for the explanation origin. Values are tuples of lists, where the first list
        # contains all the corresponding node importance explanations for the graphs and the second list
        # contains their edge importance explanations
        # The explanation channels are automatically inferred, but all explanations have to have the same
        # number of explanation channels!
        importances_map={
            'ground truth': (
                [g['node_importances_2'] for g in graph_list],
                [g['edge_importances_2'] for g in graph_list]
            )
        },
        # The PDF file will be saved to this location.
        output_path=output_path,
        # Since this process also takes some time, it is possible to track the progress through a logger
        # instance here as well
        logger=e.logger,
        log_step=10,
    )


experiment.run_if_main()
