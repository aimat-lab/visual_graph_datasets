"""
This is the example "quickstart" code from the project "README"
"""
import os
import typing as t
import matplotlib.pyplot as plt

from visual_graph_datasets.config import Config
from visual_graph_datasets.web import ensure_dataset
from visual_graph_datasets.data import load_visual_graph_dataset
from visual_graph_datasets.visualization.base import draw_image
from visual_graph_datasets.visualization.importances import plot_node_importances_border
from visual_graph_datasets.visualization.importances import plot_edge_importances_border

# This object will load the settings from the main config file. This config file contains options
# such as changing the default datasets folder and defining custom alternative file share providers
config = Config()
config.load()

# First of all we need to make sure that the dataset exists locally, this function will download it from
# the default file share provider if it does not exist.
ensure_dataset('rb_dual_motifs', config)

# Afterwards we can be sure that the datasets exists and can now load it from the default datasets path.
# The data will be loaded as a dictionary whose int keys are the indices of the corresponding elements
# and the values are dictionaries which contain all the relevant data about the dataset element,
# (Dataset format is explained below)
dataset_path = os.path.join(config.get_datasets_path(), 'rb_dual_motifs')
data_index_map: t.Dict[int, dict] = {}
_, data_index_map = load_visual_graph_dataset(dataset_path)

# Using this information we can visualize the ground truth importance explanation annotations for one
# element of the dataset like this.
index = 0
data = data_index_map[index]
# This is the dictionary which represents the graph structure of the dataset element. Descriptive
# string keys and numpy array values.
g = data['metadata']['graph']
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
draw_image(ax, image_path=data['image_path'])
plot_node_importances_border(
    ax=ax,
    g=g,
    node_positions=g['image_node_positions'],
    node_importances=g['node_importances_2'][:, 0],
)
plot_edge_importances_border(
    ax=ax,
    g=g,
    node_positions=g['image_node_positions'],
    edge_importances=g['edge_importances_2'][:, 0],
)
fig_path = os.path.join(os.getcwd(), 'importances.pdf')
fig.savefig(fig_path)
