"""
This example shows the quickstart functionality of loading a visual graph dataset, 
accessing the individual elements, plotting the graph visualizations and 
visualizing graph importance masks.
"""
import os
import typing as t
import matplotlib.pyplot as plt

from visual_graph_datasets.config import Config
from visual_graph_datasets.web import ensure_dataset
from visual_graph_datasets.data import VisualGraphDatasetReader
from visual_graph_datasets.visualization.base import draw_image
from visual_graph_datasets.visualization.importances import plot_node_importances_border
from visual_graph_datasets.visualization.importances import plot_edge_importances_border

# This object will load the settings from the main config file. This config file contains options
# such as changing the default datasets folder and defining custom alternative file share providers
config = Config()
config.load()

# First of all we need to make sure that the dataset exists locally, this function will 
# download it from the default file share provider if it does not exist.
ensure_dataset('rb_dual_motifs', config)

# RBMOTIFS DATASET
# In this example we use the "rb_dual_motifs" dataset. This is a synthetically created graph regression dataset 
# which consists of randomly generated color graphs. In a "color graph" each node is associated with 3 float 
# node features that together represent an RGB color code. The graphs in this dataset are seeded with special 
# sub-graph motifs which determine the regression target value that is associated with each graph.
# Therefore, these motfis can be considered as the perfect ground truth explanations for those target values 
# which is why every graph is additionally annotated with the information about this "explanation" sub-graph 
# in the form of a binary node and edge mask that mark the locations of the nodes and edges that belong to such 
# a motif.

# Afterwards we can be sure that the datasets exists and can now load it from the default datasets path.
# The data will be loaded as a dictionary whose int keys are the indices of the corresponding elements
# and the values are dictionaries which contain all the relevant data about the dataset element,
dataset_path = os.path.join(config.get_datasets_path(), 'rb_dual_motifs')
reader = VisualGraphDatasetReader(path=dataset_path)
data_index_map: t.Dict[int, dict] = reader.read()

# Using this information we can visualize the ground truth explanation mask for one of the graphs 
# of this dataset. We can get the information about a specifc element by simply indexing the dataset 
# dict with a specific element index.
index = 0
data = data_index_map[index]
# This is the dictionary which represents the graph structure of the dataset element. Descriptive
# string keys and numpy array values.
g = data['metadata']['graph']

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
# "draw_image" can be used to draw the visualization of a given graph onto a plt.Axes
draw_image(ax, image_path=data['image_path'])
plot_node_importances_border(
    ax=ax,
    g=g,
    # This is a list of the 2D pixel coordinates of each of the graphs nodes within the 
    # visualization image. This is how the function knows where on the image to draw explanation
    # masks.
    node_positions=g['node_positions'],
    node_importances=g['node_importances_2'][:, 0],
)
plot_edge_importances_border(
    ax=ax,
    g=g,
    node_positions=g['node_positions'],
    edge_importances=g['edge_importances_2'][:, 0],
)
fig_path = os.path.join(os.getcwd(), 'importances.pdf')
fig.savefig(fig_path)
