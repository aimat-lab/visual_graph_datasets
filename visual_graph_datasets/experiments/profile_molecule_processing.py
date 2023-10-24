"""
This experiment is more of an extended functionality test.

**MOTIVATION**

There is another experiment called "generate_molecule_dataset_from_csv" which generates a new visual graph 
dataset folder based on a csv file of SMILES representations of molecules. For very large molecules, it has 
at some point become clear, that the runtime of the generation process for the molecular graphs from the 
SMILES representation severely degrades over time. So if the average creation speed at the beginning of this 
process is perhaps 0.1s then after about 100_000 elements this already becomes 1.0s. This indicates some sort 
of inherent problem / bug in the process.

This experiment was created to track this problem down. In this experiment we plot the runtime of the different 
components of the creation process to analyze where exactly that runtime problem originates from.
"""
import os
import time
import pathlib
import tempfile
import typing as t

import numpy as np
import matplotlib.pyplot as plt
from pycomex.functional.experiment import Experiment
from pycomex.util import file_namespace, folder_path

from visual_graph_datasets.visualization.base import create_frameless_figure
from visual_graph_datasets.processing.molecules import MoleculeProcessing
from visual_graph_datasets.data import VisualGraphDatasetWriter

# == PROFILING PARAMETERS ==
# These parameters actually are relevant for the profiling itself, such as the number of iterations
# to perform the profiling for.

# :param NUM_ITERATIONS:
#       The number of iterations that the corresponding functions will consequetively be executed such that 
#       there is the number of elements in that thing.
NUM_ITERATIONS: int = 100
# :param VALUE:
#       This is the SMILES string representation of the molecule that will be used (processed to graph / visualized) for 
#       the testing purposes.
VALUE: str = 'Cc1ccc(cc1N\\N=C2\\C(=O)C(=Cc3ccccc23)C(=O)Nc4cc(C)c(NC(=O)C5=Cc6ccccc6C(=N/Nc7c(C)cccc7C(=O)OCCCl)/C5=O)cc4C)C(=O)OCCCl'
# :param WIDTH:
#       The width of the graph images in pixels
WIDTH: int = 1000
# :param HEIGHT:
#       The height of the graph images in pixels
HEIGHT: int = 1000

# == PLOTTING PARAMETERS ==
# These are the parameters that are used for the plotting of the function 

# :param WINDOW_SIZE:
#       The size of the window for the sliding average computation during the plotting. We do a sliding average to plot the 
#       runtime over the iterations to be better able to see the trend rather than just the noise. The larger this value 
#       the more "smoothed" the plot will be.
WINDOW_SIZE: int = 50

# Function to calculate the moving average
def moving_average(data, window_size):
    cumsum = np.cumsum(data)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size

# == EXPERIMENT PARAMETERS ==
__DEBUG__ = True

@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)
def experiment(e: Experiment):
    e.log('starting the profiling of the "MoleculeProcessing" class...')
    
    e.log('creating processing object...')
    processing = MoleculeProcessing()
    
    e['iterations'] = list(range(e.NUM_ITERATIONS))
    
    # ~ Checking the "process" function
    e['duration/process'] = []
    e.log(f'starting the profiling for the "process" operation with {e.NUM_ITERATIONS} iterations...')
    for index in range(e.NUM_ITERATIONS):
        time_start = time.time()
        graph = processing.process(e.VALUE)
        duration = time.time() - time_start
        e['duration/process'].append(duration)
        if index % 100 == 0:
            e.log(f' * ({index:04d}/{e.NUM_ITERATIONS}) done - duration: {duration:.3f}s')
            
    # ~ Checking the "create_frameless_figure" function
    e['duration/figure'] = []
    e.log(f'starting the profiling for the "create_frameless_figure" operation with {e.NUM_ITERATIONS} iterations...')
    for index in range(e.NUM_ITERATIONS):
        time_start = time.time()
        fig, ax = create_frameless_figure(
            width=e.WIDTH,
            height=e.HEIGHT,
        )
        plt.close(fig)
        del fig, ax
        
        duration = time.time() - time_start
        e['duration/figure'].append(duration)
        if index % 100 == 0:
            e.log(f' * ({index:04d}/{e.NUM_ITERATIONS}) done - duration: {duration:.3f}s')
             
    # ~ Checking the "visualize" function
    # Actually in this case we will just check the visualize_as_figure function which means that 
    # we will not actually write the image to the disk. The advantage of doing it like this will be 
    # that this will compute a lot faster without the IO operations and we can still reason about 
    # that case based on the results we get here.
    e['duration/visualize'] = []
    e.log(f'starting the profiling for the "visualize_as_figure" operation with {e.NUM_ITERATIONS} iterations...')
    for index in range(e.NUM_ITERATIONS):
        time_start = time.time()
        fig, _ = processing.visualize_as_figure(
            value=e.VALUE,
            width=e.WIDTH,
            height=e.HEIGHT
        )
        
        plt.close('all')
        duration = time.time() - time_start
        e['duration/visualize'].append(duration)
        if index % 100 == 0:
            e.log(f' * ({index:04d}/{e.NUM_ITERATIONS}) done - duration: {duration:.3f}s')
    
    # ~ Checking the "process" function
    # The create function will do everyting: process the smiles into a molecular graph, create the visualization 
    # and save both of those things onto the disk.
    create_folder = os.path.join(e.path, 'create')
    os.mkdir(create_folder)
    e['duration/create'] = []
    e.log(f'starting the profiling for the "create" operation with {e.NUM_ITERATIONS} iterations...')
    for index in range(e.NUM_ITERATIONS):
        time_start = time.time()
        processing.create(
            value=e.VALUE,
            index=index,
            name=f'{index:04d}',
            output_path=create_folder,
        )
        duration = time.time() - time_start
        e[f'duration/create'].append(duration)
        if index % 100 == 0:
            e.log(f' * ({index:04d}/{e.NUM_ITERATIONS}) done - duration: {duration:.3f}s')
    
    # ~ Checking the "create" function usign a VgdWriter object
    # It could make a difference if we use just the
    writer_folder = os.path.join(e.path, 'writer')
    os.mkdir(writer_folder)
    writer = VisualGraphDatasetWriter(writer_folder)

    e['duration/writer'] = []
    e.log(f'starting the profiling for the "writer" operation with {e.NUM_ITERATIONS} iterations...')
    for index in range(e.NUM_ITERATIONS):
        time_start = time.time()
        processing.create(
            value=e.VALUE,
            index=index,
            name=f'{index:04d}',
            writer=writer,
        )
        duration = time.time() - time_start
        e[f'duration/writer'].append(duration)
        if index % 100 == 0:
            e.log(f' * ({index:04d}/{e.NUM_ITERATIONS}) done - duration: {duration:.3f}s')
                
        
@experiment.analysis    
def analysis(e: Experiment):
    
    # Now that we have collected all the durations we can plot them over the iterations.
    # We plot them over the iterations here because we are mainly interested in seeing whether 
    # there actually is an increase over time
    for key in ['create', 'writer', 'process', 'figure', 'visualize']:
        
        if key in e['duration'].keys():
            e.log(f'creating plots for key "{key}"...')
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
            ax.plot(
                e['iterations'], 
                e[f'duration/{key}'], 
                label='raw',
                alpha=0.5,
            )
            ax.plot(
                e['iterations'][e.WINDOW_SIZE - 1:], 
                moving_average(e[f'duration/{key}'], e.WINDOW_SIZE), 
                label='avg'
            )
            ax.set_xlabel('iteration')
            ax.set_ylabel('duration')
            
            values_avg = np.mean(e[f'duration/{key}'])
            values_std = np.std(e[f'duration/{key}'])
            ax.set_title(f'Avg: {values_avg:.3f} - Std: {values_std:.3f}')
            
            y_min = np.percentile(e[f'duration/{key}'], 10)
            y_max = np.percentile(e[f'duration/{key}'], 90)
            y_diff = abs(y_max - y_min)
            ax.set_ylim([y_min - 0.1 * y_diff, y_max + 0.1 * y_diff])
            
            ax.legend()
            fig.savefig(os.path.join(e.path, f'{key}_time_over_iterations.pdf'))
    
experiment.run_if_main()