"""

"""
import datetime
import json
import os.path
import shutil
import time

import numpy as np
import matplotlib.pyplot as plt
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

import visual_graph_datasets.typing as tv
from visual_graph_datasets.util import dynamic_import
from visual_graph_datasets.data import load_visual_graph_dataset
from visual_graph_datasets.data import NumericJsonEncoder
from visual_graph_datasets.processing.base import ProcessingBase
from visual_graph_datasets.processing.molecules import MoleculeProcessing
from visual_graph_datasets.processing.base import create_processing_module
from visual_graph_datasets.visualization.base import create_frameless_figure


VISUAL_GRAPH_DATASET_PATH: str = '/media/ssd/.visual_graph_datasets/datasets/aggregators_binary'
DOMAIN_REPR_KEY: str = 'smiles'
UPDATE_ELEMENTS: bool = True
REPLACE_PROCESSING: bool = True

IMAGE_WIDTH = 1000
IMAGE_HEIGHT = 1000

LOG_STEP = 100
__DEBUG__ = True


@Experiment(base_path=folder_path(__file__),
            namespace=file_namespace(__file__),
            glob=globals())
def experiment(e: Experiment):

    e.log('loading the dataset...')
    metadata_map, index_data_map = load_visual_graph_dataset(
        e.VISUAL_GRAPH_DATASET_PATH,
        logger=e.logger,
        log_step=10000,
    )
    dataset_length = len(index_data_map)

    @e.hook('get_processing')
    def get_processing(e, processing: dict):
        return processing

    # The try-except here is a bit for backwards compatibility. There might be the chance of working with
    # a legacy dataset that does not have a processing module yet. In that case the hook below
    # definitely has to be implemented to supply one.
    try:
        module = dynamic_import(metadata_map['process_path'])
        processing: ProcessingBase = module.processing
    except Exception as exc:
        print(exc)
        e.log('The target VGD does not have a processing module!')
        processing = None

    processing: ProcessingBase = e.apply_hook('get_processing', processing=processing)

    @e.hook('write_metadata')
    def write_metadata(e, metadata: dict, path: str):
        # NOTE FOR FUTURE: It's super important to NOT put the json dumps into the file context manager
        # because if there is an error with the JSON encoding this version will not actually modify the
        # file, but if it is inside the context manager it will delete the entire content of that file!
        content = json.dumps(metadata, cls=NumericJsonEncoder)
        with open(data['metadata_path'], mode='w') as file:
            file.write(content)

    e.log('starting to re draw the visualizations...')
    start_time = time.time()
    for index, data in index_data_map.items():

        value: tv.DomainRepr = data['metadata'][e.DOMAIN_REPR_KEY]

        if UPDATE_ELEMENTS:

            # ~ Updating the visualization
            fig, node_positions, *_ = processing.visualize_as_figure(
                value,
                width=e.IMAGE_WIDTH,
                height=e.IMAGE_HEIGHT,
            )
            #fig.savefig(data['image_path'])
            #plt.close(fig)

            # ~ Updating the metadata
            metadata = data['metadata']
            graph = metadata['graph']
            new_graph = processing.process(
                value
            )

            graph.update({
                'node_positions': node_positions,
                **new_graph,
            })

            metadata.update({
                'graph': graph,
                'repr': value
            })

            #e.apply_hook('write_metadata', path=data['metadata_path'], metadata=metadata)

        if index % e.LOG_STEP == 0:
            time_elapsed = time.time() - start_time
            time_per_element = time_elapsed / (index + 1)
            time_remaining = (dataset_length - index) * time_per_element
            eta = datetime.datetime.now() + datetime.timedelta(seconds=time_remaining)
            e.log(f' * processed ({index}/{dataset_length})'
                  f' - elapsed time: {time_elapsed:.2f}s'
                  f' - remaining time: {time_remaining:.2f}s'
                  f' - ETA: {eta:%A %d %H:%M}')

    if e.REPLACE_PROCESSING:
        # We absolutely want to save a copy of the current module though in case something goes wrong!
        backup_path = os.path.join(e.path, 'process.py.backup')
        shutil.copy(metadata_map['process_path'], backup_path)

        code = create_processing_module(processing)
        with open(metadata_map['process_path'], mode='w') as file:
            file.write(code)


experiment.run_if_main()
