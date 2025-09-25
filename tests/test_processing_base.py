"""
Unittests for ``processing.base``
"""
import os
import sys
import json
import subprocess
import typing as t

import numpy as np
import matplotlib.pyplot as plt
from click.testing import CliRunner
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import visual_graph_datasets.typing as tc
from visual_graph_datasets.processing.base import ProcessingBase
from visual_graph_datasets.processing.base import create_processing_module
from visual_graph_datasets.processing.base import OneHotEncoder
from visual_graph_datasets.generation.graph import GraphGenerator
from visual_graph_datasets.visualization.base import create_frameless_figure
from visual_graph_datasets.visualization.base import layout_node_positions
from visual_graph_datasets.visualization.colors import visualize_grayscale_graph
from visual_graph_datasets.util import dynamic_import

from .util import ARTIFACTS_PATH


class MockProcessing(ProcessingBase):
    """
    This is a mock implementation of a subclass for the ProcessingBase interface. This is used to test
    the general functionality.
    """

    def process(self,
                size: int,
                graph_labels: list = [0]):
        """
        Given a node count SIZE returns a graph dict.

        :param size: The number of nodes for the graph to have
        :param graph_labels: Ground truth target labels to associate the graph with
        """
        generator = GraphGenerator(
            num_nodes=size,
            num_additional_edges=0
        )
        generator.reset()
        g = generator.generate()
        return g

    def visualize_graph(self,
                        g: dict,
                        width: int,
                        height: int):
        fig, ax = create_frameless_figure(width=width, height=height, ratio=1)
        node_positions = layout_node_positions(g)
        g['node_positions'] = node_positions
        visualize_grayscale_graph(ax, g, node_positions)

        return fig, ax

    def visualize(self,
                  size: int,
                  width: int = 200,
                  height: int = 200,
                  ) -> np.ndarray:
        """
        Given a node count SIZE returns image array for visualization.
        """
        g = self.process(size)

        fig, ax = self.visualize_graph(g, width, height)

        # https://stackoverflow.com/questions/35355930/matplotlib-figure-to-image-as-a-numpy-array
        canvas = FigureCanvas(fig)
        canvas.draw()
        array = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        array = array.reshape((height, width, 4))[:, :, :3]  # Remove alpha channel
        return array

    def create(self,
               size: int,
               width: int = 200,
               height: int = 200,
               name: str = 'created',
               output_path: str = os.getcwd(),
               ) -> str:
        """
        Given the SIZE, creates the corresponding randomly generated graph and creates two files to
        represent it: (1) a JSON file containing all metadata and the graph representation itself and
        (2) a PNG file containing the visualiztion for it.
        """
        g = self.process(size)
        fig, ax = self.visualize_graph(g, width, height)

        if os.path.exists(output_path) and os.path.isdir(output_path):
            fig_path = os.path.join(output_path, f'{name}.png')
            self.save_figure(fig, fig_path)

            metadata = {
                'name': name,
                'graph': g
            }
            metadata_path = os.path.join(output_path, f'{name}.json')
            self.save_metadata(metadata, metadata_path)

    def get_description_map(self) -> dict:
        return {
            'node_attributes': {
                0: 'The grayscale value assigned with that node',
            },
            'edge_attributes': {
                0: 'A constant weight of 1 for all edges'
            }
        }

    def __imports__(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from visual_graph_datasets.generation.graph import GraphGenerator
        from visual_graph_datasets.visualization.base import create_frameless_figure
        from visual_graph_datasets.visualization.base import layout_node_positions
        from visual_graph_datasets.visualization.colors import visualize_grayscale_graph


# == UNIT TESTS ==

def test_processing_base_parse_docstring_basically_works():
    """
    21.03.2023 - ``ProcessingBase.parse_docstring`` is supposed to take the docstring of a method as
    an input and then return two results: The first one is the string content of the main bulk of the
    docstring and the second one is a dictionary containing the parameter descriptions (:param:)
    """

    docstring = """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et 
    dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip 
    ex ea commodo consequat. 
    
    Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore 
    eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia 
    deserunt mollit anim id est laborum.
    
    :param foo: This is is a common variable name used throughout software engineering and computer science 
        what makes this harder is that it is two lines
    :param bar: This is another common naming convention
    """

    content, param_map = ProcessingBase.parse_docstring(docstring)
    assert isinstance(content, str)
    assert len(content) > 10

    assert isinstance(param_map, dict)
    assert 'foo' in param_map
    assert 'bar' in param_map
    assert ':param' not in param_map['foo']
    assert ':param' not in param_map['bar']


def test_processing_base_construction_basically_works():
    """
    20.03.2023 - ``ProcessingBase`` inherits from click.MultiCommand and an instance should thus be directly
    usable as a command line interface, which supports at the very least the --help option.
    """
    processing = ProcessingBase()
    assert isinstance(processing, ProcessingBase)

    # We also test if the object works correctly as a click CLI base by trying to invoke the help command
    runner = CliRunner()
    result = runner.invoke(processing, ['--help'])
    print('output', result.output)
    assert result.exit_code == 0
    assert 'help' in result.output


def test_mock_processing_construction_basically_works():
    """
    20.03.2023 - ``MockProcessing`` inherits from ProcessingBase and should thus also be able to be used
    directly as a CLI. Additionally it implements the abstract methods which should then be dynamically
    converted into commands upon construction and appear in the help text!
    """
    processing = MockProcessing()
    runner = CliRunner()

    # Here we try if the object actually fulfills it's function as a valid CLI base by running the --help
    # option.
    result = runner.invoke(processing, ['--help'])
    assert result.exit_code == 0
    assert 'help' in result.output
    # These two commands have to appear in the help text!
    assert 'visualize' in result.output
    assert 'process' in result.output
    assert 'create' in result.output


def test_mock_processing_process_command_works():
    """
    20.03.2023 - ``MockProcessing.process`` should generate a valid graph dict structure when given an
    integer node count for the graph.
    """
    processing = MockProcessing()
    runner = CliRunner()

    # First of all we are going to run the --help option for the process command to see if it was
    # correctly dynamically created from the process method
    result = runner.invoke(processing, ['process', '--help'])
    print(result.output)
    assert result.exit_code == 0
    assert 'help' in result.output
    assert 'SIZE' in result.output

    # Then we are going to actually invoke the command itself and the resulting output should be a json
    # string which we can convert back into a graph dict.
    result = runner.invoke(processing, ['process', '10'])
    assert result.exit_code == 0

    graph_dict = json.loads(result.output)
    assert isinstance(graph_dict, dict)


def test_mock_processing_visualize_command_works():
    """
    20.03.2023 - ``MockProcessing.visualize`` should create a numpy array which represents a visualization
    of the graph when given a node count. The array is an rgb image array and should thus have the shape
    (width, height, 3)
    """
    processing = MockProcessing()
    runner = CliRunner()

    # Help command first again
    result = runner.invoke(processing, ['visualize', '--help'])
    print(result.output)
    assert result.exit_code == 0
    assert 'help' in result.output
    assert 'SIZE' in result.output

    # Now we actually invoke it
    width, height = 500, 500
    result = runner.invoke(processing, ['visualize', f'--width={width}', f'--height={height}', '10'])
    assert result.exit_code == 0

    array = np.array(json.loads(result.output))
    assert isinstance(array, np.ndarray)
    assert len(array.shape) == 3
    assert array.shape == (width, height, 3)

    # Now that we have recovered the numpy array which represent the image we can use that to actually
    # save the image as a file which we can then validate after the test case is finished
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    ax.imshow(array)
    fig_path = os.path.join(ARTIFACTS_PATH, 'mock_processing_visualize.png')
    fig.savefig(fig_path)


def test_mock_processing_create_command():
    """
    20.03.2023 - ``MockProcessing.create`` should create two files given a node count: The png visualization
    of the graph and the json metadata file.
    """
    processing = MockProcessing()
    runner = CliRunner()

    # Help command first again
    result = runner.invoke(processing, ['create', '--help'])
    print(result.output)
    assert result.exit_code == 0
    assert 'help' in result.output
    assert 'SIZE' in result.output

    # Now actually invoking it
    result = runner.invoke(processing, ['create', f'--output_path={ARTIFACTS_PATH}', '10'])
    assert result.exit_code == 0


def test_mock_processing_create_processing_module():
    """
    20.03.2023 - ``create_processing_module`` should take an instance implementing ProcessingBase and
    automatically generate code for a standalone python module which essentially implements the same
    pre-processing functionality as that class. That module should also act as a command line application
    for that functionality.
    """
    processing = MockProcessing()

    code_string = create_processing_module(processing)
    assert isinstance(code_string, str)
    assert code_string != ''

    path = os.path.join(ARTIFACTS_PATH, 'process.py')
    with open(path, mode='w') as file:
        file.write(code_string)

    assert os.path.exists(path)

    # Now that we are sure that the file was properly created we can try to invoke it over the command line
    # which should be working
    help_command = f'{sys.executable} {path} --help'
    proc = subprocess.run(
        help_command,
        shell=True,
        stdout=subprocess.PIPE,
        cwd=ARTIFACTS_PATH
    )
    output = proc.stdout.decode()
    assert 'help' in output
    # all the default commands also have to be listed in there!
    assert 'process' in output
    assert 'visualize' in output
    assert 'create' in output

    # Now as the last thing we can try to dynamically import that module here and then try to use the
    # processing functions directly without having to rely on the string conversion that necessarily
    # happens with CLIs
    module = dynamic_import(path)
    g = module.processing.process(10)
    assert isinstance(g, dict)
    assert len(g['node_indices']) == 10


def test_mock_processing_get_description_map_basically_works():
    """
    21.03.2022 - the ``get_description_map`` function should return a dictionary which contains several
    sub dictionaries, each of which mapping the individual indices of node, edge and graph attributes to
    a natural language description.
    """
    processing = MockProcessing()

    description_map: dict = processing.get_description_map()
    assert isinstance(description_map, dict)
    assert 'node_attributes' in description_map
    assert 'edge_attributes' in description_map


class TestOneHotEncoder:
    
    def test_encode_basically_works(self):
        """
        The "encode" method should be able to encode elements from the "values" list into lists 
        (vectors) of one-hot encoded float values.
        """
        encoder = OneHotEncoder(
            values=['H', 'C', 'O', 'N'],
            dtype=str,
        )
        
        assert isinstance(encoder, OneHotEncoder)
        
        # It should now be possible to encode elements from the "values" list into one-hot 
        # locations in the output vector
        encoded_1 = encoder.encode('H')
        assert isinstance(encoded_1, list)
        assert len(encoded_1) == 4
        assert encoded_1 == [1., 0., 0., 0.]
        
        # we can test it for a second element from the input set as well to see
        # if the encoding is working correctly
        encoded_2 = encoder.encode('C')
        assert encoded_2 == [0., 1., 0., 0.]
        
    def test_decode_basically_works(self):
        """
        Using the "decode" vector it should be possible to obtain the corresponding value from the
        "values" list when providing a one-hot encoded list.
        """
        encoder = OneHotEncoder(
            values=['H', 'C', 'O', 'N'],
            dtype=str,
        )
        
        decoded_1 = encoder.decode([1., 0., 0., 0.])
        assert isinstance(decoded_1, str)
        assert decoded_1 == 'H'
        
        decoded_2 = encoder.decode([0., 0., 0., 1.])
        assert decoded_2 == 'N'
        
    def test_unkown_value_basically_works(self):
        """
        It should be possible to use the "add_unknown" flag to also cover the umbrella case of any 
        value that is not explicitly in the "values" list. Also providing a fall back "unknown" value 
        will return that value in the case of decoding such an unknown one-hot encoded vector.
        """
        encoder = OneHotEncoder(
            values=['H', 'C', 'O', 'N'],
            dtype=str,
            add_unknown=True,
            unknown='UNK'
        )
        
        # it knows H so that should be encoded correctly. Althouth the one-hot encoding 
        # should now have a length of 5 since there is an additional entry that represents 
        # the unknown case
        encoded_1 = encoder.encode('H')
        assert isinstance(encoded_1, list)
        assert len(encoded_1) == 5
        assert encoded_1 == [1., 0., 0., 0., 0.]
        
        # it does not know 'S' so that should be encoded as the unknown value as well 
        # as 'F' which is not in the list of known values either.
        encoded_2 = encoder.encode('S')
        assert encoded_2 == [0., 0., 0., 0., 1.]
        
        encoded_3 = encoder.encode('F')
        assert encoded_3 == [0., 0., 0., 0., 1.]
        
        # Now when decoding the unknown value it should return the 'unknown' value that was
        # provided as a fallback
        decoded_1 = encoder.decode([0., 0., 0., 0., 1.])
        assert decoded_1 == 'UNK'
        
    def test_encode_decode_round_trip_works(self):
        """
        It should be possible to encode and decode a value and get back the original input value.
        """
        values = ['H', 'C', 'O', 'N']
        encoder = OneHotEncoder(
            values=values,
            dtype=str,
            add_unknown=True,
            unknown='UNK'
        )
        
        # we can now test if the encoding and decoding works correctly for all values
        # in the "values" list. The result should be the same as the input value.
        for value in values:
            encoded = encoder.encode(value)
            decoded = encoder.decode(encoded)
            
            assert decoded == value
        
        
    def test_encode_string_basically_works(self):
        """
        Using the "encode_string" method it should be possible to simply encode a domain value from 
        the "values" list into a simple human-readable string representation.
        """
        encoder = OneHotEncoder(
            values=['H', 'C', 'O', 'N'],
            dtype=str,
            add_unknown=True,
            unknown='UNK'
        )
        
        # In the case of a str "values" list this is trivial since those are already string values.
        string = encoder.encode_string('H')
        assert isinstance(string, str)
        assert string == 'H'
        
        # However it should also be possible to use non-string values and still get a human readable 
        # string encoding when additionally passing the "string_values" list.
        encoder = OneHotEncoder(
            values=[0, 1, 2, 3],
            string_values=['zero', 'one', 'two', 'three'],
            dtype=int,
            add_unknown=True,
            unknown='unknown'
        )
        
        # Encoder should work normally
        encoded = encoder.encode(2)
        assert encoded == [0., 0., 1., 0., 0.]
        
        string = encoder.encode_string(2)
        assert string == 'two'
        
        