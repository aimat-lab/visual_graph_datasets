"""Testing utilities and mock classes for visual graph datasets.

This module provides utilities and mock implementations to support unit testing
of the visual graph datasets package. It includes functions for generating random
graphs, managing test environments, and mock processing classes for testing purposes.

The module contains:

- Graph generation utilities for creating test data
- File management utilities for test cleanup
- Isolated configuration management for testing
- Mock processing implementations for unit tests

Example:

    .. code-block:: python

        from visual_graph_datasets.testing import generate_random_graph, IsolatedConfig

        # Generate a random graph for testing
        graph = generate_random_graph(num_nodes=5, num_node_features=2)

        # Use isolated config for testing
        with IsolatedConfig() as config:
            # Your test code here
            pass
"""

import os
import itertools
import tempfile
import typing as t

import matplotlib.pyplot as plt
import numpy as np
from visual_graph_datasets.data import VisualGraphDatasetWriter
import visual_graph_datasets.typing as tv
from visual_graph_datasets.util import TEMPLATE_ENV
from visual_graph_datasets.config import Config
from visual_graph_datasets.processing.base import ProcessingBase
from visual_graph_datasets.generation.graph import GraphGenerator


def generate_random_graph(
    num_nodes: int = 10,
    num_node_features: int = 3,
    num_edge_features: int = 1,
) -> tv.GraphDict:
    """
    Generate a random acyclic graph for testing purposes.

    Creates a randomly generated graph dictionary representation with the specified
    number of nodes and features. The graph is guaranteed to be acyclic (tree-like)
    and all node and edge features are randomly generated using numpy's random module.

    This function is primarily used for creating test data in unit tests where a
    predictable but varied graph structure is needed.

    Example:

        .. code-block:: python

            # Generate a small test graph
            graph = generate_random_graph(num_nodes=5, num_node_features=3)
            print(f"Generated graph with {len(graph['node_indices'])} nodes")

            # Generate a larger graph with more features
            large_graph = generate_random_graph(
                num_nodes=20,
                num_node_features=10,
                num_edge_features=5
            )

    :param num_nodes: The number of nodes the graph should contain
    :param num_node_features: The number of feature dimensions for each node
    :param num_edge_features: The number of feature dimensions for each edge

    :returns: A graph dictionary containing node_indices, node_attributes,
              edge_indices, and edge_attributes arrays
    """
    generator = GraphGenerator(
        num_nodes=num_nodes,
        num_additional_edges=0,
        node_attributes_cb=lambda *args, **kwargs: np.random.random(size=(num_node_features,)),
        edge_attributes_cb=lambda *args, **kwargs: np.random.random(size=(num_edge_features,)),
    )
    generator.reset()
    graph = generator.generate()
    return graph


def clear_files(file_paths: t.List[str]) -> None:
    """
    Remove multiple files from the filesystem safely.

    This utility function ensures that all files specified in the file_paths list
    are completely removed from the filesystem. It safely handles cases where files
    may not exist or may not be regular files. This is particularly useful for
    cleaning up test artifacts and temporary files created during unit testing.

    The function performs existence and file type checks before attempting removal
    to avoid errors when files don't exist or are directories.

    Example:

        .. code-block:: python

            # Clean up test files after a test
            test_files = [
                '/tmp/test_graph.json',
                '/tmp/test_visualization.png',
                '/tmp/test_metadata.yaml'
            ]
            clear_files(test_files)

    :param file_paths: List of absolute file paths to be removed from the filesystem
    """
    for file_path in file_paths:
        if os.path.exists(file_path) and os.path.isfile(file_path):
            os.remove(file_path)


class IsolatedConfig:
    """
    Context manager for creating isolated test configurations.

    This context manager creates a completely isolated testing environment by:
    1. Creating a temporary directory for all test-related files
    2. Generating a fresh configuration file from the default template
    3. Setting up a temporary datasets directory within the isolated environment
    4. Loading the configuration into the global Config singleton

    The isolation ensures that tests don't interfere with existing user configurations
    or datasets, and provides a clean slate for each test run. All temporary files
    and directories are automatically cleaned up when exiting the context.

    This is particularly useful for integration tests that need to test dataset
    downloading, configuration management, or file I/O operations without affecting
    the user's actual environment.

    Example:

        .. code-block:: python

            # Use in a test to ensure isolation
            def test_dataset_operations():
                with IsolatedConfig() as config:
                    # All operations here use the isolated config
                    assert config.datasets_path.startswith('/tmp')
                    # Download or create test datasets
                    # Perform operations that modify config
                # Cleanup happens automatically

    Attributes:
        dir: TemporaryDirectory object managing the test environment
        folder_path: Path to the temporary directory root
        datasets_folder: Path to the temporary datasets directory
        config_path: Path to the temporary configuration file
        config: Config instance used during the test
    """

    def __init__(self) -> None:
        """
        Initialize the isolated configuration context manager.

        Sets up the temporary directory and initializes all path attributes to None.
        The actual directory creation and configuration setup happens in __enter__.
        """
        self.dir = tempfile.TemporaryDirectory()
        self.folder_path: t.Optional[str] = None
        self.datasets_folder: t.Optional[str] = None
        self.config_path: t.Optional[str] = None
        self.config = Config()

    def __enter__(self) -> Config:
        """
        Enter the isolated configuration context.

        Creates the temporary directory structure, generates a configuration file
        from the template, and loads it into the Config singleton. This ensures
        that all subsequent operations within the context use the isolated
        configuration.

        The setup process:
        1. Initialize the temporary directory
        2. Create a datasets subdirectory for test datasets
        3. Generate a config.yaml file from the template with the temp datasets path
        4. Load the generated configuration into the Config singleton

        :returns: The Config instance configured for the isolated environment
        """
        # First of all we need to init the temporary directory
        self.folder_path = self.dir.__enter__()

        # Then we need to create the folder which will contain all the datasets that are potentially
        # downloaded:
        self.datasets_folder = os.path.join(self.folder_path, 'datasets')
        os.mkdir(self.datasets_folder)

        # Next we need to create the config file from the template. In this step it is important that we
        # pass in the custom dataset folder to be used in the config
        self.config_path = os.path.join(self.folder_path, 'config.yaml')
        template = TEMPLATE_ENV.get_template('config.yaml.j2')
        with open(self.config_path, mode='w') as file:
            content = template.render(datasets_path=self.datasets_folder)
            file.write(content)

        # And finally we load that config file into the config singleton
        self.config.load(self.config_path)

        return self.config

    def __exit__(self, *args, **kwargs) -> None:
        """
        Exit the isolated configuration context and clean up.

        Ensures that the temporary directory and all its contents are completely
        removed from the filesystem. This cleanup is guaranteed to happen even
        if exceptions occur within the context.

        :param args: Exception type, value, and traceback (if an exception occurred)
        :param kwargs: Additional keyword arguments from context manager protocol
        """
        # In the end we need to make sure to destroy the temporary directory again
        self.dir.__exit__(*args, **kwargs)


class MockProcessing(ProcessingBase):
    """
    Mock implementation of ProcessingBase for unit testing.

    This mock class provides a simplified processing implementation that can be used
    in unit tests without requiring complex domain-specific processing logic. It
    creates fully-connected graphs where the number of nodes is determined by
    parsing an integer from a string input.

    The mock implementation has these characteristics:
    - Domain representation: String containing an integer (e.g., "5" for 5 nodes)
    - Graph structure: Fully-connected (every node connected to every other node)
    - Node attributes: Constant feature vectors filled with 1.0 values
    - Edge attributes: Constant feature vectors filled with 1.0 values

    This design allows for predictable test scenarios while maintaining compatibility
    with the ProcessingBase interface. The deterministic nature of the generated
    graphs makes it ideal for testing graph processing pipelines, dataset creation,
    and visualization workflows.

    Example:

        .. code-block:: python

            # Create a mock processor
            processor = MockProcessing()

            # Process a domain value to create a 4-node fully-connected graph
            graph = processor.process("4", num_node_attributes=5, num_edge_attributes=2)

            # The resulting graph will have:
            # - 4 nodes with indices [0, 1, 2, 3]
            # - 12 edges (4 * 3 permutations)
            # - All node features = [1.0, 1.0, 1.0, 1.0, 1.0]
            # - All edge features = [1.0, 1.0]
    """

    def process(
        self,
        value: str,
        num_node_attributes: int = 10,
        num_edge_attributes: int = 10,
    ) -> tv.GraphDict:
        """
        Process a domain value into a fully-connected graph representation.

        Converts a string representation of an integer into a fully-connected graph
        where each node is connected to every other node. All node and edge attributes
        are set to constant values of 1.0 for predictable testing behavior.

        The graph structure is deterministic based on the input, making it suitable
        for unit tests that need consistent graph properties across test runs.

        Example:

            .. code-block:: python

                processor = MockProcessing()

                # Create a 3-node graph with custom attribute dimensions
                graph = processor.process("3", num_node_attributes=2, num_edge_attributes=1)

                # Results in:
                # - node_indices: [0, 1, 2]
                # - edge_indices: [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)]
                # - All attributes filled with 1.0

        :param value: String representation of the number of nodes (e.g., "5")
        :param num_node_attributes: Number of feature dimensions per node
        :param num_edge_attributes: Number of feature dimensions per edge

        :returns: Graph dictionary with node_indices, node_attributes,
                  edge_indices, and edge_attributes
        """
        num_nodes = int(value)

        node_indices = list(range(num_nodes))
        node_attributes = [[1.0] * num_node_attributes for _ in node_indices]

        # The graph will also be fully-connected, which means that all nodes are connected to all
        # other nodes with an edge.
        edge_indices = list(itertools.permutations(node_indices, 2))
        edge_attributes = [[1.0] * num_edge_attributes for _ in edge_indices]

        graph = {
            'node_indices': np.array(node_indices, dtype=int),
            'node_attributes': np.array(node_attributes, dtype=float),
            'edge_indices': np.array(edge_indices, dtype=int),
            'edge_attributes': np.array(edge_attributes, dtype=float),
        }
        return graph

    def visualize(self, value: tv.DomainRepr, width: int, height: int, **kwargs) -> np.ndarray:
        """
        Create a visualization of the domain representation.

        This method is required by the ProcessingBase interface but is not implemented
        in this mock class. In a real implementation, this would generate a visual
        representation of the domain value (e.g., molecular structure, color pattern).

        :param value: The domain representation to visualize
        :param width: Width of the output visualization in pixels
        :param height: Height of the output visualization in pixels
        :param kwargs: Additional visualization parameters

        :returns: Numpy array representing the visualization image

        :raises NotImplementedError: This mock implementation does not provide visualization
        """
        pass

    def create(
        self,
        value: tv.DomainRepr,
        index: str,
        width: int,
        height: int,
        output_path: str,
        additional_graph_data: dict,
        additional_metadata: dict,
        writer: t.Optional[VisualGraphDatasetWriter] = None,
        *args,
        **kwargs,
    ) -> tv.MetadataDict:
        """
        Create a complete dataset entry from a domain representation.

        This method is required by the ProcessingBase interface but is not implemented
        in this mock class. In a real implementation, this would process the domain
        value, create visualizations, and write the complete dataset entry to disk.

        :param value: The domain representation to process
        :param index: Unique identifier for this dataset entry
        :param width: Width for the visualization in pixels
        :param height: Height for the visualization in pixels
        :param output_path: Base path where dataset files should be written
        :param additional_graph_data: Extra graph data to include in the output
        :param additional_metadata: Extra metadata to include in the output
        :param writer: Optional dataset writer instance for output
        :param args: Additional positional arguments
        :param kwargs: Additional keyword arguments

        :returns: Metadata dictionary describing the created dataset entry

        :raises NotImplementedError: This mock implementation does not provide dataset creation
        """
        pass
