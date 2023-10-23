"""
The unittests in this module test the functionality of the ``visual_graph_datasets.testing`` module, which 
ironically provides utilities for testing.
"""

from visual_graph_datasets.processing.base import ProcessingBase
from visual_graph_datasets.testing import MockProcessing
from visual_graph_datasets.typing import assert_graph_dict


def test_mock_processing_basically_works():
    """
    If the MockProcessing object can be properly constructed.
    """
    processing = MockProcessing()
    assert isinstance(processing, MockProcessing)
    assert isinstance(processing, ProcessingBase)
    
    # The domain representation of the mock processing is simply the string version of an integer number
    # which will determine the number of nodes within that graph. These graphs will have a deterministic 
    # topology but random node attributes.
    num_nodes = 10
    value = str(num_nodes)
    graph = processing.process(value)
    assert isinstance(graph, dict)
    assert_graph_dict(graph)
    assert len(graph['node_indices']) == num_nodes