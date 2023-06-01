import visual_graph_datasets.typing as tc
from visual_graph_datasets.generation.graph import GraphGenerator


def test_graph_generator_basically_works():
    generator = GraphGenerator(
        num_nodes=10,
        num_additional_edges=2,
    )

    generator.reset()
    graph: tc.GraphDict = generator.generate()
    # This function contains all the assertions to make sure that this is indeed a valid GraphDict
    tc.assert_graph_dict(graph)
