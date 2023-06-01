"""
This module contains functionality which is specifically useful for the generation of color-based datasets.
"""
import typing as t

import parsimonious
import numpy as np

import visual_graph_datasets.typing as tc


# Global variables for RGB codes
WHITE = [1., 1., 1.]
GRAY = [0.8, 0.8, 0.8]
BLACK = [0., 0., 0.]
RED = [1., 0., 0.]
YELLOW = [1., 1., 0.]
GREEN = [0., 1., 0.]
CYAN = [0., 1., 1.]
BLUE = [0., 0., 1.]
MAGENTA = [1., 0., 1.]


def make_star_motif(inner_color: tc.ColorList,
                    outer_color: tc.ColorList,
                    k: int = 3,
                    edge_attributes: t.List[float] = [1.0],
                    ) -> tc.GraphDict:
    """
    Creates a color motif of a star, which consists of a center node and ``k`` outer nodes, which are all
    exclusively connected the center node. The center node has the color ``inner_color`` and all the outer
    nodes share the same color ``outer_color``.
    """
    graph = {
        'node_indices': [0],
        'node_attributes': [
            inner_color,
        ],
        'edge_indices': [],
        'edge_attributes': []
    }

    for i in range(1, k + 1):
        graph['node_indices'].append(i)
        graph['node_attributes'].append(outer_color)

        # Adding the edges from the just added outer "star" node to the inner node
        graph['edge_indices'] += [[0, i], [i, 0]]
        graph['edge_attributes'] += [edge_attributes, edge_attributes]

    return graph


def make_ring_motif(start_color: tc.ColorList,
                    ring_color: tc.ColorList,
                    k: int = 3,
                    edge_attributes: t.List[int] = [1]
                    ) -> tc.GraphDict:
    graph = {
        'node_indices': [0],
        'node_attributes': [
            start_color,
        ],
        'edge_indices': [],
        'edge_attributes': []
    }

    prev_index = 0
    for i in range(1, k + 1):
        graph['node_indices'].append(i)
        graph['node_attributes'].append(ring_color)

        # Adding an edge from the previous element in the ring to the current one
        graph['edge_indices'] += [[prev_index, i], [i, prev_index]]
        graph['edge_attributes'] += [edge_attributes, edge_attributes]
        prev_index = i

    # At the end we need to add an additional edge from the end of the ring to the starting node
    graph['edge_indices'] += [[prev_index, 0], [0, prev_index]]
    graph['edge_attributes'] += [edge_attributes, edge_attributes]

    return graph


def make_grid_motif(color_1: tc.ColorList,
                    color_2: tc.ColorList,
                    n: int = 2,
                    m: int = 2,
                    edge_attributes: t.List[float] = [1.]):
    graph = {
        'node_indices': [],
        'node_attributes': [],
        'edge_indices': [],
        'edge_attributes': []
    }

    colors = [color_1, color_2]

    prev_row = None
    prev_index = None
    index = 0
    for j in range(m):

        color_index = int(j % 2)
        row = []
        for i in range(n):
            graph['node_indices'].append(index)
            graph['node_attributes'].append(colors[color_index])

            if prev_index is not None:
                graph['edge_indices'] += [[prev_index, index], [index, prev_index]]
                graph['edge_attributes'] += [edge_attributes, edge_attributes]

            if prev_row is not None:
                graph['edge_indices'] += [[prev_row[i], index], [index, prev_row[i]]]
                graph['edge_attributes'] += [edge_attributes, edge_attributes]

            row.append(index)
            color_index = int(not color_index)
            prev_index = index
            index += 1

        prev_row = row
        prev_index = None

    return graph


cogiles_grammar = parsimonious.Grammar(
    r"""
    graph           = (branch_node / anchor_node / node)*
    
    branch_node     = (anchor_node / node) branch+
    anchor_node     = node anchor+
    
    branch          = lpar (branch_node / anchor_node / node)* rpar
    lpar            = "("
    rpar            = ")"
    
    anchor          = ~r"-[\d]+"
    node            = ~r"[RGBYMCH]"
    """
)


def squeeze_list(lst):
    if isinstance(lst, list):
        if len(lst) == 1:
            return squeeze_list(lst[0])
        else:
            return [squeeze_list(value) for value in lst]
    return lst


NODE_TYPE_ATTRIBUTE_MAP = {
    'R': (1.0, 0.0, 0.0),
    'G': (0.0, 1.0, 0.0),
    'B': (0.0, 0.0, 1.0),
    'H': (0.8, 0.8, 0.8),
    'Y': (1.0, 1.0, 0.0),
    'M': (1.0, 0.0, 1.0),
    'C': (0.0, 1.0, 1.0),
}

NODE_ATTRIBUTE_TYPE_MAP = {
    value: key for key, value in NODE_TYPE_ATTRIBUTE_MAP.items()
}


class CogilesVisitor(parsimonious.NodeVisitor):

    def __init__(self, *args, **kwargs):
        super(CogilesVisitor, self).__init__(*args, **kwargs)
        self.node_index = 0
        self.index_node_map: t.Dict[int, t.Any] = {}

        self.anchor_map: t.Dict[int, int] = {}
        # In this list we will save all the "completed" connections between two nodes.
        self.anchor_edges: t.List[t.Tuple[int, int]] = []

        self.current_nesting = 0
        self.node_sequence_map = {}

        self.graph = {}

    def process(self) -> tc.GraphDict:
        node_indices = list(self.index_node_map.keys())
        node_attributes = [self.get_node_attributes(self.index_node_map[i]) for i in node_indices]

        edge_indices = []
        for i, j in self.node_sequence_map.items():
            edge_indices += [[i, j], [j, i]]

        for i, j in self.anchor_edges:
            edge_indices += [[i, j], [j, i]]

        edge_attributes = [1 for e in edge_indices]

        graph = {
            'node_indices':         np.array(node_indices),
            'node_attributes':      np.array(node_attributes),
            'edge_indices':         np.array(edge_indices),
            'edge_attributes':      np.array(edge_attributes),
        }
        return graph

    def get_node_attributes(self, node):
        node_type = str(node.text)
        return NODE_TYPE_ATTRIBUTE_MAP[node_type]

    def visit_graph(self, node, visited_children):
        return self.process()

    def visit_branch(self, node, visited_children):
        # The children of a branch will include the parentheses tokens which are used to delimit a branch
        branch = squeeze_list(visited_children[1:-1])
        return branch

    def visit_node(self, node, visited_children):
        # This method is called whenever we visit the most fundamental representation of a graph node
        # which is a letter that determines the node type / color. For every such node we visit we use these
        # two dictionaries here to establish a two-sided mapping from node index to the actual node.
        node_index = self.node_index
        self.index_node_map[node_index] = node

        if node_index != 0:
            self.node_sequence_map[node_index] = node_index - 1

        self.node_index += 1
        print(node)
        return node_index

    def visit_anchor(self, node, visited_children):
        return int(node.text)

    def visit_anchor_node(self, node, visited_children):
        node_index = visited_children[0]
        anchor_indices = visited_children[1:]
        anchor_indices = [v[0] for v in anchor_indices]

        for anchor_index in anchor_indices:
            if anchor_index not in self.anchor_map:
                self.anchor_map[anchor_index] = node_index
            else:
                self.anchor_edges.append((self.anchor_map[anchor_index], node_index))

        return node_index

    def visit_branch_node(self, node, visited_children):
        node_index = squeeze_list(visited_children[0])
        branches = visited_children[1:][0]
        print('branches', branches)
        for branch in branches:
            branch_start_index = branch[0]
            print('start', branch_start_index)
            self.node_sequence_map[branch_start_index] = node_index

        return node_index

    def generic_visit(self, node, visited_children):
        """ The generic visit method. """
        # return visited_children or node
        return visited_children or node


def graph_from_cogiles(value: str) -> tc.GraphDict:
    tree = cogiles_grammar.parse('RRGR-1(GGH-1)(GGGHHR(BBR-2)YYC-1)RR-2')
    visitor = CogilesVisitor()
    return visitor.visit(tree)

def node_type_from_attributes(attributes: np.array) -> str:
    attributes_tuple = tuple([round(value, 1) for value in attributes])
    return NODE_ATTRIBUTE_TYPE_MAP[attributes_tuple]


class CogilesEncoder:

    def __init__(self, graph: tc.GraphDict):
        self.graph = graph
        self.node_indices = self.graph['node_indices']
        self.node_attributes = self.graph['node_attributes']
        self.edge_indices = self.graph['edge_indices']
        self.edge_attributes = self.graph['edge_attributes']

        self.num_nodes = len(graph['node_indices'])
        self.num_edges = len(graph['edge_indices'])
        self.node_adjacency = np.zeros(shape=(self.num_nodes, self.num_nodes))
        for i, j in self.edge_indices:
            # We want to absolutely prevent self-loops for the encoding process because that will interfere
            # with the algorithm!
            if i != j:
                self.node_adjacency[i][j] = 1

        self.index = 0
        # This will be the final, encoded COGILES string
        self.value: str = ''
        self.current_anchor_index: int = 1
        self.index_anchor_map: t.Dict[int, int] = {}

    def encode(self):
        return self.encode_branch(0)

    def encode_branch(self, node_index: int):
        value: str = self.node_type_from_attributes(self.node_attributes[node_index])

        num_edges = sum(self.node_adjacency[node_index, :])
        if num_edges > 1:
            value += self.create_anchor(node_index)

        neighbors = sorted([i for i, v in enumerate(self.node_adjacency[node_index, :]) if v])
        for neighbor_index in [i for i in neighbors if i in self.index_anchor_map]:
            self.node_adjacency[node_index, neighbor_index] = 0
            self.node_adjacency[neighbor_index, node_index] = 0
            value += f'-{self.index_anchor_map[neighbor_index]}'

        for neighbor_index in [i for i in neighbors if i not in self.index_anchor_map]:
            self.node_adjacency[node_index, neighbor_index] = 0
            self.node_adjacency[neighbor_index, node_index] = 0

            if len(neighbors) == 1:
                value += self.encode_branch(neighbor_index)
            else:
                value += f'({self.encode_branch(neighbor_index)})'

        return value

    def node_type_from_attributes(self, attributes: np.array) -> str:
        attributes_tuple = tuple([round(value, 1) for value in attributes])
        return NODE_ATTRIBUTE_TYPE_MAP[attributes_tuple]

    def create_anchor(self, node_index: int) -> str:
        self.index_anchor_map[node_index] = self.current_anchor_index
        anchor_string = f'-{self.current_anchor_index}'
        self.current_anchor_index += 1

        return anchor_string


def graph_to_cogiles(graph: tc.GraphDict) -> str:
    encoder = CogilesEncoder(graph)
    return encoder.encode()
