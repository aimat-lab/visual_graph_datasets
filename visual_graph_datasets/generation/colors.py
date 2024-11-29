"""
This module contains functionality which is specifically useful for the generation of color-based datasets.
"""
import typing as t

import parsimonious
import numpy as np

import visual_graph_datasets.typing as tc


# Global variables for RGB codes
WHITE = (1., 1., 1.)
GRAY = (0.8, 0.8, 0.8)
LIGHT_GRAY = (0.3, 0.3, 0.3)
BLACK = (0., 0., 0)
RED = (1., 0., 0.)
YELLOW = (1., 1., 0.)
GREEN = (0., 1., 0.)
CYAN = (0., 1., 1.)
BLUE = (0., 0., 1.)
MAGENTA = (1., 0., 1.)
INDIGO = (0.5, 0.5, 1.0)
ORANGE = (1.0, 0.5, 0.0)
TEAL = (0.0, 0.5, 0.5)
PURPLE = (0.5, 0.0, 0.5)

DEFAULT_COLORS = [
    GRAY,
    RED,
    GREEN,
    BLUE,
    YELLOW,
    CYAN,
    MAGENTA,
]


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


"""
The COGILES format
==================

The following section implements the COGILES (Colored Graph Input Line Entry System) string format, which is 
a method to define colored graphs using a simple human-readable string syntax, strongly inspired by the 
SMILES format for molecular graphs; although, the COGILES format differs in certain aspects.

Node types
----------

The colored graphs that are included in the basic COGILES language can only consist of a limited number of 
distinct node types, each of which is represented by a upper case letter:

* R - red - rgb color code (0, 0, 1)
* G - green - rgb color code (0, 1, 0)
* B - blue - rgb color code (0, 0, 1)
* C - cyan - rgd color code (0, 1, 1)
* M - magenta - rgb color code (1, 0, 1)
* Y - yellow - rgb color code (1, 1, 0)
* H - gray - rgb color code (0.8, 0.8, 0.8)

By default nodes will be connected to their immediate predecessor in the COGILES string. This is the most 
basic method of creating structure from the nodes of the aforementioned types.

Therefore the example "RRRGGG" would be a graph consisting of 3 red and 3 green nodes where they are simply 
connected as a single *line*

branches
--------

On top of the connection of immediately neighboring nodes, it is also possible to create branching graphs 
by using brackets "(" ")" to enclose certain parts of the graph. Inside of a bracket, all of the other 
rules still apply, so by default all the nodes within the bracket will be connected into a single line, 
but that line itself is only connected to the greater graph through the first node that comes immediately 
after the opening bracket. That first node is only connected to the node which comes immediately *before* 
the opening bracket.

Consider the example "RRR(BB)RRRR". This is again a graph with the 

anchor connections
------------------

With the two previous rules it would not be possible to define arbitrarily connected graphs. It would only 
be possible to define *trees* as they do not yet include any method of creating *cycles* of any kind. This 
is where *anchors* come in.

Anchors are defined by a dash character followed by an integer *anchor index* - for example "-1".
Such an anchor can be placed *after* a node character to connect it to that node. All anchors with the same 
number will be connected with an edge. That means that placing only a single anchor of the same number 
will have not effect - only as soon as there are at least two will they affect the graph connections. 
All the nodes that have an anchor of the same number attached will be connected with an *additional edge*, 
regardless of other branches or default connections already applying to them as well.

Consider the example "R-1RRRRR-1". Since the first and the last node are attached to an anchor of the same 
number, they will be connected and the overall graph will be a cycle consisting of 6 red nodes.

It is also possible to use an anchor number more than twice. In that case, all the anchored nodes will be 
connected to the node where the anchor *appeared first*

It is also possible that a single node has multiple anchors like the example "R-1-2GGG-2BBB-1" where the 
first node has two anchors attached to it and consequently in the resulting tree that node will have three 
edges connected to it. 

breaking - creating disconnected graphs
---------------------------------------

It is also possible to specify multiple disconnected graphs with the COGILES grammar by using the colon "." 
character. Inserting this character between two nodes will break the default connection which would usually 
result from the two being directly placed next to each other.

The cogiles "R-1RR-1.RR(GG)R", for example, defines two separate graphs.
"""


# COGILES works by implementing a new "grammar" which is then parsed with parsimonious library and used to
# construct the tree representation.
# The grammar defines the syntactical rules of how the various elements have to look like and are connected
# with each other with regex like syntax here.
cogiles_grammar = parsimonious.Grammar(
    r"""
    graph           = (branch / node / anchor / break)*

    branch_node     = (break_node / node) branch+
    anchor_node     = (branch_node / break_node / node) anchor+
    break_node      = break node

    branch          = lpar graph rpar
    lpar            = "("
    rpar            = ")"

    break           = "."
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


# The keys of this dict are all the allowed string characters representing the node types and the values
# are the corresponding feature vectors which those nodes should then have in the resulting GraphDict
# representation. These feature vectors are simply the RGB color codes of the node type colors.
NODE_TYPE_ATTRIBUTE_MAP = {
    'R': (1.0, 0.0, 0.0),
    'G': (0.0, 1.0, 0.0),
    'B': (0.0, 0.0, 1.0),
    'H': (0.8, 0.8, 0.8),
    'Y': (1.0, 1.0, 0.0),
    'M': (1.0, 0.0, 1.0),
    'C': (0.0, 1.0, 1.0),
}
# This mapping is exactly the inverse. Given the feature vectors, this method will map back to the
# corresponding string node types.
NODE_ATTRIBUTE_TYPE_MAP = {
    value: key for key, value in NODE_TYPE_ATTRIBUTE_MAP.items()
}


# Parsimonious
class CogilesVisitor(parsimonious.NodeVisitor):
    """
    This class is a visitor for the parsimonious token tree that results from a parsing of the 
    COGILES string grammar.
    
    The COGILES grammar defines a system of encoding "color graphs" as a simple and human-readable string 
    format. This grammer is being parsed into a token tree by the parsimonious language and ultimately 
    a visitor instance of this class is then used to process that token tree into the actual graph dict 
    representation of the graph that is represented by the original COGILES string.
    
    To convert a COGILES string, first use the ``visit`` method on the token tree and then use the 
    ``process`` method which will return the resulting graph dict.
    
    ..code-block:
    
        tree = cogiles_grammar.parse(value)
        visitor = CogilesVisitor()
        visitor.visit(tree)
        graph = visitor.process()

    """
    def __init__(self, *args, **kwargs):
        super(CogilesVisitor, self).__init__(*args, **kwargs)
        self.node_index = 0
        self.index_node_map: t.Dict[int, t.Any] = {}

        self.anchor_map: t.Dict[int, int] = {}
        # In this list we will save all the "completed" connections between two nodes.
        self.anchor_edges: t.List[t.Tuple[int, int]] = []

        self.current_nesting = 0
        self.node_sequence_map = {}

        self.prev_node = 0
        self.is_break = False

        self.graph = {}

    def process(self) -> tc.GraphDict:
        """
        Returns the graph dict representation of the given token tree that was previously visited 
        by this visitor instance.
        
        :returns: The graph dict representation
        """
        node_indices = list(self.index_node_map.keys())
        node_attributes = [self.get_node_attributes(self.index_node_map[i]) for i in node_indices]

        edge_indices = []
        for i, j in self.node_sequence_map.items():
            edge_indices += [[i, j], [j, i]]

        for i, j in self.anchor_edges:
            edge_indices += [[i, j], [j, i]]

        # 12.06.23 - There was a bug here where we used the edge attributes directly which resulted in the 
        # numpy array being one dimensional instead of two-dimensional which ultimately led to problems with the 
        # models; they expect it to be 2d.
        edge_attributes = [[1.0] for e in edge_indices]
        graph = {
            'node_indices':         np.array(node_indices, dtype=int),
            'node_attributes':      np.array(node_attributes, dtype=float),
            'edge_indices':         np.array(edge_indices, dtype=int),
            'edge_attributes':      np.array(edge_attributes, dtype=float),
        }
        return graph

    def get_node_attributes(self, node):
        """
        Given a token tree Node object ``node`` which represents a "node" token of the cogiles grammar, this  
        method does returns the appropriate node attributes vector which belongs the the corresponding node type 
        that is represented by hat node.
        
        :returns: a list of float values representing the node attributes vector for the node 
        """
        node_type = str(node.text)
        return NODE_TYPE_ATTRIBUTE_MAP[node_type]

    def visit_graph(self, node, visited_children):
        return visited_children

    def visit_branch(self, node, visited_children):
        # The children of a branch will include the parentheses tokens which are used to delimit a branch
        # so we will need to remove those here
        branch = squeeze_list(visited_children[1:-1])
        if not isinstance(branch, list):
            branch = [branch]

        self.prev_node = self.node_sequence_map[branch[0]]

        return branch

    def visit_node(self, node, visited_children):
        # This method is called whenever we visit the most fundamental representation of a graph node
        # which is a letter that determines the node type / color. For every such node we visit we use these
        # two dictionaries here to establish a two-sided mapping from node index to the actual node.
        node_index = self.node_index
        self.index_node_map[node_index] = node

        if node_index != 0:
            if self.is_break:
                self.is_break = False
            else:
                self.node_sequence_map[node_index] = self.prev_node

        self.prev_node = self.node_index
        self.node_index += 1
        return node_index

    def visit_break(self, node, visited_children):
        # We set this flag, so that the node which is parsed immediately after this break token 
        # knows that it should not form a connection with the previous node.
        self.is_break = True
        return True

    def visit_anchor(self, node, visited_children):
        anchor_index = int(node.text.replace('-', ''))
        node_index = self.prev_node
        if anchor_index not in self.anchor_map:
            self.anchor_map[anchor_index] = node_index
        else:
            self.anchor_edges.append((self.anchor_map[anchor_index], node_index))

        return anchor_index

    def generic_visit(self, node, visited_children):
        """ The generic visit method. """
        # return visited_children or node
        return visited_children or node


def graph_from_cogiles(value: str) -> tc.GraphDict:
    tree = cogiles_grammar.parse(value)
    visitor = CogilesVisitor()
    visitor.visit(tree)
    return visitor.process()


def node_type_from_attributes(attributes: np.array) -> str:
    attributes_tuple = tuple([round(value, 1) for value in attributes])
    return NODE_ATTRIBUTE_TYPE_MAP[attributes_tuple]


class CogilesEncoder:
    """
    This class can be used to encode GraphDict representations of color graphs into the COGILES string
    format.

    To do this a new encoder instance has to be created for a specific graph and then the ``encoder``
    method has to be called which will return the resulting encoded COGILES string.

    .. code-block:

        encoder = CogilesEncoder(graph)
        cogiles: str = encoder.encode()

    The algorithm idea
    ------------------

    This section will give a brief overview of the general idea behind the encoding algorithm. This
    algorithm is explicitly deterministic so that there is a definitively canonical COGILES representation
    for every graph.

    The encoding is a recursive procedure which starts at a single root node (deterministically
    the node with the lowest numeric index value), converts the numeric RGB code into the representative
    string character for that number and adds that to the string. Then all the neighbors of that node are
    collected and on each of them the procedure is recursively called as well. If a node has already been
    processed a "visited" flag will be set and all visited neighbors will be excluded. By exploring the
    graph iteratively like this, eventually all branches will be visited.
    """
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

        self.node_visited = [False for _ in range(self.num_nodes)]

        self.index = 0

        # This will be the final, encoded COGILES string
        self.value: str = ''
        self.current_anchor_index: int = 1
        self.index_anchor_map: t.Dict[int, int] = {}

    def encode(self):
        """
        This method will encode the ``self.graph`` with which the encoder has been constructed and return
        corresponding COGILES string representation of it.

        :returns: str
        """
        # This list will be used to collect the individual graph strings. Multiple individual strings
        # may happen when there is a
        components: t.List[str] = []

        # get_unvisited_branch will return the index with the lowest numeric index value, which has not
        # been visited yet. If all the nodes have been visited already, this will return None.
        index = self.get_unvisited_branch()
        while index is not None:
            # encode_branch will ultimately return the string representation for the ENTIRE graph structure
            # that is CONNECTED to the initial index which we provide as an argument here.
            # This explains the loop: It only works for connected graph structures. There may be several
            # unconnected graphs in which case we need to loop until all of them have been visited.
            components.append(self.encode_branch(index))
            index = self.get_unvisited_branch()

        # At the end we join all those unconnected fragments with the corresponding colon syntax.
        self.value = '.'.join(components)
        return self.value

    def get_unvisited_branch(self) -> t.Optional[int]:
        """
        This method will return the integer index of that unvisited node with the lowest numeric index.
        If there are no unvisited nodes anymore, this method will return None.

        :returns: integer node index.
        """
        unvisited_nodes = [index for index in range(self.num_nodes) if not self.node_visited[index]]
        unvisited_nodes = list(sorted(unvisited_nodes))

        if len(unvisited_nodes) > 0:
            return unvisited_nodes[0]
        else:
            return

    def encode_branch(self, node_index: int) -> str:
        """
        Given a ``node_index`` of the subject graph, this method will recursively construct the COGILES
        representation of the entire graph structure CONNECTED to that node and return the
        resulting constructed string value.

        :param node_index: The node index from which to start the encoding

        :returns: str
        """
        # To process the current node at hand is the easy part. We just have to map the graph attribute
        # vector back into the corresponding node type string.
        value: str = self.node_type_from_attributes(self.node_attributes[node_index])
        
        # The neighbors we can simply be determined from the corresponding row of the adjacency matrix. 
        # We split the neighbors here into those neighbors that are attached to an anchor index and those 
        # who are not, because in the further process there are multiple instances where this distinction 
        # is needed.
        neighbors = sorted([i for i, v in enumerate(self.node_adjacency[node_index, :]) if v])
        neighbors_anchor = [ni for ni in neighbors if ni in self.index_anchor_map]
        neighbors_normal = [ni for ni in neighbors if ni not in self.index_anchor_map]
        
        # If this node that we are currently looking at is connected to more than one NON-ANCHORED node than 
        # we are taking a pre-caution here and pre-emptively attach an anchor to it which may or may not be
        # used in the future.
        # These connections to neighboring non-anchored nodes can really easily be determined by the length 
        # of the corresponding neighbors list.
        num_edges = len(neighbors_normal)
        if num_edges > 1:
            # This method will return a new anchoring string that we'll just add right behind the
            # current node type string.
            value += self.create_anchor(node_index)

        # But then comes the more difficult part where we have to recursively process the rest of the
        # connected graph structure.
        
        # Note: One might ask themselves why we are iterating over this weird sum of both of the split 
        # neighbor lists here instead of "neighbors" directly since both of these should amount to essentially 
        # the same list. The reason is the ordering! By doing it like this, we iterate all of the anchor nodes 
        # first which makes it so that we pre-load all of the anchor connections in the resulting string before 
        # starting the branching connections.
        for neighbor_index in neighbors_anchor + neighbors_normal:

            # We modify the node_adjacency matrix here to effectively delete the connection we have just
            # processed so that this connection will not be considered in the future when gathering the
            # neighbor nodes.
            self.node_adjacency[node_index, neighbor_index] = 0
            self.node_adjacency[neighbor_index, node_index] = 0

            # 06.06.23 - This was a bug; without checking for the visitation status of the neighbor nodes
            # here, certain nodes in certain configurations were actually duplicated in the resulting
            # output string representation.
            if self.node_visited[neighbor_index]:
                continue

            # If the neighbor has an anchor attached to it then connecting to that neighbor is actually
            # relatively easy and just involves adding the corresponding anchor string to the current
            # working string.
            if neighbor_index in self.index_anchor_map:
                value += f'-{self.index_anchor_map[neighbor_index]}'
                continue

            # If the neighbor in question has in fact not already been visited then we now need to do so
            # by recursively calling this very method on the neighboring index.
            # The trivial case is if there is only a single immediate neighbor in which case we can just
            # add it to the string and exploit the implicit connection rule of the COGILES syntax.
            if len(neighbors) == 1:
                value += self.encode_branch(neighbor_index)
            else:
                # If there are multiple neighbors however we are going to play it safe and attach EVERY
                # one of them as a separate branch!
                value += f'({self.encode_branch(neighbor_index)})'

        self.node_visited[node_index] = True
        
        return value

    def node_type_from_attributes(self, attributes: np.array) -> str:
        distances = [np.linalg.norm(np.array(attributes) - np.array(key)) for key in NODE_ATTRIBUTE_TYPE_MAP.keys()]
        index = np.argmin(distances)
        node_type = list(NODE_ATTRIBUTE_TYPE_MAP.values())[index]
        return node_type

    def create_anchor(self, node_index: int) -> str:
        """
        This method will create a new anchor at the node with the given ``node_index``

        Anchors are the "-{number}" which link nodes together even
        when the corresponding nodes are not directly next to each other within the COGILES string.

        This has to be done with this function because creating a new anchor isn't just about creating the
        string, the node at which the anchor is created also has to be saved into an internal map so that
        in the future other nodes "will" know that it is possible to connect to that node via an anchor!

        :returns: The string representation of JUST the anchor syntax.
        """
        self.index_anchor_map[node_index] = self.current_anchor_index
        anchor_string = f'-{self.current_anchor_index}'
        self.current_anchor_index += 1

        return anchor_string


def graph_to_cogiles(graph: tc.GraphDict) -> str:
    encoder = CogilesEncoder(graph)
    return encoder.encode()
