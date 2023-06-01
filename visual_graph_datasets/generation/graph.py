import random
import typing as t
from copy import copy

import numpy as np

import visual_graph_datasets.typing as tc


class GraphGenerator:

    DEFAULT_NODE_ATTRIBUTES_CB = lambda *args: np.array([1])
    DEFAULT_EDGE_ATTRIBUTES_CB = lambda *args: np.array([1])
    DEFAULT_EDGE_VALID_CB = lambda *args: True

    def __init__(self,
                 num_nodes: int,
                 num_additional_edges: int,
                 node_attributes_cb: t.Optional[t.Callable] = DEFAULT_NODE_ATTRIBUTES_CB,
                 edge_attributes_cb: t.Optional[t.Callable] = DEFAULT_EDGE_ATTRIBUTES_CB,
                 edge_valid_cb: t.Optional[t.Callable] = DEFAULT_EDGE_VALID_CB,
                 seed_graphs: t.List[tc.GraphDict] = [],
                 is_directed: bool = False,
                 prevent_edges_in_seed_graphs: bool = True):

        assert num_additional_edges >= len(seed_graphs) - 1, \
            (f'num_additional_edges ({num_additional_edges}) has to be at least as high as the '
             f'number of provided seed graphs - 1 ({len(seed_graphs) - 1}) to ensure connectivity!')

        # TODO: Check the seed graphs for shape issues to prevent very weird issues later on.

        self.num_nodes = num_nodes
        self.num_additional_edges = num_additional_edges
        self.node_attributes_cb = node_attributes_cb
        self.edge_attributes_cb = edge_attributes_cb
        self.edge_valid_cb = edge_valid_cb
        self.seed_graphs = seed_graphs
        self.is_directed = is_directed
        self.prevent_edges_in_seed_graphs = prevent_edges_in_seed_graphs

        # ~ computed properties

        # This is the number of edges which are contributed by the seed graphs. If there are no seed
        # graphs then we set that to -1 because then in that case the graph is grown from a single seed
        # node, which in itself does not contribute an edge to the graph which is actually removing one
        # edge under the assumption we make that in the graph growing process every node contributes one
        # edge to the node
        self.seed_node_count = sum(len(g['node_indices']) for g in seed_graphs)
        if seed_graphs and self.seed_node_count > 0:
            self.seed_edge_count = sum(len(g['edge_indices']) for g in seed_graphs)
        else:
            if self.is_directed:
                self.seed_edge_count = -1
            else:
                self.seed_edge_count = -2

        # undirected graphs will be assumed to just be directed graphs where nodes are connected by two
        # edges pointing both ways.
        if self.is_directed:
            self.num_edges = self.seed_edge_count + \
                              (self.num_nodes - self.seed_node_count) + self.num_additional_edges
        else:
            self.num_edges = self.seed_edge_count + \
                              ((self.num_nodes - self.seed_node_count) + self.num_additional_edges) * 2

        # ~ Graph properties
        # these following values are the array-like data structures which have to be constructed and which
        # directly represent the graph that is to be generated
        self.node_indices: t.List[int] = list(range(num_nodes))
        self.node_attributes: t.List[t.List[float]] = []
        self.edge_attributes: t.List[t.List[float]] = []
        self.node_adjacency: t.List[t.List[int]] = []
        self.edge_indices: t.List[t.List[int]] = []

        self.graph: tc.GraphDict = {}

        # ~ Generation properties
        # These properties are helpers which are needed for the construction process

        # These two lists will be modified during the generation process. The first one will be empty at
        # first and it will store all of the node indices which have already been inserted into the
        # graph structure. The second list contains all those indices which are yet remaining during each
        # step.
        self.node_indices_inserted: t.List[int] = []
        self.node_indices_remaining: t.List[int] = []
        # The keys of this dict will be the indices of the seed graphs in the given list of seed graphs,
        # the values will be mappings which map the index internal to the seed graph to an index of
        # this graph which is generated. Basically these are the translation dictionaries to know which
        # nodes of the generated graph have been originally contributed by the seed graphs.
        self.seed_graph_index_maps: t.Dict[int, t.Dict[int, int]] = {}
        # This is a list where the index is the node index and the value is the index of the seed graph
        # from which that node was grown. Basically, if we have multiple seed graphs we start with multiple
        # seeds which are not connected at first, with this list we keep track of which nodes are associated
        # with which of those seeds.
        self.seed_graph_association: t.List[int] = []
        # This will be a list mostly filled with the value "-1" which indicates that the node with the
        # corresponding index is not part of any seed graph. If the value is not -1 then it represents the
        # index of the seed graph (w.r.t the list of seed graphs given as parameter) to which that node
        # belongs.
        self.seed_graph_indices: t.List[t.Optional[int]] = []
        # To keep track of the index where to insert the next edge element into edge_indices and
        # edge_attributes lists
        self.current_edge_index = 0

    def reset(self) -> None:
        """
        Resets the internal state of the generation process and the constructed graph.

        This class implements the graph generation process via internal states. After a generation process
        was executed, the internal properties of the object will be populated and it will not be possible to
        execute a new generation process right away. To prevent that a new class has to be created for each
        graph to be generated, this method implements a reset of the internal state, such that after this
        method was called, a new graph can be generated.

        :return: None
        """
        # node_indices never get modified that is why they do not need to get reset here

        # We initialize all the property lists here with essentially empty items because during the
        # construction process we want to use index assignment at certain positions and that is not
        # possible with empty lists
        self.node_attributes = [[] for _ in range(self.num_nodes)]
        self.node_adjacency = [[0 for _ in range(self.num_nodes)] for _ in self.node_indices]

        self.edge_indices = [[] for _ in range(self.num_edges)]
        self.edge_attributes = [[] for _ in range(self.num_edges)]

        # Resetting the computation state
        self.node_indices_remaining = copy(self.node_indices)

        # We shuffle the node index list here already so that we get random node choices by simply popping
        # the first element in the other steps of the construction process.
        random.shuffle(self.node_indices_remaining)
        self.node_indices_inserted = []
        self.seed_graph_association = [0 for _ in range(self.num_nodes)]
        self.seed_graph_indices = [-1 for _ in range(self.num_nodes)]
        self.current_edge_index = 0

        self.graph = {
            'node_indices': self.node_indices,
            'node_attributes': self.node_attributes,
            'edge_indices': self.edge_indices,
            'edge_attributes': self.edge_attributes,
            'seed_graph_indices': self.seed_graph_indices
        }

    def generate(self) -> tc.GraphDict:
        # ~ Seeding
        # The whole generation procedure is a seeding method. The graph starts out with a seed center and
        # the rest of the graph is then "grown" around that. It is possible to define subgraphs which act
        # as seeds. If that is not the case, then one node is chosen at random to be the seed
        if len(self.seed_graphs) == 0 or self.seed_node_count == 0:
            self.insert_seed_node()
        else:
            self.insert_seed_graphs()

        # ~ Growing the rest of the graph
        self.grow_nodes()
        self.add_edges()

        self.node_adjacency = self.node_adjacency_from_edge_indices(self.num_nodes, self.edge_indices)

        return {
            'node_indices': np.array(self.node_indices, dtype=np.int32),
            'node_attributes': np.array(self.node_attributes, dtype=np.float32),
            'node_adjacency': np.array(self.node_adjacency, dtype=np.int32),
            'edge_indices': np.array(self.edge_indices, dtype=np.int32),
            'edge_attributes': np.array(self.edge_attributes, dtype=np.float32),
            'seed_graph_indices': np.array(self.seed_graph_indices, dtype=np.int32),
        }

    def insert_seed_node(self) -> None:
        """
        Inserts a random node from the list of nodes as the seed graph, from which the rest of the graph
        will be grown.

        :return: None
        """
        index = self.node_indices_remaining.pop(0)
        node_attributes: t.List[t.Any] = self.node_attributes_cb(self.graph, 0)
        self.node_attributes[index] = node_attributes
        self.node_indices_inserted.append(index)

    def insert_seed_graphs(self) -> None:
        """
        Inserts the seed graphs given in the list :attr:`.GraphGenerator.seed_graphs` into the graph as
        separate / unconnected seeds, from which nodes can be grown in subsequent steps.

        :return: None
        """
        # graph_index - The integer index of the seed graph within the given list of (multiple) seed graphs
        for graph_index, g in enumerate(self.seed_graphs):
            seed_size = len(g['node_indices'])
            seed_indices = random.sample(self.node_indices_remaining, k=seed_size)

            # This is a mapping whose keys are the node indices local to the seed graph and the values are
            # the corresponding values are the node indices chosen for this graph to be generated
            seed_index_map: t.Dict[int, int] = dict(zip(g['node_indices'], seed_indices))
            self.seed_graph_index_maps[graph_index] = seed_index_map

            # seed_index - The integer index of the node within the seed graph node enumeration
            # index - The integer index of the node within the enumeration of the generated graph
            for seed_index, index in seed_index_map.items():
                self.node_attributes[index] = g['node_attributes'][seed_index]
                self.seed_graph_association[index] = graph_index
                self.seed_graph_indices[index] = graph_index

                # Technically the node indices are now inserted, now we need to establish the corresponding
                # edge connections according to the seed graph.
                for (i, j), edge_attributes in zip(g['edge_indices'], g['edge_attributes']):
                    if seed_index == i:
                        edge = [seed_index_map[i], seed_index_map[j]]
                        self.edge_indices[self.current_edge_index] = edge
                        self.edge_attributes[self.current_edge_index] = edge_attributes
                        self.current_edge_index += 1

                # We now have to remove this node from the list of remaining nodes and also have to add it
                # to the list of inserted nodes
                self.node_indices_remaining.remove(index)
                self.node_indices_inserted.append(index)

    def insert_edge(self, i: int, j: int) -> None:
        # if the directed flag is set then we only insert a single edge. If it is not set then that means
        # that the graph should be undirected and in this framework we will interpret undirected edges as
        # two exactly same directed edges pointing in opposite directions.
        if self.is_directed:
            edges = [[i, j]]
        else:
            edges = [[i, j], [j, i]]

        for k, l in edges:
            self.edge_indices[self.current_edge_index] = [k, l]

            # This callback will create the list if edge attributes given information about the already
            # existing graph.
            edge_attributes: t.List[t.Any] = self.edge_attributes_cb(self.graph, k, l)
            self.edge_attributes[self.current_edge_index] = edge_attributes
            self.current_edge_index += 1

    def grow_nodes(self):
        # The general procedure is simple: we pop node indices from the remaining indices list and
        # randomly attach them to the already existing graph structure until there are none left
        while len(self.node_indices_remaining) > 0:
            index = self.node_indices_remaining.pop(0)

            # now we choose a random already inserted node to attach this node to
            anchor_index = random.choice(self.node_indices_inserted)

            # This callback will generate the node attributes given information about the already existing
            # graph structure.
            node_attributes: t.List[t.Any] = self.node_attributes_cb(self.graph, anchor_index)
            self.node_attributes[index] = node_attributes

            self.seed_graph_association[index] = self.seed_graph_association[anchor_index]

            # This method will create an edge between the two given indices, generating the edge attributes
            # along the way as well
            self.insert_edge(index, anchor_index)
            self.node_indices_inserted.append(index)

    def add_edges(self):
        # This method is supposed to add additional edges to the grown graph, because the base growing
        # procedure will always be a tree (no cycles).

        # The most important part of this will be to connect the multiple non-connected graphs grown from
        # potentially multiple seed graphs. All additional edges beyond that will be inserted between
        # random nodes.
        additional_edge_count = self.num_additional_edges

        # Obviously we only need to connect up the disconnected seeds IF there is more than one seed
        if len(self.seed_graphs) > 1:
            # We essentially want to look at pairs of seed graphs here and connect up the corresponding
            # still disconnected graphs. These two are the running seed graph indices for the two elements
            # of that pairwise consideration.
            gi = 0
            gj = 1
            while gj < len(self.seed_graphs):
                # Because there is the real possibility of completely empty seed graphs, this first section
                # of the loop is dedicated to find a pairs of seed graphs which are both not empty!
                if len(self.seed_graphs[gi]['node_indices']) == 0:
                    gi += 1
                    gj += 1
                    continue

                if len(self.seed_graphs[gj]['node_indices']) == 0:
                    gj += 1
                    continue

                # We choose a random node from both subgraph seeded clusters of nodes and insert an edge
                # there
                i = random.choice([k for k, gk in enumerate(self.seed_graph_association)
                                   if gk == gi and self.seed_graph_indices[k] < 0])
                j = random.choice([k for k, gk in enumerate(self.seed_graph_association)
                                   if gk == gj])
                self.insert_edge(i, j)
                additional_edge_count -= 1

                gj += 1

        while additional_edge_count > 0:
            i, j = random.sample(self.node_indices_inserted, k=2)

            # This first condition here makes sure that the edge is only inserted when that exact same
            # edge does not already exist.
            edge_index_tuples = [tuple(e) for e in self.edge_indices]
            edge_exist = ((i, j) in edge_index_tuples or
                          (j, i) in edge_index_tuples)
            # The second condition makes sure that if the prevent_edges_in_seed_graphs flag is set that we
            # do not insert an additional edge between two nodes of the seed graph, as that would kind of
            # disturb the subgraph structure
            seed_cond = (self.seed_graph_indices[i] == -1 or
                         self.seed_graph_indices[j] == -1 or
                         not self.prevent_edges_in_seed_graphs)

            # It is additionally possible to pass a callback to the constructor of this class which takes a
            # possible edge as arguments and then depending on some criteria returns the boolean value of
            # whether this edge is to be considered a valid insertion or not
            valid_cond = self.edge_valid_cb(self.graph, i, j)

            if not edge_exist and seed_cond and valid_cond:
                self.insert_edge(i, j)
                additional_edge_count -= 1

    # -- Utility Methods --

    def get_seed_graph_node_indication(self, seed_graph_index: int) -> t.List[int]:
        """
        Returns a list with the dimension of the node indices, where the element is 1 if the node of the
        corresponding index belongs to the seed graph with the given ``seed_graph_index`` and 0 otherwise.
        In short: a list of boolean integers indicating which nodes belong to the specified seed graph.

        :param seed_graph_index:
        :return:
        """
        indication = [0 for _ in self.node_indices]

        for i in self.node_indices:
            if self.seed_graph_indices[i] == seed_graph_index:
                indication[i] = 1

        return indication

    def get_seed_graph_edge_indication(self, seed_graph_index: int) -> t.List[int]:
        """
        Returns a list with the dimension of the edge indices, where the element is 1 if the edge of the
        corresponding index belongs to the seed graph with the given ``seed_graph_index`` and 0 otherwise.
        In short: a list of boolean integers indicating which edges belong to the specified seed graph.

        :param seed_graph_index:
        :return:
        """
        indication = [0 for _ in self.edge_indices]

        for e, (i, j) in enumerate(self.edge_indices):
            i_condition = self.seed_graph_indices[i] == seed_graph_index
            j_condition = self.seed_graph_indices[j] == seed_graph_index
            if i_condition and j_condition:
                indication[e] = 1

        return indication

    @classmethod
    def node_adjacency_from_edge_indices(cls,
                                         node_count: int,
                                         edge_indices: t.List[t.List[int]]) -> t.List[t.List[int]]:
        """
        Creates a node adjacency matrix, given the edges of the graph in the representation of an edge
        index list ``edge_indices``.

        :param node_count:
        :param edge_indices:
        :return:
        """
        node_adjacency = [[0 for _ in range(node_count)] for _ in range(node_count)]
        for i, j in edge_indices:
            node_adjacency[i][j] = 1

        return node_adjacency

