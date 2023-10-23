import typing as t

import numpy as np
import networkx as nx
import visual_graph_datasets.typing as tv


def copy_graph_dict(graph: tv.GraphDict):
    """
    Copies the given ``graph`` dict and returns the copy. 
    
    The copy is a true copy, meaning that modifying the attributes of one graph dict 
    does not influence the attributes of the other.
    All attributes of the copied graph dict will already be numpy arrays.
    
    :param graph: The graph dict to be copied
    
    :returns: graph dict
    """
    copy = {}
    for key, value in graph.items():
        copy[key] = np.array(value).copy()
        
    return copy


def graph_node_adjacency(graph: tv.GraphDict) -> np.ndarray:
    """
    given a ``graph`` dict returns the binary adjacency matrix for that graph, which is a 
    quadratic binary numpy array.
    
    :param graph: The graph dict for which to create the adjacency matrix
    
    :returns: a numpy array of shape (N, N) where N is the number of nodes in the graph and a 
        location (i, j) in the array is 1 if the two nodes with indices i and j share a directed edge 
        and otherwise 0
    """
    num_nodes = len(graph['node_indices'])
    node_adjacency = np.zeros(shape=(num_nodes, num_nodes), dtype=float)
    for (i, j) in graph['edge_indices']:
        node_adjacency[i, j] = 1
        
    return node_adjacency


def graph_edge_set(graph: tv.GraphDict) -> t.Set[t.Tuple[int, int]]:
    """
    given a ``graph`` dict, this method will return a set of tuples, where each tuple contains two 
    integer node indices which are connected by an edge in the graph.
    
    The purpose of this function is essentially to get rid of duplicate edges during an iteration. 
    Undirected graphs are represented by two directed edges in two directions. If one wants to 
    iterate simply all undirected edges then that will cause duplication.
    
    :param graph: The graph dict for which to create the edge set
    
    :returns: A set of tuples of two integers
    """
    return set([tuple(sorted([i, j])) for i, j in graph['edge_indices']])


def graph_remove_edge(graph: tv.GraphDict,
                      node_index_1: int,
                      node_index_2: int,
                      directed: bool = False,
                      ) -> tv.GraphDict:
    """
    In the given ``graph`` dict, removes the edge between the nodes given by their two integer indices 
    ``node_index_1`` and ``node_index_2``.
    
    This function is in place and will modify the given graph dict directly. If the specified edge does not 
    exist, nothing will happen.
    
    :param graph: The graph dict to be edited
    :param node_index_1: The start node of the edge
    :param node_index_2: The end node of the edge
    :param directed: If this flag is true then the function will only remove the directed edge that is 
        directly going from node 1 to node 2. If this flag is false, the function will remove ALL edges 
        that are found directly between the two given nodes.
    
    :returns: the graph
    """
    removed_indices = set()
    edge_indices = []
    for e, (i, j) in enumerate(graph['edge_indices']):
        if node_index_1 == i and node_index_2 == j:
            removed_indices.add(e)
        elif not directed and node_index_1 == j and node_index_2 == i:
            removed_indices.add(e)
        else:
            edge_indices.append((i, j))
            
    graph['edge_indices'] = np.array(edge_indices)
    
    edge_keys = [key for key in graph.keys() if key.startswith('edge') and key != 'edge_indices']
    for key in edge_keys:
        values = [value for _e, value in enumerate(graph[key].tolist()) if _e not in removed_indices]
        graph[key] = np.array(values)
        
    return graph


def graph_add_edge(graph: tv.GraphDict,
                   node_index_1: int,
                   node_index_2: int,
                   directed: bool = False,
                   attributes: t.Optional[dict] = None
                   ) -> tv.GraphDict:
    """
    In the given ``graph`` dict, this function will add a new edge between the two nodes given by their 
    integer node indices ``node_index_1`` and ``node_index_2``.
    
    :param graph: The graph dict to be edited
    :param node_index_1: The start node of the edge
    :param node_index_2: The end node of the edge
    :param directed: If this flag is true, the function will only insert the single directed edge going from 
        the start node to the end node. If false, the function will insert a second directed edge in the 
        opposite direction as well, thus making an "undirected" edge.
    :param attributes: An optional dictionary which can be used to specify additional edge related properties 
        to be added for this new edge to various additional attributes of the graph. The keys of this dict 
        should be valid names of graph dict properties and the values of the dict should be the values to be 
        inserted for the new edge in those corresponding arrays of the graph.
    
    :returns: the graph dict
    """
    edge_indices = graph['edge_indices'].tolist()
    edge_indices.append([node_index_1, node_index_2])
    if not directed:
        edge_indices.append([node_index_2, node_index_1])
        
    graph['edge_indices'] = np.array(edge_indices)
        
    if attributes is not None:
        for key, value in attributes.items():
            values = graph[key].tolist()
            values.append(value)
            if not directed:
                values.append(value)
                
            graph[key] = np.array(values)
                
    return graph


def nx_from_graph(graph: tv.GraphDict) -> nx.Graph:
    """
    Given a GraphDict ``graph``, this method will convert the graph dict representation into
    a networkx Graph object and return it.

    This networkx representation will also contain custom dynamically attached node and edge properties
    of the given graph dict.

    :param graph: The graph dict to be converted

    :return: nx.Graph
    """
    nx_graph = nx.Graph()

    node_keys = [key for key in graph.keys() if key.startswith('node')]
    edge_keys = [key for key in graph.keys() if key.startswith('edge')]

    for i in graph['node_indices']:
        nx_graph.add_node(i, **{key: graph[key][i] for key in node_keys})

    for e, (i, j) in enumerate(graph['edge_indices']):
        nx_graph.add_edge(i, j, **{key: graph[key][e] for key in edge_keys})

    return nx_graph