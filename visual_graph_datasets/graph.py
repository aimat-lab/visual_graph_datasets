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
    keep_indices: t.List[int] = []
    
    for e, (i, j) in enumerate(graph['edge_indices']):
        
        if (node_index_1 == i) and (node_index_2 == j):
            continue
        
        if (not directed) and (node_index_1 == j) and (node_index_2 == i):
            continue
        
        keep_indices.append(e)
            
    edge_indices = np.array([graph['edge_indices'][e] for e in keep_indices])
    graph['edge_indices'] = edge_indices
    
    # Now we also want to 
    for key in graph.keys():
        
        if key.startswith('edge') and key != 'edge_indices':
            value = graph[key].tolist()
            value = np.array([value[e] for e in keep_indices])
            graph[key] = value
        
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


def graph_attach_node(graph: tv.GraphDict,
                      anchor_index: int,
                      node_attribute: np.ndarray,
                      edge_attribute: np.ndarray,
                      directed: bool = False,
                      ) -> tv.GraphDict:
    """
    Given a ``graph`` dict representation, this method will attach an additional node with the given 
    ``node_attributes`` to the graph. The new node will be attached via a single edge to the node that 
    is specified by the ``anchor_index``. The edge between the two nodes will have the given 
    ``edge_attributes``.
    
    :param graph: The graph to be modified
    :param anchor_index: The integer index of the node that already exists in the graph to which the new 
        node will be attached to
    :param node_attributes: The node attributes of the new node
    :param edge_attributes: The edge attributes of the edge that is used to attach the new node to 
        the anchor node.
    :param directed: If this is False, the additional edge will be duplicated in both directions.
    
    :returns: The modified graph dict which contains the additional edge
    """
    node_index = int(np.max(graph['node_indices']) + 1)
    
    graph['node_indices'] = np.concatenate((graph['node_indices'], np.array([node_index])), axis=0)
    node_keys = [key for key in graph.keys() if key.startswith('node') and key not in ['node_indices', 'node_attributes']]
    for key in node_keys:
        graph[key] = np.concatenate((graph[key], np.zeros(shape=(1, *graph[key][0].shape))), axis=0)
    
    graph['node_attributes'] = np.concatenate((graph['node_attributes'], np.array([node_attribute])), axis=0)

    # ~ adding the edges
    edge_indices = graph['edge_indices'].tolist()
    edge_attributes = graph['edge_attributes'].tolist()

    edge_keys = [key for key in graph.keys() if key.startswith('edge') and key not in ['edge_indices', 'edge_attributes']]
    for key in edge_keys:
        if directed:
            graph[key] = np.concatenate((graph[key], np.zeros(shape=(1, *graph[key][0].shape))), axis=0)
        if not directed:
            graph[key] = np.concatenate((graph[key], np.zeros(shape=(2, *graph[key][0].shape))), axis=0)

    edge_indices.append([anchor_index, node_index])
    edge_attributes.append(edge_attribute)
    if not directed:
        edge_indices.append([node_index, anchor_index])
        edge_attributes.append(edge_attribute)
    
    graph['edge_indices'] = np.array(edge_indices, dtype=int)
    graph['edge_attributes'] = np.array(edge_attributes, dtype=float)
    
    return graph
    

def graph_expand_mask(graph: tv.GraphDict,
                      node_mask: np.ndarray,
                      num_hops: int = 1,
                      ) -> np.ndarray:
    """
    Given a ``graph`` dict representation and a ``node_mask`` which defines a binary integer mask (0 or 1) 
    on the graph ndoes, this function performs a ``num_hops`` expansion of that mask along the adjacent nodes.
    The function returns the numpy array that is the expanded mask.
    
    For every hop, the mask is expanded to all the nodes that are directly connected to at least one other 
    node that is already part of the mask. 
    
    :param graph: This is the graph dict representation
    :param node_mask: This is an integer numpy array in the shape (V, ) where V is the number of nodes 
        in the graph.
    :param num_hops: This is the number of expansions of the mask into the neighboring nodes
   
    :returns: The new and expanded mask array of the shape (V, )
    """
    # node_adjacency: (N, N)
    node_adjacency = graph_node_adjacency(graph).astype(int)
    
    # We copy this mask here so that we do not accidentally introduce side effects to the original 
    # mask that is passed as the parameter
    # mask: (N, )
    mask = node_mask.copy().astype(int)
    
    # prev_mask: (N, )
    prev_mask = node_mask.copy().astype(int)
    
    for h in range(num_hops):
        
        for i in graph['node_indices']:
            
            # So for every node of the graph that is masked in the previous mask, we go through all 
            # its direct neighbors and also mask those, but only in the updated mask
            if prev_mask[i]:
                neighbors = [j for j, value in enumerate(node_adjacency[i]) if value]
                for j in neighbors:
                    mask[j] = 1
        
        prev_mask = mask.copy()

    return mask


def extract_subgraph(graph: tv.GraphDict,
                     node_mask: np.ndarray
                     ) -> tv.GraphDict:
    """
    Given a ``graph`` dict representation, this function will extract the subgraph that is specified 
    by the binary integer ``node_mask``, which assignes either a 0 or 1 value to every node in the 
    graph. Nodes assigned with a value of 1 will be included in the extracted subgraph. The function 
    returns a new graph dict representation only representing the extracted subgraph.
    
    NOTE: An edge will only be extracted from the original graph if BOTH of the nodes are masked to 
        be part of the extracted structure - otherwise edges will be omitted.
    
    :param graph: The original graph structure from which to extract
    :param node_mask: The node mask containing 0 or 1 values which defines the parts of the graph 
        to be extracted.
    
    :returns: The extracted subgraph structure as a graph dict
    """
    sub_graph = {}
    
    node_mask = node_mask.copy().astype(int)
    
    # Here in the first step we construct a mapping from the "old" node indices of 
    # the original graph to the node indices of the "new" subgraph indicing system.
    # So the keys of this map will be the node indices of the original graph and the 
    # the values will be the corresponding node indices of the subgraph. Most 
    # importantly this dict will contain only values for those nodes that are actually 
    # masked (aka part of the new graph)
    node_index_map: t.Dict[int, int] = {}
    j: int = 0
    for i in graph['node_indices']:
        if node_mask[i]:
            node_index_map[i] = j
            j += 1
            
    # the node indices of the new graph can very easily be constructed as just the values 
    # of that mapping data structure
    node_indices = np.array(list(node_index_map.values()))
    sub_graph['node_indices'] = node_indices
    
    # We can do the same thing for the edge indices. So here we construct a mapping whose keys 
    # are the integer indices of the edge tuples in the original graph and the values are the new 
    # indices within the sub_graph
    edge_index_map: t.Dict[int, int] = {}
    q: int = 0
    for e, (i, j) in enumerate(graph['edge_indices']):
        if node_mask[i] and node_mask[j]:
            edge_index_map[e] = q
            q += 1
    
    edge_indices = np.array([
        [node_index_map[i] for i in graph['edge_indices'][e]] 
        for e, q in edge_index_map.items()
    ])
    sub_graph['edge_indices'] = edge_indices

    # With this data structure we can now iterate through all the properties of the 
    # graph and essentially translat
    for key, value in graph.items():
        
        if isinstance(value, np.ndarray):
            value = value.copy()
    
            if key.startswith('node') and key != 'node_indices':
                masked_value = np.array([value[i] for i, j in node_index_map.items()])
                sub_graph[key] = masked_value
            
            elif key.startswith('edge') and key != 'edge_indices':
                masked_value = np.array([value[e] for e, q in edge_index_map.items()])
                sub_graph[key] = masked_value

    return sub_graph, node_index_map, edge_index_map


def graph_remove_node(graph: tv.GraphDict,
                      node_index: int,
                      ) -> tv.GraphDict:
    """
    Given a ``graph`` dict representation, this method removes the node with the given ``node_index`` 
    from that graph and returns the modified graph dict representation as a result.
    
    NOTE: All the edges that are connected to that node will also be removed!
    
    :param graph: The graph from which to remove the node
    :param node_index: the integer index of the node to be removed
    
    :returns: A modified graph dict representation where the node and all its edges are removed.
    """
    # The implementation of node removal is actually very easy because we can simply cast it as a 
    # special case of the much more generic graph extraction. 
    # We simply create an extraction mask here that includes all the original graph's nodes except 
    # the one node that is to be removed! 
    node_mask = np.ones(shape=graph['node_indices'].shape)
    node_mask[node_index] = 0
    
    new_graph, _, _ = extract_subgraph(
        graph=graph,
        node_mask=node_mask,
    )
    return new_graph


def graph_find_connected_regions(graph: tv.GraphDict,
                                 node_mask: np.ndarray,
                                 ) -> np.ndarray:
    """
    Given a ``graph`` dict representation and a binary integer ``node_mask`` (consisting of only 1 and 0 values)
    this function will determine which of the nodes in that mask are connected into a region. There must exist 
    some connection between two nodes of a region - however many edges would have to be traversed.
    
    The function will return a ``region_mask`` array with the same shape as the given node mask. This region mask 
    will consist of special -1 values for every node that was NOT included in the given node mask. All nodes that 
    were somehow part of the mask will be assigned a != -1 integer index that identifies the region that they were 
    assigned to.
    
    A possible example may look like this:
    
    node_mask   = [ 0, 1, 1, 1,  0,  0, 1, 1,  0]
    region_mask = [-1, 0, 0, 0, -1, -1, 1, 1, -1]

    This result indicates that the given node mask defines two disconnected regions with the indices 0 and 1
    
    :param graph: The graph dict representation of the graph for which to determine the connected regions
    :param node_mask: An array of shape (V, ) which contains a "1" for all nodes that should be included and 
        "0" value otherwise.
    
    :returns: The ``region_mask`` array of shape (V, )
    """
    graph_size = len(graph['node_indices'])
    
    # node_adjacency: (B, B)
    node_adjacency = graph_node_adjacency(graph)
    
    # The algorithm to calculate the regions is quite simple. At first we initialize the region index of each node 
    # that is defined by the mask as a unique integer - we can simpy use their node index for this. And then we 
    # do a bunch of "message passing" operations where always the smaller index is dominant and replaces the 
    # larger indices of the neighboring nodes that are also part of the mask. If this step is repeated for as 
    # many nodes as there are then every possible region will be converged to only the smallest index that 
    # was originally assigned to one of its nodes.
    
    region_mask = np.full(shape=graph['node_indices'].shape, fill_value=-1)
    region_mask[node_mask > 0.5] = graph['node_indices'][node_mask > 0.5]
    
    for _ in range(graph_size):
        
        for i in graph['node_indices']:

            for j in graph['node_indices']:
                
                if node_adjacency[i, j] and region_mask[i] > -1 and region_mask[j] > -1:
                    min_value = min(region_mask[i], region_mask[j])
                    region_mask[i] = min_value
                    region_mask[j] = min_value    

    # After the main loop is finished the region_mask will consist of -1 values as well as the 
    # left over indices that defines the different regions. Now these indices could be any number 
    # but for convencience sake we would like them to be monotonically increasing indices starting 
    # at 0. 
    # So in the following section we define a mapping to change those indices.
    
    regions = [v for v in set(region_mask) if v > -1]

    region_map = {r: i for i, r in enumerate(regions)}
    region_map[-1] = -1
    
    region_mask = np.array([region_map[v] for v in region_mask], dtype=int)
    
    return region_mask


def graph_is_connected(graph: tv.GraphDict) -> bool:
    """
    Given a ``graph`` dict representation, this method returns the boolean value of whether or not 
    that graph is connected or not. Returns False if the graph consists of more than 1 disconnected 
    region.
    
    :param graph: the graph to be checked
    
    :returns: boolean value of whether or not the graph is connected
    """
    # THe implementation of this is rather easy because we can simply use a full mask and 
    # calculate the connected regions and if there is more than one region in the graph, 
    # it is obviously not all connected.
    mask = np.ones(shape=graph['node_indices'].shape)
    region_mask = graph_find_connected_regions(
        graph=graph,
        node_mask=mask,
    )
    return np.all(region_mask == 0)


def graph_has_isolated_node(graph: tv.GraphDict) -> bool:
    """
    Checks whether or not the given ``graph`` dict representation contains at least one isolated node, which is 
    a node that is not part of any edge.
    
    :param graph: The graph to be checked
    
    :returns: boolean value of whether or not the graph contains an isolated node
    """
    # We simply go through all the edges and every node that is part of at least one edge we will consider as 
    # a non-isolated node aka connected node.
    connected_nodes = set()
    for i, j in graph['edge_indices']:
        connected_nodes.add(i)
        connected_nodes.add(j)
    
    # And if this set of connected nodes is not as long as all the node indices then we know that there must be 
    # at least one isolated node.
    return len(connected_nodes) < len(graph['node_indices'])


def nx_from_graph(graph: tv.GraphDict) -> nx.Graph:
    """
    Given a GraphDict ``graph``, this method will convert the graph dict representation into
    a networkx Graph object and return it.

    This networkx representation will also contain custom dynamically attached node and edge properties
    of the given graph dict.

    :param graph: The graph dict to be converted

    :returns: nx.Graph
    """
    nx_graph = nx.Graph()

    node_keys = [key for key in graph.keys() if key.startswith('node')]
    edge_keys = [key for key in graph.keys() if key.startswith('edge')]
    print(node_keys, edge_keys)

    for i in graph['node_indices']:
        nx_graph.add_node(i, **{key: graph[key][i] for key in node_keys})

    for e, (i, j) in enumerate(graph['edge_indices']):
        nx_graph.add_edge(i, j, **{key: graph[key][e] for key in edge_keys})

    return nx_graph