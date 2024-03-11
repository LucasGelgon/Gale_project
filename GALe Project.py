#!/usr/bin/env python
# coding: utf-8

# # Graph Isomorphism

# Lucas Gelgon and Pavlo Vilkhoviy

# Study various approaches/algorithms (different exact algorithms or approximation algorithms with different heursitics etc.) to find graph isomorphism. Choose two comparable approximation algorithms and implemenet these algorithms. Do experiments to verify the theoretical properties as (primarly) time/space complexities.

# #### What is Isomorphism ? 

# Graph isomorphism is a concept in graph theory. Two graphs are said to be isomorphic if they are essentially the same 
# from a structural point of view, even though they may have different labels on their nodes or edges.
# 

# ## Weisfeiler-Lehman

# First, we will look at the Weisfeiler-Lehman algorithm.
# This is an algorithm used to test the isomorphism of graphs. 

# The central idea of the algorithm is to iteratively label the nodes of the graph using local information about the nodes' neighbours.   
# 
# Here's how the algorithm works:
# 
# First, we need to iteratively label the nodes of a graph based on the local structure of their neighbours. 
# The function takes as input a graph and a specified number of iterations, updating the node labels according to the Weisfeiler-Lehman algorithm. 
# 
# The example demonstrates its application to a small graph, where the initial node labels ('A', 'B', 'C', 'D') are transformed into labels after the algorithm has been run. 
# 
# Finally, the plot_graph function is used to visualise the graph. The code effectively presents the WL algorithm for labelling graphs and provides a visual representation of the resulting graph.

# In[51]:


import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import hashlib

import time #for time properties
import psutil #for storage properties


# In[52]:


def weisfeiler_lehman(graph, iterations=3):
    for _ in range(iterations):
        labels = nx.get_node_attributes(graph, 'label')
        new_labels = {}

        for node in graph.nodes():
            neighbors = sorted(graph.neighbors(node))
            neighbor_labels = [labels.get(neighbor, '') for neighbor in neighbors]
            new_label = str(labels.get(node, '')) + ''.join(map(str, neighbor_labels))
            new_labels[node] = int(hashlib.sha256(new_label.encode('utf-8')).hexdigest(), 16)

        nx.set_node_attributes(graph, values=new_labels, name='label')


# Example : 

# In[53]:


G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4)])

labels = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}  
nx.set_node_attributes(G, values=labels, name='label')

weisfeiler_lehman(G)

final_labels = nx.get_node_attributes(G, 'label')
print(final_labels)


# In[54]:


def plot_graph(graph):
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True)
    plt.show()


# In[55]:


plot_graph(G)


# Now, we will create a function that takes two graphes in parameter and says if they are isomorphic or not

# In[56]:


def are_graphs_isomorphic(graph1, graph2, iterations=3):
    weisfeiler_lehman(graph1, iterations)
    weisfeiler_lehman(graph2, iterations)

    labels1 = list(nx.get_node_attributes(graph1, 'label').values())
    labels2 = list(nx.get_node_attributes(graph2, 'label').values())

    signature1 = hash(tuple(labels1))
    signature2 = hash(tuple(labels2))

    return signature1 == signature2


# Example : 

# In[57]:


isomorphic_graph1 = nx.Graph()
isomorphic_graph1.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4)])

isomorphic_graph2 = nx.Graph()
isomorphic_graph2.add_edges_from([(10, 20), (10, 30), (20, 40), (30, 40)])

non_isomorphic_graph1 = nx.Graph()
non_isomorphic_graph1.add_edges_from([(1, 2), (1, 3), (2, 4)])

non_isomorphic_graph2 = nx.Graph()
non_isomorphic_graph2.add_edges_from([(1, 2), (2, 3), (3, 4)])

# test the function
print(are_graphs_isomorphic(isomorphic_graph1, isomorphic_graph2))  # should return True
print(are_graphs_isomorphic(non_isomorphic_graph1, non_isomorphic_graph2))  # should return False


# ### Measuring time complexity

# The theorical complexity of this algorithm is O(n),where n is the number of nodes

# In[58]:


def generate_random_graph(size, probability):
    G = nx.erdos_renyi_graph(size, probability)
    labels = {node: str(node) for node in G.nodes()}
    nx.set_node_attributes(G, values=labels, name='label')
    return G

def generate_random_tree(size):
    G = nx.random_tree(size)
    labels = {node: str(node) for node in G.nodes()}
    nx.set_node_attributes(G, values=labels, name='label')
    return G

# Measuring time complexity
def measure_time_complexity(graph_generator, sizes, iterations=3):
    for size in sizes:
        if graph_generator == generate_random_graph:
            G1 = graph_generator(size, 0.1)
            G2 = graph_generator(size, 0.1)
        else:
            G1 = graph_generator(size)
            G2 = graph_generator(size)

        start_time = time.time()
        are_isomorphic = are_graphs_isomorphic(G1, G2, iterations)
        end_time = time.time()

        execution_time = end_time - start_time
        print(f"Graph size: {size}, Time elapsed: {execution_time} seconds")

sizes_to_experiment = [100, 200, 300, 400, 500]
measure_time_complexity(generate_random_graph, sizes_to_experiment)
measure_time_complexity(generate_random_tree, sizes_to_experiment)



# ### Measuring space complexity

# This measures space consumption before and after the execution of the algorithm. 
# However, this method is fairly crude and may not exhaustively reflect actual space consumption,
# but it does give a general idea of the impact on memory.

# In[59]:


def measure_space_complexity(graph_generator, sizes, iterations=3):
    for size in sizes:
        if graph_generator == generate_random_graph:
            G1 = graph_generator(size, 0.1)
            G2 = graph_generator(size, 0.1)
        else:
            G1 = graph_generator(size)
            G2 = graph_generator(size)

        # Measure space used before execution
        start_memory = psutil.Process().memory_info().rss

        are_isomorphic = are_graphs_isomorphic(G1, G2, iterations)

        # Mesurer space used after execution
        end_memory = psutil.Process().memory_info().rss

        space_used = end_memory - start_memory

        print(f"Graph size: {size}, Space used: {space_used} bytes")

sizes_to_experiment = [100, 200, 300, 400, 500]
measure_space_complexity(generate_random_graph, sizes_to_experiment)
measure_space_complexity(generate_random_tree, sizes_to_experiment)


# ## VF2 algorithm

# ![image.png](attachment:image.png)

# A matching process between two graphs G1=(N1,B1) and G2=(N2,B2) consists in the determination of a mapping M which associates nodes of G1 with nodes of G2 and vice versa, according to some predefined constraints. Generally, the mapping M is expressed as the set of pairs (n,m) (with n∈G1 and m∈G2) each representing the mapping of a node n of G1 with a node m of G2. A mapping M⊂N1×N2 is said to be an isomorphism iff M is a bijective function that preserves the branch structure of the two graphs. A mapping M⊂N1×N2 is said to be a graph-subgraph isomorphism iff M is an isomorphism between G2 and a subgraph of G1.

# In[1]:


import networkx as nx
import time 
import psutil
#function to check if two graphs can be isomorphic before execution
def my_function(Graph1, Graph2):
    list_of_degree_G1 = []
    list_of_degree_G2 = []
    
    if Graph1.number_of_nodes() == 0 or Graph2.number_of_nodes() == 0:
        return False
    if Graph1.number_of_nodes() != Graph2.number_of_nodes():
        return False
    for x in Graph1.nodes:
        list_of_degree_G1.append(Graph1.degree[x])
        continue
    for x in Graph2.nodes:
        list_of_degree_G2.append(Graph2.degree[x])
        continue
    for x in list_of_degree_G1:
        if x not in list_of_degree_G2:
            return False
        
def is_feasible(graph1, graph2, node1, node2, partial_mapping):
    # Check if the nodes are compatible based on degrees of neighbors

    # Get the neighbors of the input nodes in both graphs
    neighbors1 = set(graph1.neighbors(node1))
    neighbors2 = set(graph2.neighbors(node2))

    # Check if the degrees of neighbors are matching
    if not check_degree_of_neighbors(graph1, graph2, neighbors1, neighbors2, partial_mapping):
        return False

    # Check if the nodes are compatible based on partial mapping
    for neighbor1 in neighbors1:
        if neighbor1 in partial_mapping:
            mapped_neighbor = partial_mapping[neighbor1]
            if mapped_neighbor not in neighbors2 or not graph2.has_edge(node2, mapped_neighbor):
                return False

    return True

def check_degree_of_neighbors(graph1, graph2, neighbors1, neighbors2, partial_mapping):
    # Check if the degrees of neighbors match

    for neighbor1 in neighbors1:
        mapped_neighbor1 = partial_mapping.get(neighbor1, None)
        if mapped_neighbor1:
            # If the neighbor is already mapped, check the degrees
            degree1 = graph1.degree(neighbor1)
            degree2 = graph2.degree(mapped_neighbor1)
            if degree1 != degree2:
                return False
    return True

def isomorphism_search(graph1, graph2, times_to_check, partial_mapping=None):
    if times_to_check <0:
        return False
    if partial_mapping is None:

        partial_mapping = {}

    if len(partial_mapping) == len(graph1.nodes):
        # All nodes in graph1 are mapped, isomorphism found
        return True

    for node1 in graph1.nodes:
        if node1 not in partial_mapping:
            for node2 in graph2.nodes:
                if node2 not in set(partial_mapping.values()):
                    if is_feasible(graph1, graph2, node1, node2, partial_mapping):
                        partial_mapping[node1] = node2
                        if isomorphism_search(graph1, graph2, times_to_check-1, partial_mapping):
                            return True
                        partial_mapping.pop(node1)

    return False

# Example usage
graph1 = nx.Graph([(0, 6), (1, 3), (1, 2), (1, 4), (2, 7), (2, 6), (2, 8), (2, 3), (3, 4), (3, 9), (3, 7), (4, 6), (4, 9), (4, 7), (5, 9)])
graph2 = nx.Graph([(0, 7), (0, 5), (0, 6), (0, 3), (1, 5), (3, 6), (3, 4), (3, 9), (3, 5), (4, 7), (5, 8), (5, 6), (6, 8), (7, 8), (8, 9)])
G = nx.Graph(
    [
        
        ("g", "b"),
        ("g", "c"),
        ("b", "h"),
        ("b", "j"),
        ("h", "d"),
        ("c", "i"),
        ("c", "j"),
        ("i", "d"),
        ("d", "j"),
        ("a", "g"),
        ("a", "h"),
        ("a", "i"),
    ]
)

H = nx.Graph(
    [
        (1, 2),
        (1, 5),
        (1, 4),
        (2, 6),
        (2, 3),
        (3, 7),
        (3, 4),
        (4, 8),
        (5, 6),
        (5, 8),
        (6, 7),
        (7, 8),
    ]
)
if my_function(G, H) is not False:
    result = isomorphism_search(G, H, len(G))
    print("Is isomorphic:", result)
else:
    print("Graphs are not isomorphic")
    
if my_function(graph1, graph2) is not False:
    start_time = time.time()
    result2 = isomorphism_search(graph1, graph2, len(graph1) )
    end_time = time.time()
    print("Is isomorphic:", result2)
else:
    print("Graphs are not isomorphic")



# ### graph randomizer and time complexity
# 

# In[2]:


#graph randomizer
test_graph_list = []
for i in range(10):
    GRA = nx.gnm_random_graph(7,15)
    m = GRA.size()
    test_graph_list.append(GRA)
#testing time of the algorithm with random graphs
for m in test_graph_list:
    print(m.edges)
for m in test_graph_list:
    print("===================================================")
    print("The number of a graph in a list to compare:",test_graph_list.index(m))
    print("===================================================")

    for n in test_graph_list:
        if my_function(m, n) is not False:
            start_time = time.time()
            res = isomorphism_search(m, n, len(m))
            end_time = time.time()
            execution_time = end_time - start_time
            print("Is isomorphic:", res)
            print("For graph number:", test_graph_list.index(n), "Time is:", execution_time)
        else:
            print("Checked by my_function if graphs are isomorphic: False", test_graph_list.index(n))

            
            

    


# ### graph randomizer and space complexity
# 

# In[3]:


for m in test_graph_list:
    print(m.edges)
for m in test_graph_list:
    print("===================================================")
    print("The number of a graph in a list to compare:",test_graph_list.index(m))
    print("===================================================")

    for n in test_graph_list:
        if my_function(m, n) is not False:
            start_space = psutil.Process().memory_info().rss
            res = isomorphism_search(m, n, len(m))
            end_space = psutil.Process().memory_info().rss
            space_used = end_space - start_space
            print("Is isomorphic:", res)
            print("For graph number:", test_graph_list.index(n), "Space used:", space_used)
        else:
            print("Checked by my_function if graphs are isomorphic: False", test_graph_list.index(n))


# In[ ]:




