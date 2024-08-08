from typing import List
from Node import Node
from connected_components import connected_components
import matplotlib.pyplot as plt
import numpy as np
from adjacency_graph import adjacency_graph
from collections import defaultdict


def propagate_label(node, node_neighbors, adj):
    """
    Update node's label based on the weights of its neighbor's labels.
    
    Parameters:
    ----------
    node : Node
        Random node in graph of nodes
    node_neighbors : List
        List of the node's neighbors
    adj : numpy.ndarray, shape=(N, N)
        The adjacency-matrix for the graph. Nonzero entries indicate
        the presence of edges.
    """
    labels_to_weights = defaultdict(lambda: 0)  # maps the label of a node to the weights of its neighbor's labels
    for node_neighbor in node_neighbors:
        # print(adj)
        weight = adj[node.id][node_neighbor.id]
        # print(weight)
        labels_to_weights[node_neighbor.label] += weight
        # print(labels_to_weights)
    if labels_to_weights:
        node.label = max(labels_to_weights, key=labels_to_weights.get)  # get key of max value in labels_to_weights


def whispers(graph, adj):
    """
    - Calls the propagate_label function some specified number of times on your graph. 
    - A node should be randomly selected each time the propagate_labels function is called.
    - Uses the connected_components function, which records / plot how the number of connected components 
    changes across iterations. It takes in your graph's list of nodes. 
    and returns a list of lists - each inner list contains all nodes with a common label.
    - At first each node should have its own unique label, and thus there should be as many connected 
    components as nodes. But soon the number of connected components should decrease and converge to a 
    stable number.

    Parameters
    ----------
    graph : Tuple[Node, ...]
        This is simple a tuple of the nodes in the graph.
        Each element should be an instance of the `Node`-class.

    adj : numpy.ndarray, shape=(N, N)
        The adjacency-matrix for the graph. Nonzero entries indicate
        the presence of edges.
    
    """
    import random
    num_connected_components_list = []
    num = 100
    while len(num_connected_components_list) < num or num_connected_components_list[-num] != \
            num_connected_components_list[-1]:
        node_relations = connected_components(graph)  # returns node_relations : dict{label : [node_1, node_2, . . .],
        #                               label2 : [node4, node7, ...]}
        # count number of connections between nodes
        num_connected_components_list.append(len(node_relations.keys()))
        # update node_relations
        random_id = random.randint(0, len(graph) - 1)  # Return a random integer x such that 0 <= x <= N-1.
        random_node = graph[random_id]
        neighbors_of_random_node = [graph[i] for i in random_node.neighbors]
        # print("adjacency matrix: ", adj)
        propagate_label(random_node, neighbors_of_random_node, adj)
    fig, ax = plt.subplots()
    ax.plot(num_connected_components_list)
    ax.set_xlabel("# of iterations")
    ax.set_ylabel("# of connected components")
    plt.show()


def cluster_photos(img_paths):
    """
    Call the adjacency_graph function that takes in a list of image-paths, and 
    returns a list of nodes and an adjacency graph (ndarray) that describes the weighted connections 
    between your nodes.

    Call the whispers function, which takes in the list of nodes and an adjacency graph. 
    It will update the nodes' labels.

    Use the list of nodes to automatically organize the photos into subfolders according to these 
    connected component groupings
    """
    graph, adj = adjacency_graph(img_paths)
    whispers(graph, adj)
    node_relations = connected_components(graph)

    from pathlib import Path
    import shutil
    root = Path(".")
    for label, nodes in node_relations.items():
        # print("label", label)
        # show image
        img = plt.imread(nodes[0].file_path)
        imgplot = plt.imshow(img)
        plt.show()

        name = input("Input the name of the person: ")
        new_dir = root / "Clustered_Images" / name
        new_dir.mkdir(exist_ok=True)
        for node in nodes:  # copy image to correct subfolder
            old_img_path = node.file_path
            dst_dir = str(new_dir)
            shutil.copy(old_img_path, dst_dir)
            # old_img_path.rename(root / new_dir / old_img_path.name)
