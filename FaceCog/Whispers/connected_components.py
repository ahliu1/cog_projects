from Node import Node 
from collections import defaultdict

def connected_components(node_list):
    """
    Takes in your graph’s list of nodes
    , and returns a list of lists 
    – each inner list contains all nodes with a common label.

    Parameters:
    ----------
    node_list : List[Label_1, Label_2, . . . ]

    Returns
    -------
    common_dict : dict{label : [node_1, node_2, . . .]}
    """
    
    

    # DefaultDict[int, List[Node]] = defaultdict(list)
    common_dict = defaultdict(list)
    for item in node_list:
        common_dict[item.label].append(item)

    return common_dict



