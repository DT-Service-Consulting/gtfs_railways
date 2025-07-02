import networkx as nx
import random
from collections import deque
import time
from functools import wraps
import random
import matplotlib.pyplot as plt

from gtfs_railways.functions.utils import P_space, get_all_GTC_refactored

def compute_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Function '{func.__name__}' completed.")
        print(f"Execution time: {end_time - start_time:.2f} seconds\n")
        return result
    return wrapper

def get_random_removal_nodes(graph, num_to_remove, seed=None):
    """
    Returns a list of nodes randomly selected from G for removal.

    Parameters:
    - G: NetworkX graph
    - num_to_remove: Number of nodes to remove (int)
    - seed: Optional random seed for reproducibility (int or None)

    Returns:
    - List of node IDs selected for removal
    """
    if num_to_remove > graph.number_of_nodes() - 2:
        raise ValueError("Cannot remove all or almost all nodes. Reduce 'num_to_remove'.")

    if seed is not None:
        random.seed(seed)

    return random.sample(list(graph.nodes()), num_to_remove)

def eg(L, gtfs_data):
    start_p = time.time()
    P = P_space(gtfs_data, L,
                start_hour=5,
                end_hour=24,
                mode="Rail")
    end_p = time.time()
    print(f"Time taken for P_space: {end_p - start_p:.2f} seconds")

    sp = get_all_GTC_refactored(L, P, 3, 2, [5])
    
    eg = 0
    for n1 in sorted(L.nodes()):
        for n2 in sorted(L.nodes()):
            if n1 != n2:
                if sp[n1][n2]:
                    tt = sp[n1][n2]["GTC"]
                    eg += 1 / tt

    return eg / (L.number_of_nodes() * (L.number_of_nodes() - 1))

def simulate_fixed_node_removal_efficiency(
    L_graph,
    original_efficiency,
    removal_nodes,
    gtfs_data=None,
    verbose=True
):
    import copy

    G = copy.deepcopy(L_graph)
    efficiencies = []
    num_removed = []
    removed_nodes = []

    if verbose:
        print(f"Original efficiency: {original_efficiency:.4f}")

    for i, node_to_remove in enumerate(removal_nodes):
        if node_to_remove not in G:
            if verbose:
                print(f"Node {node_to_remove} not in graph, skipping.")
            continue

        if verbose:
            print(f"\nIteration {i+1}: Removing node → {node_to_remove}")

        G.remove_node(node_to_remove)
        removed_nodes.append(node_to_remove)

        try:
            eff = eg(G, gtfs_data)
        except Exception as e:
            if verbose:
                print(f"Error after removing {i+1} nodes: {e}")
            break

        normalized_eff = eff / original_efficiency
        efficiencies.append(normalized_eff)
        num_removed.append(i + 1)

        if verbose:
            print(f"Removed {i+1} node(s) → Normalized Efficiency: {normalized_eff:.4f}\n")

    return efficiencies, num_removed, removed_nodes

def plot_efficiency_results(num_removed, efficiencies, title="Impact of Node Removal on Normalized Network Efficiency"):
    """
    Plots the change in normalized efficiency as nodes are removed.

    Parameters:
    - num_removed: List of number of nodes removed
    - efficiencies: Corresponding list of normalized efficiencies
    - title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(num_removed, efficiencies, marker='o')
    plt.xlabel("Number of Nodes Removed")
    plt.ylabel("Normalized Efficiency")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()