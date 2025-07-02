import pandas as pd
import numpy as np
import networkx as nx
from collections import deque
from functools import wraps
from IPython.display import display
import matplotlib.pyplot as plt
import copy
import random
import time

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

def eg(gtfs_data, L):
    start_p = time.time()
    P = P_space(gtfs_data, L,
                start_hour=5,
                end_hour=24,
                mode="Rail")
    end_p = time.time()
    # print(f"Time taken for P_space: {end_p - start_p:.2f} seconds")

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

def random_node_removal(g, G, num_to_remove, seed=None, verbose=False):
    """
    Removes nodes from the graph in a random order and tracks the impact on global efficiency.

    Parameters:
        G (networkx.Graph): The input graph to modify (passed by reference).
        num_to_remove (int): Number of nodes to remove from the graph.
        seed (int, optional): Seed for reproducible random node selection.
        verbose (bool): Whether to print detailed logs during execution.

    Returns:
        original_efficiency (float): The initial global efficiency before any removals.
        efficiencies (list of float): Normalized global efficiencies after each removal.
        num_removed (list of int): Step count corresponding to each node removal.
        removed_nodes (list of node): List of nodes removed in the order of removal.
        removal_times (list of float): Time taken (in seconds) for each removal step.
    """
    if seed is not None:
        random.seed(seed)
    removal_nodes = random.sample(list(G.nodes()), num_to_remove)

    if verbose:
        print(f"Random removal order: {removal_nodes}")

    original_efficiency = eg(g, G)
    efficiencies = []
    num_removed = []
    removed_nodes = []
    removal_times = []

    for i, node in enumerate(removal_nodes):
        start_time = time.perf_counter()

        G.remove_node(node)
        removed_nodes.append(node)

        try:
            eff = eg(g, G)
        except Exception as e:
            if verbose:
                print(f"Error after {i + 1} removals: {e}")
            break

        elapsed = time.perf_counter() - start_time
        normalized_eff = eff / original_efficiency

        efficiencies.append(normalized_eff)
        num_removed.append(i + 1)
        removal_times.append(round(elapsed, 4))

        if verbose:
            print(f"Removed {node} → Normalized Efficiency: {normalized_eff:.4f}")
            print(f"Time taken: {elapsed:.4f} seconds\n")

    return original_efficiency, efficiencies, num_removed, removed_nodes, removal_times


def targeted_node_removal(g, G, num_to_remove, verbose=False):
    """
    Removes nodes from the graph using a greedy strategy that selects the node whose removal
    results in the largest drop in global efficiency at each step.

    Parameters:
        G (networkx.Graph): The input graph to modify (passed by reference).
        num_to_remove (int): Number of nodes to remove from the graph.
        verbose (bool): Whether to print detailed logs including time per step.

    Returns:
        original_efficiency (float): The initial global efficiency before any removals.
        efficiencies (list of float): Normalized global efficiencies after each removal.
        num_removed (list of int): Step count corresponding to each node removal.
        removed_nodes (list of node): List of nodes removed in the order of removal.
        removal_times (list of float): Time taken (in seconds) for each full removal step,
                                       including node evaluation and removal.
    """
    original_efficiency = eg(g, G)
    efficiencies = []
    num_removed = []
    removed_nodes = []
    removal_times = []

    for step in range(num_to_remove):
        start_time = time.perf_counter()

        current_eff = eg(g, G)
        max_drop = -1
        best_node = None

        for node in G.nodes():
            temp_G = G.copy()
            temp_G.remove_node(node)

            try:
                eff = eg(g, temp_G)
            except:
                continue

            drop = current_eff - eff
            if drop > max_drop:
                max_drop = drop
                best_node = node

        if best_node is None:
            if verbose:
                print("No valid node to remove.")
            break

        G.remove_node(best_node)
        removed_nodes.append(best_node)

        try:
            eff = eg(g, G)
        except Exception as e:
            if verbose:
                print(f"Error after {step + 1} removals: {e}")
            break

        elapsed = time.perf_counter() - start_time
        normalized_eff = eff / original_efficiency

        efficiencies.append(normalized_eff)
        num_removed.append(step + 1)
        removal_times.append(round(elapsed, 4))

        if verbose:
            print(f"Step {step + 1}: Removed {best_node} → Normalized Efficiency: {normalized_eff:.4f}")
            print(f"Time taken: {elapsed:.4f} seconds\n")

    return original_efficiency, efficiencies, num_removed, removed_nodes, removal_times


def betweenness_node_removal(g, G, num_to_remove, verbose=False):
    """
    Removes nodes from the graph in descending order of weighted betweenness centrality
    and tracks the impact on global efficiency.

    Parameters:
        G (networkx.Graph): The input graph to modify (passed by reference).
        num_to_remove (int): Number of nodes to remove from the graph.
        verbose (bool): Whether to print detailed logs during execution.

    Returns:
        original_efficiency (float): The initial global efficiency before any removals.
        efficiencies (list of float): Normalized global efficiencies after each removal.
        num_removed (list of int): Step count corresponding to each node removal.
        removed_nodes (list of node): List of nodes removed in the order of removal.
        removal_times (list of float): Time taken (in seconds) for each removal step.
    """
    original_efficiency = eg(g, G)
    efficiencies = []
    num_removed = []
    removed_nodes = []
    removal_times = []

    for step in range(num_to_remove):
        start_time = time.perf_counter()

        # Compute weighted betweenness centrality
        try:
            centrality = nx.betweenness_centrality(G, weight='duration_avg')
        except Exception as e:
            if verbose:
                print(f"Failed to compute betweenness at step {step + 1}: {e}")
            break

        if not centrality:
            if verbose:
                print("No centrality values computed. Possibly empty graph.")
            break

        # Select node with highest centrality
        node_to_remove = max(centrality, key=centrality.get)

        G.remove_node(node_to_remove)
        removed_nodes.append(node_to_remove)

        try:
            eff = eg(g, G)
        except Exception as e:
            if verbose:
                print(f"Error after removing {node_to_remove}: {e}")
            break

        elapsed = time.perf_counter() - start_time
        normalized_eff = eff / original_efficiency

        efficiencies.append(normalized_eff)
        num_removed.append(step + 1)
        removal_times.append(round(elapsed, 4))

        if verbose:
            print(f"Step {step + 1}: Removed {node_to_remove} (Centrality: {centrality[node_to_remove]:.4f})")
            print(f"Normalized Efficiency: {normalized_eff:.4f}")
            print(f"Time taken: {elapsed:.4f} seconds\n")

    return original_efficiency, efficiencies, num_removed, removed_nodes, removal_times


def simulate_fixed_node_removal_efficiency(
    g,
    L_graph,
    num_to_remove=None,
    pct_to_remove=None,  # priority over num_to_remove
    method='random', # random or targeted or betweenness
    seed=None,
    verbose=False
):
    """
    Simulates the impact of fixed sequential node removals on the global efficiency of a graph.
    
    Parameters:
        L_graph (networkx.Graph): The subgraph from which nodes will be removed.
        num_to_remove (int, optional): Number of nodes to remove. Ignored if percentage is given.
        pct_to_remove (int, optional): Percentage of nodes to remove (between 1 and 100).
        seed (int, optional): Random seed for node selection.
        verbose (bool): Whether to print progress and debug information.
    """
    G = copy.deepcopy(L_graph)
    total_nodes = G.number_of_nodes()

    if pct_to_remove is not None:
        if not (1 <= pct_to_remove <= 100):
            raise ValueError("Percentage must be an integer between 1 and 100.")
        num_to_remove = int(total_nodes * (pct_to_remove / 100))
    elif num_to_remove is None:
        raise ValueError("You must specify either num_to_remove or percentage.")

    if method == "random":
        return random_node_removal(g, G, num_to_remove, seed, verbose)
    elif method == "targeted":
        return targeted_node_removal(g, G, num_to_remove, verbose)
    elif method == "betweenness":
        return betweenness_node_removal(g, G, num_to_remove, verbose)
    else:
        raise ValueError("Invalid method. Choose 'random' or 'targeted'.")

def run_removal_simulations(g, subgraphs_by_size, num_to_remove=None, pct_to_remove=None, method='random', seed=42, verbose=False):
    """
    Run node removal simulations across all subgraphs grouped by size and collect efficiency and timing metrics.

    Parameters:
        g (networkx.Graph): The original graph used to compute baseline efficiency.
        subgraphs_by_size (dict): A dictionary where each key is a subgraph size and each value is a list of subgraphs (networkx.Graph).
        num_to_remove (int): Number of nodes to remove from each subgraph. Default is 5.
        seed (int): Random seed for reproducibility. Default is 42.
        verbose (bool): Whether to print detailed output during simulation. Default is False.

    Returns:
        pd.DataFrame: A DataFrame where each row corresponds to one subgraph simulation and contains:
            - graph_index: Index of the subgraph within its group
            - num_nodes: Number of nodes in the subgraph
            - num_edges: Number of edges in the subgraph
            - runtime_seconds: Total time taken for the simulation
            - original_efficiency: Efficiency before any node removal
            - final_efficiency: Efficiency after all removals
            - efficiency_after_each_removal: List of normalized efficiencies after each removal (excluding original)
            - removed_nodes: List of removed node IDs
            - removal_times: List of cumulative times after each removal
            - eff_after_{i}: Normalized efficiency after i-th removal, where i=0 is the original
    """
    results = []

    for size, graphs in subgraphs_by_size.items():
        for idx, L in enumerate(graphs):
            start = time.perf_counter()
            # try:
            original_efficiency, efficiencies, num_removed, removed_nodes, removal_times = simulate_fixed_node_removal_efficiency(
                g,
                L_graph=L,
                num_to_remove=num_to_remove,
                pct_to_remove=pct_to_remove, # priority over num_to_remove
                method=method, # random or targeted
                 seed=seed,
                verbose=verbose
                )
            # except Exception as e:
            #     # print(f"Error on graph size {size}, index {idx}: {e}")
            #     continue
            end = time.perf_counter()
            elapsed = end - start

            result = {
                "graph_index": idx,
                "num_nodes": L.number_of_nodes(),
                "num_edges": L.number_of_edges(),
                "runtime_seconds": round(elapsed, 3),
                "original_efficiency": original_efficiency,
                "final_efficiency": efficiencies[-1] if efficiencies else None,
                "efficiency_after_each_removal": efficiencies[0:] if len(efficiencies) > 1 else [],
                "removed_nodes": removed_nodes,
                "removal_times": removal_times
            }

            for i, eff in enumerate(efficiencies):
                result[f"eff_after_{i}"] = eff

            results.append(result)

    return pd.DataFrame(results)