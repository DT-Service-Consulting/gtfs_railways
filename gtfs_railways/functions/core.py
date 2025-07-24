"""
This file contains all the core functions, which do not depend on the optimization version of the library.
"""

import time
import copy
import pickle
import os
from gtfspy import import_gtfs, gtfs, networks # type: ignore
import networkx as nx # type: ignore
import random
import pandas as pd # type: ignore

def load_gtfs(imported_database_path, gtfs_path=None, name=""):
    if not os.path.exists(imported_database_path):  # reimport only if the imported database does not already exist
        print("Importing gtfs zip file")
        import_gtfs.import_gtfs([gtfs_path],  # input: list of GTFS zip files (or directories)
                                imported_database_path,  # output: where to create the new sqlite3 database
                                print_progress=True,  # whether to print progress when importing data
                                location_name=name)
    return gtfs.GTFS(imported_database_path)

def load_graph(path):
    #return nx.read_gpickle(path)
    with open(path, 'rb') as f:
        G = pickle.load(f)
        return G

def efficiency_graph(L, sp):
    eg = 0
    count = 0
    for n1 in sorted(L.nodes()):
        for n2 in sorted(L.nodes()):
            if n1 == n2:
                continue
            count += 1  # Always count this pair

            if n1 not in sp or n2 not in sp[n1]:
                # No path between n1 and n2
                continue  # efficiency += 0 implicitly

            val = sp[n1][n2]

            # Case 1: val is a dict with "GTC"
            if isinstance(val, dict) and "GTC" in val:
                gtc = val["GTC"]
                if gtc > 0:
                    eg += 1 / gtc

            # Case 2: val is a non-empty list of dicts with "GTC"
            elif isinstance(val, list):
                if len(val) == 0:
                    continue  # no path, efficiency += 0
                if isinstance(val[0], dict) and "GTC" in val[0]:
                    gtc = val[0]["GTC"]
                    if gtc > 0:
                        eg += 1 / gtc
                else:
                    raise ValueError(f"Unexpected structure at sp[{n1}][{n2}]: {val}")

            else:
                raise ValueError(f"Unexpected structure at sp[{n1}][{n2}]: {val}")

    if count == 0:
        raise ZeroDivisionError("No node pairs to evaluate.")

    return eg / count

def make_sp_func(attributes, get_all_GTC_func, P_space_func):
    def sp_func(G):
        P_graph = P_space_func(attributes, G, "Rail", 5, 24, None)
        return get_all_GTC_func(G, P_graph, k=1, wait_pen=2, transfer_pen=[5, 7, 9])
    return sp_func

def simulate_fixed_node_removal_efficiency(
    L_graph,
    sp_func,
    num_to_remove=None,
    pct_to_remove=None,  # priority over num_to_remove
    method='random',  # random or targeted or betweenness
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
            raise ValueError("Percentage must be between 1 and 100.")
        num_to_remove = int(total_nodes * (pct_to_remove / 100))
    elif num_to_remove is None:
        raise ValueError("Specify num_to_remove or pct_to_remove.")

    if num_to_remove > total_nodes:
        if verbose:
            print(f"Adjusting number of nodes to remove from {num_to_remove} to {total_nodes - 2}.")
        num_to_remove = max(total_nodes - 2, 1)

    if method == "random":
        # Here pass L_graph as g to random_node_removal (or whichever is expected)
        return random_node_removal(L_graph, G, num_to_remove, sp_func, seed, verbose)
    elif method == "targeted":
        return targeted_node_removal(L_graph, G, num_to_remove, sp_func, verbose)
    elif method == "betweenness":
        return betweenness_node_removal(L_graph, G, num_to_remove, sp_func, verbose)
    else:
        raise ValueError("Invalid method. Choose 'random', 'targeted', or 'betweenness'.")

def random_node_removal(g, G, num_to_remove, sp_func, seed=None, verbose=False):
    """
    Removes edges connected to nodes in a random order and tracks the impact on global efficiency.
    The nodes themselves remain in the graph.

    Parameters:
        g: Base attributes or data required by efficiency_graph (not the networkx graph).
        G (networkx.Graph): The input graph to modify (passed by reference).
        num_to_remove (int): Number of nodes whose edges will be removed.
        sp_func (callable): Function to compute shortest path structure given a graph.
        seed (int, optional): Seed for reproducible random node selection.
        verbose (bool): Whether to print detailed logs during execution.

    Returns:
        original_efficiency (float): The initial global efficiency before any removals.
        efficiencies (list of float): Normalized global efficiencies after each removal.
        num_removed (list of int): Step count corresponding to each edge-removal step.
        removed_nodes (list of node): List of nodes whose edges were removed in the order of removal.
        removal_times (list of float): Time taken (in seconds) for each removal step.
    """
    if seed is not None:
        random.seed(seed)

    removal_nodes = random.sample(list(G.nodes()), num_to_remove)

    if verbose:
        print(f"Random removal order: {removal_nodes}")

    # Compute initial global efficiency on original graph
    sp = sp_func(G)
    original_efficiency = efficiency_graph(g, sp)  # Use g here, not G
    if verbose:
        print(f"Original Efficiency: {original_efficiency}")

    efficiencies = [1.0]
    num_removed = [0]
    removed_nodes = []
    removal_times = []

    for i, node in enumerate(removal_nodes):
        start_time = time.perf_counter()

        # Skip if node already isolated
        if G.degree(node) == 0:
            if verbose:
                print(f"Step {i + 1}: Node {node} already isolated, skipping.")
            efficiencies.append(efficiencies[-1])
            num_removed.append(num_removed[-1])
            continue

        # Remove edges connected to node
        edges_to_remove = list(G.in_edges(node)) + list(G.out_edges(node))
        G.remove_edges_from(edges_to_remove)
        removed_nodes.append(node)

        try:
            sp = sp_func(G)
            eff = efficiency_graph(g, sp)  # Use g here, not G
        except Exception as e:
            if verbose:
                print(f"Error after removing edges of {node}: {e}")
            break

        elapsed = time.perf_counter() - start_time
        normalized_eff = eff / original_efficiency

        efficiencies.append(normalized_eff)
        num_removed.append(i + 1)
        removal_times.append(round(elapsed, 4))

        if verbose:
            print(f"Removed edges of {node} → Normalized Efficiency: {normalized_eff:.4f}")
            print(f"Time taken: {elapsed:.4f} seconds\n")

    return original_efficiency, efficiencies, num_removed, removed_nodes, removal_times


def targeted_node_removal(g, G, num_to_remove, sp_func, verbose=False):
    """
    Greedy edge removal: at each step, remove the edges of the node that causes the largest drop in global efficiency.

    Parameters:
        g (nx.DiGraph): Attributes graph (used inside sp_func).
        G (nx.DiGraph): The mutable working graph (will be changed in-place).
        num_to_remove (int): Number of node edge-removals to perform.
        sp_func (function): Function that recomputes the shortest-path structure from G.
        verbose (bool): Whether to print logs.

    Returns:
        original_efficiency (float): Efficiency before any removal.
        efficiencies (list): Normalized efficiencies.
        num_removed (list): Removal steps.
        removed_nodes (list): Nodes removed in order.
        removal_times (list): Step-wise durations.
    """
    # Compute initial SP and efficiency
    sp = sp_func(G)
    original_efficiency = efficiency_graph(G, sp)
    if verbose:
        print(f"Original Efficiency: {original_efficiency:.4f}")

    efficiencies = [1.0]
    num_removed = [0]
    removed_nodes = []
    removal_times = []

    removals_done = 0
    step = 0

    while removals_done < num_to_remove:
        step += 1
        start_time = time.perf_counter()

        sp = sp_func(G)
        current_eff = efficiency_graph(G, sp)
        max_drop = -1
        best_node = None

        for node in G.nodes():
            if G.degree(node) == 0:
                continue

            temp_G = G.copy()
            temp_G.remove_edges_from(list(temp_G.in_edges(node)) + list(temp_G.out_edges(node)))

            try:
                sp_temp = sp_func(temp_G)
                eff_temp = efficiency_graph(temp_G, sp_temp)
            except:
                continue

            drop = current_eff - eff_temp
            if drop > max_drop:
                max_drop = drop
                best_node = node

        if best_node is None:
            if verbose:
                print("No valid node to remove at step", step)
            break

        G.remove_edges_from(list(G.in_edges(best_node)) + list(G.out_edges(best_node)))
        removed_nodes.append(best_node)
        removals_done += 1

        try:
            sp_new = sp_func(G)
            eff = efficiency_graph(G, sp_new)
        except Exception as e:
            if verbose:
                print(f"Error after removing {best_node}: {e}")
            break

        normalized_eff = eff / original_efficiency
        elapsed = round(time.perf_counter() - start_time, 4)

        efficiencies.append(normalized_eff)
        num_removed.append(removals_done)
        removal_times.append(elapsed)

        if verbose:
            print(f"Step {step}: Removed edges of {best_node} → Normalized Efficiency: {normalized_eff:.4f}")
            print(f"Time taken: {elapsed:.4f} seconds\n")

    return original_efficiency, efficiencies, num_removed, removed_nodes, removal_times


def betweenness_node_removal(g, G, num_to_remove, sp_func, verbose=False):
    """
    Removes edges of nodes based on descending betweenness centrality,
    tracking the impact on global efficiency (normalized by initial value).

    Parameters:
        g (nx.DiGraph): Full original graph used for path logic, passed to sp_func.
        G (nx.DiGraph): Working graph to be modified (edges removed).
        num_to_remove (int): Number of nodes to process.
        sp_func (function): Function to generate shortest-path structure from G.
        verbose (bool): If True, prints per-step logs.

    Returns:
        original_efficiency (float): Efficiency before any removal.
        efficiencies (list): Normalized efficiencies at each step.
        num_removed (list): Step counter.
        removed_nodes (list): Node IDs removed (edge-deleted).
        removal_times (list): Time taken per step (in seconds).
    """
    sp = sp_func(G)
    original_efficiency = efficiency_graph(G, sp)
    if verbose:
        print(f"Original Efficiency: {original_efficiency:.4f}")

    efficiencies = [1.0]
    num_removed = [0]
    removed_nodes = []
    removal_times = []

    removals_done = 0
    step = 0

    while removals_done < num_to_remove:
        step += 1
        start_time = time.perf_counter()

        try:
            centrality = nx.betweenness_centrality(G, weight="duration_avg")
        except Exception as e:
            if verbose:
                print(f"Step {step} failed to compute centrality: {e}")
            break

        # Filter isolated nodes
        centrality = {
            node: score for node, score in centrality.items()
            if (G.in_degree(node) > 0 or G.out_degree(node) > 0)
        }

        if not centrality:
            if verbose:
                print("No non-isolated nodes left.")
            break

        node_to_remove = max(centrality, key=centrality.get)
        edges_to_remove = list(G.in_edges(node_to_remove)) + list(G.out_edges(node_to_remove))
        G.remove_edges_from(edges_to_remove)

        removed_nodes.append(node_to_remove)
        removals_done += 1

        try:
            sp = sp_func(G)
            eff = efficiency_graph(G, sp)
        except Exception as e:
            if verbose:
                print(f"Error evaluating efficiency after removing {node_to_remove}: {e}")
            break

        normalized_eff = eff / original_efficiency
        elapsed = round(time.perf_counter() - start_time, 4)

        efficiencies.append(normalized_eff)
        num_removed.append(removals_done)
        removal_times.append(elapsed)

        if verbose:
            print(f"Step {step}: Removed edges of {node_to_remove} (Centrality: {centrality[node_to_remove]:.4f})")
            print(f"Normalized Efficiency: {normalized_eff:.4f}")
            print(f"Time taken: {elapsed:.4f} seconds\n")

    return original_efficiency, efficiencies, num_removed, removed_nodes, removal_times


def run_removal_simulations(
    subgraphs_by_size,
    num_to_remove=None,
    pct_to_remove=None,
    method='random',
    seed=42,
    verbose=False,
    sp_func=None,
):
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

            try:
                original_efficiency, efficiencies, num_removed, removed_nodes, removal_times = (
                    simulate_fixed_node_removal_efficiency(
                        L_graph=L,
                        num_to_remove=num_to_remove,
                        pct_to_remove=pct_to_remove,
                        method=method,
                        seed=seed,
                        verbose=verbose,
                        sp_func=sp_func
                    )
                )
            except Exception as e:
                if verbose:
                    print(f"Error on graph size {size}, index {idx}: {e}")
                continue

            end = time.perf_counter()
            elapsed = end - start

            result = {
                "graph_index": idx,
                "num_nodes": L.number_of_nodes(),
                "num_edges": L.number_of_edges(),
                "runtime_seconds": round(elapsed, 3),
                "original_efficiency": original_efficiency,
                "final_efficiency": efficiencies[-1] if efficiencies else None,
                "efficiency_after_each_removal": efficiencies[1:] if len(efficiencies) > 1 else [],
                "removed_nodes": removed_nodes,
                "removal_times": removal_times
            }

            for i, eff in enumerate(efficiencies):
                result[f"eff_after_{i}"] = eff

            results.append(result)

    return pd.DataFrame(results)