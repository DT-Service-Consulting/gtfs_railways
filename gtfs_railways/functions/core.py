"""
This file contains all the core functions, which do not depend on the optimization version of the library.
"""

import time
import copy
import pickle
import os
from gtfspy import import_gtfs, gtfs, networks

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
    for n1 in sorted(L.nodes()):
        for n2 in sorted(L.nodes()):
            if n1 != n2:
                try:
                    entry_list = sp[n1][n2]
                    if isinstance(entry_list, list) and entry_list:
                        gtc = entry_list[0].get("GTC")
                        if gtc and gtc > 0:
                            eg += 1 / gtc
                except KeyError:
                    continue  # skip if no path
    return eg / (L.number_of_nodes() * (L.number_of_nodes() - 1))

def simulate_fixed_node_removal_efficiency(
    L_graph,
    original_efficiency,
    removal_nodes,
    gtfs_data=None,
    verbose=True
):

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
            original_efficiency, efficiencies, num_removed, removed_nodes, removal_times = (
                simulate_fixed_node_removal_efficiency(
                g,
                L_graph=L,
                num_to_remove=num_to_remove,
                pct_to_remove=pct_to_remove, # priority over num_to_remove
                method=method, # random or targeted
                 seed=seed,
                verbose=verbose
                ))
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


