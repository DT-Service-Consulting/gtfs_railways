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
from collections import deque
from functools import wraps

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
    
def extract_directed_subgraph(G, target_size, min_edges=3, seed=None):
    if seed is not None:
        random.seed(seed)

    nodes = list(G.nodes())
    random.shuffle(nodes)
    seen_node_sets = set()

    for seed_node in nodes:
        visited = set([seed_node])
        queue = deque([seed_node])

        while queue and len(visited) < target_size:
            current = queue.popleft()
            neighbors = list(G.successors(current))
            random.shuffle(neighbors)

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                if len(visited) == target_size:
                    break

        if len(visited) == target_size:
            node_tuple = tuple(sorted(visited))
            if node_tuple in seen_node_sets:
                continue

            subG = G.subgraph(visited).copy()
            if subG.number_of_edges() >= min_edges:
                seen_node_sets.add(node_tuple)
                yield subG
    
def generate_subgraph_batches(G, sizes=(5, 10, 15), num_per_size=10, seed=42, min_edges=3):
    all_subgraphs = {size: [] for size in sizes}
    rng = random.Random(seed)

    for size in sizes:
        count = 0
        attempt = 0
        while count < num_per_size and attempt < 1000:
            sub_seed = rng.randint(0, 100000)
            for subG in extract_directed_subgraph(G, size, min_edges, seed=sub_seed):
                all_subgraphs[size].append(subG)
                count += 1
                break
            attempt += 1

        if count < num_per_size:
            print(f"Warning: Only found {count} subgraphs of size {size} after {attempt} attempts.")
    
    return all_subgraphs

def save_subgraphs_by_size(subgraphs_by_size, base_dir="../data/pkl"):
    """
    Saves subgraphs grouped by size into separate folders under the specified base directory.

    Parameters:
        subgraphs_by_size (dict): Dictionary where keys are sizes (e.g. number of nodes)
                                  and values are lists of graphs.
        base_dir (str): Base directory where the folders and graphs will be saved.
    """
    os.makedirs(base_dir, exist_ok=True)

    for size, graphs in subgraphs_by_size.items():
        folder_path = os.path.join(base_dir, str(size))
        os.makedirs(folder_path, exist_ok=True)

        for i, graph in enumerate(graphs):
            file_path = os.path.join(folder_path, f"graph_{i}.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(graph, f)

    print(f"Saved all subgraphs by size into '{base_dir}'.")


def load_all_subgraphs(base_dir="../data/pkl", max_per_type=2):
    """
    Loads pickled subgraphs organized in subfolders named by number of nodes.

    Parameters:
        base_dir (str): Base directory containing subfolders of graphs.
        max_per_type (int or None): Max graphs to load per num_node type. 
                                    If None, loads all available.

    Returns:
        dict: {num_nodes: [graph1, graph2, ...]}
    """
    subgraphs_by_size = {}

    for size_folder in os.listdir(base_dir):
        size_path = os.path.join(base_dir, size_folder)
        if os.path.isdir(size_path) and size_folder.isdigit():
            size = int(size_folder)
            subgraphs = []

            pkl_files = sorted([
                f for f in os.listdir(size_path) 
                if f.endswith(".pkl") and f.startswith("graph_")
            ])

            if max_per_type is not None:
                pkl_files = pkl_files[:max_per_type]

            for filename in pkl_files:
                file_path = os.path.join(size_path, filename)
                with open(file_path, "rb") as f:
                    graph = pickle.load(f)
                    subgraphs.append(graph)

            subgraphs_by_size[size] = subgraphs

    return subgraphs_by_size

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
        """
        Returns the Global Travel Commute of all possible nodes.
        Parameters:
            G (networkx.Graph): is the most recent L_space graph.
        """
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
                "remova l_times": removal_times
            }

            for i, eff in enumerate(efficiencies):
                result[f"eff_after_{i}"] = eff

            results.append(result)

    return pd.DataFrame(results)

def get_runtime(version_label, run_func, subgraphs, 
                method, sp_func, seed, num_to_remove=None, pct_to_remove=None,
                target_sizes=None, verbose=False):
    """
    Runs `run_func` on subgraphs and measures runtimes per subgraph.

    Parameters:
    - version_label: str label for the run (e.g. "v1")
    - run_func: function to run (e.g. run_removal_simulations)
    - subgraphs: dict {size: [subgraph1, subgraph2, ...]}
    - num_to_remove, pct_to_remove, method, sp_func, seed: params to pass to run_func
    - target_sizes: list or int or None to specify sizes to run, else all sizes
    - verbose: whether to print timing info

    Returns:
    - runtimes: dict {size: [time_per_subgraph]}
    """
    runtimes = {}

    # Prepare sizes to run
    if target_sizes is None:
        sizes_to_run = sorted(subgraphs.keys())
    elif isinstance(target_sizes, int):
        sizes_to_run = [target_sizes]
    elif isinstance(target_sizes, (list, tuple)):
        sizes_to_run = list(target_sizes)
    else:
        raise TypeError("target_sizes must be None, int, or list/tuple of ints")

    for size in sizes_to_run:
        if size not in subgraphs:
            print(f"Warning: size {size} not found in subgraphs, skipping.")
            continue

        runtimes[size] = []
        if verbose:
            print(f"Starting runs for size {size}...")

        for idx, sg in enumerate(subgraphs[size]):
            start = time.perf_counter()
            run_func(
                {size: [sg]},
                num_to_remove=num_to_remove,
                pct_to_remove=pct_to_remove,
                method=method,
                sp_func=sp_func,
                seed=seed
            )
            end = time.perf_counter()
            duration = end - start
            runtimes[size].append(duration)

            if verbose:
                print(f"{version_label} - Size {size} - Subgraph {idx + 1}: {duration:.4f} seconds")

        if verbose:
            total_time = sum(runtimes[size])
            print(f"{version_label} - Size {size}: Total runtime {total_time:.4f} seconds\n")

    return runtimes

def compute_graph_features(L):
    """
    Compute various graph features of a subgraph L:
    - number of unique route-direction pairs
    - average in-degree
    - average out-degree
    - number of strongly connected components
    - size of largest strongly connected component
    - diameter (of largest SCC, or None if not connected)
    - average shortest path length (of largest SCC, or None if not connected)
    - average clustering coefficient
    - number of bridges (considering undirected version)

    Args:
        L: networkx.DiGraph with edges having 'route_I_counts' and optionally 'direction_id'

    Returns:
        dict: features with keys:
            'route_dir_pairs': int
            'avg_in_degree': float
            'avg_out_degree': float
            'num_scc': int
            'largest_scc_size': int
            'diameter': int or None
            'avg_shortest_path_len': float or None
            'avg_clustering_coeff': float
            'num_bridges': int
    """
    # Route-direction pairs
    route_dir_pairs = set()
    for _, _, edge_data in L.edges(data=True):
        route_counts = edge_data.get('route_I_counts', {})
        dir_dict = edge_data.get('direction_id', {})
        for route in route_counts.keys():
            if dir_dict:
                for direction in dir_dict.keys():
                    route_dir_pairs.add((route, direction))
            else:
                route_dir_pairs.add((route, None))

    # Degrees
    avg_in_degree = sum(dict(L.in_degree()).values()) / L.number_of_nodes() if L.number_of_nodes() > 0 else 0
    avg_out_degree = sum(dict(L.out_degree()).values()) / L.number_of_nodes() if L.number_of_nodes() > 0 else 0

    # Strongly connected components (SCCs)
    sccs = list(nx.strongly_connected_components(L))
    num_scc = len(sccs)
    largest_scc = max(sccs, key=len) if sccs else set()
    largest_scc_size = len(largest_scc)

    # Subgraph of largest SCC
    L_scc = L.subgraph(largest_scc).copy() if largest_scc else None

    # Diameter and average shortest path length (only if connected)
    diameter = None
    avg_shortest_path_len = None
    if L_scc and largest_scc_size > 1:
        # For directed graphs, check if strongly connected
        if nx.is_strongly_connected(L_scc):
            diameter = nx.diameter(L_scc.to_undirected())
            avg_shortest_path_len = nx.average_shortest_path_length(L_scc)
        else:
            # Diameter and avg shortest path undefined if not strongly connected
            diameter = None
            avg_shortest_path_len = None

    # Average clustering coefficient (undirected)
    L_undirected = L.to_undirected()
    avg_clustering_coeff = nx.average_clustering(L_undirected)

    # Bridges (cut edges) in undirected graph
    bridges = list(nx.bridges(L_undirected))
    num_bridges = len(bridges)

    return {
        'route_dir_pairs': len(route_dir_pairs),
        'avg_in_degree': avg_in_degree,
        'avg_out_degree': avg_out_degree,
        'num_scc': num_scc,
        'largest_scc_size': largest_scc_size,
        'diameter': diameter,
        'avg_shortest_path_len': avg_shortest_path_len,
        'avg_clustering_coeff': avg_clustering_coeff,
        'num_bridges': num_bridges
    }