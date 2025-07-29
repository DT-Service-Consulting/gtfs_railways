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
from statistics import mean
from scipy import stats # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np
from bokeh.plotting import figure, show, from_networkx # type: ignore
from bokeh.models import HoverTool, Circle, MultiLine, WheelZoomTool # type: ignore
from bokeh.tile_providers import Vendors # type: ignore
from pyproj import Transformer # type: ignore

# GTFS Modes
mode_name={0: 'Tram',
    1: 'Subway',
    2: 'Rail', 
    3: 'Bus', 
    4: 'Ferry',
    5: 'Cable Car',
    6: 'Gondola', 
    7: 'Funicular',
    8: 'Horse Carriage',
    9: 'Intercity Bus',
    10: 'Commuter Train',
    11: 'Trolleybus', 
    12: 'Monorail', 
    99: 'Aircraft',
    100: 'Railway Service',
    101: 'High Speed Rail',
    102: 'Long Distance Trains',
    103: 'Inter Regional Rail Service',
    105: 'Sleeper Rail Service', 
    106: 'Regional Rail Service',
    107: 'Tourist Railway Service',
    108: 'Rail Shuttle', 
    109: 'Suburban Railway',
    200: 'CoachService', 
    201: 'InternationalCoach',
    202: 'NationalCoach',
    204: 'RegionalCoach',
    208: 'CommuterCoach',
    400: 'UrbanRailwayService',
    401: 'Metro', 
    402: 'Underground', 
    403: 'Urban Railway Service',
    405: 'Monorail', 
    700: 'BusService',
    701: 'RegionalBus',
    702: 'ExpressBus',
    704: 'LocalBus',
    715: 'Demand and Response Bus Service',
    717: 'Share Taxi Service', 
    800: 'TrolleybusService',
    900: 'TramService', 
    1000: 'WaterTransportService', 
    1100: 'AirService', 
    1300: 'TelecabinService', 
    1400: 'FunicularService', 
    1500: 'TaxiService',
    1501: 'CommunalTaxi',
    1700: 'MiscellaneousService',
    1701: 'CableCar', 
    1702: 'HorseDrawnCarriage'}
    
mode_code = {v: k for k, v in mode_name.items()}

def mode_to_string(mode):
    return mode_name[mode]

def mode_from_string(mode_str):
    return mode_code[mode_str]

#####################################################

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
    
def save_graph(G,path):
    #Rename nodes to 0..n
    G_res=nx.convert_node_labels_to_integers(G)
    #nx.write_gpickle(G_res,path)    

    with open(path, 'wb') as f:
        pickle.dump(G_res, f)
    
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
        percent_remaining (list of float): Percentage of nodes remaining at each step.
    """
    if seed is not None:
        random.seed(seed)

    total_nodes = G.number_of_nodes()  # Save original node count for percentage calculation
    removal_nodes = random.sample(list(G.nodes()), num_to_remove)

    if verbose:
        print(f"Random removal order: {removal_nodes}")

    # Compute initial global efficiency on original graph
    sp = sp_func(G)
    original_efficiency = efficiency_graph(g, sp)
    if verbose:
        print(f"Original Efficiency: {original_efficiency}")

    efficiencies = [1.0]
    num_removed = [0]
    percent_remaining = [100.0]  # Start at 100%
    removed_nodes = []
    removal_times = []

    for i, node in enumerate(removal_nodes):
        start_time = time.perf_counter()

        if G.degree(node) == 0:
            if verbose:
                print(f"Step {i + 1}: Node {node} already isolated, skipping.")
            efficiencies.append(efficiencies[-1])
            num_removed.append(num_removed[-1])
            percent_remaining.append(100 * (1 - num_removed[-1] / total_nodes))
            continue

        edges_to_remove = list(G.in_edges(node)) + list(G.out_edges(node))
        G.remove_edges_from(edges_to_remove)
        removed_nodes.append(node)

        try:
            sp = sp_func(G)
            eff = efficiency_graph(g, sp)
        except Exception as e:
            if verbose:
                print(f"Error after removing edges of {node}: {e}")
            break

        elapsed = time.perf_counter() - start_time
        normalized_eff = eff / original_efficiency

        efficiencies.append(normalized_eff)
        num_removed.append(num_removed[-1] + 1)
        percent_remaining.append(100 * (1 - num_removed[-1] / total_nodes))
        removal_times.append(round(elapsed, 4))

        if verbose:
            print(f"Removed edges of {node} → Normalized Efficiency: {normalized_eff:.4f}")
            print(f"Time taken: {elapsed:.4f} seconds\n")

    return original_efficiency, efficiencies, percent_remaining, removed_nodes, removal_times


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
    total_nodes = G.number_of_nodes()  # for percent remaining calculation
    sp = sp_func(G)
    original_efficiency = efficiency_graph(G, sp)
    if verbose:
        print(f"Original Efficiency: {original_efficiency:.4f}")

    efficiencies = [1.0]
    num_removed = [0]
    percent_remaining = [100.0]
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
        percent_remaining.append(100 * (1 - removals_done / total_nodes))
        removal_times.append(elapsed)

        if verbose:
            print(f"Step {step}: Removed edges of {best_node} → Normalized Efficiency: {normalized_eff:.4f}")
            print(f"Time taken: {elapsed:.4f} seconds\n")

    return original_efficiency, efficiencies, percent_remaining, removed_nodes, removal_times


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
    total_nodes = G.number_of_nodes()
    sp = sp_func(G)
    original_efficiency = efficiency_graph(G, sp)
    if verbose:
        print(f"Original Efficiency: {original_efficiency:.4f}")

    efficiencies = [1.0]
    num_removed = [0]
    percent_remaining = [100.0]
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
        percent_remaining.append(100 * (1 - removals_done / total_nodes))
        removal_times.append(elapsed)

        if verbose:
            print(f"Step {step}: Removed edges of {node_to_remove} (Centrality: {centrality[node_to_remove]:.4f})")
            print(f"Normalized Efficiency: {normalized_eff:.4f}")
            print(f"Time taken: {elapsed:.4f} seconds\n")

    return original_efficiency, efficiencies, percent_remaining, removed_nodes, removal_times


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

def get_efficiency_curves(
    subgraphs,
    versions_sp_func,
    target_size=15,
    num_seeds=5,
    seeds=None
):
    """
    Run node removal simulations across multiple versions and multiple seeds for specific subgraph sizes.

    Parameters:
        attributes: node/graph attributes used by sp_func
        subgraphs: dict of size -> list of subgraphs (networkx graphs)
        versions_sp_func: dict mapping version label to sp_func
        target_size: int or list of ints; subgraph sizes to run simulations on
        num_seeds: number of random seeds to run
        seeds: optional list of seeds; if None, generated internally

    Returns:
        dict of version -> dict of size -> list of dict with keys 'curve', 'removed_nodes', 'time'
        list of seeds used
    """
    import numpy as np
    import time

    if isinstance(target_size, int):
        target_sizes = [target_size]
    else:
        target_sizes = target_size

    # Validate target sizes exist in subgraphs
    for size in target_sizes:
        if size not in subgraphs:
            raise ValueError(f"Target size {size} not found in subgraphs")

    if seeds is None:
        seeds = list(np.random.SeedSequence(1234).generate_state(num_seeds))

    version_curves = {v: {size: [] for size in target_sizes} for v in versions_sp_func.keys()}

    for label, sp_func in versions_sp_func.items():
        print(f"Running simulations for version {label} on subgraph sizes {target_sizes}")

        for seed in seeds:
            start_time = time.time()

            # Select all subgraphs for all target sizes
            subgraphs_to_run = {size: subgraphs[size] for size in target_sizes}

            df = run_removal_simulations(
                subgraphs_by_size=subgraphs_to_run,
                pct_to_remove=50,
                method='random',
                seed=seed,
                sp_func=sp_func,
                verbose=False,
            )

            elapsed = time.time() - start_time

            # Collect results per size
            for size in target_sizes:
                df_size = df[df['num_nodes'] == size]

                for idx, row in df_size.iterrows():
                    version_curves[label][size].append({
                        'curve': [1.0] + row['efficiency_after_each_removal'],
                        'removed_nodes': row['removed_nodes'],
                        'time': elapsed,
                        'seed': seed,
                        'graph_index': row['graph_index']
                    })

            print(f"  Seed {seed} finished in {elapsed:.2f} seconds")

    return version_curves, seeds

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

def average_waiting_time_per_line_per_direction(P):
    routes={}
    for e in P.edges(data=True):
        for r in e[2]["veh"]:
            for d in e[2]["veh"][r]:
                if r not in routes:
                    routes[r]={}
                if d not in routes[r]:
                    routes[r][d]=[]
                routes[r][d].append(e[2]["veh"][r][d])

    #Average all number of vehicles per line per direction
    #Compute waiting time as half the headway
    for r in routes:
        for d in routes[r]:
            routes[r][d]=(60/mean(routes[r][d]))/2
    return routes
    
    
def average_speed_network(L):
    speeds=[]
    for e in L.edges(data=True):
        speeds.append((e[2]["d"]/1000)/(e[2]["duration_avg"]/3600))
    return mean(speeds)

def get_events(gtfs_feed,
               mode,
               start_hour=5, 
               end_hour=24):
               
    '''Gets all events for the most suitable day from GTFS data. Parameters:
    gtfs_feed: a gtfspy gtfs feed object
    mode: string corresponding to the transport mode that we want to consider
    start_hour: integer with the earliest hour we want to consider (in 0..24)
    end_hour: integer with the latest hour we want to consider (in 0..24, larger that start_hour)'''

    if not (start_hour>=0 and end_hour>=0):
        raise AssertionError("Start/end hour should be larger or equal to 0")
    if not (start_hour<=24 and end_hour<=24):
        raise AssertionError("Start/end hour should be smaller or equal to 24")
    if not (start_hour<end_hour):
        raise AssertionError("Start hour should be smaller than end hour")
    if not (isinstance(start_hour, int) and isinstance(end_hour, int)):
        raise AssertionError("Start/end hours should be int")
    if not (mode in mode_code and mode_from_string(mode) in gtfs_feed.get_modes()):
        raise AssertionError("Mode is not available for the city")    
    
    day_start=gtfs_feed.get_suitable_date_for_daily_extract(ut=True)
    range_start= day_start + start_hour*3600
    range_end = day_start + end_hour*3600-1
    
    print("Considering trips between %s and %s"%(gtfs_feed.unixtime_seconds_to_gtfs_datetime(range_start),
                                         gtfs_feed.unixtime_seconds_to_gtfs_datetime(range_end)))

    events = gtfs_feed.get_transit_events(start_time_ut=range_start,
                                end_time_ut=range_end,
                                route_type=mode_from_string(mode))
    return events

def save_gtc_to_pkl(gtc, filename):
    """
    Save the Global Transit Cost (GTC) data to a pickle file.

    Parameters:
        gtc: The GTC data to save (the output of get_all_GTC).
        filename: The name of the pickle file where the GTC data will be saved (default is "gtc_data.pkl").
    """
    with open(filename, 'wb') as f:
        pickle.dump(gtc, f)
    print(f"GTC data saved to {filename}")

def load_gtc_from_pkl(filename):
    """
    Load the Global Transit Cost (GTC) data from a pickle file.

    Parameters:
        filename: The name of the pickle file to load the GTC data from (default is "gtc_data.pkl").

    Returns:
        gtc: The loaded GTC data.
    """
    with open(filename, 'rb') as f:
        gtc = pickle.load(f)
    print(f"GTC data loaded from {filename}")
    return gtc

def betweenness_fit_revised(L, weight=None, confidence=.99, plot=False):
    raw = list(nx.betweenness_centrality(L,weight=weight,normalized=False).values())
    
    #data = [float(i)/sum(raw) for i in raw]
    data = [float(i)/max(raw) for i in raw]
    
    # Fitting an exponential distribution to the data
    params = stats.expon.fit(data)

    #print(params)
    #print("Lambda: %f"%(1/params[1]))

    fitted_distribution = stats.expon(*params)

    # Performing the K-S test
    ks_statistic, p_value = stats.kstest(data, fitted_distribution.cdf)

    # Printing the results
    #print("KS Statistic:", ks_statistic)
    #print("P-Value:", p_value)
    
    ###############
    
    if plot:
        plt.clf()
        # Create an array of values for the x-axis
        x = np.linspace(0, max(data), 1000)

        # Calculate the ECDF of the original data
        ecdf_data = np.arange(1, len(data) + 1) / len(data)

        # Calculate the CDF of the fitted exponential distribution
        cdf_fitted = stats.expon.cdf(x, *params)

        # Plot the ECDF of the original data
        plt.step(sorted(data), ecdf_data, label='ECDF of Original Data', color='b')

        # Plot the CDF of the fitted exponential distribution
        plt.plot(x, cdf_fitted, 'r-', lw=2, label='Fitted Exponential CDF')

        # Add labels and a legend
        plt.xlabel('Value')
        plt.ylabel('Probability')
        if p_value>(1-confidence):
            plt.title("Lambda: %f"%(1/params[1]))
        else:
            plt.title("Lambda: NaN")     
        
        plt.legend()

        # Show the plot
        # if weight:
        #     plt.savefig("plot_fit_w_%s.png"%plot,bbox_inches="tight")
        # else:
        #     plt.savefig("plot_fit_%s.png"%plot,bbox_inches="tight")
            
    
    if p_value>(1-confidence):
        return 1/params[1]
    else:
        return np.NaN
    
def meshedness(graph):
    """
    Calculates meshedness of a graph.
    """
    graph2=graph.to_undirected() #Convert graph to undirected
    e = graph2.number_of_edges()
    v = graph2.number_of_nodes()
    return (e - v + 1) / (2 * v - 5)

def plot_graph_highlight_node(G, highlight_nodes=None, back_map="OSM"):
    p = figure(
        height=600,
        width=950,
        toolbar_location='below',
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )
    p.toolbar.active_scroll = p.select_one(WheelZoomTool)

    # Convert lat/lon to Web Mercator if OSM is used
    pos_dict = {}
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857")
    for i, d in G.nodes(data=True):
        if back_map == "OSM":
            x2, y2 = transformer.transform(float(d["lat"]), float(d["lon"]))
        else:
            x2, y2 = float(d["lon"]), float(d["lat"])
        pos_dict[int(i)] = (x2, y2)

    graph = from_networkx(G, layout_function=pos_dict)

    # Prepare node renderer data (include all node attributes)
    node_data = {key: [] for key in list(next(iter(G.nodes(data=True)))[1].keys())}
    node_data['index'] = []
    node_data['color'] = []
    node_data['size'] = []

    for node, attrs in G.nodes(data=True):
        node_data['index'].append(node)
        for key in node_data.keys():
            if key in ['index', 'color', 'size']:
                continue
            node_data[key].append(attrs.get(key, None))
        node_data['color'].append("red" if node in highlight_nodes else "skyblue")
        node_data['size'].append(15 if node in highlight_nodes else 5)

    graph.node_renderer.data_source.data = node_data
    graph.node_renderer.glyph = Circle(size="size", fill_color="color")
    graph.edge_renderer.glyph = MultiLine(line_color="gray", line_alpha=0.4, line_width=1)

    # Dynamically create tooltips from available node attributes
    tooltips = [(k, f"@{k}") for k in node_data if k not in ['color', 'size']]
    p.add_tools(HoverTool(tooltips=tooltips))

    if back_map == "OSM":
        p.add_tile(Vendors.CARTODBPOSITRON)  # Use this directly instead of get_provider

    p.renderers.append(graph)
    show(p)

def plot_top_hubs(graph, top_n=10, seed=42):
    """
    Identify and visualize the top hubs, highlight them, and return a DataFrame with node details.
    
    Parameters:
        graph (nx.Graph): Input graph.
        top_n (int): Number of top hubs to highlight.
        seed (int): Random seed for layout consistency.
    """
    # Compute degree and identify top hubs
    core_degrees = dict(graph.degree())
    top_hubs = sorted(core_degrees.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_hub_nodes = [node for node, _ in top_hubs]
    
    # Create a DataFrame for the top hubs with node and degree information
    top_hub_df = pd.DataFrame(top_hubs, columns=['Node', 'Degree'])
    
    # Add 'Node Name'
    top_hub_df['Node Name'] = top_hub_df['Node'].apply(lambda node: graph.nodes[node].get('name', f"Node {node}"))
    
    # Select only the columns we want: Node, Node Name, Degree
    top_hub_df = top_hub_df[['Node', 'Node Name', 'Degree']]
    
    # Sort by degree in descending order
    top_hub_df = top_hub_df.sort_values(by="Degree", ascending=False)
    
    # Highlight the top hubs in the graph
    plot_graph_highlight_node(graph, highlight_nodes=top_hub_nodes)

    return top_hub_df