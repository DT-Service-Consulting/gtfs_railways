from gtfs_railways.functions.utils import mode_from_string, get_routes_for_mode, get_color_per_route

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import copy
import random
import time


def get_all_GTC_refactored_(L_space, P_space, k, wait_pen, transfer_pen):
    # Precompute all attributes
    P_veh = nx.get_edge_attributes(P_space, "veh")
    P_wait = nx.get_edge_attributes(P_space, "avg_wait")
    L_dur = nx.get_edge_attributes(L_space, "duration_avg")
    L_dist = nx.get_edge_attributes(L_space, "d")

    # Precompute route directions as sets to avoid redundant set conversions
    routes_dirs = {}
    for e in P_veh:
        routes_dirs[e] = set()
        for ro in P_veh[e]:
            for dr in P_veh[e][ro]:
                routes_dirs[e].add(str(ro) + str(dr))

    # Compute all shortest paths using Dijkstra's algorithm
    paths = dict(nx.all_pairs_dijkstra_path(L_space, weight="duration_avg"))
    shortest_paths = {}

    for n1 in L_space.nodes:
        for target in L_space.nodes:
            if n1 == target:
                continue

            if n1 not in shortest_paths:
                shortest_paths[n1] = {}

            tt_paths = []
            only_tts = []

            # We consider just one path
            if target in paths[n1]:
                k_paths = [paths[n1][target]]
            else:
                k_paths = []

            for p in k_paths:
                possible_routes = routes_dirs.get((p[0], p[1]), set()).copy()

                dist = 0
                tt = 0
                wait = 0
                tf = 0
                t_stations = [n1]

                for l1, l2 in zip(p, p[1:]):
                    tt += L_dur[(l1, l2)]
                    dist += L_dist[(l1, l2)]

                    routes = routes_dirs.get((l1, l2), set())
                    possible_routes.intersection_update(routes)

                    if not possible_routes:
                        possible_routes = routes.copy()
                        tf += 1
                        t_stations.append(l1)

                t_stations.append(target)
                tt = round(tt / 60)

                for t1, t2 in zip(t_stations, t_stations[1:]):
                    wait += P_wait[(t1, t2)]

                wait = round(wait)
                transfer_cost = sum([transfer_pen[i] if i < len(transfer_pen) else transfer_pen[-1] for i in range(tf)])
                total_tt = tt + wait * wait_pen + transfer_cost

                only_tts.append(total_tt)
                tt_paths.append({
                    'path': p,
                    'GTC': total_tt,
                    'in_vehicle': tt,
                    'waiting_time': wait,
                    'n_transfers': tf,
                    'traveled_distance': dist
                })

            if k_paths:
                min_path_tt = min(only_tts)
                min_path = tt_paths[only_tts.index(min_path_tt)]
                shortest_paths[n1][target] = min_path
            else:
                shortest_paths[n1][target] = []

    return shortest_paths

def P_space_(g, L, mode, start_hour=5, end_hour=24, dir_indicator=None):
    '''
    Create P-space graph given:
    g: gtfs feed
    L: L-space
    Optional:
        start_hour: start hour considered when building L-space. Defaults to 5 am
        end_hour: end hour considered when building L-space. Defaults to midnight.
        dir_indicator: override which indicator direction_id, headsign, or shape_id should be used.
    '''

    # Validate inputs
    if not (0 <= start_hour < end_hour <= 24):
        raise AssertionError("Start/end hour must be in [0, 24] and start < end")
    if not (isinstance(start_hour, int) and isinstance(end_hour, int)):
        raise AssertionError("Start/end hours must be integers")

    time = end_hour - start_hour

    backup_colors = [
        '0000FF', '008000', 'FF0000', '00FFFF', 'FF00FF', 'FFFF00', '800080', 'FFC0CB', 'A52A2A',
        'FFA500', 'FF7F50', 'ADD8E6', '00FF00', 'E6E6FA', '40E0D0', 
        '006400', 'D2B48C', 'FA8072', 'FFD700'
    ]

    # Prepare graph and data
    P_G = nx.DiGraph()
    P_G.add_nodes_from(L.nodes(data=True))

    location = g.get_location_name()
    mode_val = mode_from_string(mode)
    routes = get_routes_for_mode(g, mode)

    colors = get_color_per_route(g, routes)
    L_edges = list(L.edges(data=True))

    # Precompute final route-to-color mapping
    route_colors = {}
    for i, r in enumerate(routes):
        c = colors.get(r)
        if not c or len(c) != 6:
            c = backup_colors[i % len(backup_colors)]
        route_colors[r] = '#' + c

    # Determine dir_indicator
    if not dir_indicator:
        dir_indicator = 'empty'
        if L_edges:
            sample_edge = L_edges[0][2]
            if sample_edge.get('direction_id'):
                dir_indicator = 'direction_id'
            elif sample_edge.get('headsign'):
                dir_indicator = 'headsign'
            elif sample_edge.get('shape_id'):
                dir_indicator = 'shape_id'

    # Main loop over routes
    for r_idx, r in enumerate(routes):
        color = route_colors[r]

        # Get all direction indicators for this route
        dirs = set()
        for _, _, edge_data in L_edges:
            if r in edge_data.get('route_I_counts', {}):
                for d in edge_data.get(dir_indicator, {}).keys():
                    dirs.add(d)

        # For each direction, build subgraph and add edges
        for d in dirs:
            sub = nx.DiGraph()
            sub_edges = []

            for a, b, edge_data in L_edges:
                if r in edge_data.get('route_I_counts', {}) and d in edge_data.get(dir_indicator, {}):
                    sub_edges.append((a, b, edge_data))

            if not sub_edges:
                continue

            sub.add_edges_from(sub_edges)

            for n1 in sub:
                try:
                    paths = nx.single_source_shortest_path(sub, n1)
                except nx.NetworkXError:
                    continue

                for n2, path in paths.items():
                    if n1 == n2 or len(path) < 2:
                        continue

                    path_set = set(path)

                    out_e = next(((a, b, c) for a, b, c in sub.out_edges(n1, data=True)
                                  if a in path_set and b in path_set), None)
                    in_e = next(((a, b, c) for a, b, c in sub.in_edges(n2, data=True)
                                 if a in path_set and b in path_set), None)

                    if not out_e or not in_e:
                        continue

                    veh_out = out_e[2]['route_I_counts'][r]
                    veh_in = in_e[2]['route_I_counts'][r]
                    veh = min(veh_out, veh_in)

                    veh_per_hour = veh / time
                    avg_wait = 60 / veh_per_hour / 2

                    if P_G.has_edge(n1, n2):
                        P_G[n1][n2]['edge_color'] = '#000000'
                        if r not in P_G[n1][n2]['veh']:
                            P_G[n1][n2]['veh'][r] = {d: veh_per_hour}
                        else:
                            P_G[n1][n2]['veh'][r][d] = veh_per_hour

                        tot_veh = sum(
                            v for route_data in P_G[n1][n2]['veh'].values()
                            for v in route_data.values()
                        )
                        P_G[n1][n2]['avg_wait'] = 60 / tot_veh / 2
                    else:
                        P_G.add_edge(n1, n2, veh={r: {d: veh_per_hour}},
                                     avg_wait=avg_wait, edge_color=color)

    return P_G

def eg_(g, L):
    P = P_space_(g, L,
                start_hour=5,
                end_hour=24,
                mode="Rail")

    sp = get_all_GTC_refactored_(L, P, 3, 2, [5])
    
    eg = 0
    for n1 in sorted(L.nodes()):
        for n2 in sorted(L.nodes()):
            if n1 != n2:
                if sp[n1][n2]:
                    tt = sp[n1][n2]["GTC"]
                    eg += 1 / tt

    return eg / (L.number_of_nodes() * (L.number_of_nodes() - 1))
    

def random_node_removal(g, G, num_to_remove, seed=None, verbose=False):
    """
    Removes edges connected to nodes in a random order and tracks the impact on global efficiency.
    The nodes themselves remain in the graph.

    Parameters:
        G (networkx.Graph): The input graph to modify (passed by reference).
        num_to_remove (int): Number of nodes whose edges will be removed.
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

    original_efficiency = eg_(g, G)
    efficiencies = []
    num_removed = []
    removed_nodes = []
    removal_times = []

    for i, node in enumerate(removal_nodes):
        start_time = time.perf_counter()

        # Skip if node is already isolated (no edges)
        if G.in_degree(node) == 0 and G.out_degree(node) == 0:
            if verbose:
                print(f"Step {i + 1}: Node {node} already isolated, skipping.")
            continue

        edges_to_remove = list(G.in_edges(node)) + list(G.out_edges(node))
        G.remove_edges_from(edges_to_remove)
        removed_nodes.append(node)

        try:
            eff = eg_(g, G)
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


def targeted_node_removal(g, G, num_to_remove, verbose=False):
    """
    Removes edges connected to nodes using a greedy strategy that selects the node whose edge
    removal results in the largest drop in global efficiency at each step. The node itself is retained.

    Parameters:
        G (networkx.Graph): The input graph to modify (passed by reference).
        num_to_remove (int): Number of nodes whose edges will be removed.
        verbose (bool): Whether to print detailed logs including time per step.

    Returns:
        original_efficiency (float): The initial global efficiency before any removals.
        efficiencies (list of float): Normalized global efficiencies after each removal.
        num_removed (list of int): Step count corresponding to each edge-removal step.
        removed_nodes (list of node): List of nodes whose edges were removed.
        removal_times (list of float): Time taken (in seconds) for each step.
    """
    original_efficiency = eg_(g, G)
    efficiencies = []
    num_removed = []
    removed_nodes = []
    removal_times = []

    removals_done = 0
    step = 0

    while removals_done < num_to_remove:
        start_time = time.perf_counter()
        step += 1

        current_eff = eg_(g, G)
        max_drop = -1
        best_node = None

        for node in G.nodes():
            # Skip isolated nodes
            if G.in_degree(node) == 0 and G.out_degree(node) == 0:
                continue

            temp_G = G.copy()
            edges_to_remove = list(temp_G.in_edges(node)) + list(temp_G.out_edges(node))
            temp_G.remove_edges_from(edges_to_remove)

            try:
                eff = eg_(g, temp_G)
            except:
                continue

            drop = current_eff - eff
            if drop > max_drop:
                max_drop = drop
                best_node = node

        if best_node is None:
            if verbose:
                print("No valid node to isolate. Stopping early.")
            break

        edges_to_remove = list(G.in_edges(best_node)) + list(G.out_edges(best_node))
        G.remove_edges_from(edges_to_remove)
        removed_nodes.append(best_node)
        removals_done += 1

        try:
            eff = eg_(g, G)
        except Exception as e:
            if verbose:
                print(f"Error after {removals_done} removals: {e}")
            break

        elapsed = time.perf_counter() - start_time
        normalized_eff = eff / original_efficiency

        efficiencies.append(normalized_eff)
        num_removed.append(removals_done)
        removal_times.append(round(elapsed, 4))

        if verbose:
            print(f"Step {step}: Removed edges of {best_node} → Normalized Efficiency: {normalized_eff:.4f}")
            print(f"Time taken: {elapsed:.4f} seconds\n")

    return original_efficiency, efficiencies, num_removed, removed_nodes, removal_times


def betweenness_node_removal(g, G, num_to_remove, verbose=False):
    """
    Removes edges connected to nodes in descending order of weighted betweenness centrality
    and tracks the impact on global efficiency. The nodes themselves are retained.

    Parameters:
        G (networkx.Graph): The input graph to modify (passed by reference).
        num_to_remove (int): Number of nodes whose edges will be removed.
        verbose (bool): Whether to print detailed logs during execution.

    Returns:
        original_efficiency (float): The initial global efficiency before any removals.
        efficiencies (list of float): Normalized global efficiencies after each removal.
        num_removed (list of int): Step count corresponding to each edge-removal step.
        removed_nodes (list of node): List of nodes whose edges were removed in order.
        removal_times (list of float): Time taken (in seconds) for each step.
    """
    original_efficiency = eg_(g, G)
    efficiencies = []
    num_removed = []
    removed_nodes = []
    removal_times = []

    removals_done = 0
    step = 0

    while removals_done < num_to_remove:
        step += 1
        start_time = time.perf_counter()

        try:
            centrality = nx.betweenness_centrality(G, weight='duration_avg')
        except Exception as e:
            if verbose:
                print(f"Failed to compute betweenness at step {step}: {e}")
            break

        # Remove isolated nodes from consideration
        centrality = {
            node: cent for node, cent in centrality.items()
            if (G.is_directed() and (G.in_degree(node) > 0 or G.out_degree(node) > 0)) or
               (not G.is_directed() and G.degree(node) > 0)
        }

        if not centrality:
            if verbose:
                print("No non-isolated nodes left to remove.")
            break

        node_to_remove = max(centrality, key=centrality.get)

        # Remove all edges connected to the node
        if G.is_directed():
            edges_to_remove = list(G.in_edges(node_to_remove)) + list(G.out_edges(node_to_remove))
            G.remove_edges_from(edges_to_remove)
        else:
            G.remove_edges_from(list(G.edges(node_to_remove)))

        removed_nodes.append(node_to_remove)
        removals_done += 1

        try:
            eff = eg_(g, G)
        except Exception as e:
            if verbose:
                print(f"Error after removing edges of {node_to_remove}: {e}")
            break

        elapsed = time.perf_counter() - start_time
        normalized_eff = eff / original_efficiency

        efficiencies.append(normalized_eff)
        num_removed.append(removals_done)
        removal_times.append(round(elapsed, 4))

        if verbose:
            print(f"Step {step}: Removed edges of {node_to_remove} (Centrality: {centrality[node_to_remove]:.4f})")
            print(f"Normalized Efficiency: {normalized_eff:.4f}")
            print(f"Time taken: {elapsed:.4f} seconds\n")

    return original_efficiency, efficiencies, num_removed, removed_nodes, removal_times


def simulate_fixed_node_removal_efficiency(
    g,
    L_graph,
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
            raise ValueError("Percentage must be an integer between 1 and 100.")
        num_to_remove = int(total_nodes * (pct_to_remove / 100))
    elif num_to_remove is None:
        raise ValueError("You must specify either num_to_remove or percentage.")

    if num_to_remove > total_nodes:
        print(f"Requested number of nodes to remove ({num_to_remove}) exceeds total nodes ({total_nodes}).")
        num_to_remove = max(total_nodes - 2, 1)
        if verbose:
            print(f"Adjusting number of nodes to remove to {num_to_remove}.")

    if method == "random":
        return random_node_removal(g, G, num_to_remove, seed, verbose)
    elif method == "targeted":
        return targeted_node_removal(g, G, num_to_remove, verbose)
    elif method == "betweenness":
        return betweenness_node_removal(g, G, num_to_remove, verbose)
    else:
        raise ValueError("Invalid method. Choose 'random' or 'targeted' or 'betweenness'.")


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
            try:
                original_efficiency, efficiencies, num_removed, removed_nodes, removal_times = simulate_fixed_node_removal_efficiency(
                    g,
                    L_graph=L,
                    num_to_remove=num_to_remove,
                    pct_to_remove=pct_to_remove, # priority over num_to_remove
                    method=method, # random or targeted
                    seed=seed,
                    verbose=verbose
                )
            except Exception as e:
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
                "efficiency_after_each_removal": efficiencies[0:] if len(efficiencies) > 1 else [],
                "removed_nodes": removed_nodes,
                "removal_times": removal_times
            }

            for i, eff in enumerate(efficiencies):
                result[f"eff_after_{i}"] = eff

            results.append(result)

    return pd.DataFrame(results)