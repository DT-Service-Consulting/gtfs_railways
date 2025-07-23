from collections import defaultdict
from gtfs_railways.functions.utils import mode_from_string, get_routes_for_mode, get_color_per_route

import pandas as pd # type: ignore
import networkx as nx # type: ignore
import copy
import random
import time


def get_all_GTC(L_space, P_space, k, wait_pen, transfer_pen):
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

def P_space(g, L, mode, start_hour=5, end_hour=24, dir_indicator=None):
    if not (0 <= start_hour < end_hour <= 24):
        raise AssertionError("Start/end hour must be in [0, 24] and start < end")
    if not (isinstance(start_hour, int) and isinstance(end_hour, int)):
        raise AssertionError("Start/end hours must be integers")

    time = end_hour - start_hour

    P_G = nx.DiGraph()
    P_G.add_nodes_from(L.nodes(data=True))

    mode_val = mode_from_string(mode)
    routes = get_routes_for_mode(g, mode)
    L_edges = list(L.edges(data=True))

    # Detect direction indicator if not provided
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

    # Preprocess edges by route and direction
    route_dir_to_edges = defaultdict(list)
    for a, b, data in L_edges:
        route_counts = data.get('route_I_counts', {})
        dir_data = data.get(dir_indicator, {})
        for r in route_counts:
            for d in dir_data:
                route_dir_to_edges[(r, d)].append((a, b, data))

    for r in routes:
        directions = {d for (route_key, d) in route_dir_to_edges if route_key == r}

        for d in directions:
            sub_edges = route_dir_to_edges.get((r, d), [])
            if not sub_edges:
                continue

            sub = nx.DiGraph()
            sub.add_edges_from(sub_edges)

            for n1 in sub.nodes:
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

                    # Update P_G
                    if P_G.has_edge(n1, n2):
                        edge_data = P_G[n1][n2]
                        veh_dict = edge_data['veh']

                        if r not in veh_dict:
                            veh_dict[r] = {d: veh_per_hour}
                            edge_data['total_veh'] += veh_per_hour
                        else:
                            if d not in veh_dict[r]:
                                veh_dict[r][d] = veh_per_hour
                                edge_data['total_veh'] += veh_per_hour
                            else:
                                old_veh = veh_dict[r][d]
                                veh_dict[r][d] = veh_per_hour
                                edge_data['total_veh'] += (veh_per_hour - old_veh)

                        edge_data['avg_wait'] = 60 / edge_data['total_veh'] / 2
                    else:
                        P_G.add_edge(n1, n2,
                                     veh={r: {d: veh_per_hour}},
                                     total_veh=veh_per_hour,
                                     avg_wait=avg_wait)

    return P_G