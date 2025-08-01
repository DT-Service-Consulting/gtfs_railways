from gtfs_railways.legacy.utils import mode_from_string, get_routes_for_mode, get_color_per_route
import networkx as nx # type: ignore
from itertools import islice

# Method creating P-space with inputs gtfs-data (g), L-Space (L), and time period of L-space (time)
def P_space(g, L, mode, start_hour=5, end_hour=24, dir_indicator=None):
    '''
    Create P-space graph given:
    g: gtfs feed
    L: L-space
    Optional:
        start_hour: start hour considered when building L-space. Defaults to 5 am
    end_hour: end hour considered when building L-space. Defaults to midnight.
        dir_indicator: override which indicator direction_id,headsign,or shape_id should be used.
    '''
    
    if not (start_hour>=0 and end_hour>=0):
        raise AssertionError("Start/end hour should be larger or equal to 0")
    if not (start_hour<=24 and end_hour<=24):
        raise AssertionError("Start/end hour should be smaller or equal to 24")
    if not (start_hour<end_hour):
        raise AssertionError("Start hour should be smaller than end hour")
    if not (isinstance(start_hour, int) and isinstance(end_hour, int)):
        raise AssertionError("Start/end hours should be int")
    
    time=end_hour-start_hour
    
    # Create a list of backup colors
    backup_colors = ['0000FF', '008000', 'FF0000', '00FFFF', 'FF00FF', 'FFFF00', '800080', 'FFC0CB', 'A52A2A',
                          'FFA500', 'FF7F50', 'ADD8E6', '00FF00', 'E6E6FA', '40E0D0', 
                          '006400', 'D2B48C', 'FA8072', 'FFD700']

    # Create the P-space graph with the nodes from L-space
    P_G = nx.DiGraph()
    P_G.add_nodes_from(L.nodes(data=True))

    # Get a list of all routes of the network, with corresponding colors
    routes = get_routes_for_mode(g,mode)
    
    # Exception for Vienna metro network
    if(g.get_location_name() == 'vienna') and (mode_from_string(mode)==1):
        routes = routes[::2]
    
    colors = get_color_per_route(g, routes)
    
    if not dir_indicator:
        # Check to see if direction/headsign/shape exists
        dir_indicator = 'empty'
    
        edge_it = iter(L.edges(data=True))
        check_edge = next(edge_it, None)
        if check_edge:
            if check_edge[2]['direction_id']:
                dir_indicator = 'direction_id'    
            elif check_edge[2]['headsign']:
                dir_indicator = 'headsign'
            elif check_edge[2]['shape_id']:
                dir_indicator = 'shape_id'

            # Exception for Bilbao metro network
            if(g.get_location_name() == 'bilbao') and (mode_from_string(mode)==1):
                dir_indicator = 'headsign'

	    # Exception for Philadelphia network
            if(g.get_location_name() == 'philadelphia') and (mode_from_string(mode)==1):
                dir_indicator = 'headsign'

	    # Exception for Amsterdam network
            if(g.get_location_name() == 'amsterdam') and (mode_from_string(mode)==1):
                dir_indicator = 'headsign'

            # Exception for Paris RER
            if(g.get_location_name() == 'paris') and (mode_from_string(mode)==2):
                dir_indicator = 'headsign'

    # print("Using %s field as indicator for the direction of routes"%dir_indicator)

    # Loop through all routes
    for iter_n,r in enumerate(routes):
        
        # Get the route color (or a backup if unavailable)
        color = colors[r]
        if not color or len(color) != 6 \
           or (g.get_location_name() == 'nuremburg') and (mode_from_string(mode)==1): #All blue lines in nuremberg metro GTFS
            #color = next(backup_colors)
            color=backup_colors[iter_n%len(backup_colors)]
        
        # Create a set of the directions/headsigns/shapes for this route
        dirs = set()
        for e in L.edges(data=True):
            if r in e[2]['route_I_counts']:
                for h in e[2][dir_indicator].keys():
                    dirs.add(h)
        
        # Create a subgraph for each direction and add the edges to P-space
        for d in dirs:
            # Create an empty (directional) subgraph
            sub = nx.DiGraph()

            # Add all edges (and corresponding nodes) that are on this route and direction
            for e in L.edges(data=True):
                if r in e[2]['route_I_counts'] and d in e[2][dir_indicator]:
                    sub.add_edges_from([(e)])

            # Loop through all nodes in the subgraph that have paths between them
            for n1 in sub:
                for n2 in sub:
                    if n1 != n2 and nx.has_path(sub, n1, n2):

                        aux_out=[(a,b,c) for a,b,c in sub.out_edges(n1, data=True) if a in nx.shortest_path(sub,n1,n2) and b in nx.shortest_path(sub,n1,n2)]
                        out_e=aux_out[0]
                        
                        aux_in=[(a,b,c) for a,b,c in sub.in_edges(n2, data=True) if a in nx.shortest_path(sub,n1,n2) and b in nx.shortest_path(sub,n1,n2)]
                        in_e=aux_in[0]                            
                            
                        # Take the lowest number of vehicles between the two edges
                        veh_out = out_e[2]['route_I_counts'][r]
                        veh_in = in_e[2]['route_I_counts'][r]
                        veh = min(veh_out, veh_in)

                        # Compute the average waiting time
                        veh_per_hour = veh / time
                        max_wait = 60 / veh_per_hour
                        avg_wait = max_wait / 2

                        # If the edge already exists, append the values
                        if P_G.has_edge(n1, n2):

                            # Change the color to black to signify a shared edge
                            P_G[n1][n2]['edge_color'] = '#000000'

                            # Add the vehicles per hour for this route + direction to the wait_dir
                            if r not in P_G[n1][n2]['veh']:
                                P_G[n1][n2]['veh'][r] = {d: veh_per_hour}
                            else:
                                P_G[n1][n2]['veh'][r][d] = veh_per_hour

                            # Update the average waiting time to be the total of all routes' waiting times
                            tot_veh = 0
                            for ro in P_G[n1][n2]['veh']:
                                for di in P_G[n1][n2]['veh'][ro]:
                                    tot_veh = tot_veh + P_G[n1][n2]['veh'][ro][di]
                            P_G[n1][n2]['avg_wait'] = (60 / tot_veh) / 2

                        else:
                            P_G.add_edge(n1, n2, veh={r: {d: veh_per_hour}}, 
                                         avg_wait=avg_wait, edge_color='#'+str(color))
            
    return P_G


def k_shortest_paths(G, source, target, k, weight=None):
    try:
        return list(
            islice(nx.shortest_simple_paths(G, source, target, weight=weight), k)
        )
    except Exception:
        return []

# Given a P-space network and two nodes, retrieves all routes and corresponding directions
def get_routes_dirs(P_space, n1, n2):
    orig_routes = []
    for ro in P_space[n1][n2]['veh']:
        for dr in P_space[n1][n2]['veh'][ro]:
            orig_routes.append(str(ro) + str(dr))
    return orig_routes


def get_all_GTC(L_space, P_space, k, wait_pen, transfer_pen):

    """renamed from get_all_GTC_refactored"""

    #Precompute all attributes
    P_veh=nx.get_edge_attributes(P_space,"veh")
    P_wait=nx.get_edge_attributes(P_space,"avg_wait")
    L_dur=nx.get_edge_attributes(L_space,"duration_avg")
    L_dist=nx.get_edge_attributes(L_space,"d")

    # Precompute get routes dirs
    routes_dirs={}
    for e in P_veh:
        routes_dirs[e]=[]
        for ro in P_veh[e]:
            for dr in P_veh[e][ro]:
                routes_dirs[e].append(str(ro) + str(dr))
        
    shortest_paths={}
    paths=dict(nx.all_pairs_dijkstra_path(L_space,weight="duration_avg"))

    for n1 in L_space.nodes:
        for target in L_space.nodes:
            # Exclude self-loops
            if n1 == target:
                continue

            if n1 not in shortest_paths:
                shortest_paths[n1]={}

            # Two auxiliary datastructures to store the different shortest paths and corresponding attributes
            tt_paths = []
            only_tts = []

            # We consider just one path
            if target in paths[n1]:
                k_paths=[paths[n1][target]]
            else:
                k_paths=[]

            # Loop through all k-shortest paths and record the different travel time components
            for p in k_paths:
                possible_routes=routes_dirs[(p[0],p[1])]

                # Initialize the distance, (in-vehicle) travel time, waiting time and number of transfers as 0
                dist = 0
                tt = 0
                wait = 0
                tf = 0

                # Record the list of transfer stations, having the origin as the first "transfer station"
                t_stations = [n1]

                # Check the routes of all successive node pairs in the path,
                # if all routes of the original edge are not on the next edge, a transfer must have been made OR
                # if all routes of the previous edge are not on the next edge, a transfer must have been made
                # Route(s) on that edge become new route.
                # Also update the in-vehicle travel time for each edge passed.
                for l1, l2 in zip(p[::1], p[1::1]):
                    tt += L_dur[(l1,l2)]
                    dist += L_dist[(l1,l2)]
                    routes= routes_dirs[(l1,l2)]
                    possible_routes=set(possible_routes).intersection(set(routes))
                    if not possible_routes:
                        possible_routes = routes
                        tf +=1
                        t_stations.append(l1)

                # Add the destination node as the final transfer station
                t_stations.append(target)

                # Change travel time to minutes and round to whole minutes
                tt = round(tt / 60)

                # Find the waiting times belonging to the different routes taken by looping through all transfer station pairs
                for t1, t2 in zip(t_stations[::1], t_stations[1::1]):
                    wait += P_wait[(t1,t2)]
                    
                # Round the waiting time to whole minutes
                wait = round(wait)

                # Calculate the total travel time, take a penalty for the waiting time and per transfer
                transfer_cost=sum([transfer_pen[i] if i<len(transfer_pen) else transfer_pen[-1] for i in range(tf)])
                total_tt = tt + wait * wait_pen + transfer_cost
                only_tts.append(total_tt)
                tt_paths.append({'path': p, 'GTC': total_tt, 'in_vehicle': tt, 'waiting_time': wait, 'n_transfers': tf, 'traveled_distance': dist})

            if k_paths:
                # Find the path with the shortest total travel time
                min_path_tt = min(only_tts)
                min_path = tt_paths[only_tts.index(min_path_tt)]

                # Record that path as the shortest path belonging to nodes n1 and n2
                shortest_paths[n1][target] = min_path
            else:
                shortest_paths[n1][target]=[]

    return shortest_paths