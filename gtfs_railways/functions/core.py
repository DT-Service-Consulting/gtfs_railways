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
from collections import deque, defaultdict
from functools import wraps
from statistics import mean
from scipy import stats # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np
from bokeh.plotting import figure, show, from_networkx # type: ignore
from bokeh.models import HoverTool, Circle, MultiLine, WheelZoomTool, Range1d, DataRange1d, Button, EdgesAndLinkedNodes # type: ignore
from bokeh.layouts import column # type: ignore
from bokeh.tile_providers import Vendors # type: ignore
from pyproj import Transformer # type: ignore
from collections import Counter
import geopy.distance # type: ignore
from IPython.display import clear_output, display # type: ignore
from thefuzz import fuzz

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
    
def generate_graph(gtfs_feed,
                   mode,
                   start_hour=5, 
                   end_hour=24):
    '''Generates L-space graph considering the most suitable day from GTFS data. Parameters:
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

    G=networks.stop_to_stop_network_for_route_type(gtfs_feed,
                                                    mode_from_string(mode),
                                                    link_attributes=None,
                                                    start_time_ut=range_start,
                                                    end_time_ut=range_end)

    #Save original id in node attributes (to keep once we merge nodes)
    for n, data in G.nodes(data=True):
        data["original_ids"]=[n]

    print("Number of edges: ", len(G.edges()))
    print("Number of nodes: ", len(G.nodes()))
    return G
    
def save_graph(G,path):
    #Rename nodes to 0..n
    G_res=nx.convert_node_labels_to_integers(G)
    #nx.write_gpickle(G_res,path)    

    with open(path, 'wb') as f:
        pickle.dump(G_res, f)

def distance(G,n1,n2):
    '''Returns the distance in meters between two nodes in the graph.'''
    coords_n1=(G.nodes[n1]["lat"],G.nodes[n1]["lon"])
    coords_n2=(G.nodes[n2]["lat"],G.nodes[n2]["lon"])
    return geopy.distance.geodesic(coords_n1, coords_n2).m

def merge_nodes(G,n1,n2):
    '''Merges node n2 into n1, updates in/out edges, and merge attributes'''
    #Out edges
    for e in G.edges(n2,data=True):
        # If we get duplicated edges, average them. This should be a very odd case.
        if (n1,e[1]) in G.edges(n1):
            # Average travel time
            G[n1][e[1]]["duration_avg"]+=e[2]["duration_avg"]
            G[n1][e[1]]["duration_avg"]/=2.0 
            # Sum total n_vehicles
            G[n1][e[1]]["n_vehicles"]+=e[2]["n_vehicles"] 
            #Merge route counter
            G[n1][e[1]]["route_I_counts"]=dict(Counter(G[n1][e[1]]["route_I_counts"]) + Counter(e[2]["route_I_counts"])) 
            G[n1][e[1]]["shape_id"]=dict(Counter(G[n1][e[1]]["shape_id"]) + Counter(e[2]["shape_id"])) 
            G[n1][e[1]]["direction_id"]=dict(Counter(G[n1][e[1]]["direction_id"]) + Counter(e[2]["direction_id"])) 
            G[n1][e[1]]["headsign"]=dict(Counter(G[n1][e[1]]["headsign"]) + Counter(e[2]["headsign"])) 
        # Else, retain edge in the merged graph, except for self loops
        elif n1!=e[1]:
            G.add_edge(n1,e[1],
                       duration_avg=e[2]["duration_avg"],
                        n_vehicles=e[2]["n_vehicles"],
                       d=e[2]["d"], # We keep the original distance, which is not exactly right
                        route_I_counts=e[2]["route_I_counts"],
                          shape_id=e[2]["shape_id"],
                      direction_id=e[2]["direction_id"],
                      headsign=e[2]["headsign"])

    #In edges
    for e in G.in_edges(n2,data=True):
        # If we get duplicated edges, average them. This should be a very odd case.
        if (e[0],n1) in G.in_edges(n1):
            # Average travel time
            G[e[0]][n1]["duration_avg"]+=e[2]["duration_avg"]
            G[e[0]][n1]["duration_avg"]/=2.0 
            # Sum total n_vehicles
            G[e[0]][n1]["n_vehicles"]+=e[2]["n_vehicles"] 
            #Merge route counter
            G[e[0]][n1]["route_I_counts"]=dict(Counter(G[e[0]][n1]["route_I_counts"]) + Counter(e[2]["route_I_counts"])) 
            #Merge direction, shape_id, and headsign
            G[e[0]][n1]["shape_id"]=dict(Counter(G[e[0]][n1]["shape_id"]) + Counter(e[2]["shape_id"])) 
            G[e[0]][n1]["direction_id"]=dict(Counter(G[e[0]][n1]["direction_id"]) + Counter(e[2]["direction_id"]))
            G[e[0]][n1]["headsign"]=dict(Counter(G[e[0]][n1]["headsign"]) + Counter(e[2]["headsign"]))
            
        # Else, retain edge in the merged graph
        elif e[0]!=n1:
            G.add_edge(e[0],n1,
                       duration_avg=e[2]["duration_avg"],
                      n_vehicles=e[2]["n_vehicles"],
                      d=e[2]["d"], # We keep the original distance, which is not exactly right
                      route_I_counts=e[2]["route_I_counts"],
                      shape_id=e[2]["shape_id"],
                      direction_id=e[2]["direction_id"],
                      headsign=e[2]["headsign"])

    #Retain original ID before merging
    G.nodes[n1]["original_ids"]+=G.nodes[n2]["original_ids"]

    #Remove node
    G.remove_node(n2)


def merge_stops_with_same_name(G, delta=100, excepted=[]):
    '''Merge stops that share the same name and are
    closer to delta meters.'''
    
    #Dataframe of stops
    aux=list(zip(*G.nodes(data=True)))
    df_stops=pd.DataFrame(aux[1],index=aux[0]).reset_index()
    
    #Backup original graph
    G_res=G.copy()

    #Merge stations that share a name
    aux=list(df_stops.groupby("name").index.apply(list))
    aux2=[a for a in aux if len(a)>1]
    
    #Merge only nodes that are at most 100m away from the first node with the same name
    aux3=[]
    for group in aux2:
        clean_group=[group[0]]
        for n in group[1:]:
            if (not group[0] in excepted) and (not n in excepted):
                if distance(G,group[0],n)<=delta:
                    clean_group.append(n)
        if len(clean_group)>1:
            aux3.append(clean_group)

    for repeated in aux3:
        for i in repeated[1:]:
            print("Merged %s - %s"%(G_res.nodes[repeated[0]]["name"],G_res.nodes[i]["name"]))
            merge_nodes(G_res,repeated[0],i)
    
    return G_res

def check_islands(G):
    islands=list(nx.isolates(G))
    if islands:
        print("Found the following disconnected nodes: %s"%islands,flush=True)
        ans=input("Delete these nodes? (y/n)")
        if ans=="y":
            G.remove_nodes_from(islands)
            print("Removed the following disconnected nodes: %s"%islands)
        else:
            print("Islands were not removed. Make sure to manually create connecting edges with the appropriate labels")
    else:
        print("No disconnected nodes found")

def plot_graph_for_merge(G, n1, n2, delta=0.05):
    '''Plot graph zoomed to stops n1 and n2, which are plotted with big red circles'''

    clear_output(wait=True)
    p = figure(height = 600 ,
    width = 950, 
    toolbar_location = 'below',
    tools = "pan, wheel_zoom, box_zoom, reset, save")
    
    #Build dictionary of node positions for visualizations
    pos_dict={}
    for i,d in G.nodes(data=True):
        pos_dict[int(i)]=(float(d["lon"]),float(d["lat"]))
        
    # Plot updated graph
    graph = from_networkx(G, layout_function=pos_dict)
    
    #Create virtual graph with the two stops
    G_stops=nx.Graph()
    
    G_stops.add_node(n1)
    G_stops.add_node(n2)
    
    pos_dict_2={}
    pos_dict_2[n1]=pos_dict[n1]
    pos_dict_2[n2]=pos_dict[n2]
    
    graph_stops = from_networkx(G_stops, layout_function=pos_dict_2)

    node_hover_tool = HoverTool(tooltips=[("index", "@index"),
                                          ("name", "@name")],
                               renderers=[graph.node_renderer,
                                         graph_stops.node_renderer])

    p.add_tools(node_hover_tool)

   
    graph_stops.node_renderer.glyph = Circle(fill_color = 'red', size=8)

    p.renderers.append(graph)
    p.renderers.append(graph_stops)
    
    #TITLE
    p.title="%s <-> %s"%(G.nodes[n1]["name"],G.nodes[n2]["name"])
    p.title.text_font_size = '10pt'
    p.title.align = 'center'

    #ZOOM
    p.y_range = Range1d(min(G.nodes[n1]["lat"], G.nodes[n2]["lat"])-delta,
                       max(G.nodes[n1]["lat"], G.nodes[n2]["lat"])+delta)
    p.x_range = Range1d(min(G.nodes[n1]["lon"], G.nodes[n2]["lon"]-delta),
                         max(G.nodes[n1]["lon"], G.nodes[n2]["lon"])+delta)
    
    p.toolbar.active_scroll = p.select_one(WheelZoomTool)
    
    show(p)
    
    while True:
        ans=input("Merge? (y/n)")
        if ans=="y":
            #Merge stops
            print("Merged %s - %s"%(G.nodes[n1]["name"],G.nodes[n2]["name"]))
            merge_nodes(G,n1,n2)
            break
        elif ans=="n":
            break
    # clear_output(wait=True)


def merge_recommender(G, 
                      string_match=75, 
                      stop_distance=500):
    '''Iteratively suggest stops to merge with names closer than string_match (0,100)
    and not farther away than "distance" meters. Prompt y/n from user and merge or not.'''
    #Dataframe of stops
    aux=list(zip(*G.nodes(data=True)))
    df_stops=pd.DataFrame(aux[1],index=aux[0]).reset_index()
    stop_names=list(df_stops[["index","name"]].itertuples(index=False,name=None))

    for i,tuple_i in enumerate(stop_names):
        index_i,name_i=tuple_i
        for index_j,name_j in stop_names[i+1:]:
            #Check if node still exists (may have been merged already)
            if index_i in G.nodes() and index_j in G.nodes(): 
                #If names are similar
                if fuzz.ratio(name_i,name_j)>string_match: 
                    if distance(G,index_i,index_j)<=stop_distance:
                        plot_graph_for_merge(G,index_i,index_j)   


def manual_merge(G,
                 jupyter_url="http://localhost:8888"):
    def bkapp(doc):    
        #Build dictionary of node positions for visualizations
        pos_dict={}
        for i,d in G.nodes(data=True):
            pos_dict[int(i)]=(float(d["lon"]),float(d["lat"]))

        # source
        global graph
        graph = from_networkx(G, layout_function=pos_dict)

        def create_figure():
            back_map=False

            if back_map:
                map_options = GMapOptions(lat=list(G.nodes(data=True))[0][1]["lat"],  # type: ignore
                                          lng=list(G.nodes(data=True))[0][1]["lon"], 
                                          map_type="roadmap", 
                                          zoom=11)
                p = gmap(MAPS_API_KEY, map_options) # type: ignore
            else:
                p = figure(height = 600 ,
                width = 950, 
                toolbar_location = 'below',
                tools = "pan, tap, wheel_zoom, box_zoom, box_select, reset, save")

            #Zoom is active by default    
            p.toolbar.active_scroll = p.select_one(WheelZoomTool)

            # Plot updated graph
            global graph
            
            #Build dictionary of node positions for visualizations
            pos_dict_2={}
            for i,d in G.nodes(data=True):
                pos_dict_2[int(i)]=(float(d["lon"]),float(d["lat"]))
            
            graph = from_networkx(G, layout_function=pos_dict_2)

            #Hover tool
            node_hover_tool = HoverTool(tooltips=[("index", "@index"),
                                                  ("name", "@name")],
                                       renderers=[graph.node_renderer])

            p.add_tools(node_hover_tool)

            #Formatting
            graph.node_renderer.selection_glyph = Circle(fill_color="red")
            graph.node_renderer.glyph = Circle(size=8)
            
            p.renderers.append(graph)

            return p

        bt = Button(label='Merge nodes')        
        bt2 = Button(label='Delete edge')
        bt3= Button(label='Delete nodes')

        def change_click():
            #Get selected stops
            indices = graph.node_renderer.data_source.selected.indices
            if len(indices)==2:
                n1=graph.node_renderer.data_source.data["index"][indices[0]]
                n2=graph.node_renderer.data_source.data["index"][indices[1]]
                name_n1=graph.node_renderer.data_source.data["name"][indices[0]]
                name_n2=graph.node_renderer.data_source.data["name"][indices[1]]
                merge_nodes(G,
                            n1,
                            n2)
                print("Merged %s - %s"%(name_n1,name_n2))
                p = figure(tools="reset,pan,wheel_zoom,lasso_select")
                layout.children[0] = create_figure()
                return p
            else:
                print("Select two nodes to merge")

        def delete_edge():
            #Get selected stops
            indices = graph.node_renderer.data_source.selected.indices
            if len(indices)==2:
                n1=graph.node_renderer.data_source.data["index"][indices[0]]
                n2=graph.node_renderer.data_source.data["index"][indices[1]]
                name_n1=graph.node_renderer.data_source.data["name"][indices[0]]
                name_n2=graph.node_renderer.data_source.data["name"][indices[1]]
                if G.has_edge(n1,n2):
                    G.remove_edge(n1,n2)
                if G.has_edge(n2,n1):
                    G.remove_edge(n2,n1)           
                print("Deleted edges between %s - %s"%(name_n1,name_n2))
                p = figure(tools="reset,pan,wheel_zoom,lasso_select")
                layout.children[0] = create_figure()
                return p
            else:
                print("Select two nodes to delete an edge")
                
        def delete_nodes():
            #Get selected stops
            indices = graph.node_renderer.data_source.selected.indices
            if len(indices)==1:
                n1=graph.node_renderer.data_source.data["index"][indices[0]]
                name_n1=graph.node_renderer.data_source.data["name"][indices[0] ]
                G.remove_node(n1)
                print("Deleted node %s"%name_n1)
                p = figure(tools="reset,pan,wheel_zoom,lasso_select")
                layout.children[0] = create_figure()
            else:
                print("Select one node to delete")
                
        bt.on_click(change_click)
        bt2.on_click(delete_edge)
        bt3.on_click(delete_nodes)

        #layout=column(create_figure(),bt, bt2)
        layout=column(create_figure(),bt, bt3, bt2)

        doc.add_root(layout)

    show(bkapp,
         notebook_url=jupyter_url)

#################################################################

def merge_edges(G,edges):

    if edges==[]:
        print ("Select edges first!")
        return None
    
    #First edge is intercity
    ic1=edges[0]
    ic2=(edges[0][1],edges[0][0])

    #Get all other edges (sprinters)
    spr=[]
    for e in edges[1:]:
        spr.append(e)
        spr.append((e[1],e[0]))

    #Remove duplicates
    spr=list(set(spr)-set([ic1,ic2]))

    #Check all edges in graph
    for e in spr+[ic1,ic2]:
        if e not in G.edges():
            print("Error: edge (%d,%d) not in Graph"%(e[0],e[1]))
            return None

    if spr==[]:
        print ("Error: no sprinter lines selected")
        return None

    #Get ic1 edges
    visited_nodes=[ic1[0]]
    spr_ic1=[]
    node=ic1[0]
    #print(ic1)
    #print(spr)
    #return None
    while node!=ic1[1] or len(visited_nodes)==100:
        aux=[e for e in spr if e[0]==node and e[1] not in visited_nodes and e!=ic2 and e!=ic1]
        if aux==[]:
            print("Error. The selected edges are not connected, check: %s"%edges)
            return None
        edge=aux[0]
        spr_ic1.append(edge)
        visited_nodes.append(edge[1])
        node=edge[1]

    if len(visited_nodes)==100:
        print("Error. Check the list of selected edges: %s"%edges)
        return None

    #print(ic1)
    #print(spr_ic1)

    #Get ic2 edges
    visited_nodes=[ic2[0]]
    spr_ic2=[]
    node=ic2[0]
    while node!=ic2[1] or len(visited_nodes)==100:
        aux=[e for e in spr if e[0]==node and e[1] not in visited_nodes and e!=ic1 and e!=ic2]
        if aux==[]:
            print("Error. The selected edges are not connected, check: %s"%edges)
            return None
        edge=aux[0]
        spr_ic2.append(edge)
        visited_nodes.append(edge[1])
        node=edge[1]

    if len(visited_nodes)==100:
        print("Error. Check the list of selected edges: %s"%edges)
        return None

    #print("---")
    #print(ic2)
    #print(spr_ic2)
    #Merge IC1
    data_ic1=G[ic1[0]][ic1[1]]
    sum_times=sum([G[e[0]][e[1]]["duration_avg"] for e in spr_ic1])
    for e in spr_ic1:

        #Get proportion of IC time assigned to that edge
        data_e=G[e[0]][e[1]]
        prop_ic=data_ic1["duration_avg"]*(data_e["duration_avg"]/sum_times)

        #Weight the time based on frequency
        data_e["duration_avg"]=(data_e["duration_avg"]*data_e["n_vehicles"]+prop_ic*data_ic1["n_vehicles"])/(data_e["n_vehicles"]+data_ic1["n_vehicles"])

        #Update n_vehicles
        data_e["n_vehicles"]+=data_ic1["n_vehicles"]
        data_e["route_I_counts"]={k: data_e["route_I_counts"].get(k, 0) + data_ic1["route_I_counts"].get(k, 0) for k in set(data_e["route_I_counts"]) | set(data_ic1["route_I_counts"])}
        data_e["direction_id"]={k: data_e["direction_id"].get(k, 0) + data_ic1["direction_id"].get(k, 0) for k in set(data_e["direction_id"]) | set(data_ic1["direction_id"])}
        data_e["shape_id"]={k: data_e["shape_id"].get(k, 0) + data_ic1["shape_id"].get(k, 0) for k in set(data_e["shape_id"]) | set(data_ic1["shape_id"])}
        data_e["headsign"]={k: data_e["headsign"].get(k, 0) + data_ic1["headsign"].get(k, 0) for k in set(data_e["headsign"]) | set(data_ic1["headsign"])}
        
        #Keep log of merged edges
        if "merged_ic_edges" not in data_e:
            data_e["merged_ic_edges"]=[]
        data_e["merged_ic_edges"].append(ic1)

    
    G.remove_edge(ic1[0],ic1[1])
    print("Merged %s into %s"%(ic1,spr_ic1))

    #Merge IC2
    data_ic2=G[ic2[0]][ic2[1]]
    sum_times=sum([G[e[0]][e[1]]["duration_avg"] for e in spr_ic2])
    for e in spr_ic2:

        #Get proportion of IC time assigned to that edge
        data_e=G[e[0]][e[1]]
        prop_ic=data_ic2["duration_avg"]*(data_e["duration_avg"]/sum_times)

        #Weight the time based on frequency
        data_e["duration_avg"]=(data_e["duration_avg"]*data_e["n_vehicles"]+prop_ic*data_ic2["n_vehicles"])/(data_e["n_vehicles"]+data_ic2["n_vehicles"])

        #Update n_vehicles
        data_e["n_vehicles"]+=data_ic2["n_vehicles"]
        data_e["route_I_counts"]={k: data_e["route_I_counts"].get(k, 0) + data_ic2["route_I_counts"].get(k, 0) for k in set(data_e["route_I_counts"]) | set(data_ic2["route_I_counts"])}
        data_e["direction_id"]={k: data_e["direction_id"].get(k, 0) + data_ic2["direction_id"].get(k, 0) for k in set(data_e["direction_id"]) | set(data_ic2["direction_id"])}
        data_e["shape_id"]={k: data_e["shape_id"].get(k, 0) + data_ic2["shape_id"].get(k, 0) for k in set(data_e["shape_id"]) | set(data_ic2["shape_id"])}
        data_e["headsign"]={k: data_e["headsign"].get(k, 0) + data_ic2["headsign"].get(k, 0) for k in set(data_e["headsign"]) | set(data_ic2["headsign"])}
        
        #Keep log of merged edges
        if "merged_ic_edges" not in data_e:
            data_e["merged_ic_edges"]=[]
        data_e["merged_ic_edges"].append(ic2)

    G.remove_edge(ic2[0],ic2[1])
    print("Merged %s into %s"%(ic2,spr_ic2))        

def edge_merger(G,
                 jupyter_url="http://localhost:8888"):
    def bkapp(doc):    
        #Build dictionary of node positions for visualizations
        pos_dict={}
        for i,d in G.nodes(data=True):
            pos_dict[int(i)]=(float(d["lon"]),float(d["lat"]))


        def update_range(axis,endpoint,value):
            if axis=="x":
                if endpoint=="start":
                    global x_range_start
                    x_range_start=value
                elif endpoint=="end":
                    global x_range_end
                    x_range_end=value
            elif axis=="y":
                if endpoint=="start":
                    global y_range_start
                    y_range_start=value
                elif endpoint=="end":
                    global y_range_end
                    y_range_end=value
            
        global x_range_start
        x_range_start=None
        global x_range_end
        x_range_end=None
        global y_range_start
        y_range_start=None
        global y_range_end
        y_range_end=None
            
        # source
        global graph
        graph = from_networkx(G, layout_function=pos_dict)

        def create_figure():
            back_map=False

            if back_map:
                map_options = GMapOptions(lat=list(G.nodes(data=True))[0][1]["lat"], # type: ignore
                                          lng=list(G.nodes(data=True))[0][1]["lon"], 
                                          map_type="roadmap", 
                                          zoom=11)
                p = gmap(MAPS_API_KEY, map_options) # type: ignore
            elif not x_range_start:
                p = figure(height = 600 ,
                           width = 950,
                           toolbar_location = 'below',
                           tools = "pan, tap, wheel_zoom, box_zoom, box_select, reset, save")
            else:
                p = figure(height = 600 ,
                           width = 950,
                           toolbar_location = 'below',
                           tools = "pan, tap, wheel_zoom, box_zoom, box_select, reset, save",
                           x_range=DataRange1d(start=x_range_start,end=x_range_end),
                           y_range=DataRange1d(start=y_range_start,end=y_range_end))
                

            #Zoom is active by default    
            p.toolbar.active_scroll = p.select_one(WheelZoomTool)

            # Plot updated graph
            global graph
            
            #Build dictionary of node positions for visualizations
            pos_dict_2={}
            for i,d in G.nodes(data=True):
                pos_dict_2[int(i)]=(float(d["lon"]),float(d["lat"]))
            
            graph = from_networkx(G, layout_function=pos_dict_2)

            #Hover tool
            node_hover_tool = HoverTool(tooltips=[("index", "@index"),
                                                  ("name", "@name")],
                                       renderers=[graph.node_renderer])

            p.add_tools(node_hover_tool)

            hover_edges = HoverTool(tooltips=[("duration_avg", "@duration_avg"),
                                              ("n_vehicles","@n_vehicles"),
                                              ("merged_ic_edges","@merged_ic_edges")],
                            renderers=[graph.edge_renderer],
                           line_policy="interp")

            p.add_tools(hover_edges)
            
            #Formatting
            graph.node_renderer.selection_glyph = Circle(fill_color="red")
            graph.node_renderer.glyph = Circle(size=8)
            graph.edge_renderer.selection_glyph = MultiLine(line_width=2,line_color="red")
            graph.edge_renderer.glyph = MultiLine(line_width=2)


            graph.selection_policy = EdgesAndLinkedNodes() #NodesAndLinkedEdges()
            p.renderers.append(graph)

            p.x_range.on_change('start', lambda attr, old, new: update_range("x","start",new))
            p.x_range.on_change('end', lambda attr, old, new: update_range("x","end",new))
            p.y_range.on_change('start', lambda attr, old, new: update_range("y","start",new))
            p.y_range.on_change('end', lambda attr, old, new: update_range("y","end",new))

            return p

        bt = Button(label='Combine edges')
        
        #bt2 = Button(label='Delete edge')

        def change_click():
            edges=[]
            #Get selected stops
            indices = graph.edge_renderer.data_source.selected.indices
            for i in indices:
                start=graph.edge_renderer.data_source.data["start"][i]
                end=graph.edge_renderer.data_source.data["end"][i]
                edges.append((start,end))
            merge_edges(G,edges)
            p = figure(tools="reset,pan,wheel_zoom,lasso_select")
            layout.children[0] = create_figure()
            return p

        bt.on_click(change_click)

        layout=column(create_figure(),bt)

        doc.add_root(layout)

    show(bkapp,
         notebook_url=jupyter_url)



###############



def sanity_check(G):
    print("Checking self loops...")
    for n in G.edges:
        if n[0]==n[1]:
            print("Self loop found: %d. Consider removing it manually."%n[0])
    print("---")

    print("Checking links only on one direction...")
    for n in G.edges: 
        if (n[1], n[0]) not in G.edges:
            print("Edge exists only in one direction: ",
                  G.nodes[n[0]]['name'],
                  " (node %d) "%n[0],
                  "to", 
                  G.nodes[n[1]]['name'],
                  " (node %d) "%n[1])
    print("---")

    print("Checking edges with invalid duration...")
    for n in G.edges(data=True):
        if n[2]["duration_avg"]<=0:
           message="Edge (%d,%d) has duration_avg of %d. "%(n[0],n[1],n[2]["duration_avg"])
           if (n[1],n[0]) in G.edges() and G[n[1]][n[0]]["duration_avg"]>0:
               message+="Consider setting up the duration manually, perhaps using the duration of the opposite edge (%d,%d)=%d"%(n[1],n[0],G[n[1]][n[0]]["duration_avg"])
           else:
               message+="Consider setting up the duration manually."
           print(message)
    print("---")
    
    print("Number of edges: ", len(G.edges()))
    print("Number of nodes: ", len(G.nodes()))
    print("Number of strongly connected components: %d"%nx.number_strongly_connected_components(G))

def process_route_data(L_graph, routes, node1, node2):
    # Helper function to process a single direction
    def process_single_direction(start_node, end_node):
        edge_data = L_graph.get_edge_data(start_node, end_node)
        if edge_data is None:  # Handle case where edge doesn’t exist
            print(f"No edge data found for nodes ({start_node}, {end_node})")
            return None
        
        route_I_counts = edge_data['route_I_counts']
        route_ids = list(route_I_counts.keys())
        
        temp = routes.loc[routes['route_id'].isin(route_ids)].copy()
        temp['n_vehicles'] = temp['route_id'].map(route_I_counts)
        temp = temp.dropna(axis=1, how='all')
        
        # Print edge data for this direction
        print(f"\nData for nodes ({start_node}, {end_node}):")
        print('Average Duration:', edge_data['duration_avg'])
        print('Number of Vehicles:', edge_data['n_vehicles'])
        print('Headsign:', edge_data['headsign'])
        
        return temp

    # Process both directions
    temp_df1 = process_single_direction(node1, node2)
    temp_df2 = process_single_direction(node2, node1)

    # Print the DataFrames with labels
    if temp_df1 is not None:
        print(f"\nResult for nodes ({node1}, {node2}):")
        display(temp_df1)
    if temp_df2 is not None:
        print(f"\nResult for nodes ({node2}, {node1}):")
        display(temp_df2)

    # Return both DataFrames
    return temp_df1, temp_df2

def node_degrees_table(graph):
    data = []
    
    for node in graph.nodes:
        node_name = graph.nodes[node].get('name', f"Node {node}")  # Extract name if it exists
        in_degree = graph.in_degree(node)
        out_degree = graph.out_degree(node)
        total_degree = graph.degree(node)
        
        data.append([node, node_name, total_degree, in_degree, out_degree])
    
    df = pd.DataFrame(data, columns=['Node Number', 'Node Name', 'Total (In+Out)', 'Ingoing', 'Outgoing'])

    # Sort by total degree in descending order
    df = df.sort_values(by="Total (In+Out)", ascending=False)
    
    return df

def stations_trains_df(graph):
    data = []
    
    for node in graph.nodes:
        node_name = graph.nodes[node].get('name', f"Node {node}")
        
        degree = graph.degree(node)
        
        # Sum n_vehicles for all edges connected to this node
        num_trains = 0
        for _, _, edata in graph.edges(node, data=True):
            num_trains += edata.get('n_vehicles', 0)
        
        data.append([node, node_name, degree, num_trains])
    
    df = pd.DataFrame(data, columns=['Node Number', 'Node Name', 'Degree', 'Num Trains'])
    df = df.sort_values(by="Num Trains", ascending=False).reset_index(drop=True)
    
    return df

def select_nodes_bokeh(G, jupyter_url="http://localhost:8888"):
    """
    Interactive Bokeh tool to manually select nodes from a network graph.
    Returns the list of selected node IDs after clicking 'Confirm selection'.

    Parameters:
        G (networkx.Graph): The full graph.
        jupyter_url (str): Jupyter notebook server URL (default: localhost:8888).
    
    Returns:
        list: List of selected node IDs.
    """
    selected_nodes = []

    def bkapp(doc):    
        # Prepare node positions
        pos_dict = {int(i): (float(d["lon"]), float(d["lat"])) for i, d in G.nodes(data=True)}

        # Create network graph renderer
        graph = from_networkx(G, layout_function=pos_dict)
        p = figure(
            height=600,
            width=950,
            toolbar_location='below',
            tools="pan,tap,box_select,wheel_zoom,reset,save",
            title="Select nodes (Tap or Box Select). Click 'Confirm selection' when done."
        )

        # Activate zoom by default
        p.toolbar.active_scroll = p.select_one(WheelZoomTool)

        # Node hover info
        hover = HoverTool(tooltips=[("ID", "@index"), ("Name", "@name")], renderers=[graph.node_renderer])
        p.add_tools(hover)

        # Node appearance
        graph.node_renderer.glyph = Circle(size=8, fill_color="lightblue")
        graph.node_renderer.selection_glyph = Circle(size=10, fill_color="red")
        graph.node_renderer.nonselection_glyph = Circle(size=6, fill_color="lightgray", fill_alpha=0.5)
        p.renderers.append(graph)

        # Confirm selection button
        confirm_button = Button(label="Confirm selection", button_type="success")

        def confirm_selection():
            indices = graph.node_renderer.data_source.selected.indices
            selected = [graph.node_renderer.data_source.data["index"][i] for i in indices]
            selected_nodes.clear()
            selected_nodes.extend(selected)
            print(f"Selected nodes: {selected_nodes}")
            p.title.text = f"Selection confirmed: {len(selected_nodes)} nodes"

        confirm_button.on_click(confirm_selection)

        layout = column(p, confirm_button)
        doc.add_root(layout)

    show(bkapp, notebook_url=jupyter_url)
    return selected_nodes

def make_subgraph_from_nodes(G, node_list, copy=True):
    """
    Create a subgraph from a given list of nodes.

    Parameters:
        G (networkx.Graph): The full input graph.
        node_list (list): List of node IDs to include in the subgraph.
        copy (bool): Whether to return a deep copy of the subgraph (default: True).

    Returns:
        networkx.Graph: The resulting subgraph.
    """
    # Filter only nodes that exist in G
    valid_nodes = [n for n in node_list if n in G.nodes]
    missing_nodes = set(node_list) - set(valid_nodes)
    
    if missing_nodes:
        print(f"Warning: {len(missing_nodes)} nodes not found in graph and were skipped: {missing_nodes}")

    # Create subgraph
    subG = G.subgraph(valid_nodes)
    if copy:
        subG = nx.Graph(subG)  # returns a copy, not a view

    # Log some info
    print(f"Created subgraph with {subG.number_of_nodes()} nodes and {subG.number_of_edges()} edges.")
    return subG

def save_subgraph_as_pickle(subgraph, file_path):
    """
    Save a NetworkX subgraph as a .pkl file (compatible across NetworkX versions).

    Parameters:
        subgraph (networkx.Graph): The subgraph to save.
        file_path (str): Full path where the .pkl file should be saved.
                         Example: 'data/subgraphs/ring_structure.pkl'
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(str(file_path)), exist_ok=True)

    # Always save as pickle directly
    with open(file_path, "wb") as f:
        pickle.dump(subgraph, f)

    print(f"Subgraph saved successfully at: {file_path}")

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

def load_removal_results_df(csv_path):
    """
    Load the node/edge removal simulation results from a CSV file.

    Parameters:
        csv_path (str): Path to the results CSV.

    Returns:
        pd.DataFrame: DataFrame containing removal results.
    """
    df = pd.read_csv(csv_path)
    return df

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
    method='random',  # random, targeted, betweenness, hubs, or custom
    removal_type='node',
    custom_nodes=None,  # list of nodes to remove in order (only for method='custom')
    seed=None,
    verbose=False
):
    """
    Simulates the impact of sequential removals (node or edge) on global efficiency.

    Parameters:
        L_graph (networkx.Graph): The subgraph from which nodes/edges will be removed.
        num_to_remove (int, optional): Number of elements to remove. Ignored if percentage is given.
        pct_to_remove (int, optional): Percentage of elements to remove (1-100).
        method (str): Removal method. Options:
            'random', 'targeted', 'betweenness',
            'top_hubs_edges', 'top_hubs_trains',
            'custom' (requires custom_nodes).
        removal_type (str): 'node' or 'edge'.
        custom_nodes (list, optional): List of nodes to remove in given order (used if method='custom').
        seed (int, optional): Random seed.
        verbose (bool): Print progress/debug info.
    """
    G = copy.deepcopy(L_graph)

    if removal_type == "node":
        total_elements = G.number_of_nodes()
    elif removal_type == "edge":
        total_elements = G.number_of_edges()
    else:
        raise ValueError("Invalid removal_type. Choose 'node' or 'edge'.")

    if method != "custom":  # normal modes
        if pct_to_remove is not None:
            if not (1 <= pct_to_remove <= 100):
                raise ValueError("Percentage must be between 1 and 100.")
            num_to_remove = int(total_elements * (pct_to_remove / 100))
        elif num_to_remove is None:
            raise ValueError("Specify num_to_remove or pct_to_remove.")

        if num_to_remove > total_elements:
            if verbose:
                print(f"Adjusting number of elements to remove from {num_to_remove} to {total_elements - 2}.")
            num_to_remove = max(total_elements - 2, 1)

    # Node removals
    if removal_type == "node":
        if method == "random":
            return random_node_removal(L_graph, G, num_to_remove, sp_func, seed, verbose)
        elif method == "random_average":
            return random_node_removal_average(L_graph, G, num_to_remove, sp_func, verbose)
        elif method == "targeted":
            return targeted_node_removal(L_graph, G, num_to_remove, sp_func, verbose)
        elif method == "targeted_least":
            return targeted_node_removal_least(L_graph, G, num_to_remove, sp_func, verbose)
        elif method == "betweenness_weighted":
            return betweenness_node_removal_weighted(L_graph, G, num_to_remove, sp_func, verbose)
        elif method == "betweenness_unweighted":
            return betweenness_node_removal_unweighted(L_graph, G, num_to_remove, sp_func, verbose)
        elif method == "betweenness_weighted_least":
            return betweenness_node_removal_weighted_least(L_graph, G, num_to_remove, sp_func, verbose)
        elif method == "betweenness_unweighted_least":
            return betweenness_node_removal_unweighted_least(L_graph, G, num_to_remove, sp_func, verbose)
        elif method == "top_hubs_edges":
            return top_hubs_node_removal(L_graph, G, num_to_remove, sp_func, verbose)
        elif method == "top_hubs_trains":
            return top_train_hubs_node_removal(L_graph, G, num_to_remove, sp_func, verbose)
        elif method == "custom":
            if not custom_nodes:
                raise ValueError("Provide a list of custom_nodes when using method='custom'.")
            return custom_node_removal(L_graph, G, custom_nodes, sp_func, verbose)
        else:
            raise ValueError("Invalid method.")
    
    # Edge removals
    elif removal_type == "edge":
        if method == "random":
            return random_edge_removal(L_graph, G, num_to_remove, sp_func, seed, verbose)
        elif method == "random_average":
            return random_edge_removal_average(L_graph, G, num_to_remove, sp_func, verbose)
        elif method == "targeted":
            return targeted_edge_removal(L_graph, G, num_to_remove, sp_func, verbose)
        elif method == "targeted_least":
            return targeted_edge_removal_least(L_graph, G, num_to_remove, sp_func, verbose)
        elif method == "betweenness_weighted":
            return betweenness_edge_removal_weighted(L_graph, G, num_to_remove, sp_func, verbose)
        elif method == "betweenness_unweighted":
            return betweenness_edge_removal_unweighted(L_graph, G, num_to_remove, sp_func, verbose)
        else:
            raise ValueError("Invalid method for edge removal.")
    
    return None


def custom_node_removal(L_graph, G, node_list, sp_func, verbose=False):
    """
    Remove all incoming and outgoing edges of nodes in a user-defined order and compute efficiency at each step.
    """
    import time

    removed_nodes = []
    efficiencies = []
    removal_times = []
    num_removed = []

    sp = sp_func(G)
    original_efficiency = efficiency_graph(G, sp)
    efficiencies.append(original_efficiency)
    num_removed.append(0)
    removal_times.append(0)

    for i, node in enumerate(node_list, 1):
        if node not in G:
            if verbose:
                print(f"Node {node} not in graph, skipping.")
            continue

        start_time = time.perf_counter()
        
        # Remove all edges of the node (both incoming and outgoing if directed)
        if G.is_directed():
            edges_to_remove = list(G.out_edges(node)) + list(G.in_edges(node))
        else:
            edges_to_remove = list(G.edges(node))
        G.remove_edges_from(edges_to_remove)
        removed_nodes.append(node)

        sp = sp_func(G)
        eff = efficiency_graph(G, sp)

        elapsed = time.perf_counter() - start_time

        efficiencies.append(eff)
        num_removed.append(i)
        removal_times.append(round(elapsed, 4))

        if verbose:
            print(f"Removed edges of node {node}, efficiency = {eff:.4f}, time = {elapsed:.4f}s")

    return original_efficiency, efficiencies, num_removed, removed_nodes, removal_times


def random_node_removal(g, G, num_to_remove, sp_func, seed=None, verbose=False):
    if seed is not None:
        random.seed(seed)

    total_nodes = G.number_of_nodes()
    removal_nodes = random.sample(list(G.nodes()), num_to_remove)

    if verbose:
        print(f"Random removal order: {removal_nodes}")

    sp = sp_func(G)
    original_efficiency = efficiency_graph(g, sp)
    if verbose:
        print(f"Original Efficiency: {original_efficiency}")

    efficiencies = [1.0]
    num_removed = [0]
    percent_remaining = [100.0]
    removed_nodes = []
    removal_times = []

    for i, node in enumerate(removal_nodes):
        start_time = time.perf_counter()

        # Case 1: node already isolated — reuse last efficiency, still advance counters
        if G.degree(node) == 0:
            if verbose:
                print(f"Step {i + 1}: Node {node} already isolated — no edges removed, same efficiency recorded.")

            efficiencies.append(efficiencies[-1])
            num_removed.append(num_removed[-1] + 1)  # increment to keep visualization aligned
            percent_remaining.append(100 * (1 - num_removed[-1] / total_nodes))
            removed_nodes.append(node)
            removal_times.append(0)
            continue

        # Case 2: node still connected — remove its edges
        if isinstance(G, nx.DiGraph):
            edges_to_remove = list(G.in_edges(node)) + list(G.out_edges(node))
        else:
            edges_to_remove = list(G.edges(node))
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

def random_node_removal_average(L_graph, G, num_to_remove, sp_func, verbose=False, n_runs=10):
    """
    Runs random node removal multiple times and returns mean and standard deviation
    of efficiency evolution across runs.
    """
    all_eff = []
    total_nodes = L_graph.number_of_nodes()
    target_len = num_to_remove + 1

    for i in range(n_runs):
        seed = random.randint(0, 2**31 - 1)
        _, eff, _, _, _ = random_node_removal(L_graph, copy.deepcopy(G), num_to_remove, sp_func, seed=seed, verbose=verbose)
        eff = np.array(eff, dtype=float)
        # Pad to target length with last value
        if len(eff) < target_len:
            eff = np.pad(eff, (0, target_len - len(eff)), mode='edge')
        elif len(eff) > target_len:
            eff = eff[:target_len]
        all_eff.append(eff)

    all_eff = np.vstack(all_eff)
    mean_eff = np.mean(all_eff, axis=0)
    std_eff = np.std(all_eff, axis=0)

    percent_remaining = 100 * (1 - np.arange(target_len) / total_nodes)

    # Original efficiency (first value)
    sp = sp_func(G)
    original_eff = efficiency_graph(G, sp)

    return original_eff, mean_eff, std_eff, percent_remaining

def random_edge_removal(g, G, num_to_remove, sp_func, seed=None, verbose=False):
    """
    Randomly removes edges from a graph and tracks the impact on global efficiency.

    Parameters:
        g: Base attributes or data required by efficiency_graph.
        G (networkx.DiGraph): The working graph from which edges will be removed.
        num_to_remove (int): Number of edges to remove.
        sp_func (callable): Function to compute shortest path structure.
        seed (int, optional): Random seed.
        verbose (bool): Whether to print logs.

    Returns:
        original_efficiency (float): Efficiency before removal.
        efficiencies (list): Normalized efficiencies.
        num_removed (list): Cumulative edge removals.
        removed_edges (list): List of edges removed.
        removal_times (list): Time taken for each step.
        percent_remaining (list): Percentage of edges remaining.
    """
    if seed is not None:
        random.seed(seed)

    total_edges = G.number_of_edges()
    if num_to_remove > total_edges:
        if verbose:
            print(f"Adjusting number of edges to remove from {num_to_remove} to {total_edges}.")
        num_to_remove = total_edges

    removal_edges = random.sample(list(G.edges()), num_to_remove)
    if verbose:
        print(f"Random edge removal order: {removal_edges}")

    # Compute original efficiency
    sp = sp_func(G)
    original_efficiency = efficiency_graph(g, sp)

    efficiencies = [1.0]
    num_removed = [0]
    percent_remaining = [100.0]
    removed_edges = []
    removal_times = []

    for i, edge in enumerate(removal_edges):
        start_time = time.perf_counter()
        if not G.has_edge(*edge):
            if verbose:
                print(f"Step {i + 1}: Edge {edge} already removed, skipping.")
            efficiencies.append(efficiencies[-1])
            num_removed.append(num_removed[-1])
            percent_remaining.append(100 * (1 - num_removed[-1] / total_edges))
            continue

        G.remove_edge(*edge)
        removed_edges.append(edge)

        try:
            sp = sp_func(G)
            eff = efficiency_graph(g, sp)
        except Exception as e:
            if verbose:
                print(f"Error after removing edge {edge}: {e}")
            break

        elapsed = time.perf_counter() - start_time
        normalized_eff = eff / original_efficiency

        efficiencies.append(normalized_eff)
        num_removed.append(num_removed[-1] + 1)
        percent_remaining.append(100 * (1 - num_removed[-1] / total_edges))
        removal_times.append(round(elapsed, 4))

        if verbose:
            print(f"Removed edge {edge} → Normalized Efficiency: {normalized_eff:.4f}")
            print(f"Time taken: {elapsed:.4f} seconds\n")

    return original_efficiency, efficiencies, percent_remaining, removed_edges, removal_times

def random_edge_removal_average(L_graph, G, num_to_remove, sp_func, verbose=False, n_runs=10):
    """
    Runs random edge removal multiple times and returns mean and standard deviation
    of efficiency evolution across runs.
    Ensures all runs are aligned to the same number of removals.
    """
    all_eff = []
    total_edges = G.number_of_edges()
    target_len = num_to_remove + 1

    for i in range(n_runs):
        seed = np.random.randint(0, 1e9)
        _, eff, _, _, _ = random_edge_removal(L_graph, copy.deepcopy(G), num_to_remove, sp_func, seed, verbose)
        eff = np.array(eff, dtype=float)
        # Pad to target length with last value
        if len(eff) < target_len:
            eff = np.pad(eff, (0, target_len - len(eff)), mode='edge')
        elif len(eff) > target_len:
            eff = eff[:target_len]
        all_eff.append(eff)

    all_eff = np.vstack(all_eff)
    mean_eff = np.mean(all_eff, axis=0)
    std_eff = np.std(all_eff, axis=0)

    percent_remaining = 100 * (1 - np.arange(target_len) / total_edges)

    # Original efficiency (first value)
    sp = sp_func(G)
    original_eff = efficiency_graph(G, sp)

    return original_eff, mean_eff, std_eff, percent_remaining

def targeted_node_removal(g, G, num_to_remove, sp_func, verbose=False):
    """
    Greedy edge removal: at each step, remove all edges (incoming and outgoing) of the node
    that causes the largest drop in global efficiency.
    If all remaining nodes are isolated, the function still advances the step count using the previous efficiency.
    
    Args:
        g: unused, can be ignored or removed.
        G (nx.Graph or nx.DiGraph): graph to operate on.
        num_to_remove (int): number of nodes to remove edges from.
        sp_func (callable): function to compute shortest paths on G.
        verbose (bool): print progress.
        
    Returns:
        original_efficiency, efficiencies, percent_remaining, removed_nodes, removal_times
    """
    import time

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

        # Recompute efficiency before choosing node
        sp = sp_func(G)
        current_eff = efficiency_graph(G, sp)
        max_drop = -1
        best_node = None

        for node in G.nodes():
            if G.degree(node) == 0:
                continue  # isolated nodes can't cause efficiency drop

            temp_G = G.copy()
            if G.is_directed():
                temp_edges_to_remove = list(temp_G.out_edges(node)) + list(temp_G.in_edges(node))
            else:
                temp_edges_to_remove = list(temp_G.edges(node))
            temp_G.remove_edges_from(temp_edges_to_remove)

            try:
                sp_temp = sp_func(temp_G)
                eff_temp = efficiency_graph(temp_G, sp_temp)
            except Exception:
                continue

            drop = current_eff - eff_temp
            if drop > max_drop:
                max_drop = drop
                best_node = node

        # Handle case where all remaining nodes are isolated
        if best_node is None:
            if verbose:
                print(f"Step {step}: All remaining nodes are isolated — no edges removed, same efficiency recorded.")
            efficiencies.append(efficiencies[-1])
            num_removed.append(num_removed[-1] + 1)
            percent_remaining.append(100 * (1 - num_removed[-1] / total_nodes))
            removed_nodes.append(None)
            removal_times.append(0)
            removals_done += 1
            continue

        # Perform removal of all edges (incoming + outgoing)
        if G.is_directed():
            edges_to_remove = list(G.out_edges(best_node)) + list(G.in_edges(best_node))
        else:
            edges_to_remove = list(G.edges(best_node))
        G.remove_edges_from(edges_to_remove)
        removed_nodes.append(best_node)
        removals_done += 1

        try:
            sp_new = sp_func(G)
            eff = efficiency_graph(G, sp_new)
        except Exception as e:
            if verbose:
                print(f"Error after removing edges of {best_node}: {e}")
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


def targeted_node_removal_least(g, G, num_to_remove, sp_func, verbose=False):
    """
    Greedy node-edge removal: at each step, remove the node whose edge removals cause
    the smallest drop in global efficiency (least impactful first).
    For directed graphs, removes both incoming and outgoing edges.
    If all remaining nodes are isolated, reuse the previous efficiency so that
    visualizations remain continuous.
    """
    import time

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

        sp = sp_func(G)
        current_eff = efficiency_graph(G, sp)
        min_drop = float("inf")
        best_node = None

        for node in G.nodes():
            if G.degree(node) == 0:
                continue

            temp_G = G.copy()
            # Remove all edges of the node (incoming + outgoing for directed)
            if G.is_directed():
                temp_edges_to_remove = list(temp_G.out_edges(node)) + list(temp_G.in_edges(node))
            else:
                temp_edges_to_remove = list(temp_G.edges(node))
            temp_G.remove_edges_from(temp_edges_to_remove)

            try:
                sp_temp = sp_func(temp_G)
                eff_temp = efficiency_graph(temp_G, sp_temp)
            except Exception:
                continue

            drop = current_eff - eff_temp
            if drop < min_drop:
                min_drop = drop
                best_node = node

        # Handle case where all remaining nodes are isolated
        if best_node is None:
            if verbose:
                print(f"Step {step}: All remaining nodes are isolated — no edges removed, same efficiency recorded.")

            efficiencies.append(efficiencies[-1])
            num_removed.append(num_removed[-1] + 1)
            percent_remaining.append(100 * (1 - num_removed[-1] / total_nodes))
            removed_nodes.append(None)
            removal_times.append(0)
            removals_done += 1
            continue

        # Perform edge removal
        if G.is_directed():
            edges_to_remove = list(G.out_edges(best_node)) + list(G.in_edges(best_node))
        else:
            edges_to_remove = list(G.edges(best_node))
        G.remove_edges_from(edges_to_remove)
        removed_nodes.append(best_node)
        removals_done += 1

        try:
            sp_new = sp_func(G)
            eff = efficiency_graph(G, sp_new)
        except Exception as e:
            if verbose:
                print(f"Error after removing edges of {best_node}: {e}")
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


def targeted_edge_removal(g, G, num_to_remove, sp_func, verbose=False):
    """
    Greedily removes one edge at a time to maximize efficiency drop.

    Returns same structure as targeted_node_removal.
    """
    total_edges = G.number_of_edges()
    sp = sp_func(G)
    original_efficiency = efficiency_graph(G, sp)
    if verbose:
        print(f"Original Efficiency: {original_efficiency:.4f}")

    efficiencies = [1.0]
    num_removed = [0]
    percent_remaining = [100.0]
    removed_edges = []
    removal_times = []

    removals_done = 0
    step = 0

    while removals_done < num_to_remove and G.number_of_edges() > 0:
        step += 1
        start_time = time.perf_counter()

        sp = sp_func(G)
        current_eff = efficiency_graph(G, sp)
        max_drop = -1
        best_edge = None

        for edge in G.edges():
            temp_G = G.copy()
            temp_G.remove_edge(*edge)
            try:
                sp_temp = sp_func(temp_G)
                eff_temp = efficiency_graph(temp_G, sp_temp)
                drop = current_eff - eff_temp
                if drop > max_drop:
                    max_drop = drop
                    best_edge = edge
            except:
                continue

        if best_edge is None:
            if verbose:
                print("No valid edge to remove at step", step)
            break

        G.remove_edge(*best_edge)
        removed_edges.append(best_edge)
        removals_done += 1

        try:
            sp_new = sp_func(G)
            eff = efficiency_graph(G, sp_new)
        except Exception as e:
            if verbose:
                print(f"Error after removing {best_edge}: {e}")
            break

        normalized_eff = eff / original_efficiency
        elapsed = round(time.perf_counter() - start_time, 4)

        efficiencies.append(normalized_eff)
        num_removed.append(removals_done)
        percent_remaining.append(100 * (1 - removals_done / total_edges))
        removal_times.append(elapsed)

        if verbose:
            print(f"Step {step}: Removed edge {best_edge} → Normalized Efficiency: {normalized_eff:.4f}")
            print(f"Time taken: {elapsed:.4f} seconds\n")

    return original_efficiency, efficiencies, percent_remaining, removed_edges, removal_times

def targeted_edge_removal_least(g, G, num_to_remove, sp_func, verbose=False):
    """
    Greedy edge removal: at each step, remove the edge that causes
    the smallest drop in global efficiency (least impactful first).
    """
    total_edges = G.number_of_edges()
    sp = sp_func(G)
    original_efficiency = efficiency_graph(G, sp)
    if verbose:
        print(f"Original Efficiency: {original_efficiency:.4f}")

    efficiencies = [1.0]
    num_removed = [0]
    percent_remaining = [100.0]
    removed_edges = []
    removal_times = []

    removals_done = 0
    step = 0

    while removals_done < num_to_remove and G.number_of_edges() > 0:
        step += 1
        start_time = time.perf_counter()

        sp = sp_func(G)
        current_eff = efficiency_graph(G, sp)
        min_drop = float("inf")
        best_edge = None

        for edge in G.edges():
            temp_G = G.copy()
            temp_G.remove_edge(*edge)
            try:
                sp_temp = sp_func(temp_G)
                eff_temp = efficiency_graph(temp_G, sp_temp)
                drop = current_eff - eff_temp
                if drop < min_drop:
                    min_drop = drop
                    best_edge = edge
            except Exception:
                continue

        if best_edge is None:
            if verbose:
                print("No valid edge to remove at step", step)
            break

        G.remove_edge(*best_edge)
        removed_edges.append(best_edge)
        removals_done += 1

        try:
            sp_new = sp_func(G)
            eff = efficiency_graph(G, sp_new)
        except Exception as e:
            if verbose:
                print(f"Error after removing {best_edge}: {e}")
            break

        normalized_eff = eff / original_efficiency
        elapsed = round(time.perf_counter() - start_time, 4)

        efficiencies.append(normalized_eff)
        num_removed.append(removals_done)
        percent_remaining.append(100 * (1 - removals_done / total_edges))
        removal_times.append(elapsed)

        if verbose:
            print(f"Step {step}: Removed edge {best_edge} → Normalized Efficiency: {normalized_eff:.4f}")
            print(f"Time taken: {elapsed:.4f} seconds\n")

    return original_efficiency, efficiencies, percent_remaining, removed_edges, removal_times


def betweenness_node_removal_weighted(g, G, num_to_remove, sp_func, verbose=False):
    """
    Removes edges of nodes based on descending betweenness centrality,
    tracking the impact on global efficiency (normalized by initial value).

    Works for both directed and undirected graphs.

    Parameters:
        g (nx.Graph or nx.DiGraph): Original reference graph (for efficiency baseline).
        G (nx.Graph or nx.DiGraph): Working copy of graph (modified in-place).
        num_to_remove (int): Number of nodes (edge-removal steps) to perform.
        sp_func (function): Function that recomputes shortest-path structure from G.
        verbose (bool): Whether to print progress updates.

    Returns:
        original_efficiency (float): Baseline efficiency before any removal.
        efficiencies (list): Normalized efficiency after each step.
        percent_remaining (list): Remaining node percentage after each step.
        removed_nodes (list): Node removal order.
        removal_times (list): Time taken per step.
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

        # Updated: Do NOT stop if no non-isolated nodes remain.
        # Instead, filter out isolated nodes and continue.
        centrality = {
            node: score for node, score in centrality.items()
            if (
                (isinstance(G, nx.DiGraph) and (G.in_degree(node) > 0 or G.out_degree(node) > 0))
                or
                (not isinstance(G, nx.DiGraph) and G.degree(node) > 0)
            )
        }

        if not centrality:
            # No edges remain anywhere. The graph is fully isolated, but we continue
            # by selecting ANY node (max centrality from original dict).
            # This matches the least-version behavior: continue past isolation.
            centrality = {
                node: score for node, score in nx.betweenness_centrality(G, weight="duration_avg").items()
            }

        node_to_remove = max(centrality, key=centrality.get)

        # Collect edges to remove depending on graph type
        if isinstance(G, nx.DiGraph):
            edges_to_remove = list(G.in_edges(node_to_remove)) + list(G.out_edges(node_to_remove))
        else:
            edges_to_remove = list(G.edges(node_to_remove))

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


def betweenness_node_removal_unweighted(g, G, num_to_remove, sp_func, verbose=False):
    """
    Removes edges of nodes based on descending betweenness centrality,
    tracking the impact on global efficiency (normalized by initial value).

    Works for both directed and undirected graphs.

    Parameters:
        g (nx.Graph or nx.DiGraph): Original reference graph (for efficiency baseline).
        G (nx.Graph or nx.DiGraph): Working copy of graph (modified in-place).
        num_to_remove (int): Number of nodes (edge-removal steps) to perform.
        sp_func (function): Function that recomputes shortest-path structure from G.
        verbose (bool): Whether to print progress updates.

    Returns:
        original_efficiency (float): Baseline efficiency before any removal.
        efficiencies (list): Normalized efficiency after each step.
        percent_remaining (list): Remaining node percentage after each step.
        removed_nodes (list): Node removal order.
        removal_times (list): Time taken per step.
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
            centrality = nx.betweenness_centrality(G)
        except Exception as e:
            if verbose:
                print(f"Step {step} failed to compute centrality: {e}")
            break

        # Updated isolated-node handling to match least-logic
        centrality_valid = {
            node: score for node, score in centrality.items()
            if (
                (isinstance(G, nx.DiGraph) and (G.in_degree(node) > 0 or G.out_degree(node) > 0))
                or
                (not isinstance(G, nx.DiGraph) and G.degree(node) > 0)
            )
        }

        if centrality_valid:
            centrality_use = centrality_valid
        else:
            # No nodes with edges remain, fall back to full dictionary
            centrality_use = centrality

            if verbose:
                print("All nodes isolated. Continuing removal using full centrality set.")

        node_to_remove = max(centrality_use, key=centrality_use.get)

        if isinstance(G, nx.DiGraph):
            edges_to_remove = list(G.in_edges(node_to_remove)) + list(G.out_edges(node_to_remove))
        else:
            edges_to_remove = list(G.edges(node_to_remove))

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
            print(f"Step {step}: Removed edges of {node_to_remove} (Centrality: {centrality_use[node_to_remove]:.4f})")
            print(f"Normalized Efficiency: {normalized_eff:.4f}")
            print(f"Time taken: {elapsed:.4f} seconds\n")

    return original_efficiency, efficiencies, percent_remaining, removed_nodes, removal_times


def betweenness_node_removal_weighted_least(g, G, num_to_remove, sp_func, verbose=False):
    """
    Removes nodes starting from the lowest weighted betweenness centrality value.
    Otherwise identical to betweenness_node_removal_weighted.
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

    while removals_done < num_to_remove and G.number_of_edges() > 0:
        step += 1
        start_time = time.perf_counter()

        try:
            centrality = nx.betweenness_centrality(G, weight="duration_avg")
        except Exception as e:
            if verbose:
                print(f"Step {step} failed to compute centrality: {e}")
            break

        # Exclude isolated nodes
        centrality = {
            node: score for node, score in centrality.items()
            if (G.degree(node) > 0) or (isinstance(G, nx.DiGraph) and (G.in_degree(node) > 0 or G.out_degree(node) > 0))
        }

        if not centrality:
            if verbose:
                print("No valid nodes left.")
            break

        node_to_remove = min(centrality, key=centrality.get)

        edges_to_remove = list(G.in_edges(node_to_remove)) + list(G.out_edges(node_to_remove)) if isinstance(G, nx.DiGraph) else list(G.edges(node_to_remove))
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


def betweenness_node_removal_unweighted_least(g, G, num_to_remove, sp_func, verbose=False):
    """
    Removes nodes starting from the lowest weighted betweenness centrality value.
    Otherwise identical to betweenness_node_removal_weighted.
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

    while removals_done < num_to_remove and G.number_of_edges() > 0:
        step += 1
        start_time = time.perf_counter()

        try:
            centrality = nx.betweenness_centrality(G)
        except Exception as e:
            if verbose:
                print(f"Step {step} failed to compute centrality: {e}")
            break

        # Exclude isolated nodes
        centrality = {
            node: score for node, score in centrality.items()
            if (G.degree(node) > 0) or (isinstance(G, nx.DiGraph) and (G.in_degree(node) > 0 or G.out_degree(node) > 0))
        }

        if not centrality:
            if verbose:
                print("No valid nodes left.")
            break

        node_to_remove = min(centrality, key=centrality.get)

        edges_to_remove = list(G.in_edges(node_to_remove)) + list(G.out_edges(node_to_remove)) if isinstance(G, nx.DiGraph) else list(G.edges(node_to_remove))
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


def betweenness_edge_removal_weighted(g, G, num_to_remove, sp_func, verbose=False):
    """
    Removes edges based on descending edge betweenness centrality,
    tracking the impact on global efficiency (normalized by initial value).

    Works for both directed and undirected graphs.

    Parameters:
        g (nx.Graph or nx.DiGraph): Original reference graph (kept for API consistency).
        G (nx.Graph or nx.DiGraph): Working copy of graph (modified in-place).
        num_to_remove (int): Number of edges to remove.
        sp_func (function): Function that recomputes shortest-path structure from G.
        verbose (bool): Whether to print progress updates.

    Returns:
        original_efficiency (float): Baseline efficiency before any removal.
        efficiencies (list): Normalized efficiency after each step.
        percent_remaining (list): Remaining edge percentage after each step.
        removed_edges (list): Edge removal order.
        removal_times (list): Time taken per step.
    """
    total_edges = G.number_of_edges()
    sp = sp_func(G)
    original_efficiency = efficiency_graph(G, sp)

    if verbose:
        print(f"Original Efficiency: {original_efficiency:.4f}")

    efficiencies = [1.0]
    num_removed = [0]
    percent_remaining = [100.0]
    removed_edges = []
    removal_times = []

    removals_done = 0
    step = 0

    while removals_done < num_to_remove:
        step += 1
        start_time = time.perf_counter()

        if G.number_of_edges() == 0:
            if verbose:
                print(f"Step {step}: No edges remain to remove.")
            break

        try:
            centrality = nx.edge_betweenness_centrality(G, weight="duration_avg")
        except Exception as e:
            if verbose:
                print(f"Step {step} failed to compute edge centrality: {e}")
            break

        if not centrality:
            if verbose:
                print(f"Step {step}: No edge centrality values available.")
            break

        edge_to_remove = max(centrality, key=centrality.get)

        G.remove_edge(*edge_to_remove)
        removed_edges.append(edge_to_remove)
        removals_done += 1

        try:
            sp = sp_func(G)
            eff = efficiency_graph(G, sp)
        except Exception as e:
            if verbose:
                print(f"Error evaluating efficiency after removing edge {edge_to_remove}: {e}")
            break

        normalized_eff = eff / original_efficiency if original_efficiency != 0 else 0.0
        elapsed = round(time.perf_counter() - start_time, 4)

        efficiencies.append(normalized_eff)
        num_removed.append(removals_done)
        percent_remaining.append(100 * (1 - removals_done / total_edges))
        removal_times.append(elapsed)

        if verbose:
            print(f"Step {step}: Removed edge {edge_to_remove} (Centrality: {centrality[edge_to_remove]:.4f})")
            print(f"Normalized Efficiency: {normalized_eff:.4f}")
            print(f"Time taken: {elapsed:.4f} seconds\n")

    return original_efficiency, efficiencies, percent_remaining, removed_edges, removal_times

def betweenness_edge_removal_unweighted(g, G, num_to_remove, sp_func, verbose=False):
    """
    Removes edges based on descending edge betweenness centrality,
    tracking the impact on global efficiency (normalized by initial value).

    Works for both directed and undirected graphs.

    Parameters:
        g (nx.Graph or nx.DiGraph): Original reference graph (for efficiency baseline).
        G (nx.Graph or nx.DiGraph): Working copy of graph (modified in-place).
        num_to_remove (int): Number of edges to remove.
        sp_func (function): Function that recomputes shortest-path structure from G.
        verbose (bool): Whether to print progress updates.

    Returns:
        original_efficiency (float): Baseline efficiency before any removal.
        efficiencies (list): Normalized efficiency after each step.
        percent_remaining (list): Remaining edge percentage after each step.
        removed_edges (list): Edge removal order.
        removal_times (list): Time taken per step.
    """

    total_edges = G.number_of_edges()
    sp = sp_func(G)
    original_efficiency = efficiency_graph(G, sp)

    if verbose:
        print(f"Original Efficiency: {original_efficiency:.4f}")

    efficiencies = [1.0]
    num_removed = [0]
    percent_remaining = [100.0]
    removed_edges = []
    removal_times = []

    removals_done = 0
    step = 0

    while removals_done < num_to_remove:
        step += 1
        start_time = time.perf_counter()

        try:
            centrality = nx.edge_betweenness_centrality(G)
        except Exception as e:
            if verbose:
                print(f"Step {step} failed to compute edge centrality: {e}")
            break

        if centrality:
            centrality_use = centrality
        else:
            if verbose:
                print("No edges remain. Stopping removal.")
            break

        edge_to_remove = max(centrality_use, key=centrality_use.get)

        G.remove_edge(*edge_to_remove)
        removed_edges.append(edge_to_remove)
        removals_done += 1

        try:
            sp = sp_func(G)
            eff = efficiency_graph(G, sp)
        except Exception as e:
            if verbose:
                print(f"Error evaluating efficiency after removing {edge_to_remove}: {e}")
            break

        normalized_eff = eff / original_efficiency if original_efficiency != 0 else 0.0
        elapsed = round(time.perf_counter() - start_time, 4)

        efficiencies.append(normalized_eff)
        num_removed.append(removals_done)
        percent_remaining.append(100 * (1 - removals_done / total_edges) if total_edges > 0 else 0.0)
        removal_times.append(elapsed)

        if verbose:
            print(f"Step {step}: Removed edge {edge_to_remove} (Centrality: {centrality_use[edge_to_remove]:.4f})")
            print(f"Normalized Efficiency: {normalized_eff:.4f}")
            print(f"Time taken: {elapsed:.4f} seconds\n")

    return original_efficiency, efficiencies, percent_remaining, removed_edges, removal_times

def top_hubs_node_removal(g, G, num_to_remove, sp_func, verbose=False):
    """
    Degree based removal.
    Nodes are grouped by degree (descending).
    Within each degree group, nodes are removed using targeted efficiency drop.
    """

    # Group nodes by degree
    degree_buckets = defaultdict(list)
    for node, deg in G.degree():
        degree_buckets[deg].append(node)

    # Sort degrees descending
    sorted_degrees = sorted(degree_buckets.keys(), reverse=True)

    sp = sp_func(G)
    original_efficiency = efficiency_graph(g, sp)

    efficiencies = [1.0]
    num_removed = [0]
    percent_remaining = [100.0]
    removed_nodes = []
    removed_node_names = []
    removal_times = []

    total_nodes = G.number_of_nodes()
    removals_done = 0

    for deg in sorted_degrees:
        if removals_done >= num_to_remove:
            break

        candidates = degree_buckets[deg]

        if verbose:
            print(f"Processing degree {deg} with nodes {candidates}")

        while candidates and removals_done < num_to_remove:
            start_time = time.perf_counter()

            sp = sp_func(G)
            current_eff = efficiency_graph(g, sp)

            max_drop = -1
            best_node = None

            for node in candidates:
                if node not in G or G.degree(node) == 0:
                    continue

                temp_G = G.copy()
                if G.is_directed():
                    edges = list(temp_G.in_edges(node)) + list(temp_G.out_edges(node))
                else:
                    edges = list(temp_G.edges(node))
                temp_G.remove_edges_from(edges)

                try:
                    sp_temp = sp_func(temp_G)
                    eff_temp = efficiency_graph(g, sp_temp)
                except Exception:
                    continue

                drop = current_eff - eff_temp
                if drop > max_drop:
                    max_drop = drop
                    best_node = node

            if best_node is None:
                efficiencies.append(efficiencies[-1])
                num_removed.append(num_removed[-1] + 1)
                percent_remaining.append(100 * (1 - num_removed[-1] / total_nodes))
                removed_nodes.append(None)
                removed_node_names.append(None)
                removal_times.append(0)
                removals_done += 1
                continue

            # Remove edges of selected node
            if G.is_directed():
                edges_to_remove = list(G.in_edges(best_node)) + list(G.out_edges(best_node))
            else:
                edges_to_remove = list(G.edges(best_node))
            G.remove_edges_from(edges_to_remove)

            candidates.remove(best_node)

            try:
                sp_new = sp_func(G)
                eff = efficiency_graph(g, sp_new)
            except Exception as e:
                if verbose:
                    print(f"Error after removing edges of {best_node}: {e}")
                break

            elapsed = time.perf_counter() - start_time
            normalized_eff = eff / original_efficiency

            efficiencies.append(normalized_eff)
            removals_done += 1
            num_removed.append(removals_done)
            percent_remaining.append(100 * (1 - removals_done / total_nodes))
            removed_nodes.append(best_node)
            removed_node_names.append(g.nodes[best_node].get("name", str(best_node)))
            removal_times.append(round(elapsed, 4))

            if verbose:
                print(
                    f"Removed edges of node {best_node} "
                    f"(degree {deg}) → Normalized Efficiency: {normalized_eff:.4f}"
                )

    return (
        original_efficiency,
        efficiencies,
        percent_remaining,
        removed_nodes,
        removal_times,
        removed_node_names,
    )

def top_train_hubs_node_removal(g, G, num_to_remove, sp_func, verbose=False):
    """
    Remove edges of top train hubs (by number of trains) sequentially and track efficiency degradation.

    Works for both directed and undirected graphs. Returns:
        original_efficiency, efficiencies (normalized),
        percent_remaining, removed_nodes, removal_times, removed_node_names
    """
    import time

    # Calculate num_trains for each node
    node_trains = {}
    for node in G.nodes:
        num_trains = sum(edata.get('n_vehicles', 0) for _, _, edata in G.edges(node, data=True))
        node_trains[node] = num_trains

    # Sort nodes by num_trains descending
    hubs_sorted = sorted(node_trains.items(), key=lambda x: x[1], reverse=True)
    removal_nodes = [node for node, _ in hubs_sorted[:num_to_remove]]

    if verbose:
        print(f"Top train hubs to remove edges: {removal_nodes}")

    sp = sp_func(G)
    original_efficiency = efficiency_graph(g, sp)
    if verbose:
        print(f"Original Efficiency: {original_efficiency}")

    efficiencies = [1.0]
    num_removed = [0]
    percent_remaining = [100.0]
    removed_nodes = []
    removed_node_names = []
    removal_times = []

    total_nodes = G.number_of_nodes()

    for node in removal_nodes:
        start_time = time.perf_counter()

        if node not in G:
            if verbose:
                print(f"Node {node} missing, skipping.")
            efficiencies.append(efficiencies[-1])
            num_removed.append(num_removed[-1])
            percent_remaining.append(100 * (1 - num_removed[-1] / total_nodes))
            removed_nodes.append(None)
            removed_node_names.append(None)
            removal_times.append(0)
            continue

        # Remove all edges (incoming + outgoing) for directed graphs
        if isinstance(G, nx.DiGraph):
            edges_to_remove = list(G.in_edges(node)) + list(G.out_edges(node))
        else:
            edges_to_remove = list(G.edges(node))

        G.remove_edges_from(edges_to_remove)
        removed_nodes.append(node)
        removed_node_names.append(g.nodes[node].get('name', str(node)))

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
            print(f"Removed edges of node {node} ({removed_node_names[-1]}) → Normalized Efficiency: {normalized_eff:.4f}")
            print(f"Time taken: {elapsed:.4f} seconds\n")

    return original_efficiency, efficiencies, percent_remaining, removed_nodes, removal_times, removed_node_names


def export_removal_results_to_csv(
    output_path,
    efficiencies,
    percent_remaining,
    removed_nodes,
    removal_times,
    removed_node_names=None
):
    """
    Exports node removal simulation results to a CSV file.

    Parameters:
        output_path (str): Path to save the CSV.
        efficiencies (list of float): Normalized efficiencies at each step.
        percent_remaining (list of float): Remaining percent of nodes.
        removed_nodes (list): Nodes whose edges were removed.
        removal_times (list of float): Time taken per removal step.
        removed_node_names (list, optional): Names of removed nodes, aligned with removed_nodes.
    """
    steps = list(range(len(efficiencies)))
    
    # Pad removed_nodes and removal_times with None or 0 for step 0
    removed_nodes_full = [None] + removed_nodes
    removal_times_full = [0.0] + removal_times

    data = {
        "step": steps,
        "removed_node": removed_nodes_full,
        "normalized_efficiency": efficiencies,
        "percent_remaining": percent_remaining,
        "removal_time_seconds": removal_times_full
    }

    if removed_node_names:
        removed_node_names_full = [None] + removed_node_names
        data["removed_node_names"] = removed_node_names_full

    df = pd.DataFrame(data)

    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


def run_removal_simulations(
    subgraphs_by_size,
    num_to_remove=None,
    pct_to_remove=None,
    method='random',
    removal_type='node',
    seed=42,
    verbose=False,
    sp_func=None,
):
    """
    Run node or edge removal simulations across subgraphs and collect efficiency and timing metrics.

    Parameters:
        subgraphs_by_size (dict): Dictionary mapping subgraph sizes to lists of subgraphs (nx.Graph).
        num_to_remove (int): Fixed number of nodes or edges to remove. Optional if pct_to_remove is set.
        pct_to_remove (float): Percentage of nodes or edges to remove. Optional if num_to_remove is set.
        method (str): Removal strategy: 'random', 'targeted', or 'betweenness'.
        removal_type (str): Either 'node' or 'edge'.
        seed (int): Random seed for reproducibility.
        verbose (bool): Verbose logging for debug.
        sp_func (function): Shortest path structure calculator.

    Returns:
        pd.DataFrame: DataFrame where each row is a subgraph simulation.
    """
    results = []

    for size, graphs in subgraphs_by_size.items():
        for idx, L in enumerate(graphs):
            start = time.perf_counter()

            try:
                original_efficiency, efficiencies, num_removed, removed_entities, removal_times = (
                    simulate_fixed_node_removal_efficiency(
                        L_graph=L,
                        num_to_remove=num_to_remove,
                        pct_to_remove=pct_to_remove,
                        method=method,
                        removal_type=removal_type,
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
                "removed_entities": removed_entities,  # renamed for generality
                "removal_times": removal_times,
                "removal_type": removal_type,
                "removal_method": method,
                "removal_count": len(removed_entities)
            }

            for i, eff in enumerate(efficiencies):
                result[f"eff_after_{i}"] = eff

            results.append(result)

    return pd.DataFrame(results)

def get_runtime(version_label, run_func, subgraphs, 
                method, sp_func, seed, num_to_remove=None, pct_to_remove=None,
                removal_type='node',  # Added removal_type parameter
                target_sizes=None, verbose=False):
    """
    Runs `run_func` on subgraphs and measures runtimes per subgraph.

    Parameters:
    - version_label: str label for the run (e.g. "v1")
    - run_func: function to run (e.g. run_removal_simulations)
    - subgraphs: dict {size: [subgraph1, subgraph2, ...]}
    - num_to_remove, pct_to_remove, method, sp_func, seed: params to pass to run_func
    - removal_type: 'node' or 'edge' removal mode
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
                removal_type=removal_type,   # Pass removal_type here
                sp_func=sp_func,
                seed=seed,
                verbose=verbose
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
    removal_type='node',
    method='random',
    num_to_remove=None,
    pct_to_remove=None,
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
                removal_type=removal_type,
                method=method,
                num_to_remove=num_to_remove,
                pct_to_remove=pct_to_remove,
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
                        'removed_nodes': row['removed_entities'],
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
        p.add_tile(Vendors.CARTODBPOSITRON)

    p.renderers.append(graph)
    show(p)

    # Print highlighted node info below the plot
    if highlight_nodes:
        print("Highlighted Nodes:")
        for node in highlight_nodes:
            name = G.nodes[node].get("name", "Unknown")
            print(f"  Node {node}: {name}")


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

def plot_efficiency_results(percent_remaining, efficiencies, title="Impact of Node Removal on Network Efficiency (Normalized)"):
    """
    Plots the change in normalized efficiency as nodes are removed.

    Parameters:
    - num_removed: List of number of nodes removed
    - efficiencies: Corresponding list of normalized efficiencies
    - title: Plot title
    """
    plt.figure(figsize=(6, 4))
    plt.plot(percent_remaining, efficiencies, marker='o')
    plt.xlabel("Percentage Remaining")
    plt.ylabel("Normalized Efficiency")
    plt.title(title)
    plt.grid(True)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.show()

def plot_efficiency_results_multi(efficiency_data, size, versions=None):
    """
    Plots efficiency curves for specified versions and subgraph size.

    Parameters:
    - efficiency_data: dict from get_efficiency_curves output:
        { version_label: { size: [ { 'curve': [...], 'seed': int, ... }, ... ] } }
    - seeds: List of seed values used
    - size: Integer size of the subgraph to plot
    - versions: Optional list of version labels to plot (e.g. ['v0', 'v4']); if None, plots all
    """
    if versions is None:
        versions = sorted(efficiency_data.keys())

    plt.figure(figsize=(6 * len(versions), 5))

    for i, label in enumerate(versions, start=1):
        plt.subplot(1, len(versions), i)

        runs = efficiency_data.get(label, {}).get(size, [])
        if not runs:
            plt.title(f'{label} - Size {size} (no data)')
            continue

        for idx, run in enumerate(runs):
            curve = run['curve']
            seed = run.get('seed', 'unknown')
            plt.plot(curve, label=f'Seed {seed}, Run {idx + 1}')

        plt.title(f'{label} - Size {size}')
        plt.xlabel('Nodes removed')
        plt.ylabel('Efficiency')
        plt.legend(fontsize='small', loc='best')
        plt.grid(True)

    plt.tight_layout(rect=[0, 0, 0.9, 0.75])
    plt.show()