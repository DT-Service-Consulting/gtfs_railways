import pandas as pd
import os
from itertools import islice
from statistics import mean 
from gtfspy import import_gtfs, gtfs, networks
from bokeh.io import show, export_png
from bokeh.models import (ColorBar,
                          HoverTool,
                          LinearColorMapper, Circle, 
                          MultiLine, WheelZoomTool,GMapOptions, 
                          DataRange1d, Button, EdgesAndLinkedNodes,
                          ColorBar)
from bokeh.layouts import column
from bokeh.palettes import RdYlGn11
from bokeh.plotting import figure,from_networkx, gmap
from bokeh.tile_providers import CARTODBPOSITRON
from pyproj import Transformer
from collections import Counter
import networkx as nx
import pickle
from thefuzz import fuzz
import geopy.distance
from IPython.display import clear_output
import time
import copy
import random

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

    
def load_sqlite(imported_database_path):
    return gtfs.GTFS(imported_database_path)

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


def plot_graph(G, space="L", back_map=False, MAPS_API_KEY=None, color_by="",edge_color_by="", export_name=""):
    '''Plots a networkx graph. Arguments:
    -G: the nx graph
    -space: either "L" or "P" depending on which space you are plotting
    -back_map: either False (no map), "GMAPS" (for Google Maps) or "OSM" for OpenStreetMap
    -MAPS_API_KEY: a valid Google maps api key if back_map="GMAPS"
    -color_by: string with the name of an attribute in G.nodes that will be used to color the nodes
    -edge_color_by: string with the name of an attribute in G.edges that will be used to color the nodes'''
        
    if back_map=="GMAPS":
        map_options = GMapOptions(lat=list(G.nodes(data=True))[0][1]["lat"], 
                                  lng=list(G.nodes(data=True))[0][1]["lon"], 
                                  map_type="roadmap", 
                                  zoom=11)
        p = gmap(MAPS_API_KEY, map_options)
    else:
        p = figure(height = 600 ,
        width = 950, 
        toolbar_location = 'below',
        tools = "pan, wheel_zoom, box_zoom, reset, save")
    
    #Build dictionary of node positions for visualizations
    pos_dict={}
    #Reproject for OSM
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857")
    for i,d in G.nodes(data=True):
        if back_map=="OSM":
            x2,y2=transformer.transform(float(d["lat"]),float(d["lon"]))
        else:
            x2,y2=float(d["lon"]),float(d["lat"])
        pos_dict[int(i)]=(x2,y2)
    
    # Plot updated graph
    graph = from_networkx(G, layout_function=pos_dict)

    # Add hover tools
    node_hover_tool = HoverTool(tooltips=[("index", "@index"),
                                          ("name", "@name")],
                               renderers=[graph.node_renderer])

    hover_edges = HoverTool(tooltips=[("duration_avg", "@duration_avg")],
                            renderers=[graph.edge_renderer],
                           line_policy="interp")
    
        
    if space == 'P':
        hover_edges = HoverTool(tooltips=[("avg_wait", "@avg_wait")],
                            renderers=[graph.edge_renderer],
                           line_policy="interp")

    p.add_tools(node_hover_tool,hover_edges)

    # Define the visualization
    if color_by:
        mapper = LinearColorMapper(palette=RdYlGn11)
        graph.node_renderer.glyph = Circle(size=7,fill_color={'field': color_by, 'transform': mapper})
    else:
        graph.node_renderer.glyph = Circle(size=7)

    if edge_color_by:
        mapper = LinearColorMapper(palette=RdYlGn11)
        graph.edge_renderer.glyph = MultiLine(line_width=4, line_alpha=.5, line_color={'field': edge_color_by, 'transform': mapper})      
        color_bar = ColorBar(color_mapper=mapper, label_standoff=12, border_line_color=None, location=(0,0))
        p.add_layout(color_bar,"right")
    
    graph.node_renderer.selection_glyph = Circle(fill_color='blue')
    graph.node_renderer.hover_glyph = Circle(fill_color='red')

    #graph.selection_policy = NodesAndLinkedEdges()
    
    p.toolbar.active_scroll = p.select_one(WheelZoomTool)
    
    # Different hover and select policies depending on the space
    if space == 'P':
        graph.edge_renderer.glyph = MultiLine(line_color = 'edge_color')
        graph.edge_renderer.selection_glyph = MultiLine(line_color='edge_color', line_width=5)
        graph.edge_renderer.hover_glyph = MultiLine(line_color='edge_color', line_width=10)
    
    if space == 'L':
        graph.edge_renderer.selection_glyph = MultiLine(line_color='blue', line_width=5)
        graph.edge_renderer.hover_glyph = MultiLine(line_color='red', line_width=5)
    
    p.renderers.append(graph)
    
    if back_map=="OSM":
        p.add_tile("CartoDB Positron")

    if export_name:
        export_png(p, filename=export_name+".png")
    else:
        show(p)


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
                map_options = GMapOptions(lat=list(G.nodes(data=True))[0][1]["lat"], 
                                          lng=list(G.nodes(data=True))[0][1]["lon"], 
                                          map_type="roadmap", 
                                          zoom=11)
                p = gmap(MAPS_API_KEY, map_options)
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

#######################################################3
    
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
                map_options = GMapOptions(lat=list(G.nodes(data=True))[0][1]["lat"], 
                                          lng=list(G.nodes(data=True))[0][1]["lon"], 
                                          map_type="roadmap", 
                                          zoom=11)
                p = gmap(MAPS_API_KEY, map_options)
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

def save_graph(G,path):
    #Rename nodes to 0..n
    G_res=nx.convert_node_labels_to_integers(G)
    #nx.write_gpickle(G_res,path)    

    with open(path, 'wb') as f:
        pickle.dump(G_res, f)
    


# Method to get the routeids of the subway lines
def get_routes_for_mode(g, mode):
    
    cur = g.conn.cursor()
    subway = 1
    routes = list()
    
    # Get all routes that are of the subway type (type = 1)
    t_results = cur.execute("SELECT route_I FROM routes WHERE type={mode}".format(mode=mode_from_string(mode)))
    route_list = t_results.fetchall()
    for r in route_list:
        routes.append(r[0])
    
    return routes

# Method to get the routeids of the subway lines
def get_routes_for_rail(graph):
    cur = graph.conn.cursor()
    #subway = 1
    rail = 2
    routes = list()
    
    # Get all routes that are of the subway type (type = 1)
    t_results = cur.execute("SELECT route_I FROM routes WHERE type={rail}".format(rail=rail))
    route_list = t_results.fetchall()
    for r in route_list:
        routes.append(r[0])
    
    return routes

# Method to get a corresponding color for each route
def get_color_per_route(graph, routes):
    colors = dict()
    for r in routes:
        cur = graph.conn.cursor()
        
        # Get the color corresponding to route r
        c_results = cur.execute("SELECT color FROM routes WHERE route_I={r}"
                              .format(r=r))
        color = c_results.fetchone()
        colors[r] = color[0]

    return colors

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

    original_efficiency = eg(g, G)
    if verbose:
        print(f"Original Efficiency: {original_efficiency}")
    efficiencies = [1.0]
    num_removed = [0]
    removed_nodes = []
    removal_times = []

    for i, node in enumerate(removal_nodes):
        start_time = time.perf_counter()

        # Skip if node is already isolated (no edges)
        if G.in_degree(node) == 0 and G.out_degree(node) == 0:
            if verbose:
                print(f"Step {i + 1}: Node {node} already isolated, skipping.")
            efficiencies.append(efficiencies[-1])
            num_removed.append(num_removed[-1])
            continue

        edges_to_remove = list(G.in_edges(node)) + list(G.out_edges(node))
        G.remove_edges_from(edges_to_remove)
        removed_nodes.append(node)

        try:
            eff = eg(g, G)
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
    original_efficiency = eg(g, G)
    if verbose:
        print(f"Original Efficiency: {original_efficiency}")
    efficiencies = [1.0]
    num_removed = [0]
    removed_nodes = []
    removal_times = []

    removals_done = 0
    step = 0

    while removals_done < num_to_remove:
        start_time = time.perf_counter()
        step += 1

        current_eff = eg(g, G)
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
                eff = eg(g, temp_G)
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
            eff = eg(g, G)
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
    original_efficiency = eg(g, G)
    if verbose:
        print(f"Original Efficiency: {original_efficiency}")
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
            eff = eg(g, G)
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

