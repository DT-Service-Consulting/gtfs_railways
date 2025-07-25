"""
Minimal working example of the efficiency_graph function.
The efficiency_graph function has been optimized (version 0 to 5).
"""

from gtfs_railways.utils.imports import *

@print_processing_file
def example_03():
    attributes = load_gtfs(path_to_sqlite)
    L_graph = load_graph(L_space_path)
    P = P_space(attributes, L_graph, "Rail", 5, 24, None)
    travel_cost = get_all_GTC(L_graph, P, 3, 2, [5])
    efficiency = efficiency_graph(L_graph, travel_cost)
    # pprint(efficiency)

example_03()
