"""
Minimal working example of the GTC function.
GTC is the computational cost to the calculation of all connections between two nodes in the P-space.
The travel cost is the calculation of all possible paths between any two nodes in the P-space.
"""


from gtfs_railways.utils.imports import *

@print_processing_file
def example_02():
    attributes = load_gtfs(path_to_sqlite)
    L_graph = load_graph(L_space_path)
    P_graph = P_space(attributes, L_graph, "Rail", 5, 24, None)

    travel_cost = get_all_GTC( L_graph, P_graph,3, 2, [5])
    pprint(travel_cost)

example_02()
