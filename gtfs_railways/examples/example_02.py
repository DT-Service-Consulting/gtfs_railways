"""
Minimal working example of the GTC function.
GTC is the computational cost to the calculation of all connections between two nodes in the P-space.
The travel cost is the calculation of all possible paths between any two nodes in the P-space.
"""

print("Running examples...")

from gtfs_railways.utils.imports import *

attributes = load_gtfs(path_to_sqlite)
L_space_path = DATA_DIR / "pkl/10/graph_0.pkl"  # Path where the clean L-space graph was stored (cleaned routes)
L_graph = load_graph(L_space_path)
P_graph = P_space(attributes, L_graph, "Rail", 5, 24, None)

travel_cost = get_all_GTC( L_graph, P_graph,3, 2, [5])
pprint(travel_cost)

print("Example finished.")

