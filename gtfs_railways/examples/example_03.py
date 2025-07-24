"""
Minimal working example of the efficiency_graph function.
The efficiency_graph function has been optimised (version 0 to 5).
"""

print("Running examples...")

from gtfs_railways.utils.imports import *

attributes = load_gtfs(path_to_sqlite)
L_space_path = DATA_DIR / "pkl/10/graph_0.pkl"  # Path where the clean L-space graph was stored (cleaned routes)
L_graph = load_graph(L_space_path)
P = P_space(attributes, L_graph, "Rail", 5, 24, None)
travel_cost = get_all_GTC(L_graph, P, 3, 2, [5])
efficiency = efficiency_graph(L_graph, travel_cost)

print("Example finished.")
