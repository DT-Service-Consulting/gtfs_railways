"""
Minimal working example of the P-space function.
The P-space is the ensemble of all possible paths in the railway network.
"""

print("Running examples...")

from gtfs_railways.utils.imports import *

attributes = load_gtfs(path_to_sqlite)
L_space_path = DATA_DIR / "pkl/10/graph_0.pkl"  # Path where the clean L-space graph was stored (cleaned routes)
L_graph = load_graph(L_space_path)
P_graph = P_space(attributes, L_graph, "Rail", 5, 24, None)

print("Example finished.")

