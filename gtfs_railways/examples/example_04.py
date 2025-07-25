"""
Minimal working example of the nodes removal simulation on a single graph.
"""

print("Running example_04.py")

from gtfs_railways.utils.imports import *

@print_processing_file
def example_04():
    attributes = load_gtfs(path_to_sqlite)
    L_graph = load_graph(L_space_path)
    P = P_space(attributes, L_graph, "Rail", 5, 24, None)

    sp_func = make_sp_func(attributes, get_all_GTC, P_space)

    original_efficiency, efficiencies, num_removed, removed_nodes, removal_times = \
        simulate_fixed_node_removal_efficiency(L_graph=L_graph, sp_func=sp_func, num_to_remove=5, method='random', seed=42)

example_04()