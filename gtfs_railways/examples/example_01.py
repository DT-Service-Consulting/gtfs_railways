"""
Minimal working example of the P-space function.
The P-space is the ensemble of all possible paths in the railway network.
"""

from gtfs_railways.utils.imports import *

@print_processing_file
def example_01():
    attributes = load_gtfs(path_to_sqlite)
    L_graph = load_graph(L_space_path)
    P_graph = P_space(
        attributes,
        L_graph,
        "Rail",
        5,
        24,
        None
        )


example_01()