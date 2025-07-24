"""
This file contains imports for the examples and utility functions of the gtfs_railways package.
"""

from gtfs_railways.decorators.decorators import print_processing_file
from gtfs_railways.utils.config import DATA_DIR

from gtfs_railways.functions.core import load_graph
from gtfs_railways.functions.core import load_gtfs
from gtfs_railways.functions.core import efficiency_graph
from gtfs_railways.functions.core import make_sp_func
from gtfs_railways.functions.core import simulate_fixed_node_removal_efficiency
from gtfs_railways.functions.core import run_removal_simulations
from gtfs_railways.functions.core import load_all_subgraphs

from gtfs_railways.functions.v0 import P_space
from gtfs_railways.functions.v0 import get_all_GTC
from pprint import pprint

path_to_sqlite = str( DATA_DIR / "sqlite/belgium.sqlite" )
L_space_path = DATA_DIR / "pkl/10/graph_0.pkl"  # one of the smallest possible graphs.
L_graphs_path = str ( DATA_DIR / "pkl" )
