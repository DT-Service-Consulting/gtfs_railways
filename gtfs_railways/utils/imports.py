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