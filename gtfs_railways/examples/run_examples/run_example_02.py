print("Running examples...")
from gtfs_railways.config import EXAMPLES_DIR, DATA_DIR
from gtfs_railways.functions.core import load_graph, load_gtfs
from gtfs_railways.functions.v1 import P_space, get_all_GTC
from pprint import pprint

path_to_sqlite = str( DATA_DIR / "sqlite/belgium.sqlite" )
attributes = load_gtfs(path_to_sqlite)

L_space_path = EXAMPLES_DIR / "10/graph_0.pkl"  # Path where the clean L-space graph was stored (cleaned routes)
L_graph = load_graph(L_space_path)
P_graph = P_space(attributes, L_graph, "Rail", 5, 24, None)

travel_cost = get_all_GTC( L_graph, P_graph,3, 2, [5])
pprint(travel_cost)

print("Example finished.")

