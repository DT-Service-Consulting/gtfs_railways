print("Running examples...")
from gtfs_railways.config import EXAMPLES_DIR, DATA_DIR
from gtfs_railways.functions.core import load_graph, \
    load_gtfs, efficiency_graph
from gtfs_railways.functions.v0 import P_space as P_space_0, \
    get_all_GTC as get_all_GTC_0
from gtfs_railways.functions.v1 import P_space as P_space_1, \
    get_all_GTC as get_all_GTC_1
from pprint import pprint

path_to_sqlite = str( DATA_DIR / "sqlite/belgium.sqlite" )
attributes = load_gtfs(path_to_sqlite)

L_space_path = EXAMPLES_DIR / "10/graph_0.pkl"  # Path where the clean L-space graph was stored (cleaned routes)
L_graph = load_graph(L_space_path)

P_0 = P_space_0(attributes, L_graph, "Rail", 5, 24, None)
P_1 = P_space_1(attributes, L_graph, "Rail", 5, 24, None)

TC_0 = get_all_GTC_0( L_graph, P_space_0,3, 2, [5])
TC_1 = get_all_GTC_1( L_graph, P_space_1,3, 2, [5])


eff_0 = efficiency_graph(L_graph, TC_0)
eff_1 = efficiency_graph(L_graph, TC_1)


pprint(eff_0)
pprint(eff_1)

print("Example finished.")
