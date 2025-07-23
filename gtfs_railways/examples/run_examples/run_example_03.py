print("Running examples...")
from gtfs_railways.config import EXAMPLES_DIR, DATA_DIR
from pprint import pprint

from gtfs_railways.functions.core import load_graph, \
    load_gtfs, efficiency_graph, simulate_fixed_node_removal_efficiency
from gtfs_railways.functions.v0 import P_space as P_space_0, \
    get_all_GTC as get_all_GTC_0
from gtfs_railways.functions.v1 import P_space as P_space_1, \
    get_all_GTC as get_all_GTC_1
from gtfs_railways.functions.v2 import P_space as P_space_2, \
    get_all_GTC as get_all_GTC_2
from gtfs_railways.functions.v3 import P_space as P_space_3, \
    get_all_GTC as get_all_GTC_3
from gtfs_railways.functions.v4 import P_space as P_space_4, \
    get_all_GTC as get_all_GTC_4
from gtfs_railways.functions.v5 import P_space as P_space_5, \
    get_all_GTC as get_all_GTC_5

path_to_sqlite = str( DATA_DIR / "sqlite/belgium.sqlite" )
attributes = load_gtfs(path_to_sqlite)

L_space_path = EXAMPLES_DIR / "10/graph_0.pkl"  # Path where the clean L-space graph was stored (cleaned routes)
L_graph = load_graph(L_space_path)

P_0 = P_space_0(attributes, L_graph, "Rail", 5, 24, None)
P_1 = P_space_1(attributes, L_graph, "Rail", 5, 24, None)
P_2 = P_space_2(attributes, L_graph, "Rail", 5, 24, None)
P_3 = P_space_3(attributes, L_graph, "Rail", 5, 24, None)
P_4 = P_space_4(attributes, L_graph, "Rail", 5, 24, None)
P_5 = P_space_5(attributes, L_graph, "Rail", 5, 24, None)

TC_0 = get_all_GTC_0(L_graph, P_0, 3, 2, [5])
TC_1 = get_all_GTC_1(L_graph, P_1, 3, 2, [5])
TC_2 = get_all_GTC_2(L_graph, P_2, 3, 2, [5])
TC_3 = get_all_GTC_3(L_graph, P_3, 3, 2, [5])
TC_4 = get_all_GTC_4(L_graph, P_4, 3, 2, [5])
TC_5 = get_all_GTC_5(L_graph, P_5, 3, 2, [5])

# pprint(TC_0[298][299])
# pprint(TC_1[298][299])
# pprint(TC_2[298][299])

eff_0 = efficiency_graph(L_graph, TC_0)
eff_1 = efficiency_graph(L_graph, TC_1)
eff_2 = efficiency_graph(L_graph, TC_2)
eff_3 = efficiency_graph(L_graph, TC_3)
eff_4 = efficiency_graph(L_graph, TC_4)
eff_5 = efficiency_graph(L_graph, TC_5)

# pprint(eff_0)
# pprint(eff_1)
# pprint(eff_2)
# pprint(eff_3)
# pprint(eff_4)
# pprint(eff_5)

print("Example finished.")
