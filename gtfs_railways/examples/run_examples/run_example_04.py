print("Running examples...")
from gtfs_railways.config import EXAMPLES_DIR, DATA_DIR
from pprint import pprint

from gtfs_railways.functions.core import load_graph, load_gtfs, simulate_fixed_node_removal_efficiency, make_sp_func
from gtfs_railways.functions.v0 import P_space as P_space_0, get_all_GTC as get_all_GTC_0
from gtfs_railways.functions.v1 import P_space as P_space_1, get_all_GTC as get_all_GTC_1
from gtfs_railways.functions.v2 import P_space as P_space_2, get_all_GTC as get_all_GTC_2
from gtfs_railways.functions.v3 import P_space as P_space_3, get_all_GTC as get_all_GTC_3
from gtfs_railways.functions.v4 import P_space as P_space_4, get_all_GTC as get_all_GTC_4
from gtfs_railways.functions.v4 import P_space as P_space_5, get_all_GTC as get_all_GTC_5

path_to_sqlite = str( DATA_DIR / "sqlite/belgium.sqlite" )
attributes = load_gtfs(path_to_sqlite)

L_space_path = DATA_DIR / "pkl/10/graph_0.pkl"  # Path where the clean L-space graph was stored (cleaned routes)
L_graph = load_graph(L_space_path)

sp_func_0 = make_sp_func(attributes, get_all_GTC_0, P_space_0)
sp_func_1 = make_sp_func(attributes, get_all_GTC_1, P_space_1)
sp_func_2 = make_sp_func(attributes, get_all_GTC_2, P_space_2)
sp_func_3 = make_sp_func(attributes, get_all_GTC_3, P_space_3)
sp_func_4 = make_sp_func(attributes, get_all_GTC_4, P_space_4)
sp_func_5 = make_sp_func(attributes, get_all_GTC_5, P_space_5)

original_efficiency_0, efficiencies_0, num_removed_0, removed_nodes_0, removal_times_0 = simulate_fixed_node_removal_efficiency(L_graph=L_graph, sp_func=sp_func_0, num_to_remove=5, method='random', seed=42)
original_efficiency_1, efficiencies_1, num_removed_1, removed_nodes_1, removal_times_1 = simulate_fixed_node_removal_efficiency(L_graph=L_graph, sp_func=sp_func_1, num_to_remove=5, method='random', seed=42)
original_efficiency_2, efficiencies_2, num_removed_2, removed_nodes_2, removal_times_2 = simulate_fixed_node_removal_efficiency(L_graph=L_graph, sp_func=sp_func_2, num_to_remove=5, method='random', seed=42)
original_efficiency_3, efficiencies_3, num_removed_3, removed_nodes_3, removal_times_3 = simulate_fixed_node_removal_efficiency(L_graph=L_graph, sp_func=sp_func_3, num_to_remove=5, method='random', seed=42)
original_efficiency_4, efficiencies_4, num_removed_4, removed_nodes_4, removal_times_4 = simulate_fixed_node_removal_efficiency(L_graph=L_graph, sp_func=sp_func_4, num_to_remove=5, method='random', seed=42)
original_efficiency_5, efficiencies_5, num_removed_5, removed_nodes_5, removal_times_5 = simulate_fixed_node_removal_efficiency(L_graph=L_graph, sp_func=sp_func_5, num_to_remove=5, method='random', seed=42)

pprint(efficiencies_0[5])
pprint(efficiencies_1[5])
pprint(efficiencies_2[5])
pprint(efficiencies_3[5])
pprint(efficiencies_4[5])
pprint(efficiencies_5[5])

print("Example finished.")
