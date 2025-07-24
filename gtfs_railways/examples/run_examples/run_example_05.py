print("Running examples...")
from gtfs_railways.config import DATA_DIR
from pprint import pprint

from gtfs_railways.functions.core import load_gtfs, load_all_subgraphs, run_removal_simulations, make_sp_func
from gtfs_railways.functions.v0 import P_space as P_space_0, get_all_GTC as get_all_GTC_0
from gtfs_railways.functions.v1 import P_space as P_space_1, get_all_GTC as get_all_GTC_1
from gtfs_railways.functions.v2 import P_space as P_space_2, get_all_GTC as get_all_GTC_2
from gtfs_railways.functions.v3 import P_space as P_space_3, get_all_GTC as get_all_GTC_3
from gtfs_railways.functions.v4 import P_space as P_space_4, get_all_GTC as get_all_GTC_4
from gtfs_railways.functions.v4 import P_space as P_space_5, get_all_GTC as get_all_GTC_5

path_to_sqlite = str( DATA_DIR / "sqlite/belgium.sqlite" )
attributes = load_gtfs(path_to_sqlite)

L_graphs_path = str ( DATA_DIR / "pkl" )
L_graphs = load_all_subgraphs(base_dir=L_graphs_path)

sp_func_0 = make_sp_func(attributes, get_all_GTC_0, P_space_0)
sp_func_1 = make_sp_func(attributes, get_all_GTC_1, P_space_1)
# sp_func_2 = make_sp_func(attributes, get_all_GTC_2, P_space_2)
# sp_func_3 = make_sp_func(attributes, get_all_GTC_3, P_space_3)
# sp_func_4 = make_sp_func(attributes, get_all_GTC_4, P_space_4)
# sp_func_5 = make_sp_func(attributes, get_all_GTC_5, P_space_5)

results_random_0 = run_removal_simulations(subgraphs_by_size=L_graphs, sp_func=sp_func_0, num_to_remove=5, method='random', seed=42)
results_random_1 = run_removal_simulations(subgraphs_by_size=L_graphs, sp_func=sp_func_1, num_to_remove=5, method='random', seed=42)


# pprint(results_random_0)
pprint(results_random_1)
# pprint(efficiencies_1[5])
# pprint(efficiencies_2[5])
# pprint(efficiencies_3[5])
# pprint(efficiencies_4[5])
# pprint(efficiencies_5[5])

print("Example finished.")
