from gtfs_railways.utils.config import DATA_DIR
from gtfs_railways.functions.core import load_gtfs
from gtfs_railways.functions.core import load_graph
from gtfs_railways.functions.core import compute_time

from gtfs_railways.functions.core import make_sp_func
from gtfs_railways.functions.core import simulate_fixed_node_removal_efficiency
from gtfs_railways.functions.core import export_removal_results_to_csv

from gtfs_railways.functions.v0 import P_space as P_space_0, get_all_GTC as get_all_GTC_0
from gtfs_railways.functions.v1 import P_space as P_space_1, get_all_GTC as get_all_GTC_1
from gtfs_railways.functions.v2 import P_space as P_space_2, get_all_GTC as get_all_GTC_2
from gtfs_railways.functions.v3 import P_space as P_space_3, get_all_GTC as get_all_GTC_3
from gtfs_railways.functions.v4 import P_space as P_space_4, get_all_GTC as get_all_GTC_4
from gtfs_railways.functions.v4 import P_space as P_space_5, get_all_GTC as get_all_GTC_5

simulate_fixed_node_removal_efficiency = compute_time(simulate_fixed_node_removal_efficiency)

# attributes = load_gtfs(str(DATA_DIR / "sqlite/belgium.sqlite"))
# L_graph = load_graph(DATA_DIR / "pkl/belgium_routesCleaned.pkl")
# print("Working on BE data")

attributes = load_gtfs(str(DATA_DIR / "sqlite/NL.sqlite"))
L_graph = load_graph(DATA_DIR / "pkl/nl_merged.pkl")
print("Working on NL data")

sp_func = make_sp_func(attributes, get_all_GTC_4, P_space_4)

original_efficiency, efficiencies, pct_remaining, removed_nodes, removal_times = \
    simulate_fixed_node_removal_efficiency(
        L_graph=L_graph,
        sp_func=sp_func,
        num_to_remove=5,
        # pct_to_remove=40,
        method='random',
        seed=42,
        verbose=True
    )

results_path = DATA_DIR/"results/test.csv"

export_removal_results_to_csv(
    output_path=results_path,
    efficiencies=efficiencies,
    percent_remaining=pct_remaining,
    removed_nodes=removed_nodes,
    removal_times=removal_times
)

print("Completed")