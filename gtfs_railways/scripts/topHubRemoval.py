import sys
import time
from gtfs_railways.utils.config import DATA_DIR
from gtfs_railways.functions.core import (
    load_gtfs,
    load_graph,
    compute_time,
    make_sp_func,
    simulate_fixed_node_removal_efficiency,
    export_removal_results_to_csv
)
from gtfs_railways.functions.v4 import P_space as P_space_4, get_all_GTC as get_all_GTC_4

simulate_fixed_node_removal_efficiency = compute_time(simulate_fixed_node_removal_efficiency)

results_dir = DATA_DIR / "results/removal_runs_tophubs_node_BE"
results_dir.mkdir(parents=True, exist_ok=True)

attributes = load_gtfs(str(DATA_DIR / "sqlite/belgium.sqlite"))
L_graph = load_graph(DATA_DIR / "pkl/belgium_routesCleaned.pkl")
log_path = results_dir / "removal_tophub_BE.log"
print("Working on BE data")

# attributes = load_gtfs(str(DATA_DIR / "sqlite/NL.sqlite"))
# L_graph = load_graph(DATA_DIR / "pkl/nl_merged.pkl")
# log_path = results_dir / "removal_tophub_NL.log"
# print("Working on NL data")

sp_func = make_sp_func(attributes, get_all_GTC_4, P_space_4)

with open(log_path, "w") as log_file:
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = log_file
    sys.stderr = log_file
    try:
        num_nodes_total = L_graph.number_of_nodes()
        print(f"Total number of nodes in the full graph: {num_nodes_total}", flush=True)

        original_efficiency, efficiencies, pct_remaining, removed_nodes, removal_times, removed_node_names = \
            simulate_fixed_node_removal_efficiency(
                L_graph=L_graph,
                sp_func=sp_func,
                num_to_remove=10,  
                removal_type='node',
                method='top_hubs',
		verbose=True
            )

        output_filename = f"tophub_removal_nodes{num_nodes_total}.csv"
        output_path = results_dir / output_filename

        export_removal_results_to_csv(
            output_path=output_path,
            efficiencies=efficiencies,
            percent_remaining=pct_remaining,
            removed_nodes=removed_nodes,
            removal_times=removal_times,
	    removed_node_names=removed_node_names
        )

        print("Simulation completed. Results saved to CSV.", flush=True)

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
