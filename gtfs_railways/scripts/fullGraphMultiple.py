import time
import sys
from pathlib import Path

from gtfs_railways.utils.config import DATA_DIR
from gtfs_railways.functions.core import load_gtfs, load_graph, compute_time
from gtfs_railways.functions.core import make_sp_func
from gtfs_railways.functions.core import simulate_fixed_node_removal_efficiency
from gtfs_railways.functions.core import export_removal_results_to_csv

from gtfs_railways.functions.v4 import P_space as P_space_4, get_all_GTC as get_all_GTC_4

simulate_fixed_node_removal_efficiency = compute_time(simulate_fixed_node_removal_efficiency)

results_dir = DATA_DIR / "results/random_removal_runs_BE"
results_dir.mkdir(parents=True, exist_ok=True)

attributes = load_gtfs(str(DATA_DIR / "sqlite/belgium.sqlite"))
L_graph = load_graph(DATA_DIR / "pkl/belgium_routesCleaned.pkl")
log_path = results_dir / "random_removal_BE.log"
print("Working on BE data")

# attributes = load_gtfs(str(DATA_DIR / "sqlite/NL.sqlite"))
# L_graph = load_graph(DATA_DIR / "pkl/nl_merged.pkl")
# log_path = results_dir / "random_removal_NL.log"
# print("Working on NL data")

sp_func = make_sp_func(attributes, get_all_GTC_4, P_space_4)

with open(log_path, "w") as log_file:
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = log_file
    sys.stderr = log_file
    try:
        base_seed = int(time.time())
        seeds = [base_seed + i for i in range(10)]

        num_nodes_total = L_graph.number_of_nodes()
        print(f"Total number of nodes in the full graph: {num_nodes_total}", flush=True)

        for seed in seeds:
            print(f"\nRunning simulation with seed {seed}", flush=True)
            original_efficiency, efficiencies, pct_remaining, removed_nodes, removal_times = \
                simulate_fixed_node_removal_efficiency(
                    L_graph=L_graph,
                    sp_func=sp_func,
                    num_to_remove=5,
                    # pct_to_remove=100,
                    method='random',
                    seed=seed,
                    verbose=False
                )

            filename = f"random_removal_seed{seed}_nodes{num_nodes_total}.csv"
            output_path = results_dir / filename

            export_removal_results_to_csv(
                output_path=output_path,
                efficiencies=efficiencies,
                percent_remaining=pct_remaining,
                removed_nodes=removed_nodes,
                removal_times=removal_times
            )

        print(f"\nCompleted all {len(seeds)} simulations.", flush=True)
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
