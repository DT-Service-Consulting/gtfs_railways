print("Running examples...")


from gtfs_railways.utils.imports import *

attributes = load_gtfs(path_to_sqlite)

L_graphs_path = str ( DATA_DIR / "pkl" )
L_graphs = load_all_subgraphs(base_dir=L_graphs_path)

sp_func = make_sp_func(attributes, get_all_GTC, P_space)

results_random = run_removal_simulations(
    subgraphs_by_size=L_graphs,
    sp_func=sp_func,
    num_to_remove=5,
    method='random',
    seed=42)


pprint(results_random)

print("Example finished.")
