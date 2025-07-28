# Python

import pytest # type: ignore

from gtfs_railways.functions.core import load_gtfs
from gtfs_railways.functions.core import load_graph
from gtfs_railways.functions.core import efficiency_graph
from gtfs_railways.functions.core import make_sp_func
from gtfs_railways.functions.core import simulate_fixed_node_removal_efficiency
from gtfs_railways.functions.core import run_removal_simulations

from gtfs_railways.functions.v0 import P_space as P_space_0, get_all_GTC as get_all_GTC_0
from gtfs_railways.functions.v1 import P_space as P_space_1, get_all_GTC as get_all_GTC_1
from gtfs_railways.functions.v2 import P_space as P_space_2, get_all_GTC as get_all_GTC_2
from gtfs_railways.functions.v3 import P_space as P_space_3, get_all_GTC as get_all_GTC_3
from gtfs_railways.functions.v4 import P_space as P_space_4, get_all_GTC as get_all_GTC_4
from gtfs_railways.functions.v4 import P_space as P_space_5, get_all_GTC as get_all_GTC_5


from gtfs_railways.tests.conftest import L_space_10_path
from gtfs_railways.tests.conftest import L_space_many
from gtfs_railways.tests.conftest import attributes_path

def test_get_all_GTC(L_space_10_path, attributes_path):
    attributes = load_gtfs(attributes_path)
    L_10_graph = load_graph(L_space_10_path)

    # P Space Graph
    P_0 = P_space_0(attributes, L_10_graph, "Rail", 5, 24, None)
    P_1 = P_space_1(attributes, L_10_graph, "Rail", 5, 24, None)
    P_2 = P_space_2(attributes, L_10_graph, "Rail", 5, 24, None)
    P_3 = P_space_3(attributes, L_10_graph, "Rail", 5, 24, None)
    P_4 = P_space_4(attributes, L_10_graph, "Rail", 5, 24, None)
    P_5 = P_space_5(attributes, L_10_graph, "Rail", 5, 24, None)

    # Travel Cost
    TC_0 = get_all_GTC_0(L_10_graph, P_0, 3, 2, [5])
    TC_1 = get_all_GTC_1(L_10_graph, P_1, 3, 2, [5])
    TC_2 = get_all_GTC_2(L_10_graph, P_2, 3, 2, [5])
    TC_3 = get_all_GTC_3(L_10_graph, P_3, 3, 2, [5])
    TC_4 = get_all_GTC_4(L_10_graph, P_4, 3, 2, [5])
    TC_5 = get_all_GTC_1(L_10_graph, P_5, 3, 2, [5])

    # Check if all values are equal
    result_0 = TC_0[326][298][0]
    result_1 = TC_1[326][298]
    result_2 = TC_2[326][298]
    result_3 = TC_3[326][298]
    result_4 = TC_4[326][298]
    result_5 = TC_5[326][298]

    assert result_0['GTC'] == result_1['GTC'] == result_2['GTC'] == result_3['GTC'] == result_4['GTC'] == result_5['GTC']
    assert result_0['in_vehicle'] == result_1['in_vehicle'] == result_2['in_vehicle'] == result_3['in_vehicle'] == result_4['in_vehicle'] == result_5['in_vehicle']
    assert result_0['n_transfers'] == result_1['n_transfers'] == result_2['n_transfers'] == result_3['n_transfers'] == result_4['n_transfers'] == result_5['n_transfers']
    assert result_0['path'] == result_1['path'] == result_2['path'] == result_3['path'] == result_4['path'] == result_5['path']
    assert result_0['traveled_distance'] == result_1['traveled_distance'] == result_2['traveled_distance'] == result_3['traveled_distance'] == result_4['traveled_distance'] == result_5['traveled_distance']
    assert result_0['waiting_time'] == result_1['waiting_time'] == result_2['waiting_time'] == result_3['waiting_time'] == result_4['waiting_time'] == result_5['waiting_time']


def test_efficiency(attributes_path, L_space_10_path):
    attributes = load_gtfs(attributes_path)
    L_10_graph = load_graph(L_space_10_path)

    P_0 = P_space_0(attributes, L_10_graph, "Rail", 5, 24, None)
    P_1 = P_space_1(attributes, L_10_graph, "Rail", 5, 24, None)
    P_2 = P_space_2(attributes, L_10_graph, "Rail", 5, 24, None)
    P_3 = P_space_3(attributes, L_10_graph, "Rail", 5, 24, None)
    P_4 = P_space_4(attributes, L_10_graph, "Rail", 5, 24, None)
    P_5 = P_space_5(attributes, L_10_graph, "Rail", 5, 24, None)

    TC_0 = get_all_GTC_0(L_10_graph, P_0, 3, 2, [5]) 
    TC_1 = get_all_GTC_1(L_10_graph, P_1, 3, 2, [5])    
    TC_2 = get_all_GTC_2(L_10_graph, P_2, 3, 2, [5])    
    TC_3 = get_all_GTC_3(L_10_graph, P_3, 3, 2, [5])    
    TC_4 = get_all_GTC_4(L_10_graph, P_4, 3, 2, [5])    
    TC_5 = get_all_GTC_5(L_10_graph, P_5, 3, 2, [5])

    efficiency_0 = efficiency_graph(L_10_graph, TC_0)
    efficiency_1 = efficiency_graph(L_10_graph, TC_1)
    efficiency_2 = efficiency_graph(L_10_graph, TC_2)
    efficiency_3 = efficiency_graph(L_10_graph, TC_3)
    efficiency_4 = efficiency_graph(L_10_graph, TC_4)
    efficiency_5 = efficiency_graph(L_10_graph, TC_5)

    assert efficiency_0 == efficiency_1 == efficiency_2 == efficiency_3 == efficiency_4 == efficiency_5


def test_simulate_fixed_node_removal_efficiency(L_space_10_path, attributes_path):
    attributes = load_gtfs(attributes_path)
    L_10_graph = load_graph(L_space_10_path)

    sp_func_0 = make_sp_func(attributes, get_all_GTC_0, P_space_0)
    sp_func_1 = make_sp_func(attributes, get_all_GTC_1, P_space_1)
    sp_func_2 = make_sp_func(attributes, get_all_GTC_2, P_space_2)
    sp_func_3 = make_sp_func(attributes, get_all_GTC_3, P_space_3)
    sp_func_4 = make_sp_func(attributes, get_all_GTC_4, P_space_4)
    sp_func_5 = make_sp_func(attributes, get_all_GTC_5, P_space_5)

    original_efficiency_0, efficiencies_0, num_removed_0, removed_nodes_0, removal_times_0 = simulate_fixed_node_removal_efficiency(L_graph=L_10_graph, sp_func=sp_func_0, num_to_remove=5, method='random', seed=42)
    original_efficiency_1, efficiencies_1, num_removed_1, removed_nodes_1, removal_times_1 = simulate_fixed_node_removal_efficiency(L_graph=L_10_graph, sp_func=sp_func_1, num_to_remove=5, method='random', seed=42)
    original_efficiency_2, efficiencies_2, num_removed_2, removed_nodes_2, removal_times_2 = simulate_fixed_node_removal_efficiency(L_graph=L_10_graph, sp_func=sp_func_2, num_to_remove=5, method='random', seed=42)
    original_efficiency_3, efficiencies_3, num_removed_3, removed_nodes_3, removal_times_3 = simulate_fixed_node_removal_efficiency(L_graph=L_10_graph, sp_func=sp_func_3, num_to_remove=5, method='random', seed=42)
    original_efficiency_4, efficiencies_4, num_removed_4, removed_nodes_4, removal_times_4 = simulate_fixed_node_removal_efficiency(L_graph=L_10_graph, sp_func=sp_func_4, num_to_remove=5, method='random', seed=42)
    original_efficiency_5, efficiencies_5, num_removed_5, removed_nodes_5, removal_times_5 = simulate_fixed_node_removal_efficiency(L_graph=L_10_graph, sp_func=sp_func_5, num_to_remove=5, method='random', seed=42)

    assert original_efficiency_0 == original_efficiency_1 == original_efficiency_2 == original_efficiency_3 == original_efficiency_4 == original_efficiency_5
    assert efficiencies_0 == efficiencies_1 == efficiencies_2 == efficiencies_3 == efficiencies_4 == efficiencies_5

def test_random_removal_simulations(L_space_many, attributes_path):
    attributes = load_gtfs(attributes_path)
    L_graphs = load_graph(L_space_many)

    sp_func_0 = make_sp_func(attributes, get_all_GTC_0, P_space_0)
    sp_func_1 = make_sp_func(attributes, get_all_GTC_1, P_space_1)
    sp_func_2 = make_sp_func(attributes, get_all_GTC_2, P_space_2)
    sp_func_3 = make_sp_func(attributes, get_all_GTC_3, P_space_3)
    sp_func_4 = make_sp_func(attributes, get_all_GTC_4, P_space_4)
    sp_func_5 = make_sp_func(attributes, get_all_GTC_5, P_space_5)

    results_random_0 = run_removal_simulations(subgraphs_by_size=L_graphs, sp_func=sp_func_0, num_to_remove=5, method='random', seed=42)
    results_random_1 = run_removal_simulations(subgraphs_by_size=L_graphs, sp_func=sp_func_1, num_to_remove=5, method='random', seed=42)
    results_random_2 = run_removal_simulations(subgraphs_by_size=L_graphs, sp_func=sp_func_2, num_to_remove=5, method='random', seed=42)
    results_random_3 = run_removal_simulations(subgraphs_by_size=L_graphs, sp_func=sp_func_3, num_to_remove=5, method='random', seed=42)
    results_random_4 = run_removal_simulations(subgraphs_by_size=L_graphs, sp_func=sp_func_4, num_to_remove=5, method='random', seed=42)
    results_random_5 = run_removal_simulations(subgraphs_by_size=L_graphs, sp_func=sp_func_5, num_to_remove=5, method='random', seed=42)


    assert results_random_0['eff_after_1'][0] == results_random_1['eff_after_1'][0] == results_random_2['eff_after_1'][0] == results_random_3['eff_after_1'][0] == results_random_4['eff_after_1'][0] == results_random_5['eff_after_1'][0]