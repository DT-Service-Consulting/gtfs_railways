# Python

import pytest
import gtfspy
from gtfs_railways.functions.core import load_gtfs, \
    load_graph, efficiency_graph
import networkx as nx
from gtfs_railways.config import EXAMPLES_DIR, DATA_DIR
from gtfs_railways.functions.v1 import P_space as P_space_1, \
    get_all_GTC as get_all_GTC_1


@pytest.fixture
def L_space_10_path():
    # Use the graph_0.pkl file from the examples directory
    return str( EXAMPLES_DIR / "10/graph_0.pkl" )

@pytest.fixture
def L_space_20_path():
    # Use the graph_0.pkl file from the examples directory
    return str( EXAMPLES_DIR / "20/graph_0.pkl" )

@pytest.fixture
def attributes_path():
    return str( DATA_DIR / "sqlite/belgium.sqlite" )

@pytest.fixture
def P_graph_example(L_space_10_path, attributes_path):
    L_10_graph = load_graph(L_space_10_path)
    attributes = load_gtfs(attributes_path)
    P_graph = P_space_1(attributes, L_10_graph, "Rail", 5, 24, None)
    return P_graph

@pytest.fixture
def travel_cost_example(L_space_10_path, P_graph_example):
    L_10_graph = load_graph(L_space_10_path)
    return get_all_GTC_1( L_10_graph, P_graph_example,3, 2, [5])


def test_load_gtfs(attributes_path):
    attributes = load_gtfs(attributes_path)
    assert isinstance(attributes, gtfspy.gtfs.GTFS)

def test_load_graph(L_space_10_path, L_space_20_path):
    L_10_graph = load_graph(L_space_10_path)
    assert isinstance(L_10_graph, nx.classes.digraph.DiGraph)
    assert L_10_graph.number_of_nodes() == 10

    L_20_graph = load_graph(L_space_20_path)
    assert isinstance(L_20_graph, nx.classes.digraph.DiGraph)
    assert L_20_graph.number_of_nodes() == 20


def test_P_space(L_space_10_path, L_space_20_path, attributes_path):
    attributes = load_gtfs(attributes_path)
    L_10_graph = load_graph(L_space_10_path)
    L_20_graph = load_graph(L_space_20_path)

    P_graph = P_space_1(attributes, L_10_graph, "Rail", 5, 24, None)

    assert isinstance(P_graph, nx.classes.digraph.DiGraph)
    assert P_graph.number_of_nodes() == L_10_graph.number_of_nodes()
    assert P_graph.number_of_nodes() == 10
    assert P_graph.number_of_edges() == 76

    P_graph = P_space_1(attributes, L_20_graph, "Rail", 5, 24, None)

    assert isinstance(P_graph, nx.classes.digraph.DiGraph)
    assert P_graph.number_of_nodes() == L_20_graph.number_of_nodes()
    assert P_graph.number_of_nodes() == 20
    assert P_graph.number_of_edges() == 256


def test_travel_cost(travel_cost_example):

    travel_cost = travel_cost_example
    assert isinstance(travel_cost, dict)

    result = travel_cost[326][298][0]
    assert result['GTC'] == 217
    assert result['in_vehicle'] == 27
    assert result['n_transfers'] == 0
    assert result['path'] == [326, 327, 420, 419, 418, 298]
    assert result['traveled_distance'] == 30336
    assert result['waiting_time'] == 95

    result = travel_cost[300][299][0]
    assert result['GTC'] == 34
    assert result['in_vehicle'] == 4
    assert result['n_transfers'] == 0
    assert result['path'] == [300, 299]
    assert result['traveled_distance'] == 2863
    assert result['waiting_time'] == 15
