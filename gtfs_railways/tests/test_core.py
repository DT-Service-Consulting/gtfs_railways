# Python

import pytest # type: ignore
import gtfspy # type: ignore
import networkx as nx # type: ignore

from gtfs_railways.functions.core import load_gtfs
from gtfs_railways.functions.core import load_graph

from gtfs_railways.functions.v1 import P_space as P_space_1

from gtfs_railways.tests.conftest import L_space_10_path
from gtfs_railways.tests.conftest import L_space_20_path
from gtfs_railways.tests.conftest import attributes_path
from gtfs_railways.tests.conftest import travel_cost_example_v0
from gtfs_railways.tests.conftest import travel_cost_example

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


def test_P_space(L_space_10_path: object, L_space_20_path: object, attributes_path: object) -> None:
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


def test_travel_cost_v0(travel_cost_example_v0):

    """
    For version 0 of the travel cost function, we expect the output to be a dictionary with THREE levels
    """

    travel_cost_v0 = travel_cost_example_v0
    result_v0 = travel_cost_v0[326][298][0]

    assert isinstance(travel_cost_v0, dict)
    assert result_v0['GTC'] == 217
    assert result_v0['in_vehicle'] == 27


def test_travel_cost(travel_cost_example):

    """
    For version 1 and above of the travel cost function, we expect the output to be a dictionary with THREE levels
    """

    travel_cost = travel_cost_example
    result = travel_cost[326][298]

    assert isinstance(travel_cost, dict)
    assert result['GTC'] == 217
    assert result['in_vehicle'] == 27
    assert result['n_transfers'] == 0
    assert result['path'] == [326, 327, 420, 419, 418, 298]
    assert result['traveled_distance'] == 30336
    assert result['waiting_time'] == 95

    result = travel_cost[300][299]
    assert isinstance(travel_cost, dict)
    assert result['GTC'] == 34
    assert result['in_vehicle'] == 4
    assert result['n_transfers'] == 0
    assert result['path'] == [300, 299]
    assert result['traveled_distance'] == 2863
    assert result['waiting_time'] == 15
