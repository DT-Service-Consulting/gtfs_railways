# Python

import pytest
import gtfspy
from gtfs_railways.functions.core import load_gtfs, load_graph
from gtfs_railways.functions.v1 import P_space
import networkx as nx
from gtfs_railways.config import EXAMPLES_DIR, DATA_DIR


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

    P_graph = P_space(attributes, L_10_graph, "Rail", 5, 24, None)

    assert isinstance(P_graph, nx.classes.digraph.DiGraph)
    assert P_graph.number_of_nodes() == L_10_graph.number_of_nodes()
    assert P_graph.number_of_nodes() == 10
    assert P_graph.number_of_edges() == 76

    P_graph = P_space(attributes, L_20_graph, "Rail", 5, 24, None)

    assert isinstance(P_graph, nx.classes.digraph.DiGraph)
    assert P_graph.number_of_nodes() == L_20_graph.number_of_nodes()
    assert P_graph.number_of_nodes() == 20
    assert P_graph.number_of_edges() == 256

