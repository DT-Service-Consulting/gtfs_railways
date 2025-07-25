import pytest # type: ignore

from gtfs_railways.utils.config import DATA_DIR
from gtfs_railways.functions.core import load_gtfs
from gtfs_railways.functions.core import load_graph

from gtfs_railways.functions.v0 import get_all_GTC as get_all_GTC_0
from gtfs_railways.functions.v1 import P_space as P_space_1, get_all_GTC as get_all_GTC_1


@pytest.fixture
def L_space_10_path():
    # Use the graph_0.pkl file from the examples directory
    print(DATA_DIR)
    print(DATA_DIR)
    print(DATA_DIR)
    print(DATA_DIR)
    print(DATA_DIR)
    print(DATA_DIR)
    return str( DATA_DIR / "pkl/10/graph_0.pkl" )

@pytest.fixture
def L_space_20_path():
    # Use the graph_0.pkl file from the examples directory
    print(DATA_DIR)
    print(DATA_DIR)
    print(DATA_DIR)
    print(DATA_DIR)

    return str( DATA_DIR / "pkl/20/graph_0.pkl" )

@pytest.fixture
def L_space_many():
    # Use the graph_0.pkl file from the examples directory
    print(DATA_DIR)
    print(DATA_DIR)
    print(DATA_DIR)
    print(DATA_DIR)

    return str( DATA_DIR / "pkl/subgraphs_by_size.pkl" )

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
def travel_cost_example_v0(L_space_10_path, P_graph_example):
    L_10_graph = load_graph(L_space_10_path)
    return get_all_GTC_0(L_10_graph, P_graph_example,3, 2, [5])

@pytest.fixture
def travel_cost_example(L_space_10_path, P_graph_example):
    L_10_graph = load_graph(L_space_10_path)
    return get_all_GTC_1(L_10_graph, P_graph_example,3, 2, [5])

