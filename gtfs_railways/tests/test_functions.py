# Python

import pytest
import gtfspy
from gtfs_railways.functions.core import load_gtfs, load_graph
from gtfs_railways.config import EXAMPLES_DIR, DATA_DIR
from gtfs_railways.functions.v1 import P_space as P_space_1, get_all_GTC as get_all_GTC_1
#from gtfs_railways.functions.v1 import P_space as P_space_2
#from gtfs_railways.functions.v1 import P_space as P_space_3
#from gtfs_railways.functions.v1 import P_space as P_space_4



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

def test_get_all_GTC(L_space_10_path, attributes_path):
    attributes = load_gtfs(attributes_path)
    L_10_graph = load_graph(L_space_10_path)

    P_graph = P_space_1(attributes, L_10_graph, "Rail", 5, 24, None)
    travel_cost = get_all_GTC_1(L_10_graph, P_graph, 3, 2, [5])

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
