# Python

import pytest
import gtfspy
from gtfs_railways.functions.core import load_gtfs, load_graph, efficiency_graph
from gtfs_railways.config import EXAMPLES_DIR, DATA_DIR

from gtfs_railways.functions.v0 import P_space as P_space_0, get_all_GTC as get_all_GTC_0
from gtfs_railways.functions.v1 import P_space as P_space_1, get_all_GTC as get_all_GTC_1
from gtfs_railways.functions.v2 import P_space as P_space_2, get_all_GTC as get_all_GTC_2
#from gtfs_railways.functions.v3 import P_space as P_space_3, get_all_GTC as get_all_GTC_3
#from gtfs_railways.functions.v4 import P_space as P_space_4, get_all_GTC as get_all_GTC_4
#from gtfs_railways.functions.v4 import P_space as P_space_4, get_all_GTC as get_all_GTC_5



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

    P_0 = P_space_0(attributes, L_10_graph, "Rail", 5, 24, None)
    P_1 = P_space_1(attributes, L_10_graph, "Rail", 5, 24, None)
    P_2 = P_space_2(attributes, L_10_graph, "Rail", 5, 24, None)
    #P_3 = P_space_3(attributes, L_10_graph, "Rail", 5, 24, None)
    #P_4 = P_space_4(attributes, L_10_graph, "Rail", 5, 24, None)
    #P_5 = P_space_5(attributes, L_10_graph, "Rail", 5, 24, None)

    # Travel Cost
    TC_0 = get_all_GTC_0(L_10_graph, P_0, 3, 2, [5])
    TC_1 = get_all_GTC_1(L_10_graph, P_1, 3, 2, [5])
    TC_2 = get_all_GTC_2(L_10_graph, P_2, 3, 2, [5])
    #TC_3 = get_all_GTC_1(L_10_graph, P_graph_3, 3, 2, [5])
    #TC_4 = get_all_GTC_1(L_10_graph, P_graph_4, 3, 2, [5])
    #TC_5 = get_all_GTC_1(L_10_graph, P_graph_5, 3, 2, [5])

    # Check if all values are equal
    #print(TC_0)
    #assert P_0 == P_1 == P_2, "Not all values are equal"
    #assert TC_0 == TC_1 == TC_2, "Not all values are equal"
    #assert P_0 == P_1 == P_2 == P_3 == P_4 == P_5, "Not all values are equal"
    #assert TC_0 == TC_1 == TC_2 == TC_3 == TC_4 == TC_5, "Not all values are equal"


def test_efficiency(attributes_path, L_space_10_path):

    attributes = load_gtfs(attributes_path)
    L_10_graph = load_graph(L_space_10_path)

    P_0 = P_space_0(attributes, L_10_graph, "Rail", 5, 24, None)
    TC_0 = get_all_GTC_0(L_10_graph, P_0, 3, 2, [5])

    P_1 = P_space_1(attributes, L_10_graph, "Rail", 5, 24, None)
    TC_1 = get_all_GTC_1(L_10_graph, P_1, 3, 2, [5])

    P_2 = P_space_2(attributes, L_10_graph, "Rail", 5, 24, None)
    TC_2 = get_all_GTC_2(L_10_graph, P_2, 3, 2, [5])

    efficiency_0 = efficiency_graph(L_10_graph, TC_0)
    efficiency_1 = efficiency_graph(L_10_graph, TC_1)
    efficiency_2 = efficiency_graph(L_10_graph, TC_2)

    print(efficiency_1)
    assert efficiency_0 == efficiency_1 == efficiency_2