print("Running examples...")
from gtfs_railways.config import EXAMPLES_DIR
from gtfs_railways.functions.core import load_graph, efficiency_graph
from gtfs_railways.functions.v1 import get_all_GTC

print(EXAMPLES_DIR)
L_space_path = EXAMPLES_DIR / "10/graph_0.pkl"  # Path where the clean L-space graph was stored (cleaned routes)
print(L_space_path)

L_graph = load_graph(L_space_path)
print(L_graph)


efficiency_graph(    L_graph,
    sp=L_graph.shortest_path,
)
