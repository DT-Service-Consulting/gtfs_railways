print("Running examples...")
from gtfs_railways.config import DATA_DIR
from gtfs_railways.functions.core import load_graph, load_gtfs
from gtfs_railways.functions.v1 import P_space

path_to_sqlite = "/Users/marco/Library/CloudStorage/OneDrive-DTServicesandConsultingSRL/BRAIN/PROJECTS/Railways_Resilience/belgium 1.sqlite"
attributes = load_gtfs(path_to_sqlite)

L_space_path = DATA_DIR / "pkl/10/graph_0.pkl"  # Path where the clean L-space graph was stored (cleaned routes)
L_graph = load_graph(L_space_path)
P_graph = P_space(attributes, L_graph, "Rail", 5, 24, None)


print("Example finished.")

