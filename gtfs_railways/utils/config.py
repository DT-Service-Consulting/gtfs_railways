from pathlib import Path

# Start from this file and walk up until we find the project root
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]  # Assuming utils/ inside gtfs_railways/

# Define paths
FUNCTION_DIR = PROJECT_ROOT / 'gtfs_railways/functions'
DATA_DIR = PROJECT_ROOT / 'gtfs_railways/data'
EXAMPLES_DIR = PROJECT_ROOT / 'gtfs_railways/examples'

print(PROJECT_ROOT)
print(PROJECT_ROOT)
print(PROJECT_ROOT)
