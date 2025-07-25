from pathlib import Path

# Start from this file and walk up until we find the project root
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]  # Assuming utils/ inside gtfs_railways/

# Optional: assert something exists to be safe
assert (PROJECT_ROOT / 'setup.py').exists() or (PROJECT_ROOT / '.git').exists(), "Can't find project root"

# Define paths
FUNCTION_DIR = PROJECT_ROOT / 'functions'
DATA_DIR = PROJECT_ROOT / 'data'
EXAMPLES_DIR = PROJECT_ROOT / 'examples'