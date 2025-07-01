from pathlib import Path

# Get the base directory (directory where this file resides)
BASE_DIR = Path(__file__).parent.parent

# Paths to various resources
FUNCTION_DIR = BASE_DIR / 'functions'
DATA_DIR = BASE_DIR / 'data'
GRAPH_DIR = BASE_DIR / 'graphs'
NB_DIR = BASE_DIR / 'notebooks'
SETUP_PATH = BASE_DIR / 'setup.py'
