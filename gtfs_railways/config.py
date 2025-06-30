from pathlib import Path

# Get the base directory (directory where this file resides)
BASE_DIR = Path(__file__).parent.parent

# Paths to various resources
SRC_DIR = BASE_DIR / 'src'
FUNCTION_DIR = SRC_DIR / 'functions'
DATA_DIR = BASE_DIR / 'data'
GRAPH_DIR = BASE_DIR / 'graphs'
NB_DIR = BASE_DIR / 'notebooks'
ENVFILE = BASE_DIR / '.env'
