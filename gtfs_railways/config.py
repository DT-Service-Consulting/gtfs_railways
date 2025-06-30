from pathlib import Path

# Get the base directory (directory where this file resides)
BASE_DIR = Path(__file__).parent.parent

# Paths to various resources
DATA_DIR = BASE_DIR / 'data'
NB_DIR = BASE_DIR / 'notebooks'
ENVFILE = BASE_DIR / '.env'
