from pathlib import Path

# Get the base directory (directory where this file resides)
BASE_DIR = Path(__file__).parent.parent

# Paths to various resources
FUNCTION_DIR = BASE_DIR / 'functions'
DATA_DIR = BASE_DIR / 'data'
EXAMPLES_DIR = BASE_DIR / 'examples'
#SETUP_PATH = BASE_DIR / 'setup.py'
#SETUP_PATH = BASE_DIR