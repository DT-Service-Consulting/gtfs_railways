from gtfs_railways.utils.config import DATA_DIR, EXAMPLES_DIR, PROJECT_ROOT, FUNCTION_DIR


def test_project_structure():
    """
    Test to ensure the project structure is correct.
    """
    assert (PROJECT_ROOT / 'setup.py').exists() or (PROJECT_ROOT / '.git').exists(), "Can't find project root"
    assert DATA_DIR.exists(), "Can't find data directory"
    assert EXAMPLES_DIR.exists(), "Can't find example directory"
    assert FUNCTION_DIR.exists(), "Can't find function directory"
