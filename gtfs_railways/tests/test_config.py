from gtfs_railways.utils.config import DATA_DIR, EXAMPLES_DIR, PROJECT_ROOT, FUNCTION_DIR


def test_project_structure():
    """
    Test to ensure the project structure is correct.
    """

    assert (PROJECT_ROOT / 'setup.py').exists() or (PROJECT_ROOT / '.git').exists(), "Can't find project root"
    assert DATA_DIR.exists(), "Can't find data directory"
    assert EXAMPLES_DIR.exists(), "Can't find example directory"
    assert FUNCTION_DIR.exists(), "Can't find function directory"

def test_data_dir_structure():
    required_dirs = ['belgium', 'pkl', 'sqlite']
    for subdir in required_dirs:
        sub_path = DATA_DIR / subdir
        assert sub_path.is_dir(), f"Missing expected subdirectory: {sub_path}"

