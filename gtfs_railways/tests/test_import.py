# tests/test_setup.py
import sys
import os
import pytest
import setuptools

from pathlib import Path

this_folder = Path.cwd()
base_dir = this_folder.parent
path_setup = base_dir
this_file = this_folder / 'test_import.py'

#print(this_file)
# Add parent directory to sys.path
sys.path.insert(0, base_dir)

print(base_dir)
from config import SETUP_PATH  
print(SETUP_PATH)
print(this_file)


@pytest.fixture
def captured_setup(monkeypatch):
    captured = {}

    def fake_setup(**kwargs):
        captured.update(kwargs)

    # Monkeypatch setuptools.setup
    monkeypatch.setattr(setuptools, "setup", fake_setup)

    # Simulate running setup.py
    with open(SETUP_PATH, encoding="utf-8") as f:
        code = compile(f.read(), SETUP_PATH, 'exec')
        exec(code, {"__name__": "__main__"})

    return captured

def test_setup_metadata(captured_setup):
    assert captured_setup["name"] == "gtfs_railways"
##    assert captured_setup["version"] == "0.1.4"
##    assert "gtfs_railways" in captured_setup["packages"]
##    assert "pandas" in captured_setup["install_requires"]
##    assert captured_setup["python_requires"] == ">=3.6, <3.9"
##    assert isinstance(captured_setup["project_urls"], dict)
##    assert "Homepage" in captured_setup["project_urls"]
##    assert captured_setup["author"] == "Praneesh Sharma"
##    assert "Marco" in captured_setup["maintainer"]  # rough check
