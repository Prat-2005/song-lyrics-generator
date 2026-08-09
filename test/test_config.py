import unittest
from pathlib import Path
import sys
from config import DATA_PATH, INDEX_PATH, METADATA_PATH, BASE_DIR

# Ensure src is in the python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

class TestConfig(unittest.TestCase):
    def test_paths_are_absolute(self):
        """Test that paths are resolved to absolute paths"""
        self.assertTrue(Path(DATA_PATH).is_absolute(), f"DATA_PATH {DATA_PATH} is not absolute")
        self.assertTrue(Path(INDEX_PATH).is_absolute(), f"INDEX_PATH {INDEX_PATH} is not absolute")
        self.assertTrue(Path(METADATA_PATH).is_absolute(), f"METADATA_PATH {METADATA_PATH} is not absolute")
        
    def test_paths_relative_to_base_dir(self):
        """Test that relative paths from .env are resolved relative to BASE_DIR"""
        self.assertTrue(str(BASE_DIR) in DATA_PATH)
        self.assertTrue(str(BASE_DIR) in INDEX_PATH)
        self.assertTrue(str(BASE_DIR) in METADATA_PATH)

if __name__ == '__main__':
    unittest.main()
