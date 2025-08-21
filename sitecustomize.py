"""Ensure 'src' directory is on sys.path for test execution without PYTHONPATH."""
import sys
from pathlib import Path


root = Path(__file__).resolve().parent
src = root / "src"
if src.is_dir() and str(src) not in sys.path:
    sys.path.insert(0, str(src))
