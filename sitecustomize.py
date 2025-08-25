"""Ensure repo root on sys.path so tests import top-level folders.

Allows importing `backend` / `engine` before packaging refactor.
"""
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
