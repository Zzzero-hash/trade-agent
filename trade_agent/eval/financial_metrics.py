"""Re-export financial metrics from src/eval for backward compatibility."""
from __future__ import annotations

import importlib


_mod = importlib.import_module("src.eval.financial_metrics")
for _k in dir(_mod):
    if not _k.startswith("_"):
        globals()[_k] = getattr(_mod, _k)

__all__ = [k for k in globals() if not k.startswith("_")]
