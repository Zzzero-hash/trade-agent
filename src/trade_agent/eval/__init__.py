"""Alias legacy eval namespace to evaluation_extra after migration."""
from __future__ import annotations

from importlib import import_module as _imp


_extra = _imp("trade_agent.evaluation_extra")
for _k in list(dir(_extra)):
    if not _k.startswith("_"):
        globals()[_k] = getattr(_extra, _k)
__all__ = [k for k in globals() if not k.startswith("_")]
