"""Training shim.

Aggregates training entry points from legacy supervised (``sl.train``)
and reinforcement learning modules.
"""
from __future__ import annotations

from importlib import import_module as _imp
from typing import Any, Dict, Tuple


_sl_train = _imp("sl.train") if True else None  # noqa: F841
_rl_train_ppo = _imp("rl.train_ppo") if True else None  # noqa: F841
_rl_train_sac = _imp("rl.train_sac") if True else None  # noqa: F841

__all__: list[str] = []  # type: ignore[var-annotated]

# Documented entry points (attributes are looked up lazily)
_entry_points: dict[str, tuple[str, str]] = {
    "train_sl": ("sl.train", "main"),
    "train_ppo": ("rl.train_ppo", "main"),
    "train_sac": ("rl.train_sac", "main"),
}


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name in _entry_points:
        mod_name, attr = _entry_points[name]
        mod = _imp(mod_name)
        return getattr(mod, attr)
    raise AttributeError(name)
