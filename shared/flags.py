from __future__ import annotations
from typing import Dict, Any

# Simple feature flag store (Phase 0) - will evolve to provider pattern.

_FLAGS: Dict[str, Any] = {
    "premium.large_models": False,
    "premium.cloud_execution": False,
}


def is_enabled(name: str) -> bool:
    return bool(_FLAGS.get(name, False))


def dump_all() -> Dict[str, bool]:
    return {k: bool(v) for k, v in _FLAGS.items()}
