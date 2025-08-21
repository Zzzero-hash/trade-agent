"""Utility shim package.

Exports consolidated helpers used across the project.
"""
from .seed import set_seed  # noqa: F401


__all__: list[str] = [
    "set_seed",
]
