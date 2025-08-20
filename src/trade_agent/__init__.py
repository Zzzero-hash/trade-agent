"""trade_agent package root.

Version is supplied dynamically by packaging metadata so semantic-release can
update it without editing source files directly.
"""
from __future__ import annotations

from importlib import metadata


try:  # pragma: no cover - simple metadata fetch
	__version__ = metadata.version("trade-agent")
except metadata.PackageNotFoundError:  # local editable fallback
	__version__ = "0.0.0"

__all__: list[str] = ["__version__"]
