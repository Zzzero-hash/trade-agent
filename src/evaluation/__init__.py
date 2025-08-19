"""Lightweight evaluation utilities (metrics, backtest, reporting).

These modules provide a vectorised backtesting path independent of the
object-oriented engine under ``trade_agent.eval``. They are intentionally
minimal and focus on:

* Fast, dependency‑light metric computation for unit tests / research notebooks
* Deterministic, purely functional interfaces (no hidden state)
* Graceful handling of edge cases (empty / flat / gapped series)

The canonical / feature‑rich implementation continues to live under
``trade_agent.eval``; this namespace can be viewed as a convenience layer.
"""
from .backtest import run_backtest  # noqa: F401
from .metrics import compute_metrics  # noqa: F401
from .report import generate_report  # noqa: F401


__all__ = [
    "compute_metrics",
    "run_backtest",
    "generate_report",
]
