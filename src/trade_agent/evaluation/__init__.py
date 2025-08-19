"""Lightweight evaluation utilities (metrics, backtest, reporting).

Vectorised, dependency‑light helpers for quick research loops and unit tests.
All legacy shim layers have been removed; this is the canonical location for
these utilities.
"""
from .backtest import run_backtest  # noqa: F401
from .metrics import compute_metrics  # noqa: F401
from .report import generate_report  # noqa: F401
from .temporal_cv import (  # noqa: F401
    PurgedTimeSeriesSplit,
    optuna_objective_factory,
    temporal_cv_scores,
)


__all__ = [
    "compute_metrics",
    "run_backtest",
    "generate_report",
    "PurgedTimeSeriesSplit",
    "temporal_cv_scores",
    "optuna_objective_factory",
]
