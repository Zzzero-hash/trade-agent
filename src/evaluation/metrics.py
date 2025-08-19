"""Core performance metrics (vectorised, minimal API).

The functions here wrap / reuse the canonical implementations in
``trade_agent.eval.financial_metrics`` where possible to avoid duplicated
logic, adding a few lightweight strategy level statistics commonly needed in
quick experiments.
"""
from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from trade_agent.eval import financial_metrics as fm


_EPS = 1e-12


@dataclass(frozen=True)
class EvaluationMetrics:
    """Container for strategy level evaluation metrics.

    All rate/ratio style metrics return 0.0 when undefined (e.g. division by
    zero, insufficient data) to keep downstream usage simple; callers can
    choose to post‑process if a different sentinel is desired.
    """

    pnl: float
    sharpe: float
    sortino: float
    max_drawdown: float
    win_rate: float
    exposure: float
    turnover: float

    def as_dict(self) -> dict[str, float]:  # pragma: no cover - trivial
        return {k: float(v) for k, v in self.__dict__.items()}


def _coerce_series(series: pd.Series | list[float] | np.ndarray) -> pd.Series:
    if isinstance(series, pd.Series):
        return series.astype(float)
    return pd.Series(series, dtype="float64")


def compute_metrics(
    returns: pd.Series | list[float] | np.ndarray,
    positions: pd.Series | list[float] | np.ndarray | None = None,
) -> Mapping[str, float]:
    """Compute core evaluation metrics.

    Args:
        returns: Per‑period strategy returns (already net of costs).
        positions: (Optional) position exposure per period expressed as a
            fraction of capital (e.g. -1.0 .. 1.0). If omitted, exposure /
            turnover / win rate fall back to calculations on returns alone.

    Returns:
        Mapping of metric name -> value.
    """
    r = _coerce_series(returns)
    if positions is None:
        pos = pd.Series(np.where(r != 0, 1.0, 0.0), index=r.index)
    else:
        pos = _coerce_series(positions).reindex(r.index)
        # Forward fill then replace any residual NaNs with 0
        pos = pos.ffill().fillna(0.0)

    if r.empty:
        return EvaluationMetrics(
            pnl=0.0,
            sharpe=0.0,
            sortino=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            exposure=0.0,
            turnover=0.0,
        ).as_dict()

    pnl = float(r.cumsum().iloc[-1]) if not r.empty else 0.0

    # Reuse canonical implementations for ratios
    sharpe = float(fm.sharpe_ratio(r))
    sortino = float(fm.sortino_ratio(r))
    max_dd = float(fm.max_drawdown(r))

    active_mask = pos.abs() > 0
    active_returns = r[active_mask]
    win_rate = 0.0 if active_returns.empty else float((active_returns > 0).mean())

    exposure = float(pos.abs().mean()) if not pos.empty else 0.0
    turnover = float(pos.diff().abs().sum() / (len(pos) - 1)) if len(pos) > 1 else 0.0

    return EvaluationMetrics(
        pnl=pnl,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        win_rate=win_rate,
        exposure=exposure,
        turnover=turnover,
    ).as_dict()


def enrich_results(result: MutableMapping[str, object]) -> MutableMapping[str, object]:
    """Convenience helper to attach metrics to a backtest result dict.

    The function mutates the provided mapping (if mutable) and also returns it
    to allow fluent style usage.
    """
    returns = result.get("returns")  # type: ignore[assignment]
    positions = result.get("positions")  # type: ignore[assignment]
    if isinstance(returns, pd.Series):
        result["metrics"] = compute_metrics(returns, positions)  # type: ignore[index]
    return result

__all__ = ["compute_metrics", "enrich_results", "EvaluationMetrics"]
