"""Vectorised backtest utility.

This module provides a *pure functional* style backtest for rapid iteration
and testability. It complements (does not replace) any richer OO engine.

Key design points
-----------------
* Signals represent target position *fraction of capital* (1.0 == 100% long,
    -1.0 == 100% short). Think of them as *desired* end‑of‑bar exposures.
* Strategy (gross) return for bar ``t`` uses the position held *during* that
    bar, which is the (already applied) signal at ``t-1``.
* Transaction costs & slippage are modelled as proportional (bps style)
    penalties on *turnover* (absolute position change). Both ``fee_rate`` and
    ``slippage`` are simple additive proportional costs.
* Positions are clipped to ``[-max_leverage, max_leverage]`` (hard risk guard).
* All computations are fully vectorised with Pandas / NumPy.

Returned fields
---------------
prices: original aligned prices
signals: clipped target signals
positions: effective position per bar (== signals here, but separated for
        future flexibility)
gross_returns: pre‑cost strategy returns
turnover: absolute change in position each bar
trading_cost: per‑bar proportional cost ( (fee_rate+slippage)*turnover )
returns: net returns after costs
equity_curve: cumulative equity indexed to 1.0
metrics: dictionary from :func:`compute_metrics`
"""
from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

import pandas as pd

from .metrics import compute_metrics


def _align(
    prices: pd.Series, signals: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Align signals to price index (forward fill & fill missing with 0)."""
    signals = signals.reindex(prices.index).ffill().fillna(0.0)
    return prices.astype(float), signals.astype(float)


def run_backtest(
    prices: pd.Series,
    signals: pd.Series,
    fee_rate: float = 0.0,
    slippage: float = 0.0,
    max_leverage: float = 1.0,
) -> Mapping[str, Any]:
    """Execute a vectorised backtest.

    Parameters
    ----------
    prices : pd.Series
        Price series (must be sortable index; irregular gaps allowed).
    signals : pd.Series
        Target position fraction in ``[-max_leverage, max_leverage]``.
    fee_rate : float, default 0.0
        Proportional transaction fee applied to absolute position change.
    slippage : float, default 0.0
        Additional proportional cost (simple linear model) applied to turnover.
    max_leverage : float, default 1.0
        Hard clip bound for absolute position size.

    Returns
    -------
    Mapping[str, Any]
        Dictionary containing prices, signals, positions, gross_returns,
        turnover, trading_cost, returns (net), equity_curve and metrics.
    """
    # Basic defensive checks (types enforced by signature in most call paths).
    if not isinstance(prices, pd.Series) or not isinstance(signals, pd.Series):  # type: ignore[unreachable]
        raise TypeError("prices and signals must be pandas Series")
    if prices.empty:
        empty_series = pd.Series(dtype=float)
        return {
            "prices": prices,
            "signals": signals,
            "positions": empty_series,
            "gross_returns": empty_series,
            "turnover": empty_series,
            "trading_cost": empty_series,
            "returns": empty_series,
            "equity_curve": empty_series,
            "metrics": compute_metrics(empty_series),
        }

    prices, signals = _align(prices.sort_index(), signals)
    clipped_signals = signals.clip(lower=-max_leverage, upper=max_leverage)

    price_returns = prices.pct_change().fillna(0.0)
    # Strategy return uses *previous* period's position.
    positions = clipped_signals
    prev_pos = positions.shift(1).fillna(0.0)
    gross_returns = prev_pos * price_returns

    # Turnover: absolute change in position (first bar vs 0 position)
    turnover = positions.diff().abs().fillna(positions.abs())
    trading_cost = (fee_rate + slippage) * turnover
    net_returns = gross_returns - trading_cost

    equity_curve = (1.0 + net_returns).cumprod()

    result: MutableMapping[str, Any] = {
        "prices": prices,
        "signals": clipped_signals,
        "positions": positions,
        "gross_returns": gross_returns,
        "turnover": turnover,
        "trading_cost": trading_cost,
        "returns": net_returns,
        "equity_curve": equity_curve,
    }
    result["metrics"] = compute_metrics(net_returns, positions)
    return result


__all__ = ["run_backtest"]
