"""Vectorised backtest utility.

This module provides a *pure functional* style backtest for rapid iteration.
It complements (does not replace) the richer OO engine in
``trade_agent.eval.backtest``.

Design choices:
* Signals represent target position fraction of capital (e.g. 1 = 100% long).
* Returns are computed as ``prev_position * price_pct_change``.
* Transaction costs & slippage are modelled as proportional (bps style)
  penalties applied on *notional turnover* (absolute position change).
* All computations are vectorised; no iterative Python loops over rows.
"""
from __future__ import annotations

from collections.abc import Mapping, MutableMapping

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
) -> Mapping[str, object]:
    """Execute a vectorised backtest.

    Args:
        prices: Price series (must be sortable index; irregular gaps allowed).
        signals: Target position fraction in [-max_leverage, max_leverage].
        fee_rate: Proportional transaction cost applied to *change* in
            position (e.g. 0.001 = 10 bps round turn approximately if flip).
        slippage: Additional proportional cost on turnover (modelled same as
            fee_rate for simplicity).
        max_leverage: Maximum absolute position allowed; signals are clipped.

    Returns:
        Dict with keys: prices, signals (post‑clip), positions, returns,
        equity_curve, metrics (dict).
    """
    if prices is None or signals is None:
        raise ValueError("prices and signals must be provided")
    if not isinstance(prices, pd.Series) or not isinstance(signals, pd.Series):
        raise TypeError("prices and signals must be pandas Series")
    if prices.empty:
        return {
            "prices": prices,
            "signals": signals,
            "positions": pd.Series(dtype=float),
            "returns": pd.Series(dtype=float),
            "equity_curve": pd.Series(dtype=float),
            "metrics": compute_metrics(pd.Series(dtype=float)),
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

    result: MutableMapping[str, object] = {
        "prices": prices,
        "signals": clipped_signals,
        "positions": positions,
        "returns": net_returns,
        "equity_curve": equity_curve,
    }
    result["metrics"] = compute_metrics(net_returns, positions)
    return result


__all__ = ["run_backtest"]
