"""Canonical financial ratio & performance metric implementations.

This module replaces all legacy shim locations (previously under
``trade_agent.evaluation_extra`` / ``trade_agent.eval``). Only a compact
set of stable, dependency-light functions are retained. All functions are
pure, vectorised where reasonable, and defensive against empty / NaN
input. Undefined ratios return 0.0 (or ``float('inf')`` for profit factor
when there are no losses) to simplify downstream aggregation.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


ReturnLike = Sequence[float] | pd.Series | np.ndarray  # type: ignore[type-arg]

__all__ = [
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "profit_factor",
    "cagr",
    "calculate_metrics",
]

_TRADING_DAYS = 252
_EPS = 1e-12


def _coerce(r: ReturnLike) -> pd.Series:
    if isinstance(r, pd.Series):
        return r.astype(float)
    if isinstance(r, np.ndarray):
        return pd.Series(r.astype(float))
    return pd.Series(list(r), dtype="float64")


def sharpe_ratio(returns: ReturnLike, risk_free: float = 0.0) -> float:
    r = _coerce(returns)
    if r.empty:
        return 0.0
    # Convert annual risk_free to per-period approximation
    excess = r - risk_free / max(len(r), 1)
    std = r.std(ddof=0)
    mean = excess.mean()
    if std <= _EPS or np.isnan(std):
        return float(np.sqrt(_TRADING_DAYS) * mean) if mean > 0 else 0.0
    return float(np.sqrt(_TRADING_DAYS) * mean / std)


def sortino_ratio(returns: ReturnLike, risk_free: float = 0.0) -> float:
    r = _coerce(returns)
    if r.empty:
        return 0.0
    downside = r[r < 0]
    dd = downside.std(ddof=0)
    excess = r - risk_free / max(len(r), 1)
    mean = excess.mean()
    if dd <= _EPS or np.isnan(dd):
        return float(np.sqrt(_TRADING_DAYS) * mean) if mean > 0 else 0.0
    return float(np.sqrt(_TRADING_DAYS) * mean / dd)


def max_drawdown(returns: ReturnLike, as_percent: bool = True) -> float:
    """Maximum (peak to trough) drawdown.

    Returns negative value when ``as_percent`` (default) so tests can assert
    bounds like -15 < dd < -5. If ``as_percent`` is False returns fraction.
    """
    r = _coerce(returns)
    if r.empty:
        return 0.0 if as_percent else 0.0
    equity = (1 + r).cumprod()
    peaks = equity.cummax()
    dd_series = equity / peaks - 1.0  # negative or 0
    min_dd = dd_series.min()  # already negative
    if as_percent:
        return float(min_dd * 100.0)
    return float(min_dd)


def profit_factor(returns: ReturnLike) -> float:
    r = _coerce(returns)
    if r.empty:
        return 0.0
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()  # positive value
    if losses <= _EPS:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def cagr(returns: ReturnLike, periods_per_year: int = _TRADING_DAYS) -> float:
    r = _coerce(returns)
    if r.empty:
        return 0.0
    equity = (1 + r).cumprod()
    total_return = equity.iloc[-1]
    n_periods = len(r)
    if n_periods == 0 or total_return <= 0:
        return 0.0
    years = n_periods / periods_per_year
    if years <= 0:
        return 0.0
    return float(total_return ** (1 / years) - 1)


def calculate_metrics(
    prices: pd.Series,
    positions: pd.Series | None = None,
) -> Mapping[str, float]:
    """Compute a richer set of metrics from price series (and positions).

    Parameters
    ----------
    prices : pd.Series
        Price time series (assumed equally spaced, cleaned).
    positions : pd.Series | None
        Position exposure (-1..1). If omitted a fully invested (1) series is
        assumed for win rate / exposure style metrics.
    """
    if prices.empty:
        return {
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "cagr": 0.0,
            "total_return": 0.0,
            "win_rate": 0.0,
            "risk_adjusted_return": 0.0,
        }

    returns = prices.pct_change().fillna(0.0)  # type: ignore[call-overload]
    if positions is None:
        positions = pd.Series(1.0, index=prices.index)
    else:
        positions = (
            positions.reindex(prices.index)  # type: ignore[call-overload]
            .ffill()
            .fillna(0.0)  # type: ignore[call-overload]
        )

    pos_shift = positions.shift().fillna(0.0)  # type: ignore[call-overload]
    strat_returns = returns * pos_shift

    s = sharpe_ratio(strat_returns)
    so = sortino_ratio(strat_returns)
    dd = max_drawdown(strat_returns, as_percent=True)  # negative percent
    pf = profit_factor(strat_returns)
    cg = cagr(strat_returns)
    equity = (1 + strat_returns).cumprod()
    total_ret = float(equity.iloc[-1]) - 1.0
    wins = strat_returns[strat_returns > 0]
    active_source = (
        positions.shift().fillna(0.0)  # type: ignore[call-overload]
    )
    active = strat_returns[active_source != 0]
    win_rate = (
        float(wins.count() / max(active.count(), 1))
        if not active.empty
        else 0.0
    )
    # Simple risk-adjusted metric: total return / abs drawdown (fraction)
    rad = total_ret / (abs(dd) / 100.0 + _EPS)

    return {
        "sharpe_ratio": s,
        "sortino_ratio": so,
        "max_drawdown": dd,
        "profit_factor": pf,
        "cagr": cg,
        "total_return": total_ret,
        "win_rate": win_rate,
        "risk_adjusted_return": rad,
    }
