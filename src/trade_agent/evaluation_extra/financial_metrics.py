from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


__all__ = [
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
]

ReturnLike = Sequence[float] | pd.Series


def _coerce(r: ReturnLike) -> pd.Series:
    if isinstance(r, pd.Series):
        return r.astype(float)
    return pd.Series(list(r), dtype="float64")


def sharpe_ratio(returns: ReturnLike, risk_free: float = 0.0) -> float:
    r = _coerce(returns)
    if r.empty:
        return 0.0
    excess = r - risk_free / max(len(r), 1)
    std = r.std(ddof=0)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.sqrt(252) * excess.mean() / std)


def sortino_ratio(returns: ReturnLike, risk_free: float = 0.0) -> float:
    r = _coerce(returns)
    if r.empty:
        return 0.0
    downside = r[r < 0]
    dd = downside.std(ddof=0)
    if dd == 0 or np.isnan(dd):
        return 0.0
    excess = r - risk_free / max(len(r), 1)
    return float(np.sqrt(252) * excess.mean() / dd)


def max_drawdown(returns: ReturnLike) -> float:
    r = _coerce(returns)
    if r.empty:
        return 0.0
    eq = (1 + r).cumprod()
    peaks = eq.cummax()
    dd = (eq / peaks - 1).min()
    return float(abs(dd))
