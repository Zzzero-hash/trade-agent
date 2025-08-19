"""Tests for financial metrics utility module."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trade_agent.evaluation.financial_metrics import (
    cagr,
    calculate_metrics,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
)


def test_sharpe_and_sortino_positive_sequence() -> None:
    returns = pd.Series([0.01] * 10)
    s = sharpe_ratio(returns)
    so = sortino_ratio(returns)
    assert s > 0
    assert so >= s  # no downside returns => Sortino >= Sharpe


def test_max_drawdown_simple() -> None:
    # Up then down 10%
    returns = pd.Series([0.1, -0.05, -0.05])
    dd = max_drawdown(returns)  # negative percent
    # Roughly -9.52% ( (1.1*0.95*0.95)/1.1 -1 ) *100
    assert dd < 0
    assert -15 < dd < -5


def test_profit_factor_and_cagr() -> None:
    returns = pd.Series([0.02, -0.01, 0.03, -0.005])
    pf = profit_factor(returns)
    cg = cagr(returns)
    assert pf > 0
    assert isinstance(cg, float)


def test_calculate_metrics_smoke() -> None:
    prices = pd.Series(np.linspace(100, 110, 50))
    positions = pd.Series([1] * 50)
    metrics = calculate_metrics(prices, positions)
    expected = {
        'sharpe_ratio',
        'sortino_ratio',
        'max_drawdown',
        'profit_factor',
        'cagr',
        'total_return',
        'win_rate',
        'risk_adjusted_return',
    }
    assert expected.issubset(metrics.keys())
