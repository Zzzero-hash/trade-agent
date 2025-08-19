import numpy as np
import pandas as pd
import pytest

from trade_agent.evaluation.metrics import compute_metrics


def test_metrics_empty_series() -> None:
    metrics = compute_metrics(pd.Series(dtype=float))
    assert metrics["pnl"] == 0.0
    assert metrics["sharpe"] == 0.0
    assert metrics["turnover"] == 0.0


def test_metrics_flat_series() -> None:
    returns = pd.Series([0.0] * 10)
    metrics = compute_metrics(returns)
    assert metrics["pnl"] == 0.0
    assert metrics["sharpe"] == 0.0
    assert metrics["sortino"] == 0.0
    assert metrics["max_drawdown"] == 0.0
    assert metrics["win_rate"] == 0.0
    assert metrics["exposure"] in (0.0, 1.0)


def test_metrics_positive_returns() -> None:
    returns = pd.Series([0.01] * 20)
    metrics = compute_metrics(returns)
    assert metrics["pnl"] == pytest.approx(0.2, rel=1e-6)
    assert metrics["win_rate"] == 1.0
    assert metrics["sharpe"] > 0


def test_metrics_with_positions_turnover() -> None:
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.001, 50))
    positions = pd.Series([(-1) ** i for i in range(50)], index=returns.index)
    metrics = compute_metrics(returns, positions)
    assert metrics["turnover"] > 0.0
    assert metrics["exposure"] <= positions.abs().max()
