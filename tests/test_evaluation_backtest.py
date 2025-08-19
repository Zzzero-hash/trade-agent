import numpy as np
import pandas as pd

from trade_agent.evaluation.backtest import run_backtest
from trade_agent.evaluation.report import generate_report


def test_backtest_basic_long_trend() -> None:
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    prices = pd.Series(np.linspace(100, 130, len(idx)), index=idx)
    signals = pd.Series(1.0, index=idx)
    result = run_backtest(prices, signals)
    metrics = result["metrics"]
    assert metrics["pnl"] > 0
    assert metrics["max_drawdown"] <= 0


def test_backtest_flat_prices_costs_negative_pnl() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    prices = pd.Series(100.0, index=idx)
    signals = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], index=idx, dtype=float)
    result = run_backtest(prices, signals, fee_rate=0.001, slippage=0.001)
    assert result["returns"].sum() < 0


def test_backtest_with_gaps() -> None:
    idx = pd.to_datetime(
        [
            "2024-01-01",
            "2024-01-02",
            "2024-01-05",
            "2024-01-10",
            "2024-01-11",
        ]
    )
    prices = pd.Series([100, 101, 102, 103, 104], index=idx, dtype=float)
    signals = pd.Series([0, 1, 1, 0, -1], index=idx, dtype=float)
    result = run_backtest(prices, signals, fee_rate=0.0005)
    assert len(result["returns"]) == len(prices)
    assert "metrics" in result


def test_report_generation(tmp_path) -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    prices = pd.Series([100, 101, 100, 102, 101], index=idx, dtype=float)
    signals = pd.Series([0, 1, 1, 0, 0], index=idx, dtype=float)
    result = run_backtest(prices, signals)
    report = generate_report(result, html_path=str(tmp_path / "report.html"))
    assert (tmp_path / "report.html").exists()
    assert "final_equity" in report
