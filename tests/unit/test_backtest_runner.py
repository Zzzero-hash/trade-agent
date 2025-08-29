import pandas as pd

from trade_agent.backtest import run_backtest


def test_backtest_happy_path() -> None:
    data = {
        "close": [100, 101, 102, 101, 103],
        "signal": [0, 1, 1, -1, 0],  # position applied next bar
    }
    df = pd.DataFrame(data)
    res = run_backtest(df)
    # Basic sanity
    assert len(res.equity_curve) == len(df)
    assert res.trades > 0
    assert 0 <= res.max_drawdown <= 1
    assert -1 <= res.cumulative_return <= 1


def test_backtest_all_flat() -> None:
    df = pd.DataFrame({"close": [100, 101, 102], "signal": [0, 0, 0]})
    res = run_backtest(df)
    assert res.trades == 0
    assert all(v == 10000 for v in res.equity_curve)
    assert res.cumulative_return == 0


def test_backtest_single_long_trade() -> None:
    df = pd.DataFrame({"close": [100, 102, 101], "signal": [0, 1, 0]})
    res = run_backtest(df)
    # One trade open then close
    assert res.trades >= 1
    assert len(res.returns) == len(df)


def test_edge_short_series() -> None:
    df = pd.DataFrame({"close": [100], "signal": [0]})
    res = run_backtest(df)
    assert res.equity_curve == [10000]
    assert res.trades == 0
    assert res.cumulative_return == 0
