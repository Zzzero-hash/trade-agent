"""Minimal backtest runner.

Assumptions
-----------
* Single instrument DataFrame ordered chronologically.
* Requires price column (default 'close') & signal column (default 'signal').
* Signal is target position for next bar in {-1,0,1}.
* Trades applied next bar: pnl uses position[t-1] * return[t].
* No slippage / spreads / partial fills (fees ignored placeholder).

Future Work (not in this minimal version)
----------------------------------------
* Multi-asset & position sizing / cash management.
* Real fee & slippage model; maker/taker distinction.
* Event-driven fills & intraday support.
* Advanced risk metrics (Sharpe, Sortino, etc.).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(slots=True)
class BacktestResult:
    equity_curve: list[float]
    returns: list[float]
    cumulative_return: float
    max_drawdown: float
    max_run_up: float
    win_rate: float
    trades: int

    def as_dict(self):  # pragma: no cover - trivial
        return asdict(self)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return -dd.min() if not dd.empty else 0.0


def _max_run_up(equity: pd.Series) -> float:
    """
    Calculate the maximum run-up (peak gain from trough) of an equity curve.

    Assumes the equity series is chronologically ordered.

    Parameters
    ----------
    equity : pd.Series
        Series of equity values over time, ordered chronologically.

    Returns
    -------
    float
        Maximum run-up as a fraction of the trough value.
    """
    roll_min = equity.cummin()
    # Avoid division by zero by replacing zeros with np.nan
    roll_min_safe = roll_min.replace(0, pd.NA)
    ru = (equity - roll_min_safe) / roll_min_safe
    return ru.max(skipna=True) if not ru.empty else 0.0


def run_backtest(
    df: pd.DataFrame,
    price_col: str = "close",
    signal_col: str = "signal",
    initial_equity: float = 10_000.0,
    fee_bps: float = 0.0,
) -> BacktestResult:
    """Run a simple vectorized backtest.

    Parameters
    ----------
    df : DataFrame
        Must contain price_col and signal_col. Assumed chronologically ordered.
    price_col : str
        Name of the price column (close price or similar).
    signal_col : str
        Name of the signal column (target position in {-1,0,1}).
    initial_equity : float
        Starting equity.
    fee_bps : float
        Fee in basis points applied on position change notional (|delta_position| * price).

    Returns
    -------
    BacktestResult
        Aggregate metrics and equity/returns series.
    """
    if len(df) == 0:
        return BacktestResult(
            equity_curve=[],
            returns=[],
            cumulative_return=0.0,
            max_drawdown=0.0,
            max_run_up=0.0,
            win_rate=0.0,
            trades=0
        )

    if price_col not in df.columns or signal_col not in df.columns:
        missing = [c for c in (price_col, signal_col) if c not in df.columns]
        raise ValueError(f"Missing required columns: {missing}")

    prices = df[price_col].astype(float)
    signals = df[signal_col].fillna(0).astype(int)

    if len(prices) < 2:
        # Not enough data to compute returns.
        return BacktestResult(
            equity_curve=[initial_equity],
            returns=[],
            cumulative_return=0.0,
            max_drawdown=0.0,
            max_run_up=0.0,
            win_rate=0.0,
            trades=0,
            )

    rets = prices.pct_change().fillna(0.0)
    prev_pos = signals.shift(1).fillna(0).astype(int)

    raw_pnl_pct = prev_pos * rets  # position return contribution

    # Fees intentionally not modeled in minimal version.

    equity = (1 + raw_pnl_pct).cumprod() * initial_equity

    equity_curve = equity.tolist()
    returns_list = raw_pnl_pct.tolist()

    cumulative_return = (equity.iloc[-1] / initial_equity) - 1.0
    max_dd = _max_drawdown(equity / initial_equity)
    max_run_up = _max_run_up(equity / initial_equity)

    trade_events = (signals != prev_pos).astype(int)
    trades = int(trade_events.sum())

    pnl_positive = raw_pnl_pct[prev_pos != 0] > 0
    wins = int(pnl_positive.sum())
    total_trading_periods = int((prev_pos != 0).sum())
    win_rate = wins / total_trading_periods if total_trading_periods else 0.0

    return BacktestResult(
        equity_curve=equity_curve,
        returns=returns_list,
        cumulative_return=float(cumulative_return),
        max_drawdown=float(max_dd),
        max_run_up=float(max_run_up),
        win_rate=float(win_rate),
        trades=trades,
    )
