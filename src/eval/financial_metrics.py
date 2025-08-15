"""
Financial Performance Metrics for Trading Models

This module implements industry-standard financial metrics
for evaluating trading strategies.
"""

from typing import Union

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate annualized Sharpe Ratio
    Args:
        returns: Daily returns series (percentage format)
        risk_free_rate: Annualized risk-free rate (e.g. 0.02 for 2%)
    """
    if len(returns) < 2:
        return 0.0

    # Convert to float and calculate excess returns
    daily_rf = risk_free_rate / 252
    excess_returns = returns.astype(float) - daily_rf

    # Calculate components with type safety
    mean_return = float(excess_returns.mean())
    std_dev = float(excess_returns.std()) + 1e-9

    return float(np.sqrt(252) * mean_return / std_dev)

def max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown in percentage terms"""
    cumulative = (1 + returns.astype(float)).cumprod()
    peak = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative / peak) - 1
    return float(drawdown.min() * 100)

def profit_factor(returns: pd.Series) -> float:
    """Ratio of gross profits to gross losses"""
    gains = returns[returns > 0].sum()
    losses = np.abs(returns[returns < 0].sum())
    return gains / losses if losses != 0 else np.nan

def cagr(returns: pd.Series) -> float:
    """Compound Annual Growth Rate"""
    if len(returns) < 1:
        return 0.0
    total_return = (1 + returns).prod()
    years = len(returns) / 252
    return (total_return ** (1/years) - 1) * 100

def calculate_metrics(prices: Union[pd.Series, pd.DataFrame],
                     positions: pd.Series,
                     transaction_cost: float = 0.0001) -> dict:
    """
    Calculate comprehensive financial metrics for trading strategy results

    Args:
        prices: DataFrame with 'close' prices or Series of prices
        positions: Series of positions (-1, 0, 1)
        transaction_cost: Per-trade cost as percentage of notional

    Returns:
        Dictionary of calculated metrics
    """
    if isinstance(prices, pd.DataFrame):
        returns = prices['close'].pct_change().dropna()
    else:
        returns = prices.pct_change().dropna()

    # Calculate strategy returns with transaction costs
    strategy_returns = positions.shift(1) * returns
    trades = positions.diff().abs()
    strategy_returns -= trades * transaction_cost

    return {
        'sharpe_ratio': sharpe_ratio(strategy_returns),
        'max_drawdown': max_drawdown(strategy_returns),
        'profit_factor': profit_factor(strategy_returns),
        'cagr': cagr(strategy_returns),
        'total_return': (strategy_returns + 1).prod() - 1,
        'win_rate': (strategy_returns > 0).mean(),
        'risk_adjusted_return': sharpe_ratio(strategy_returns) * (strategy_returns + 1).prod()
    }
