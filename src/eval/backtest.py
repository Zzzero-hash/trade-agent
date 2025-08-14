from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd


class FeeModel(ABC):
    """Abstract base class for fee models."""

    @abstractmethod
    def calculate_fee(self, trade: dict[str, Any]) -> float:
        """
        Calculate the fee for a given trade.

        Args:
            trade (Dict[str, Any]): Trade details dictionary

        Returns:
            float: The fee amount
        """
        pass


class FixedFeeModel(FeeModel):
    """Fixed fee per trade model."""

    def __init__(self, fee_amount: float = 1.0):
        """
        Initialize the fixed fee model.

        Args:
            fee_amount (float): Fixed fee amount per trade
        """
        self.fee_amount = fee_amount

    def calculate_fee(self, trade: dict[str, Any]) -> float:
        """Calculate fixed fee."""
        return self.fee_amount


class PercentageFeeModel(FeeModel):
    """Percentage-based fee on trade value."""

    def __init__(self, percentage: float = 0.001):
        """
        Initialize the percentage fee model.

        Args:
            percentage (float): Percentage fee (e.g., 0.001 for 0.1%)
        """
        self.percentage = percentage

    def calculate_fee(self, trade: dict[str, Any]) -> float:
        """Calculate percentage fee on total trade value."""
        entry_value = trade['entry_price'] * trade['quantity']
        exit_value = trade['exit_price'] * trade['quantity']
        return (entry_value + exit_value) * self.percentage


class SlippageModel(ABC):
    """Abstract base class for slippage models."""

    @abstractmethod
    def calculate_slippage(self, trade: dict[str, Any]) -> float:
        """
        Calculate the slippage for a given trade.

        Args:
            trade (Dict[str, Any]): Trade details dictionary

        Returns:
            float: The slippage amount (price adjustment per share)
        """
        pass


class FixedSlippageModel(SlippageModel):
    """Fixed slippage amount per share."""

    def __init__(self, slippage_per_share: float = 0.01):
        """
        Initialize the fixed slippage model.

        Args:
            slippage_per_share (float): Fixed slippage amount per share
        """
        self.slippage_per_share = slippage_per_share

    def calculate_slippage(self, trade: dict[str, Any]) -> float:
        """Calculate fixed slippage."""
        return self.slippage_per_share


class PercentageSlippageModel(SlippageModel):
    """Percentage-based slippage on entry price."""

    def __init__(self, percentage: float = 0.0005):
        """
        Initialize the percentage slippage model.

        Args:
            percentage (float): Percentage slippage (e.g., 0.0005 for 0.05%)
        """
        self.percentage = percentage

    def calculate_slippage(self, trade: dict[str, Any]) -> float:
        """Calculate percentage slippage."""
        return trade['entry_price'] * self.percentage


class MarketImpactSlippageModel(SlippageModel):
    """Market impact based slippage model."""

    def __init__(self, impact_coefficient: float = 0.1):
        """
        Initialize the market impact slippage model.

        Args:
            impact_coefficient (float): Coefficient for market impact calculation
        """
        self.impact_coefficient = impact_coefficient

    def calculate_slippage(self, trade: dict[str, Any]) -> float:
        """Calculate market impact slippage based on trade size and market volume."""
        # Simple model: slippage proportional to trade size relative to average volume
        avg_volume = trade.get('avg_volume', 1000000)  # Default average volume
        quantity = trade['quantity']
        price = trade['entry_price']

        # Market impact as a percentage of price
        impact_ratio = (quantity / avg_volume) * self.impact_coefficient
        return price * impact_ratio


class TradingRegime(Enum):
    """Enumeration of trading regimes."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class PerformanceMetrics:
    """Data class to hold performance metrics."""
    initial_capital: float
    final_capital: float
    cumulative_pnl: float
    return_percentage: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    exposure: float
    turnover: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    num_trades: int


class BacktestEngine:
    """
    A brokerage-style backtesting engine for P&L reconciliation,
    including configurable hooks for fees and slippage.
    """

    def __init__(self, initial_capital: float = 100000.0):
        """
        Initializes the BacktestEngine with a starting capital.

        Args:
            initial_capital (float): The starting capital for the backtest.
        """
        self.initial_capital = initial_capital
        self.trades = []
        self.equity_curve = pd.Series(dtype=float)
        self.current_capital = initial_capital
        self.fee_model: Optional[FeeModel] = None
        self.slippage_model: Optional[SlippageModel] = None

    def set_fee_model(self, fee_model: FeeModel) -> None:
        """
        Set the fee model for the backtest.

        Args:
            fee_model (FeeModel): The fee model to use
        """
        self.fee_model = fee_model

    def set_slippage_model(self, slippage_model: SlippageModel) -> None:
        """
        Set the slippage model for the backtest.

        Args:
            slippage_model (SlippageModel): The slippage model to use
        """
        self.slippage_model = slippage_model

    def run_backtest(
        self,
        trade_data: list[dict[str, Any]]
    ):
        """
        Runs the backtest simulation over the provided trade data.

        Args:
            trade_data (List[Dict[str, Any]]): A list of trade dictionaries,
                                                each containing 'entry_price', 'exit_price',
                                                'quantity', 'entry_timestamp', 'exit_timestamp',
                                                etc.
        """
        self.trades = []
        self.equity_curve = pd.Series(index=pd.to_datetime([]), dtype=float)
        self.current_capital = self.initial_capital
        # Placeholder for initial equity
        self.equity_curve[pd.Timestamp.min] = self.initial_capital

        for trade in trade_data:
            # Apply slippage before calculating actual entry/exit prices
            adjusted_entry_price = trade['entry_price']
            adjusted_exit_price = trade['exit_price']

            if self.slippage_model:
                slippage_amount = self.slippage_model.calculate_slippage(trade)
                # Assuming slippage_amount is a price adjustment per share
                if trade.get('trade_type') == 'long':
                    adjusted_entry_price += slippage_amount  # Higher entry price for long
                    adjusted_exit_price -= slippage_amount   # Lower exit price for long
                elif trade.get('trade_type') == 'short':
                    adjusted_entry_price -= slippage_amount  # Lower entry price for short
                    adjusted_exit_price += slippage_amount   # Higher exit price for short

            # Calculate gross P&L
            pnl_per_share = adjusted_exit_price - adjusted_entry_price
            gross_pnl = pnl_per_share * trade['quantity']

            # Apply fees
            fees = 0.0
            if self.fee_model:
                fees = self.fee_model.calculate_fee(trade)

            net_pnl = gross_pnl - fees

            # Update current capital
            self.current_capital += net_pnl

            # Record trade details
            trade_record = {
                'entry_timestamp': trade['entry_timestamp'],
                'exit_timestamp': trade['exit_timestamp'],
                'entry_price': trade['entry_price'],
                'exit_price': trade['exit_price'],
                'adjusted_entry_price': adjusted_entry_price,
                'adjusted_exit_price': adjusted_exit_price,
                'quantity': trade['quantity'],
                'trade_type': trade.get('trade_type', 'unknown'),
                'gross_pnl': gross_pnl,
                'fees': fees,
                'net_pnl': net_pnl,
                'cumulative_pnl': self.current_capital - self.initial_capital,
                'capital_after_trade': self.current_capital,
                'regime': trade.get('regime', 'unknown')
            }
            self.trades.append(trade_record)

            # Update equity curve
            entry_ts = pd.to_datetime(trade['entry_timestamp'])
            exit_ts = pd.to_datetime(trade['exit_timestamp'])

            # Ensure equity curve is monotonically increasing in index
            if (entry_ts not in self.equity_curve.index or
                    self.equity_curve.index.max() < entry_ts):
                # Capital before this trade
                self.equity_curve[entry_ts] = self.current_capital - net_pnl
            if (exit_ts not in self.equity_curve.index or
                    self.equity_curve.index.max() < exit_ts):
                self.equity_curve[exit_ts] = self.current_capital

        # Sort equity curve by index (timestamp)
        self.equity_curve = self.equity_curve.sort_index()
        # Remove the placeholder
        self.equity_curve = self.equity_curve[self.equity_curve.index != pd.Timestamp.min]

    def get_equity_curve(self) -> pd.Series:
        """
        Returns the equity curve of the backtest.

        Returns:
            pd.Series: A time-indexed Series representing the capital over time.
        """
        return self.equity_curve

    def get_trades(self) -> list[dict[str, Any]]:
        """
        Returns the detailed records of all trades executed during the backtest.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a trade.
        """
        return self.trades

    def get_cumulative_pnl(self) -> float:
        """
        Returns the total cumulative P&L from the backtest.

        Returns:
            float: The total profit or loss.
        """
        return self.current_capital - self.initial_capital

    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """
        Calculate the Sharpe ratio.

        Args:
            returns (pd.Series): Series of returns
            risk_free_rate (float): Risk-free rate

        Returns:
            float: Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - risk_free_rate
        std_dev = excess_returns.std()

        if std_dev == 0:
            return 0.0

        return (excess_returns.mean() / std_dev) * np.sqrt(252)  # Annualized

    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> tuple:
        """
        Calculate maximum drawdown and drawdown duration.

        Args:
            equity_curve (pd.Series): Equity curve

        Returns:
            tuple: (max_drawdown, max_drawdown_duration)
        """
        if len(equity_curve) < 2:
            return 0.0, 0

        # Calculate running maximum
        running_max = equity_curve.expanding().max()

        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max

        # Maximum drawdown
        max_dd = drawdown.min()

        # Drawdown duration
        # Find periods in drawdown
        in_drawdown = drawdown < 0
        if not in_drawdown.any():
            max_duration = 0
        else:
            # Calculate duration of each drawdown period
            drawdown_periods = (~in_drawdown).cumsum()
            drawdown_grouped = in_drawdown.groupby(drawdown_periods).cumsum()
            max_duration = int(drawdown_grouped.max()) if not drawdown_grouped.empty else 0

        return abs(max_dd), max_duration

    def _calculate_exposure(self) -> float:
        """
        Calculate market exposure as the average capital at risk.

        Returns:
            float: Average market exposure
        """
        if not self.trades:
            return 0.0

        total_exposure = 0.0
        for trade in self.trades:
            entry_value = trade['entry_price'] * trade['quantity']
            total_exposure += entry_value

        return total_exposure / len(self.trades) if self.trades else 0.0

    def _calculate_turnover(self) -> float:
        """
        Calculate portfolio turnover.

        Returns:
            float: Portfolio turnover
        """
        if not self.trades:
            return 0.0

        total_value_traded = 0.0
        for trade in self.trades:
            entry_value = trade['entry_price'] * trade['quantity']
            exit_value = trade['exit_price'] * trade['quantity']
            total_value_traded += entry_value + exit_value

        # Turnover = total value traded / average portfolio value
        avg_portfolio_value = (self.initial_capital + self.current_capital) / 2
        if avg_portfolio_value == 0:
            return 0.0

        return total_value_traded / avg_portfolio_value

    def _calculate_win_rate(self) -> float:
        """
        Calculate the win rate of trades.

        Returns:
            float: Win rate as a percentage
        """
        if not self.trades:
            return 0.0

        winning_trades = sum(1 for trade in self.trades if trade['net_pnl'] > 0)
        return (winning_trades / len(self.trades)) * 100

    def _calculate_profit_factor(self) -> float:
        """
        Calculate the profit factor (gross profits / gross losses).

        Returns:
            float: Profit factor
        """
        if not self.trades:
            return 0.0

        gross_profits = sum(trade['gross_pnl'] for trade in self.trades if trade['gross_pnl'] > 0)
        gross_losses = abs(sum(trade['gross_pnl'] for trade in self.trades if trade['gross_pnl'] < 0))

        if gross_losses == 0:
            return float('inf') if gross_profits > 0 else 0.0

        return gross_profits / gross_losses

    def _calculate_avg_win_loss(self) -> tuple:
        """
        Calculate average win and average loss.

        Returns:
            tuple: (average_win, average_loss)
        """
        if not self.trades:
            return 0.0, 0.0

        winning_trades = [trade['net_pnl'] for trade in self.trades if trade['net_pnl'] > 0]
        losing_trades = [abs(trade['net_pnl']) for trade in self.trades if trade['net_pnl'] < 0]

        avg_win = np.mean(winning_trades) if winning_trades else 0.0
        avg_loss = np.mean(losing_trades) if losing_trades else 0.0

        return avg_win, avg_loss

    def get_performance_metrics(self, risk_free_rate: float = 0.0) -> PerformanceMetrics:
        """
        Calculates and returns key performance metrics.

        Args:
            risk_free_rate (float): Risk-free rate for Sharpe ratio calculation

        Returns:
            PerformanceMetrics: A dataclass containing performance metrics
        """
        # Calculate returns from equity curve
        if len(self.equity_curve) > 1:
            returns = self.equity_curve.pct_change().dropna()
        else:
            returns = pd.Series(dtype=float)

        # Calculate metrics
        cumulative_pnl = self.get_cumulative_pnl()
        sharpe_ratio = self._calculate_sharpe_ratio(returns, risk_free_rate)
        max_drawdown, max_drawdown_duration = self._calculate_max_drawdown(self.equity_curve)
        exposure = self._calculate_exposure()
        turnover = self._calculate_turnover()
        win_rate = self._calculate_win_rate()
        profit_factor = self._calculate_profit_factor()
        avg_win, avg_loss = self._calculate_avg_win_loss()

        return PerformanceMetrics(
            initial_capital=self.initial_capital,
            final_capital=self.current_capital,
            cumulative_pnl=cumulative_pnl,
            return_percentage=(cumulative_pnl / self.initial_capital) * 100 if self.initial_capital else 0,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            exposure=exposure,
            turnover=turnover,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            num_trades=len(self.trades)
        )

    def get_regime_performance(self) -> dict[str, PerformanceMetrics]:
        """
        Calculate performance metrics for each trading regime.

        Returns:
            Dict[str, PerformanceMetrics]: Dictionary mapping regime names to performance metrics
        """
        regime_metrics = {}

        # Group trades by regime
        regimes = {}
        for trade in self.trades:
            regime = trade.get('regime', 'unknown')
            if regime not in regimes:
                regimes[regime] = []
            regimes[regime].append(trade)

        # Calculate metrics for each regime
        for regime, trades in regimes.items():
            if not trades:
                continue

            # Create a temporary backtest engine for this regime
            regime_engine = BacktestEngine(self.initial_capital)
            regime_engine.trades = trades

            # Calculate cumulative P&L for this regime
            regime_pnl = sum(trade['net_pnl'] for trade in trades)
            regime_engine.current_capital = self.initial_capital + regime_pnl

            # Create a simple equity curve for this regime
            regime_equity = pd.Series(dtype=float)
            capital = self.initial_capital
            for trade in trades:
                entry_ts = pd.to_datetime(trade['entry_timestamp'])
                exit_ts = pd.to_datetime(trade['exit_timestamp'])
                capital_before = capital
                capital += trade['net_pnl']
                regime_equity[entry_ts] = capital_before
                regime_equity[exit_ts] = capital

            regime_engine.equity_curve = regime_equity.sort_index()

            # Calculate metrics
            regime_metrics[regime] = regime_engine.get_performance_metrics()

        return regime_metrics

    def get_trade_statistics(self) -> dict[str, Any]:
        """
        Get detailed trade statistics.

        Returns:
            Dict[str, Any]: Dictionary containing trade statistics
        """
        if not self.trades:
            return {}

        pnls = [trade['net_pnl'] for trade in self.trades]
        gross_pnls = [trade['gross_pnl'] for trade in self.trades]
        fees = [trade['fees'] for trade in self.trades]

        return {
            'total_trades': len(self.trades),
            'winning_trades': len([pnl for pnl in pnls if pnl > 0]),
            'losing_trades': len([pnl for pnl in pnls if pnl < 0]),
            'win_rate': (len([pnl for pnl in pnls if pnl > 0]) / len(pnls)) * 100,
            'avg_pnl': np.mean(pnls),
            'avg_gross_pnl': np.mean(gross_pnls),
            'avg_fee': np.mean(fees),
            'largest_win': max(pnls) if pnls else 0,
            'largest_loss': min(pnls) if pnls else 0,
            'total_fees': sum(fees)
        }
