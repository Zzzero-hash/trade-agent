"""
Backtesting Framework for Trading Strategies.

This module provides comprehensive backtesting functionality with:
- Out-of-sample testing with costs/slippage modeling
- Performance metrics calculation (CAGR, Sharpe, Calmar, maxDD, turnover, etc.)
- Stress testing capabilities
- Report generation in HTML/CSV formats
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

# Import existing modules
try:
    # Try absolute import first
    from src.data.loaders import load_ohlcv_data
except ImportError:
    try:
        # Try relative import
        from ..data.loaders import load_ohlcv_data
    except ImportError:
        # For direct execution, import from current directory
        import os
        import sys
        sys.path.append(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        from src.data.loaders import load_ohlcv_data


class BacktestEngine:
    """Core backtesting engine with comprehensive evaluation capabilities."""

    def __init__(self,
                 transaction_cost: float = 0.001,
                 slippage: float = 0.0005,
                 initial_capital: float = 100000.0,
                 risk_free_rate: float = 0.02):
        """
        Initialize backtesting engine.

        Args:
            transaction_cost: Transaction cost per trade (default: 0.1%)
            slippage: Slippage per trade (default: 0.05%)
            initial_capital: Initial capital for backtest (default: $100,000)
            risk_free_rate: Annual risk-free rate (default: 2%)
        """
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        # Daily risk-free rate
        self.daily_rf = (1 + risk_free_rate) ** (1/252) - 1

    def run_backtest(self,
                     signals: pd.Series,
                     prices: pd.Series,
                     benchmark: Optional[pd.Series] = None) -> dict[str, Any]:
        """
        Run backtest with given signals and prices.

        Args:
            signals: Trading signals (-1 to 1)
            prices: Asset prices
            benchmark: Benchmark returns for comparison (optional)

        Returns:
            Dict containing backtest results
        """
        # Align data
        df = pd.DataFrame({
            'signal': signals,
            'price': prices
        }).dropna()

        # Calculate positions (assuming signal represents position size)
        # Shift to avoid look-ahead bias
        df['position'] = df['signal'].shift(1).fillna(0)

        # Calculate returns
        df['returns'] = df['price'].pct_change().fillna(0)

        # Calculate position changes for transaction cost calculation
        df['position_change'] = df['position'].diff().fillna(0)

        # Calculate transaction costs
        transaction_cost_per_trade = self.transaction_cost + self.slippage
        # Calculate transaction costs
        df['transaction_costs'] = (
            np.abs(df['position_change']) * transaction_cost_per_trade
        )

        # Calculate strategy returns (gross and net)
        df['gross_return'] = df['position'] * df['returns']
        df['net_return'] = df['gross_return'] - df['transaction_costs']

        # Calculate equity curve
        # Calculate equity curves
        df['gross_equity'] = (
            (1 + df['gross_return']).cumprod() * self.initial_capital
        )
        df['net_equity'] = (
            (1 + df['net_return']).cumprod() * self.initial_capital
        )

        # Calculate trades
        trades = self._extract_trades(df)

        # Calculate metrics
        metrics = self._calculate_metrics(df, trades, benchmark)

        return {
            'data': df,
            'trades': trades,
            'metrics': metrics,
            'gross_equity': df['gross_equity'],
            'net_equity': df['net_equity']
        }

    def _extract_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract individual trades from position changes.

        Args:
            df: DataFrame with position data

        Returns:
            DataFrame with trade information
        """
        trades = []
        current_position = 0
        entry_price = 0
        entry_date = None

        for idx, row in df.iterrows():
            # Check for position change
            if row['position'] != current_position:
                # If we were in a position, close the trade
                if current_position != 0 and entry_date is not None:
                    # Calculate PnL and return for this trade
                    price_ratio = row['price'] / entry_price - 1
                    pnl_value = price_ratio * current_position
                    return_value = price_ratio * current_position
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': idx,
                        'entry_price': entry_price,
                        'exit_price': row['price'],
                        'position': current_position,
                        'pnl': pnl_value,
                        'return': return_value
                    })

                # If new position is not zero, start a new trade
                if row['position'] != 0:
                    entry_date = idx
                    entry_price = row['price']
                else:
                    entry_date = None

                current_position = row['position']

        return pd.DataFrame(trades)

    def _calculate_metrics(
        self,
        df: pd.DataFrame,
        trades: pd.DataFrame,
        benchmark: Optional[pd.Series] = None
    ) -> dict[str, float]:
        """
        Calculate comprehensive performance metrics.

        Args:
            df: Backtest results DataFrame
            trades: Trades DataFrame
            benchmark: Benchmark returns (optional)

        Returns:
            Dictionary of performance metrics
        """
        metrics = {}

        # Basic return metrics
        net_returns = df['net_return']

        # Number of periods
        n_periods = len(net_returns)
        if n_periods == 0:
            return {}

        # Annualization factor (assuming daily data)
        annualization_factor = 252

        # CAGR (Compound Annual Growth Rate)
        total_return = df['net_equity'].iloc[-1] / self.initial_capital - 1
        years = n_periods / annualization_factor
        # CAGR (Compound Annual Growth Rate)
        metrics['cagr'] = (
            (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        )

        # Volatility (annualized standard deviation)
        # Volatility (annualized standard deviation)
        # Volatility (annualized standard deviation)
        volatility_annualized = np.sqrt(annualization_factor)
        metrics['volatility'] = net_returns.std() * volatility_annualized

        # Sharpe Ratio
        excess_return = net_returns.mean() - self.daily_rf
        if metrics['volatility'] > 0:
            # Sharpe Ratio
            std_returns = net_returns.std()
            metrics['sharpe_ratio'] = (
                excess_return / std_returns * np.sqrt(annualization_factor)
            ) if std_returns > 0 else 0
        else:
            metrics['sharpe_ratio'] = 0

        # Maximum Drawdown
        equity_curve = df['net_equity']
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        metrics['max_drawdown'] = drawdown.min()

        # Calmar Ratio
        if metrics['max_drawdown'] < 0:
            # Calmar Ratio
            max_dd = metrics['max_drawdown']
            metrics['calmar_ratio'] = (
                metrics['cagr'] / abs(max_dd)
            ) if max_dd < 0 else 0
        else:
            metrics['calmar_ratio'] = 0

        # Turnover (annualized)
        position_changes = df['position_change'].abs().sum()
        avg_position = df['position'].abs().mean()
        if avg_position > 0:
            # Turnover (annualized)
            turnover_factor = annualization_factor / n_periods
            metrics['turnover'] = (
                (position_changes / avg_position) * turnover_factor
            ) if avg_position > 0 else 0
        else:
            metrics['turnover'] = 0

        # Hit Ratio (winning trades / total trades)
        if len(trades) > 0:
            winning_trades = (trades['return'] > 0).sum()
            metrics['hit_ratio'] = winning_trades / len(trades)

            # Profit Factor
            gross_profits = trades[trades['return'] > 0]['return'].sum()
            gross_losses = abs(trades[trades['return'] < 0]['return'].sum())
            if gross_losses > 0:
                metrics['profit_factor'] = gross_profits / gross_losses
            else:
                metrics['profit_factor'] = np.inf if gross_profits > 0 else 0
        else:
            metrics['hit_ratio'] = 0
            metrics['profit_factor'] = 0

        # Skewness and Kurtosis
        metrics['skewness'] = skew(net_returns)
        metrics['kurtosis'] = kurtosis(net_returns)

        # PnL Autocorrelation (1st order)
        if len(net_returns) > 1:
            metrics['pnl_autocorr'] = net_returns.autocorr()
        else:
            metrics['pnl_autocorr'] = 0

        # Benchmark comparison if provided
        if benchmark is not None:
            # Align benchmark data
            bench_df = pd.DataFrame({
                'benchmark': benchmark,
                'strategy': net_returns
            }).dropna()

            if len(bench_df) > 0:
                # Information Ratio
                active_returns = bench_df['strategy'] - bench_df['benchmark']
                if active_returns.std() > 0:
                    # Information Ratio
                    std_active_returns = active_returns.std()
                    metrics['information_ratio'] = (
                        active_returns.mean() / std_active_returns *
                        np.sqrt(annualization_factor)
                    ) if std_active_returns > 0 else 0
                else:
                    metrics['information_ratio'] = 0

                # Tracking Error
                # Tracking Error
                std_active_returns = active_returns.std()
                metrics['tracking_error'] = (
                    std_active_returns * np.sqrt(annualization_factor)
                )

        return metrics


class StressTester:
    """Stress testing functionality for backtesting results."""

    def __init__(self, backtest_engine: BacktestEngine):
        """
        Initialize stress tester.

        Args:
            backtest_engine: BacktestEngine instance
        """
        self.backtest_engine = backtest_engine

    def run_stress_tests(
        self,
        signals: pd.Series,
        prices: pd.Series
    ) -> dict[str, Any]:
        """
        Run comprehensive stress tests.

        Args:
            signals: Trading signals
            prices: Asset prices

        Returns:
            Dictionary with stress test results
        """
        results = {}

        # Baseline test
        baseline_result = self.backtest_engine.run_backtest(signals, prices)
        results['baseline'] = baseline_result

        # ±50% transaction costs
        # Double costs
        high_cost_engine = BacktestEngine(
            transaction_cost=self.backtest_engine.transaction_cost * 1.5,
            slippage=self.backtest_engine.slippage * 1.5,
            initial_capital=self.backtest_engine.initial_capital
        )
        high_cost_result = high_cost_engine.run_backtest(signals, prices)
        results['high_costs'] = high_cost_result

        # Half costs
        low_cost_engine = BacktestEngine(
            transaction_cost=self.backtest_engine.transaction_cost * 0.5,
            slippage=self.backtest_engine.slippage * 0.5,
            initial_capital=self.backtest_engine.initial_capital
        )
        low_cost_result = low_cost_engine.run_backtest(signals, prices)
        results['low_costs'] = low_cost_result

        # Fill delay +1 bar (shift signals by one more bar)
        delayed_signals = signals.shift(1).fillna(0)
        # Run backtest with delayed signals
        delayed_result = self.backtest_engine.run_backtest(
            delayed_signals, prices
        )
        results['fill_delay'] = delayed_result

        return results


class ReportGenerator:
    """Report generation functionality."""

    def __init__(self, output_dir: str = "reports"):
        """
        Initialize report generator.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_csv_report(
        self,
        backtest_results: dict[str, Any],
        filename: str = "backtest_results.csv"
    ) -> str:
        """
        Generate CSV report.

        Args:
            backtest_results: Backtest results
            filename: Output filename

        Returns:
            Path to generated file
        """
        # Create trades DataFrame
        trades_df = backtest_results['trades'].copy()

        # Create metrics DataFrame
        metrics_df = pd.DataFrame([backtest_results['metrics']])

        # Create equity curve DataFrame
        equity_df = pd.DataFrame({
            'date': backtest_results['data'].index,
            'gross_equity': backtest_results['data']['gross_equity'].values,
            'net_equity': backtest_results['data']['net_equity'].values
        })

        # Save to CSV files
        trades_path = self.output_dir / f"trades_{filename}"
        metrics_path = self.output_dir / f"metrics_{filename}"
        equity_path = self.output_dir / f"equity_{filename}"

        trades_df.to_csv(trades_path, index=False)
        metrics_df.to_csv(metrics_path, index=False)
        equity_df.to_csv(equity_path, index=False)

        return str(trades_path)

    def generate_html_report(
        self,
        backtest_results: dict[str, Any],
        stress_results: Optional[dict[str, Any]] = None,
        filename: str = "backtest_report.html"
    ) -> str:
        """
        Generate HTML report.

        Args:
            backtest_results: Backtest results
            stress_results: Stress test results (optional)
            filename: Output filename

        Returns:
            Path to generated file
        """
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .metric-value {{ text-align: right; }}
        .positive {{ color: green; }}
        .negative {{ color: red; }}
    </style>
</head>
<body>
    <h1>Backtest Report</h1>
    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <h2>Performance Metrics</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
"""

        # Add metrics to HTML
        metrics = backtest_results['metrics']
        for metric, value in metrics.items():
            formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
            metric_name = metric.replace('_', ' ').title()
            html_content += (
                f"        <tr><td>{metric_name}</td>"
                f"<td class='metric-value'>{formatted_value}</td></tr>\n"
            )

        html_content += "    </table>\n"

        # Add stress test results if available
        if stress_results:
            html_content += "    <h2>Stress Test Results</h2>\n"
            html_content += "    <table>\n"
            html_content += (
                "        <tr><th>Scenario</th><th>CAGR</th>"
                "<th>Sharpe Ratio</th><th>Max Drawdown</th></tr>\n"
            )

            for scenario, results in stress_results.items():
                metrics = results['metrics']
                cagr = metrics.get('cagr', 0)
                sharpe = metrics.get('sharpe_ratio', 0)
                max_dd = metrics.get('max_drawdown', 0)

                scenario_name = scenario.replace('_', ' ').title()
                html_content += f"        <tr><td>{scenario_name}</td>"
                html_content += f"<td class='metric-value'>{cagr:.4f}</td>"
                html_content += f"<td class='metric-value'>{sharpe:.4f}</td>"
                html_content += (
                    f"<td class='metric-value'>{max_dd:.4f}</td></tr>\n"
                )

            html_content += "    </table>\n"

        html_content += """
</body>
</html>
"""

        # Save HTML file
        html_path = self.output_dir / filename
        with open(html_path, 'w') as f:
            f.write(html_content)

        return str(html_path)

    def plot_equity_curve(
        self,
        backtest_results: dict[str, Any],
        filename: str = "equity_curve.png"
    ) -> str:
        """
        Plot equity curve.

        Args:
            backtest_results: Backtest results
            filename: Output filename

        Returns:
            Path to generated file
        """
        plt.figure(figsize=(12, 6))

        # Plot equity curves
        plt.plot(
            backtest_results['data'].index,
            backtest_results['data']['gross_equity'],
            label='Gross Equity', alpha=0.7
        )
        plt.plot(
            backtest_results['data'].index,
            backtest_results['data']['net_equity'],
            label='Net Equity (with costs)', linewidth=2
        )

        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save plot
        plot_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        return str(plot_path)


def load_backtest_data(
    data_path: str
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Load data for backtesting.

    Args:
        data_path: Path to data file

    Returns:
        Tuple of (features DataFrame, signals Series, prices Series)
    """
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    elif data_path.endswith('.csv'):
        df = load_ohlcv_data(data_path)
        # Convert to features format if needed
        # This is a simplified example - in practice, you'd have a more complex feature engineering pipeline
        df['signal'] = np.random.randn(len(df))  # Placeholder signals
    else:
        raise ValueError("Unsupported file format. Use .parquet or .csv")

    # Extract signals and prices
    # In a real implementation, you would have a model that generates signals
    # For this example, we'll use a simple moving average crossover strategy
    if 'signal' not in df.columns:
        # Generate simple signals based on price data
        df['signal'] = np.where(df['close'].pct_change(20) > 0, 1, -1)

    signals = df['signal']
    prices = df['close'] if 'close' in df.columns else df.iloc[:, 0]  # Use first column as price if close not available

    return df, signals, prices


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Backtesting Framework")
    parser.add_argument("--data", required=True, help="Path to data file")
    parser.add_argument("--output-dir", default="reports", help="Output directory")
    parser.add_argument(
        "--transaction-cost", type=float, default=0.001,
        help="Transaction cost per trade (default: 0.001)"
    )
    parser.add_argument(
        "--slippage", type=float, default=0.0005,
        help="Slippage per trade (default: 0.0005)"
    )
    parser.add_argument(
        "--initial-capital", type=float, default=100000.0,
        help="Initial capital (default: 100000.0)"
    )
    parser.add_argument(
        "--no-stress-test", action="store_true",
        help="Skip stress testing"
    )

    args = parser.parse_args()

    try:
        # Load data
        print("Loading data...")
        df, signals, prices = load_backtest_data(args.data)
        print(f"Loaded {len(df)} data points")

        # Initialize backtest engine
        engine = BacktestEngine(
            transaction_cost=args.transaction_cost,
            slippage=args.slippage,
            initial_capital=args.initial_capital
        )

        # Run backtest
        print("Running backtest...")
        results = engine.run_backtest(signals, prices)
        print("Backtest completed")

        # Run stress tests
        stress_results = None
        if not args.no_stress_test:
            print("Running stress tests...")
            stress_tester = StressTester(engine)
            stress_results = stress_tester.run_stress_tests(signals, prices)
            print("Stress tests completed")

        # Generate reports
        print("Generating reports...")
        report_generator = ReportGenerator(args.output_dir)

        # CSV report
        csv_path = report_generator.generate_csv_report(
            results, "backtest_results.csv"
        )
        print(f"CSV report saved to: {csv_path}")

        # HTML report
        html_path = report_generator.generate_html_report(
            results, stress_results, "backtest_report.html"
        )
        print(f"HTML report saved to: {html_path}")

        # Equity curve plot
        plot_path = report_generator.plot_equity_curve(
            results, "equity_curve.png"
        )
        print(f"Equity curve plot saved to: {plot_path}")

        # Print key metrics
        print("\n=== Key Performance Metrics ===")
        metrics = results['metrics']
        print(f"CAGR: {metrics.get('cagr', 0):.4f}")
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
        print(f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.4f}")
        print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.4f}")
        print(f"Hit Ratio: {metrics.get('hit_ratio', 0):.4f}")
        print(f"Profit Factor: {metrics.get('profit_factor', 0):.4f}")

        if stress_results:
            print("\n=== Stress Test Results ===")
            for scenario, stress_result in stress_results.items():
                stress_metrics = stress_result['metrics']
                print(f"{scenario.replace('_', ' ').title()}:")
                print(f"  CAGR: {stress_metrics.get('cagr', 0):.4f}")
                sharpe_ratio = stress_metrics.get('sharpe_ratio', 0)
                max_dd = stress_metrics.get('max_drawdown', 0)
                print(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
                print(f"  Max Drawdown: {max_dd:.4f}")

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
