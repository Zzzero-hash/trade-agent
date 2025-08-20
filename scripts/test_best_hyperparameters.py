#!/usr/bin/env python3
"""
Test Evaluation Script for Best Hyperparameters from Optuna Study

This script:
1. Loads the best parameters from the Optuna study
2. Creates an untouched test slice from the data (last 20%)
3. Trains a PPO agent with the optimal reward parameters
4. Evaluates performance using comprehensive metrics
5. Saves results in structured format with documentation

The test slice is truly "untouched" as it uses the final 20% of the data
that was not used during the original hyperparameter tuning process.
"""

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback


# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.backtest import BacktestEngine, ReportGenerator, StressTester
from trade_agent.agents.sl.models.base import set_all_seeds
from trade_agent.envs.trading_env import TradingEnvironment


# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class BestHyperparameterTestEvaluator:
    """Comprehensive test evaluator for best hyperparameters."""

    def __init__(self,
                 data_file: str = "data/features.parquet",
                 best_params_file: str = "reports/reward_params_best_params.json",
                 output_dir: str = "reports",
                 test_split_ratio: float = 0.2,
                 seed: int = 42) -> None:
        """
        Initialize the test evaluator.

        Args:
            data_file: Path to the features data file
            best_params_file: Path to the best parameters JSON file
            output_dir: Directory for output files
            test_split_ratio: Ratio of data to use for testing (default: 20%)
            seed: Random seed for reproducibility
        """
        self.data_file = data_file
        self.best_params_file = best_params_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.test_split_ratio = test_split_ratio
        self.seed = seed

        # Set seeds for reproducibility
        set_all_seeds(seed)

        # Load data and best parameters
        self.df = pd.read_parquet(data_file)
        self.best_params = self._load_best_parameters()

        # Create test slice
        self.train_data, self.test_data = self._create_test_slice()


    def _load_best_parameters(self) -> dict[str, float]:
        """Load best parameters from Optuna study."""
        with open(self.best_params_file) as f:
            return json.load(f)

    def _create_test_slice(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create truly untouched test slice (last 20% of data).

        Returns:
            Tuple of (train_data, test_data)
        """
        # Calculate split point for test slice
        split_point = int(len(self.df) * (1 - self.test_split_ratio))

        # Split data chronologically
        train_data = self.df.iloc[:split_point].copy()
        test_data = self.df.iloc[split_point:].copy()

        return train_data, test_data

    def train_agent(self,
                   total_timesteps: int = 200000,
                   eval_freq: int = 10000) -> PPO:
        """
        Train PPO agent with best reward parameters.

        Args:
            total_timesteps: Total training timesteps
            eval_freq: Evaluation frequency

        Returns:
            Trained PPO model
        """

        # Save training data temporarily
        train_file = self.output_dir / "temp_train_data.parquet"
        self.train_data.to_parquet(train_file)

        try:
            # Create training environment with best reward parameters
            train_env = TradingEnvironment(
                data_file=str(train_file),
                initial_capital=100000.0,
                transaction_cost=0.001,
                seed=self.seed,
                window_size=30,
                reward_config=self.best_params
            )

            # Load PPO configuration
            with open("configs/ppo_config.json") as f:
                config = json.load(f)

            ppo_config = config.get('ppo', {})

            # Create PPO model with configuration
            model = PPO(
                "MlpPolicy",
                train_env,
                learning_rate=ppo_config.get('learning_rate', 3e-4),
                n_steps=ppo_config.get('n_steps', 2048),
                batch_size=ppo_config.get('batch_size', 64),
                n_epochs=ppo_config.get('n_epochs', 10),
                gamma=ppo_config.get('gamma', 0.99),
                gae_lambda=ppo_config.get('gae_lambda', 0.95),
                clip_range=ppo_config.get('clip_range', 0.2),
                ent_coef=ppo_config.get('ent_coef', 0.0),
                vf_coef=ppo_config.get('vf_coef', 0.5),
                max_grad_norm=ppo_config.get('max_grad_norm', 0.5),
                seed=self.seed,
                verbose=1
            )

            # Create evaluation environment for monitoring
            eval_env = TradingEnvironment(
                data_file=str(train_file),
                initial_capital=100000.0,
                transaction_cost=0.001,
                seed=self.seed + 1,
                window_size=30,
                reward_config=self.best_params
            )

            # Set up evaluation callback
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(self.output_dir),
                log_path=str(self.output_dir),
                eval_freq=eval_freq,
                deterministic=True,
                render=False
            )

            # Train the model
            model.learn(
                total_timesteps=total_timesteps,
                callback=eval_callback,
                progress_bar=True
            )

            return model

        finally:
            # Clean up temporary file
            if train_file.exists():
                train_file.unlink()

    def evaluate_on_test_set(self, model: PPO) -> dict[str, Any]:
        """
        Evaluate trained model on untouched test set.

        Args:
            model: Trained PPO model

        Returns:
            Dictionary containing evaluation results
        """

        # Save test data temporarily
        test_file = self.output_dir / "temp_test_data.parquet"
        self.test_data.to_parquet(test_file)

        try:
            # Create test environment with best reward parameters
            test_env = TradingEnvironment(
                data_file=str(test_file),
                initial_capital=100000.0,
                transaction_cost=0.001,
                seed=self.seed + 2,
                window_size=30,
                reward_config=self.best_params
            )

            # Run evaluation episode
            obs, _ = test_env.reset()
            signals = []
            prices = []
            rewards = []
            actions = []
            infos = []

            episode_reward = 0

            while True:
                # Get action from trained model
                action, _ = model.predict(obs, deterministic=True)

                # Store data for analysis
                prices.append(test_env.prices[test_env.current_step])
                actions.append(action[0])

                # Step environment
                obs, reward, terminated, truncated, info = test_env.step(action)

                # Store results
                rewards.append(reward)
                episode_reward += reward
                infos.append(info)

                # Convert action to signal (-1 to 1)
                signal = np.clip(action[0], -1, 1)
                signals.append(signal)

                # Break if episode is done
                if terminated or truncated:
                    break

            # Convert to pandas Series for analysis
            signals_series = pd.Series(signals, index=self.test_data.index[:len(signals)])
            prices_series = pd.Series(prices, index=self.test_data.index[:len(prices)])
            rewards_series = pd.Series(rewards, index=self.test_data.index[:len(rewards)])
            actions_series = pd.Series(actions, index=self.test_data.index[:len(actions)])


            return {
                'signals': signals_series,
                'prices': prices_series,
                'rewards': rewards_series,
                'actions': actions_series,
                'episode_reward': episode_reward,
                'final_equity': infos[-1]['equity'] if infos else 100000.0,
                'accounting_errors': infos[-1]['accounting_errors'] if infos else []
            }

        finally:
            # Clean up temporary file
            if test_file.exists():
                test_file.unlink()

    def run_comprehensive_backtest(self,
                                 signals: pd.Series,
                                 prices: pd.Series) -> dict[str, Any]:
        """
        Run comprehensive backtesting analysis.

        Args:
            signals: Trading signals from the model
            prices: Asset prices

        Returns:
            Dictionary containing backtest results
        """

        # Initialize backtest engine
        backtest_engine = BacktestEngine(
            transaction_cost=0.001,
            slippage=0.0005,
            initial_capital=100000.0,
            risk_free_rate=0.02
        )

        # Run main backtest
        results = backtest_engine.run_backtest(signals, prices)

        # Run stress tests
        stress_tester = StressTester(backtest_engine)
        stress_results = stress_tester.run_stress_tests(signals, prices)

        return {
            'backtest_results': results,
            'stress_results': stress_results
        }

    def save_results(self,
                    evaluation_results: dict[str, Any],
                    backtest_results: dict[str, Any],
                    model: PPO) -> dict[str, str]:
        """
        Save all results in structured format.

        Args:
            evaluation_results: Results from model evaluation
            backtest_results: Results from backtesting
            model: Trained model

        Returns:
            Dictionary with paths to saved files
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save model
        model_path = self.output_dir / f"best_hyperparams_model_{timestamp}.zip"
        model.save(str(model_path))

        # Save evaluation results
        eval_results_path = self.output_dir / f"best_hyperparams_evaluation_{timestamp}.json"

        # Prepare evaluation results for JSON serialization
        eval_results_serializable = {
            'episode_reward': float(evaluation_results['episode_reward']),
            'final_equity': float(evaluation_results['final_equity']),
            'accounting_errors': evaluation_results['accounting_errors'],
            'test_period': {
                'start_date': str(self.test_data.index[0]),
                'end_date': str(self.test_data.index[-1]),
                'num_samples': len(self.test_data)
            },
            'best_parameters': self.best_params,
            'timestamp': timestamp
        }

        with open(eval_results_path, 'w') as f:
            json.dump(eval_results_serializable, f, indent=2)

        # Save detailed signals and actions
        signals_path = self.output_dir / f"best_hyperparams_signals_{timestamp}.csv"
        signals_df = pd.DataFrame({
            'signal': evaluation_results['signals'],
            'price': evaluation_results['prices'],
            'reward': evaluation_results['rewards'],
            'action': evaluation_results['actions']
        })
        signals_df.to_csv(signals_path)

        # Generate comprehensive reports using existing framework
        report_generator = ReportGenerator(str(self.output_dir))

        # Generate CSV reports
        csv_path = report_generator.generate_csv_report(
            backtest_results['backtest_results'],
            f"best_hyperparams_backtest_{timestamp}.csv"
        )

        # Generate HTML report
        html_path = report_generator.generate_html_report(
            backtest_results['backtest_results'],
            backtest_results['stress_results'],
            f"best_hyperparams_report_{timestamp}.html"
        )

        # Generate equity curve plot
        plot_path = report_generator.plot_equity_curve(
            backtest_results['backtest_results'],
            f"best_hyperparams_equity_{timestamp}.png"
        )

        # Create comprehensive summary
        summary = self._create_comprehensive_summary(
            evaluation_results, backtest_results, timestamp
        )

        summary_path = self.output_dir / f"best_hyperparams_summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return {
            'model_path': str(model_path),
            'evaluation_results_path': str(eval_results_path),
            'signals_path': str(signals_path),
            'csv_report_path': csv_path,
            'html_report_path': html_path,
            'equity_plot_path': plot_path,
            'summary_path': str(summary_path)
        }

    def _create_comprehensive_summary(self,
                                    evaluation_results: dict[str, Any],
                                    backtest_results: dict[str, Any],
                                    timestamp: str) -> dict[str, Any]:
        """Create comprehensive summary of all results."""

        metrics = backtest_results['backtest_results']['metrics']

        return {
            'experiment_info': {
                'timestamp': timestamp,
                'test_period': {
                    'start_date': str(self.test_data.index[0]),
                    'end_date': str(self.test_data.index[-1]),
                    'num_samples': len(self.test_data),
                    'duration_days': (self.test_data.index[-1] - self.test_data.index[0]).days
                },
                'best_parameters': self.best_params,
                'data_file': self.data_file,
                'seed': self.seed
            },
            'performance_metrics': {
                'episode_reward': float(evaluation_results['episode_reward']),
                'final_equity': float(evaluation_results['final_equity']),
                'total_return': (float(evaluation_results['final_equity']) / 100000.0) - 1,
                'cagr': float(metrics.get('cagr', 0)),
                'sharpe_ratio': float(metrics.get('sharpe_ratio', 0)),
                'calmar_ratio': float(metrics.get('calmar_ratio', 0)),
                'max_drawdown': float(metrics.get('max_drawdown', 0)),
                'volatility': float(metrics.get('volatility', 0)),
                'hit_ratio': float(metrics.get('hit_ratio', 0)),
                'profit_factor': float(metrics.get('profit_factor', 0)),
                'turnover': float(metrics.get('turnover', 0))
            },
            'risk_metrics': {
                'max_drawdown': float(metrics.get('max_drawdown', 0)),
                'volatility': float(metrics.get('volatility', 0)),
                'skewness': float(metrics.get('skewness', 0)),
                'kurtosis': float(metrics.get('kurtosis', 0)),
                'pnl_autocorr': float(metrics.get('pnl_autocorr', 0))
            },
            'trading_metrics': {
                'hit_ratio': float(metrics.get('hit_ratio', 0)),
                'profit_factor': float(metrics.get('profit_factor', 0)),
                'turnover': float(metrics.get('turnover', 0)),
                'num_trades': len(backtest_results['backtest_results']['trades'])
            },
            'stress_test_summary': {
                scenario: {
                    'cagr': float(results['metrics'].get('cagr', 0)),
                    'sharpe_ratio': float(results['metrics'].get('sharpe_ratio', 0)),
                    'max_drawdown': float(results['metrics'].get('max_drawdown', 0))
                }
                for scenario, results in backtest_results['stress_results'].items()
            },
            'accounting_status': {
                'errors_count': len(evaluation_results['accounting_errors']),
                'errors': evaluation_results['accounting_errors']
            }
        }

    def run_full_evaluation(self,
                          total_timesteps: int = 200000,
                          eval_freq: int = 10000) -> dict[str, str]:
        """
        Run complete evaluation pipeline.

        Args:
            total_timesteps: Training timesteps
            eval_freq: Evaluation frequency

        Returns:
            Dictionary with paths to all generated files
        """

        # 1. Train agent
        model = self.train_agent(total_timesteps, eval_freq)

        # 2. Evaluate on test set
        evaluation_results = self.evaluate_on_test_set(model)

        # 3. Run comprehensive backtest
        backtest_results = self.run_comprehensive_backtest(
            evaluation_results['signals'],
            evaluation_results['prices']
        )

        # 4. Save all results
        file_paths = self.save_results(evaluation_results, backtest_results, model)

        # 5. Print summary

        backtest_results['backtest_results']['metrics']

        for _name, _path in file_paths.items():
            pass

        return file_paths


def main() -> None:
    """Main function for command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Best Hyperparameters Evaluation")
    parser.add_argument("--data", default="data/features.parquet",
                       help="Path to features data file")
    parser.add_argument("--best-params", default="reports/reward_params_best_params.json",
                       help="Path to best parameters JSON file")
    parser.add_argument("--output-dir", default="reports",
                       help="Output directory for results")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                       help="Ratio of data to use for testing")
    parser.add_argument("--timesteps", type=int, default=200000,
                       help="Training timesteps")
    parser.add_argument("--eval-freq", type=int, default=10000,
                       help="Evaluation frequency during training")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    args = parser.parse_args()

    try:
        # Create evaluator
        evaluator = BestHyperparameterTestEvaluator(
            data_file=args.data,
            best_params_file=args.best_params,
            output_dir=args.output_dir,
            test_split_ratio=args.test_ratio,
            seed=args.seed
        )

        # Run evaluation
        evaluator.run_full_evaluation(
            total_timesteps=args.timesteps,
            eval_freq=args.eval_freq
        )


    except Exception:
        raise


if __name__ == "__main__":
    main()
