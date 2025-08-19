#!/usr/bin/env python3
"""
Validation script to inspect hyperparameter tuning results and test on data.

This script:
1. Loads saved Optuna studies from reports/
2. Analyzes top 5-10 trials for parameter stability
3. Identifies if parameters make sense (no extreme/implausible values)
4. Runs best parameters on untouched test slice (last 20% of data)
5. Compares validation vs test performance
6. Flags overfitting and razor-thin optima
"""

import json
import os
import pickle
import sys
import warnings
from datetime import datetime
from typing import Any

import numpy as np
import optuna
import pandas as pd
from scipy import stats


# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules (primary path first, then fallbacks)
try:
    from src.eval.backtest import BacktestEngine  # type: ignore
    from trade_agent.agents.envs.trading_env import TradingEnvironment  # type: ignore
    from trade_agent.agents.sl.models.base import set_all_seeds  # type: ignore
    from trade_agent.agents.sl.models.factory import SLModelFactory  # type: ignore
    from trade_agent.agents.sl.train import SLTrainingPipeline  # type: ignore
    from trade_agent.data.splits import purged_walk_forward_splits  # type: ignore
except ImportError:
    try:
        from eval.backtest import BacktestEngine  # type: ignore

        from trade_agent.agents.envs.trading_env import (
            TradingEnvironment,  # type: ignore
        )
        from trade_agent.agents.sl.models.base import set_all_seeds  # type: ignore
        from trade_agent.agents.sl.models.factory import SLModelFactory  # type: ignore
        from trade_agent.agents.sl.train import SLTrainingPipeline  # type: ignore
        from trade_agent.data.splits import purged_walk_forward_splits  # type: ignore
    except ImportError:
        purged_walk_forward_splits = None  # type: ignore
        TradingEnvironment = None  # type: ignore
        BacktestEngine = None  # type: ignore
        set_all_seeds = None  # type: ignore
        SLModelFactory = None  # type: ignore
        SLTrainingPipeline = None  # type: ignore

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class TuningValidator:
    """Validates hyperparameter tuning results and tests on untouched data."""

    def __init__(self, data_file: str = "data/features.parquet") -> None:
        """Initialize tuning validator."""
        self.data_file = data_file
        self.df = pd.read_parquet(data_file)
        self.feature_columns = [
            col for col in self.df.columns
            if col not in ['mu_hat', 'sigma_hat']
        ]
        self.X = self.df[self.feature_columns].values
        self.y = self.df['mu_hat'].values

        # Set random seed for reproducibility
        if set_all_seeds:
            set_all_seeds(42)

    def load_study(self, study_path: str) -> optuna.Study:
        """
        Load Optuna study from pickle file.

        Args:
            study_path: Path to the study pickle file

        Returns:
            Loaded Optuna study
        """
        with open(study_path, 'rb') as f:
            return pickle.load(f)

    def analyze_parameter_stability(self, study: optuna.Study, top_k: int = 10) -> dict[str, Any]:
        """
        Analyze parameter stability across top trials.

        Args:
            study: Optuna study to analyze
            top_k: Number of top trials to analyze

        Returns:
            Dictionary with stability analysis results
        """
        # Get top trials
        trials = study.trials
        if len(trials) < 2:
            return {"error": "Not enough trials for stability analysis"}

        # Sort by value (descending for maximization)
        sorted_trials = sorted(trials, key=lambda t: t.value if t.value is not None else float('-inf'), reverse=True)
        top_trials = sorted_trials[:min(top_k, len(sorted_trials))]

        # Extract parameters
        param_names = list(top_trials[0].params.keys())
        param_values = {name: [] for name in param_names}
        values = []

        for trial in top_trials:
            if trial.value is not None:
                values.append(trial.value)
                for name in param_names:
                    param_values[name].append(trial.params.get(name, np.nan))

        # Calculate stability metrics
        stability_analysis = {
            "top_k": len(top_trials),
            "best_value": max(values) if values else None,
            "worst_value": min(values) if values else None,
            "mean_value": np.mean(values) if values else None,
            "std_value": np.std(values) if values else None,
            "cv_value": np.std(values) / np.mean(values) if values and np.mean(values) != 0 else None,
            "parameters": {}
        }

        # Analyze each parameter
        for name in param_names:
            values_list = param_values[name]
            if values_list and all(v is not None for v in values_list):
                # Convert to numeric for analysis
                numeric_values = []
                for v in values_list:
                    try:
                        numeric_values.append(float(v))
                    except (ValueError, TypeError):
                        numeric_values.append(np.nan)

                if numeric_values and not all(np.isnan(numeric_values)):
                    stability_analysis["parameters"][name] = {
                        "values": values_list,
                        "mean": np.nanmean(numeric_values),
                        "std": np.nanstd(numeric_values),
                        "cv": np.nanstd(numeric_values) / abs(np.nanmean(numeric_values)) if np.nanmean(numeric_values) != 0 else None,
                        "min": np.nanmin(numeric_values),
                        "max": np.nanmax(numeric_values),
                        "range": np.nanmax(numeric_values) - np.nanmin(numeric_values),
                        "is_extreme": self._check_extreme_values(numeric_values, name),
                        "distribution": self._analyze_distribution(numeric_values)
                    }

        return stability_analysis

    def _check_extreme_values(self, values: list[float], param_name: str) -> dict[str, bool]:
        """Check if parameter values are extreme or implausible."""
        if not values:
            return {}

        # Define reasonable ranges for common parameters
        ranges = {
            "learning_rate": (1e-6, 1e-1),
            "alpha": (1e-6, 1000),
            "dropout": (0.0, 0.8),
            "batch_size": (8, 2048),
            "epochs": (5, 1000),
            "window_size": (3, 100),
            "hidden_size": (8, 2048),
            "gamma": (0.8, 1.0),
            "clip_range": (0.01, 0.5),
            "ent_coef": (0.0, 1.0),
            "vf_coef": (0.0, 1.0),
            "tau": (1e-4, 0.1),
            "temperature": (1e-4, 1.0),
            "pnl_weight": (0.1, 20.0),
            "transaction_cost_weight": (1e-4, 50.0),
            "risk_adjustment_weight": (0.0, 10.0),
            "stability_penalty_weight": (0.0, 5.0),
            "drawdown_penalty_weight": (0.0, 5.0),
            "volatility_penalty_weight": (0.0, 2.0),
            "position_change_penalty": (0.0, 1.0),
            "max_position_penalty": (0.0, 2.0)
        }

        # Check if parameter is in our defined ranges
        param_range = ranges.get(param_name, (None, None))
        if param_range[0] is None:
            return {"unknown_parameter": True}

        values_array = np.array(values)
        return {
            "too_low": np.any(values_array < param_range[0]),
            "too_high": np.any(values_array > param_range[1]),
            "out_of_range": np.any((values_array < param_range[0]) | (values_array > param_range[1]))
        }


    def _analyze_distribution(self, values: list[float]) -> dict[str, Any]:
        """Analyze the distribution of parameter values."""
        if len(values) < 3:
            return {"insufficient_data": True}

        values_array = np.array(values)
        values_array = values_array[~np.isnan(values_array)]

        if len(values_array) < 3:
            return {"insufficient_data": True}

        # Basic statistics
        distribution = {
            "skewness": stats.skew(values_array),
            "kurtosis": stats.kurtosis(values_array),
            "normality_pvalue": stats.normaltest(values_array)[1] if len(values_array) >= 8 else None,
            "uniformity_pvalue": stats.kstest(values_array, 'uniform')[1],
            "iqr": np.percentile(values_array, 75) - np.percentile(values_array, 25),
            "median": np.median(values_array),
            "q1": np.percentile(values_array, 25),
            "q3": np.percentile(values_array, 75)
        }

        # Detect clustering (razor-thin optima)
        range_val = np.max(values_array) - np.min(values_array)
        if range_val > 0:
            distribution["clustering_score"] = distribution["iqr"] / range_val
            distribution["is_razor_thin"] = distribution["clustering_score"] < 0.1

        return distribution

    def detect_overfitting(self, study: optuna.Study) -> dict[str, Any]:
        """
        Detect potential overfitting in the tuning results.

        Args:
            study: Optuna study to analyze

        Returns:
            Dictionary with overfitting analysis
        """
        trials = [t for t in study.trials if t.value is not None]
        if len(trials) < 10:
            return {"error": "Not enough trials for overfitting analysis"}

        # Sort by value
        trials_sorted = sorted(trials, key=lambda t: t.value, reverse=True)

        # Split into top 20% and bottom 80%
        split_idx = max(1, len(trials_sorted) // 5)
        top_trials = trials_sorted[:split_idx]
        bottom_trials = trials_sorted[split_idx:]

        # Calculate statistics
        top_values = [t.value for t in top_trials]
        bottom_values = [t.value for t in bottom_trials]

        overfitting_analysis = {
            "top_20_percent_mean": np.mean(top_values),
            "bottom_80_percent_mean": np.mean(bottom_values),
            "performance_gap": np.mean(top_values) - np.mean(bottom_values),
            "gap_ratio": (np.mean(top_values) - np.mean(bottom_values)) / abs(np.mean(bottom_values)) if np.mean(bottom_values) != 0 else None,
            "top_20_percent_std": np.std(top_values),
            "bottom_80_percent_std": np.std(bottom_values),
            "total_trials": len(trials),
            "top_trials_count": len(top_trials),
            "bottom_trials_count": len(bottom_trials)
        }

        # Flag potential overfitting
        if overfitting_analysis["gap_ratio"] and overfitting_analysis["gap_ratio"] > 2.0:
            overfitting_analysis["overfitting_flag"] = True
            overfitting_analysis["overfitting_severity"] = "high" if overfitting_analysis["gap_ratio"] > 5.0 else "moderate"
        else:
            overfitting_analysis["overfitting_flag"] = False
            overfitting_analysis["overfitting_severity"] = "low"

        return overfitting_analysis

    def run_test_evaluation(self, study: optuna.Study, model_type: str,
                          test_size: float = 0.2) -> dict[str, Any]:
        """
        Run best parameters on untouched test slice.

        Args:
            study: Optuna study with best parameters
            model_type: Type of model (ridge, mlp, cnn_lstm, ppo, sac, reward)
            test_size: Fraction of data to use for testing

        Returns:
            Dictionary with test performance results
        """
        # Get best parameters
        best_params = study.best_params

        # Split data into train and test
        split_idx = int(len(self.df) * (1 - test_size))
        train_data = self.df.iloc[:split_idx]
        test_data = self.df.iloc[split_idx:]


        if model_type in ["ridge", "mlp", "cnn_lstm"]:
            return self._test_sl_model(best_params, model_type, train_data, test_data)
        if model_type in ["ppo", "sac"]:
            return self._test_rl_agent(best_params, model_type, train_data, test_data)
        if model_type == "reward":
            return self._test_reward_params(best_params, train_data, test_data)
        return {"error": f"Unsupported model type: {model_type}"}

    def _test_sl_model(self, params: dict[str, Any], model_type: str,
                      train_data: pd.DataFrame, test_data: pd.DataFrame) -> dict[str, Any]:
        """Test SL model on untouched test data."""
        # Prepare data
        X_train = train_data[self.feature_columns].values
        y_train = train_data['mu_hat'].values
        X_test = test_data[self.feature_columns].values
        y_test = test_data['mu_hat'].values

        # Build model configuration
        config = self._build_sl_config(params, model_type)

        try:
            # Train model
            pipeline = SLTrainingPipeline(config)
            pipeline.train(X_train, y_train)

            # Make predictions
            y_pred = pipeline.model.predict(X_test)

            # Align predictions and targets if needed
            if len(y_pred) != len(y_test):
                y_test_aligned = y_test[len(y_test) - len(y_pred):]
                y_pred_aligned = y_pred
            else:
                y_test_aligned = y_test
                y_pred_aligned = y_pred

            # Calculate returns based on predictions
            positions = np.sign(y_pred_aligned)
            returns = positions[:-1] * np.diff(y_test_aligned)

            # Calculate metrics
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
                total_return = np.sum(returns)
                volatility = np.std(returns) * np.sqrt(252)
                max_drawdown = self._calculate_max_drawdown(np.cumsum(returns))

                return {
                    "test_sharpe_ratio": float(sharpe_ratio),
                    "test_total_return": float(total_return),
                    "test_volatility": float(volatility),
                    "test_max_drawdown": float(max_drawdown),
                    "test_samples": len(test_data),
                    "train_samples": len(train_data),
                    "best_params": params
                }
            return {
                "test_sharpe_ratio": 0.0,
                "test_total_return": 0.0,
                "test_volatility": 0.0,
                "test_max_drawdown": 0.0,
                "test_samples": len(test_data),
                "train_samples": len(train_data),
                "best_params": params,
                "warning": "Insufficient data for meaningful metrics"
            }

        except Exception as e:
            return {"error": str(e)}

    def _test_rl_agent(self, params: dict[str, Any], agent_type: str,
                      train_data: pd.DataFrame, test_data: pd.DataFrame) -> dict[str, Any]:
        """Test RL agent on untouched test data."""
        import json
        import tempfile

        try:
            from stable_baselines3 import PPO, SAC
        except ImportError:
            return {"error": "RL libraries not available"}

        # Save temporary files
        train_file = tempfile.NamedTemporaryFile(suffix='.parquet', delete=False)
        test_file = tempfile.NamedTemporaryFile(suffix='.parquet', delete=False)
        train_data.to_parquet(train_file.name)
        test_data.to_parquet(test_file.name)
        train_file.close()
        test_file.close()

        try:
            # Create training environment
            train_env = TradingEnvironment(
                data_file=train_file.name,
                initial_capital=100000.0,
                transaction_cost=0.001,
                seed=42,
                window_size=30
            )

            # Update config with best parameters
            if agent_type == "ppo":
                config_path = "configs/ppo_config.json"
                with open(config_path) as f:
                    config = json.load(f)

                for key, value in params.items():
                    if key in config["ppo"]:
                        config["ppo"][key] = value

            elif agent_type == "sac":
                config_path = "configs/sac_config.json"
                with open(config_path) as f:
                    config = json.load(f)

                for key, value in params.items():
                    if key in config["sac"]:
                        config["sac"][key] = value

            # Train agent
            if agent_type == "ppo":
                model = PPO("MlpPolicy", train_env, verbose=0, seed=42)
            else:  # sac
                model = SAC("MlpPolicy", train_env, verbose=0, seed=42)

            model.learn(total_timesteps=100000)

            # Evaluate on test data
            test_env = TradingEnvironment(
                data_file=test_file.name,
                initial_capital=100000.0,
                transaction_cost=0.001,
                seed=43,
                window_size=30
            )

            # Run evaluation
            obs, _ = test_env.reset()
            signals = []
            prices = []

            while True:
                action, _ = model.predict(obs, deterministic=True)
                prices.append(test_env.prices[test_env.current_step])
                signal = np.clip(action[0], -1, 1)
                signals.append(signal)

                obs, reward, terminated, truncated, info = test_env.step(action)
                if terminated or truncated:
                    break

            # Run backtest
            signals_series = pd.Series(signals)
            prices_series = pd.Series(prices)

            backtest_engine = BacktestEngine(
                transaction_cost=0.001,
                slippage=0.0005,
                initial_capital=100000.0
            )

            results = backtest_engine.run_backtest(signals_series, prices_series)

            return {
                "test_sharpe_ratio": float(results['metrics'].get('sharpe_ratio', 0.0)),
                "test_total_return": float(results['metrics'].get('total_return', 0.0)),
                "test_volatility": float(results['metrics'].get('volatility', 0.0)),
                "test_max_drawdown": float(results['metrics'].get('max_drawdown', 0.0)),
                "test_samples": len(test_data),
                "train_samples": len(train_data),
                "best_params": params
            }

        except Exception as e:
            return {"error": str(e)}
        finally:
            # Clean up
            try:
                os.unlink(train_file.name)
                os.unlink(test_file.name)
            except FileNotFoundError:
                pass

    def _test_reward_params(self, params: dict[str, Any],
                           train_data: pd.DataFrame, test_data: pd.DataFrame) -> dict[str, Any]:
        """Test reward parameters on untouched test data."""
        import tempfile

        try:
            from stable_baselines3 import PPO
        except ImportError:
            return {"error": "RL libraries not available"}

        # Save temporary files
        train_file = tempfile.NamedTemporaryFile(suffix='.parquet', delete=False)
        test_file = tempfile.NamedTemporaryFile(suffix='.parquet', delete=False)
        train_data.to_parquet(train_file.name)
        test_data.to_parquet(test_file.name)
        train_file.close()
        test_file.close()

        try:
            # Create training environment with reward config
            train_env = TradingEnvironment(
                data_file=train_file.name,
                initial_capital=100000.0,
                transaction_cost=0.001,
                seed=42,
                window_size=30,
                reward_config=params
            )

            # Train PPO agent
            model = PPO("MlpPolicy", train_env, verbose=0, seed=42)
            model.learn(total_timesteps=100000)

            # Evaluate on test data
            test_env = TradingEnvironment(
                data_file=test_file.name,
                initial_capital=100000.0,
                transaction_cost=0.001,
                seed=43,
                window_size=30,
                reward_config=params
            )

            # Run evaluation
            obs, _ = test_env.reset()
            signals = []
            prices = []

            while True:
                action, _ = model.predict(obs, deterministic=True)
                prices.append(test_env.prices[test_env.current_step])
                signal = np.clip(action[0], -1, 1)
                signals.append(signal)

                obs, reward, terminated, truncated, info = test_env.step(action)
                if terminated or truncated:
                    break

            # Run backtest
            signals_series = pd.Series(signals)
            prices_series = pd.Series(prices)

            backtest_engine = BacktestEngine(
                transaction_cost=0.001,
                slippage=0.0005,
                initial_capital=100000.0
            )

            results = backtest_engine.run_backtest(signals_series, prices_series)

            return {
                "test_sharpe_ratio": float(results['metrics'].get('sharpe_ratio', 0.0)),
                "test_total_return": float(results['metrics'].get('total_return', 0.0)),
                "test_volatility": float(results['metrics'].get('volatility', 0.0)),
                "test_max_drawdown": float(results['metrics'].get('max_drawdown', 0.0)),
                "test_samples": len(test_data),
                "train_samples": len(train_data),
                "best_params": params
            }

        except Exception as e:
            return {"error": str(e)}
        finally:
            # Clean up
            try:
                os.unlink(train_file.name)
                os.unlink(test_file.name)
            except FileNotFoundError:
                pass

    def _build_sl_config(self, params: dict[str, Any], model_type: str) -> dict[str, Any]:
        """Build SL model configuration from parameters."""
        base_config = {
            "model_type": model_type,
            "model_config": {},
            "cv_config": {"n_splits": 3, "gap": 10},
            "tuning_config": {"enable_tuning": False},
            "random_state": 42,
            "output_dir": "models/",
            "save_model": False
        }

        # Map parameters to model config
        for key, value in params.items():
            base_config["model_config"][key] = value

        return base_config

    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown from cumulative returns."""
        if len(cumulative_returns) == 0:
            return 0.0

        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / (peak + 1e-8)
        return float(np.min(drawdown))

    def generate_report(self, study_path: str, model_type: str,
                       stability_analysis: dict[str, Any],
                       overfitting_analysis: dict[str, Any],
                       test_results: dict[str, Any]) -> str:
        """
        Generate comprehensive validation report.

        Args:
            study_path: Path to the study file
            model_type: Type of model
            stability_analysis: Parameter stability analysis
            overfitting_analysis: Overfitting analysis
            test_results: Test performance results

        Returns:
            Path to generated report file
        """
        report_path = f"reports/validation_report_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Combine all analyses
        full_report = {
            "study_path": study_path,
            "model_type": model_type,
            "timestamp": datetime.now().isoformat(),
            "stability_analysis": stability_analysis,
            "overfitting_analysis": overfitting_analysis,
            "test_results": test_results,
            "validation_vs_test_comparison": self._compare_validation_test(stability_analysis, test_results)
        }

        # Save report
        with open(report_path, 'w') as f:
            json.dump(full_report, f, indent=2, default=str)

        # Also create human-readable summary
        summary_path = report_path.replace('.json', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(self._create_human_readable_summary(full_report))

        return report_path

    def _compare_validation_test(self, stability_analysis: dict[str, Any],
                               test_results: dict[str, Any]) -> dict[str, Any]:
        """Compare validation and test performance."""
        if "error" in stability_analysis or "error" in test_results:
            return {"error": "Cannot compare due to errors in analysis"}

        val_best = stability_analysis.get("best_value", 0)
        test_sharpe = test_results.get("test_sharpe_ratio", 0)

        if val_best != 0:
            performance_drop = (val_best - test_sharpe) / abs(val_best)
            return {
                "validation_best": val_best,
                "test_sharpe": test_sharpe,
                "performance_drop": performance_drop,
                "overfitting_flag": performance_drop > 0.3
            }
        return {"error": "Cannot compare - no validation best value"}
