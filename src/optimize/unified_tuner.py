"""
Unified hyperparameter tuner for both SL and RL models using Optuna.

This module consolidates the fragmented hyperparameter optimization
implementations across the codebase into a single, coherent framework.
"""

import json
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Never

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


try:
    from trade_agent.agents.sl.models.base import set_all_seeds
    from trade_agent.agents.sl.models.factory import SLModelFactory
    from trade_agent.data.splits import purged_walk_forward_splits
    from trade_agent.envs.trading_env import TradingEnvironment
except ImportError as e:
    warnings.warn(f"Import error: {e}. Some functionality may be limited.", stacklevel=2)


class CrossValidationStrategy:
    """Base class for cross-validation strategies."""

    def split(self, X: np.ndarray, y: np.ndarray | None = None) -> Never:
        """Generate train/validation splits."""
        raise NotImplementedError


class PurgedTimeSeriesCV(CrossValidationStrategy):
    """Advanced temporal cross-validation with purging and embargo."""

    def __init__(self,
                 n_splits: int = 5,
                 embargo_days: int = 1,
                 purge_days: int = 0) -> None:
        self.n_splits = n_splits
        self.embargo_days = embargo_days
        self.purge_days = purge_days

    def split(self, X: np.ndarray, y: np.ndarray | None = None):
        """Generate purged walk-forward splits."""
        if isinstance(X, np.ndarray):
            # Convert to DataFrame for purged_walk_forward_splits
            df = pd.DataFrame(X)
            if y is not None:
                df['target'] = y
        else:
            df = X

        try:
            splits_data = list(purged_walk_forward_splits(
                df,
                n_splits=self.n_splits,
                train_ratio=0.6,
                val_ratio=0.2,
                test_ratio=0.2,
                min_gap=self.embargo_days,
                max_gap=self.purge_days
            ))

            # Convert to indices format
            for train_df, val_df, _ in splits_data:
                train_indices = train_df.index.tolist()
                val_indices = val_df.index.tolist()
                yield train_indices, val_indices

        except Exception:
            # Fallback to simple temporal splits
            n_samples = len(X)
            fold_size = n_samples // (self.n_splits + 1)

            for i in range(self.n_splits):
                train_end = (i + 1) * fold_size
                val_start = train_end + self.embargo_days
                val_end = val_start + fold_size

                if val_end <= n_samples:
                    train_indices = list(range(0, train_end))
                    val_indices = list(range(val_start, val_end))
                    yield train_indices, val_indices


class UnifiedHyperparameterTuner:
    """Unified Optuna-based hyperparameter tuner for both SL and RL models."""

    def __init__(
        self,
        study_name: str,
        cv_strategy: CrossValidationStrategy | None = None,
        storage: str | None = None,
        n_trials: int = 100,
        random_state: int = 42,
        resolved_config: Any | None = None,
        artifact_root_runs: str = "runs",
        artifact_root_models: str = "models",
    ) -> None:
        """
        Initialize the unified hyperparameter tuner.

        Args:
            study_name: Name for the Optuna study
            cv_strategy: Cross-validation strategy to use
            storage: Optuna storage backend (None for in-memory)
            n_trials: Number of optimization trials
            random_state: Random seed for reproducibility
        """
        self.study_name = study_name
        self.cv_strategy = cv_strategy or PurgedTimeSeriesCV()
        self.n_trials = n_trials
        self.random_state = random_state
        self.resolved_config = resolved_config

        # Artifact roots & timestamp namespace
        self.artifact_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.artifact_root_runs = Path(artifact_root_runs) / self.artifact_timestamp / study_name
        self.artifact_root_models = Path(artifact_root_models) / self.artifact_timestamp / study_name
        for p in [self.artifact_root_runs, self.artifact_root_models]:
            p.mkdir(parents=True, exist_ok=True)

        # Create Optuna study
        sampler = TPESampler(seed=random_state)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

        self.study = optuna.create_study(
            study_name=study_name,
            direction='maximize',  # Maximize Sharpe ratio
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=True
        )

        set_all_seeds(random_state)

    def optimize_sl_model(self,
                          model_type: str,
                          X: np.ndarray,
                          y: np.ndarray,
                          metric: str = 'sharpe') -> dict[str, Any]:
        """
        Optimize supervised learning model hyperparameters.

        Args:
            model_type: Type of SL model ('ridge', 'mlp', 'cnn_lstm', etc.)
            X: Feature matrix
            y: Target vector
            metric: Optimization metric ('sharpe', 'mse', 'mae')

        Returns:
            Dictionary with best parameters and score
        """
        def objective(trial):
            return self._sl_objective(trial, model_type, X, y, metric)

        self.study.optimize(objective, n_trials=self.n_trials)

        return {
            'best_params': self.study.best_params,
            'best_score': self.study.best_value,
            'n_trials': len(self.study.trials)
        }

    def optimize_rl_agent(self,
                         agent_type: str,
                         env_config: dict[str, Any],
                         metric: str = 'sharpe') -> dict[str, Any]:
        """
        Optimize reinforcement learning agent hyperparameters.

        Args:
            agent_type: Type of RL agent ('ppo', 'sac')
            env_config: Environment configuration
            metric: Optimization metric

        Returns:
            Dictionary with best parameters and score
        """
        def objective(trial):
            return self._rl_objective(trial, agent_type, env_config, metric)

        self.study.optimize(objective, n_trials=self.n_trials)

        return {
            'best_params': self.study.best_params,
            'best_score': self.study.best_value,
            'n_trials': len(self.study.trials)
        }

    def _sl_objective(
        self,
        trial: "optuna.trial.Trial",
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        metric: str,
    ) -> float:
        """Supervised model objective with temporal CV & pruning.

        All metrics cast to a maximization target; loss metrics negated.
        """
        # Hyperparameter spaces
        if model_type == 'ridge':
            config = {
                'alpha': trial.suggest_float('alpha', 1e-6, 1e2, log=True),
                'solver': trial.suggest_categorical(
                    'solver', ['auto', 'svd', 'cholesky', 'lsqr']
                ),
                'max_iter': trial.suggest_int('max_iter', 1000, 10000),
                'random_state': self.random_state,
            }
        elif model_type == 'mlp':
            config = {
                'hidden_layer_sizes': trial.suggest_categorical(
                    'hidden_layer_sizes',
                    [(50,), (100,), (50, 50), (100, 50), (100, 100)],
                ),
                'learning_rate': trial.suggest_float(
                    'learning_rate', 1e-5, 1e-1, log=True
                ),
                'alpha': trial.suggest_float('alpha', 1e-6, 1e-2, log=True),
                'batch_size': trial.suggest_categorical(
                    'batch_size', [32, 64, 128, 256]
                ),
                'max_iter': trial.suggest_int('max_iter', 100, 1000),
                'random_state': self.random_state,
            }
        elif model_type == 'cnn_lstm':
            config = {
                'cnn_channels': trial.suggest_categorical(
                    'cnn_channels',
                    [(16,), (32,), (64,), (16, 32), (32, 64)],
                ),
                'cnn_kernel_sizes': trial.suggest_categorical(
                    'cnn_kernel_sizes', [[3], [5], [3, 5]]
                ),
                'lstm_hidden_size': trial.suggest_int(
                    'lstm_hidden_size', 16, 256
                ),
                'lstm_num_layers': trial.suggest_int('lstm_num_layers', 1, 3),
                'dropout': trial.suggest_float('dropout', 0.0, 0.5),
                'learning_rate': trial.suggest_float(
                    'learning_rate', 1e-5, 1e-2, log=True
                ),
                'sequence_length': trial.suggest_int('sequence_length', 10, 60),
                'random_state': self.random_state,
            }
        else:  # pragma: no cover
            raise ValueError(f"Unsupported model type: {model_type}")

        # Temporal CV loop with pruning
        fold_scores: list[float] = []
        for fold_idx, (train_idx, val_idx) in enumerate(self.cv_strategy.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            try:
                model = SLModelFactory.create_model(model_type, config)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                if metric == 'sharpe':
                    signals = np.sign(y_pred)
                    returns = signals * y_val
                    fold_score = self._calculate_sharpe_ratio(returns)
                elif metric in {'mse', 'cv_mse'}:
                    fold_score = -float(np.mean((y_val - y_pred) ** 2))
                elif metric == 'mae':
                    fold_score = -float(np.mean(np.abs(y_val - y_pred)))
                else:
                    raise ValueError(f"Unsupported metric: {metric}")
            except Exception as e:  # pragma: no cover
                warnings.warn(f"Fold failed: {e}", stacklevel=2)
                fold_score = -np.inf
            fold_scores.append(fold_score)
            interim = float(np.mean(fold_scores))
            trial.report(interim, step=fold_idx)
            if trial.should_prune():  # pragma: no cover
                raise optuna.TrialPruned()

        final_score = float(np.mean(fold_scores)) if fold_scores else -np.inf

        # Persist artifacts
        try:  # pragma: no cover - IO
            trial_dir = self.artifact_root_runs / f"trial_{trial.number:04d}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            with (trial_dir / "params.json").open('w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            with (trial_dir / "scores.json").open('w', encoding='utf-8') as f:
                json.dump(
                    {
                        'fold_scores': fold_scores,
                        'final_score': final_score,
                        'metric': metric,
                    },
                    f,
                    indent=2,
                )
            if self.resolved_config is not None:
                try:
                    from omegaconf import OmegaConf  # type: ignore
                    with (trial_dir / 'hydra_config.yaml').open(
                        'w', encoding='utf-8'
                    ) as f:
                        try:
                            f.write(OmegaConf.to_yaml(self.resolved_config))  # type: ignore
                        except Exception:
                            json.dump(self.resolved_config, f, indent=2, default=str)
                except Exception:
                    pass
        except Exception:
            warnings.warn('Failed to write trial artifacts', stacklevel=2)

        return final_score

    def _rl_objective(
        self,
        trial,
        agent_type: str,
        env_config: dict[str, Any],
        metric: str,
    ) -> float:
        """Objective function for RL agent optimization."""
        try:  # Import lazily to keep core deps minimal
            import stable_baselines3  # type: ignore  # noqa: F401
        except ImportError:
            warnings.warn("stable_baselines3 not available", stacklevel=2)
            return -np.inf

        if agent_type == 'ppo':  # Hyperparameter search space
            params = {
                'learning_rate': trial.suggest_float(
                    'learning_rate', 1e-5, 1e-2, log=True
                ),
                'n_steps': trial.suggest_categorical(
                    'n_steps', [512, 1024, 2048]
                ),
                'batch_size': trial.suggest_categorical(
                    'batch_size', [32, 64, 128]
                ),
                'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
                'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
                'ent_coef': trial.suggest_float('ent_coef', 0.0, 0.1),
            }
        elif agent_type == 'sac':
            params = {
                'learning_rate': trial.suggest_float(
                    'learning_rate', 1e-5, 1e-2, log=True
                ),
                'batch_size': trial.suggest_categorical(
                    'batch_size', [64, 128, 256]
                ),
                'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
                'tau': trial.suggest_float('tau', 0.001, 0.02, log=True),
                'ent_coef': trial.suggest_categorical(
                    'ent_coef', ['auto', 0.1, 0.01]
                ),
            }
        else:  # pragma: no cover - guarded path
            raise ValueError(f"Unsupported agent type: {agent_type}")

        # Placeholder feature matrix for temporal splitting until env integration
        X_placeholder = np.random.randn(1000, 10)
        scores: list[float] = []
        for fold_idx, (train_idx, val_idx) in enumerate(
            self.cv_strategy.split(X_placeholder)
        ):
            try:
                score = self._train_and_evaluate_rl_agent(
                    agent_type, params, env_config, train_idx, val_idx
                )
                scores.append(score)
                trial.report(float(np.mean(scores)), step=fold_idx)
                if trial.should_prune():  # pragma: no cover
                    raise optuna.TrialPruned()
            except Exception as e:  # pragma: no cover
                warnings.warn(f"RL trial failed: {e}", stacklevel=2)
                scores.append(-np.inf)
        return float(np.mean(scores)) if scores else -np.inf

    def _train_and_evaluate_rl_agent(
        self,
        agent_type: str,
        params: dict[str, Any],
        env_config: dict[str, Any],
        train_idx: list[int],
        val_idx: list[int],
    ) -> float:
        """Train and evaluate RL agent on a CV fold."""
        try:
            from stable_baselines3 import PPO, SAC

            # Create temporary data files for training and validation
            # This is a simplified implementation - should be enhanced
            with tempfile.NamedTemporaryFile(
                suffix='.parquet', delete=False
            ) as train_file:
                train_file_name = train_file.name
            with tempfile.NamedTemporaryFile(
                suffix='.parquet', delete=False
            ) as val_file:
                val_file_name = val_file.name

            # Create training environment
            train_env = TradingEnvironment(
                data_file=train_file_name,
                initial_capital=100000.0,
                transaction_cost=0.001,
                seed=self.random_state
            )

            # Create and train model
            if agent_type == 'ppo':
                model = PPO(
                    "MlpPolicy",
                    train_env,
                    **params,
                    seed=self.random_state,
                    verbose=0,
                )
            elif agent_type == 'sac':
                model = SAC(
                    "MlpPolicy",
                    train_env,
                    **params,
                    seed=self.random_state,
                    verbose=0,
                )

            # Train for limited timesteps (for efficiency in hyperopt)
            model.learn(total_timesteps=10000)

            # Evaluate on validation set
            val_env = TradingEnvironment(
                data_file=val_file_name,
                initial_capital=100000.0,
                transaction_cost=0.001,
                seed=self.random_state + 1
            )

            # Run evaluation episode
            obs, _ = val_env.reset()
            returns = []

            for _ in range(min(len(val_idx), 1000)):  # Limit evaluation length
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = val_env.step(action)
                returns.append(reward)

                if terminated or truncated:
                    break

            # Calculate Sharpe ratio
            return self._calculate_sharpe_ratio(returns)

        except Exception as e:
            warnings.warn(f"RL evaluation failed: {e}", stacklevel=2)
            return -np.inf
        finally:
            # Cleanup temp files
            try:
                Path(train_file_name).unlink(missing_ok=True)
                Path(val_file_name).unlink(missing_ok=True)
            except Exception:
                pass

    def _calculate_sharpe_ratio(
        self, returns: list[float], risk_free_rate: float = 0.0
    ) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) <= 1:
            return 0.0

        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate

        if np.std(excess_returns) == 0:
            return 0.0

        # Annualize assuming daily returns
        sharpe = (
            np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        )
        return float(sharpe)

    def get_study_summary(self) -> dict[str, Any]:
        """Get summary of optimization study."""
        return {
            'study_name': self.study_name,
            'n_trials': len(self.study.trials),
            'best_value': self.study.best_value,
            'best_params': self.study.best_params,
            'trials_df': self.study.trials_dataframe()
        }
