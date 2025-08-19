"""
Training orchestrator for end-to-end pipeline execution.
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .registry import ExperimentRegistry


# Add src to path if not already there
if os.path.join(os.path.dirname(__file__), '..', '..') not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.optimize.unified_tuner import PurgedTimeSeriesCV, UnifiedHyperparameterTuner

from ..sl.models.base import set_all_seeds


class TrainingOrchestrator:
    """Orchestrates end-to-end training pipeline."""

    def __init__(self, experiment_config: ExperimentConfig) -> None:
        """
        Initialize training orchestrator.

        Args:
            experiment_config: Experiment configuration
        """
        self.config = experiment_config
        self.registry = ExperimentRegistry()
        self.experiment_id = None
        self.results = {}

        # Set random seed
        set_all_seeds(self.config.random_state)

        # Validate configuration
        self.config.validate()

    def run_full_pipeline(self) -> dict[str, Any]:
        """
        Run the complete training pipeline.

        Returns:
            Dictionary with experiment results
        """
        # Register experiment
        self.experiment_id = self.registry.register_experiment(self.config)

        try:
            # Load and prepare data
            X, y = self._load_and_prepare_data()

            # Run hyperparameter optimization if enabled
            if self.config.optimization_config.enabled:
                optimization_results = self.run_hyperparameter_optimization(X, y)
                self.results['optimization'] = optimization_results
            else:
                optimization_results = {}

            # Train models with best hyperparameters
            training_results = self.train_best_models(X, y, optimization_results)
            self.results['training'] = training_results

            # Evaluate ensemble if configured
            if (self.config.ensemble_config and
                    self.config.ensemble_config.enabled):
                ensemble_results = self.evaluate_ensemble(X, y, training_results)
                self.results['ensemble'] = ensemble_results

            # Log final results
            self._log_final_results()

            return self.results

        except Exception:
            raise

    def _load_and_prepare_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Load and prepare data for training."""
        data_path = self.config.data_config.data_path
        target_column = self.config.data_config.target_column

        # Load data
        if str(data_path).endswith('.parquet'):
            df = pd.read_parquet(data_path)
        elif str(data_path).endswith('.csv'):
            df = pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported data format: {data_path}")

        # Prepare features and targets
        y = df[target_column].values

        # Get numeric features excluding targets
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        if target_column in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=[target_column])

        # Remove other target columns
        for col in ['mu_hat', 'sigma_hat']:
            if col in numeric_df.columns and col != target_column:
                numeric_df = numeric_df.drop(columns=[col])

        X = numeric_df.values

        return X, y

    def run_hyperparameter_optimization(self, X: np.ndarray,
                                       y: np.ndarray) -> dict[str, Any]:
        """
        Run hyperparameter optimization across all models.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Dictionary with optimization results for each model
        """
        optimization_results = {}

        # Create cross-validation strategy
        if self.config.cv_config.strategy == "purged_time_series":
            cv_strategy = PurgedTimeSeriesCV(
                n_splits=self.config.cv_config.n_splits,
                embargo_days=self.config.cv_config.embargo_days,
                purge_days=self.config.cv_config.purge_days
            )
        else:
            # Fallback to simple temporal CV
            cv_strategy = PurgedTimeSeriesCV(
                n_splits=self.config.cv_config.n_splits
            )

        # Optimize each model
        for model_config in self.config.model_configs:
            model_type = model_config.model_type

            try:
                # Create tuner for this model
                tuner = UnifiedHyperparameterTuner(
                    study_name=f"{self.config.experiment_name}_{model_type}",
                    cv_strategy=cv_strategy,
                    n_trials=self.config.optimization_config.n_trials,
                    random_state=self.config.random_state
                )

                # Run optimization
                if model_type in ['ppo', 'sac']:
                    # RL optimization
                    env_config = {'data_path': self.config.data_config.data_path}
                    result = tuner.optimize_rl_agent(
                        model_type, env_config,
                        metric=self.config.optimization_config.metric
                    )
                else:
                    # SL optimization
                    result = tuner.optimize_sl_model(
                        model_type, X, y,
                        metric=self.config.optimization_config.metric
                    )

                optimization_results[model_type] = result

                # Log optimization results
                self.registry.log_results(
                    self.experiment_id,
                    f"{model_type}_optimization",
                    {
                        'best_score': result['best_score'],
                        'n_trials': result['n_trials'],
                        'optimization_metric': self.config.optimization_config.metric
                    },
                    result['best_params']
                )


            except Exception as e:
                warnings.warn(f"Optimization failed for {model_type}: {e}", stacklevel=2)
                optimization_results[model_type] = {
                    'best_params': {},
                    'best_score': 0.0,
                    'error': str(e)
                }

        return optimization_results

    def train_best_models(self, X: np.ndarray, y: np.ndarray,
                         optimization_results: dict[str, Any]) -> dict[str, Any]:
        """
        Train models with best hyperparameters.

        Args:
            X: Feature matrix
            y: Target vector
            optimization_results: Results from hyperparameter optimization

        Returns:
            Dictionary with training results for each model
        """
        training_results = {}

        # Split data for training
        val_cut = int(len(X) * (self.config.data_config.train_ratio +
                               self.config.data_config.val_ratio))
        train_cut = int(len(X) * self.config.data_config.train_ratio)

        X_train, X_val = X[:train_cut], X[train_cut:val_cut]
        y_train, y_val = y[:train_cut], y[train_cut:val_cut]

        # Train each model
        for model_config in self.config.model_configs:
            model_type = model_config.model_type

            try:
                # Get best parameters from optimization
                best_params = optimization_results.get(
                    model_type, {}).get('best_params', {})

                # Merge with base model config
                final_params = model_config.model_params.copy()
                final_params.update(best_params)
                final_params['random_state'] = self.config.random_state

                # Train model based on type
                if model_type in ['ppo', 'sac']:
                    # Train RL model
                    result = self._train_rl_model(
                        model_type, final_params, X_train, y_train, X_val, y_val
                    )
                else:
                    # Train SL model
                    result = self._train_sl_model(
                        model_type, final_params, X_train, y_train, X_val, y_val
                    )

                training_results[model_type] = result

                # Log training results
                self.registry.log_results(
                    self.experiment_id,
                    f"{model_type}_training",
                    result['metrics'],
                    final_params
                )


            except Exception as e:
                warnings.warn(f"Training failed for {model_type}: {e}", stacklevel=2)
                training_results[model_type] = {
                    'metrics': {'error': str(e)},
                    'model_path': None
                }

        return training_results

    def _train_sl_model(self, model_type: str, params: dict[str, Any],
                       X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray) -> dict[str, Any]:
        """Train supervised learning model."""
        try:
            from sklearn.metrics import mean_squared_error, r2_score

            from ..sl.models.factory import SLModelFactory

            # Create and train model
            model = SLModelFactory.create_model(model_type, params)
            model.fit(X_train, y_train)

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

            # Calculate metrics
            train_mse = mean_squared_error(y_train, y_train_pred)
            val_mse = mean_squared_error(y_val, y_val_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            val_r2 = r2_score(y_val, y_val_pred)

            # Calculate Sharpe ratio (simplified)
            train_signals = np.sign(y_train_pred)
            val_signals = np.sign(y_val_pred)
            train_returns = train_signals * y_train
            val_returns = val_signals * y_val

            train_sharpe = self._calculate_sharpe_ratio(train_returns)
            val_sharpe = self._calculate_sharpe_ratio(val_returns)

            # Save model if configured
            model_path = None
            if self.config.save_models:
                model_path = self._save_model(model, model_type)

            return {
                'metrics': {
                    'train_mse': train_mse,
                    'val_mse': val_mse,
                    'train_r2': train_r2,
                    'val_r2': val_r2,
                    'train_sharpe': train_sharpe,
                    'val_sharpe': val_sharpe,
                    'val_score': val_sharpe  # Primary metric
                },
                'model_path': model_path,
                'model': model
            }

        except Exception as e:
            raise RuntimeError(f"SL model training failed: {e}")

    def _train_rl_model(self, agent_type: str, params: dict[str, Any],
                       X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray) -> dict[str, Any]:
        """Train reinforcement learning model (PPO or SAC)."""
        try:
            from stable_baselines3 import PPO, SAC
            # Choose environment implementation
            if self.config.use_enhanced_env:
                from integrations.enhanced_trading_env import (
                    EnhancedTradingEnvironment as TradingEnvironment,
                )
            else:
                from trade_agent.agents.envs.trading_env import TradingEnvironment
            import tempfile

            from trade_agent.agents.sl.models.base import set_all_seeds

            # Set random seeds for reproducibility
            set_all_seeds(self.config.random_state)

            # Combine training and validation data for RL environment
            X_combined = np.vstack([X_train, X_val])
            y_combined = np.concatenate([y_train, y_val])

            # Create trading environment
            if self.config.use_enhanced_env:
                # Persist combined data temporarily for enhanced env ingestion
                temp_path = Path("data/temp_rl_env.parquet")
                temp_df = pd.DataFrame(X_combined).copy()
                temp_df['mu_hat'] = y_combined  # treat target as mu_hat
                temp_df['sigma_hat'] = (
                    np.abs(temp_df['mu_hat']).rolling(5).std().fillna(0.02)
                )
                temp_df.to_parquet(temp_path)
                env = TradingEnvironment(
                    data_file=str(temp_path),
                    window_size=30,
                    reward_config={
                        'position_limit': (
                            self.config.reward_config.position_limit
                        ),
                        'transaction_cost_weight': (
                            self.config.reward_config.transaction_cost_weight
                        )
                    },
                    auto_convert=False
                )
            else:
                env = TradingEnvironment(
                    data=pd.DataFrame(X_combined),
                    target=y_combined,
                    initial_balance=10000,
                    transaction_cost=0.001,
                    random_state=self.config.random_state
                )

            # Initialize RL agent based on type
            if agent_type == 'ppo':
                model = PPO(
                    "MlpPolicy", env, **params,
                    seed=self.config.random_state, verbose=0
                )
            elif agent_type == 'sac':
                model = SAC(
                    "MlpPolicy", env, **params,
                    seed=self.config.random_state, verbose=0
                )
            else:
                raise ValueError(f"Unsupported RL agent type: {agent_type}")

            # Train the model
            total_timesteps = params.get('total_timesteps', 10000)
            model.learn(total_timesteps=total_timesteps)

            # Evaluate on validation environment
            if self.config.use_enhanced_env:
                val_temp_path = Path("data/temp_rl_env_val.parquet")
                val_df = pd.DataFrame(X_val).copy()
                val_df['mu_hat'] = y_val
                val_df['sigma_hat'] = (
                    np.abs(val_df['mu_hat']).rolling(5).std().fillna(0.02)
                )
                val_df.to_parquet(val_temp_path)
                val_env = TradingEnvironment(
                    data_file=str(val_temp_path),
                    window_size=30,
                    reward_config={
                        'position_limit': (
                            self.config.reward_config.position_limit
                        ),
                        'transaction_cost_weight': (
                            self.config.reward_config.transaction_cost_weight
                        )
                    },
                    auto_convert=False
                )
            else:
                val_env = TradingEnvironment(
                    data=pd.DataFrame(X_val),
                    target=y_val,
                    initial_balance=10000,
                    transaction_cost=0.001,
                    random_state=self.config.random_state
                )

            obs = val_env.reset()
            total_reward = 0
            done = False
            steps = 0

            while not done and steps < len(X_val):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = val_env.step(action)
                total_reward += reward
                steps += 1

            # Calculate validation metrics
            val_score = total_reward
            sharpe_ratio = val_score / max(np.std([val_score]), 1e-8)

            # Save model
            model_path = None
            if self.config.save_models:
                with tempfile.NamedTemporaryFile(
                    suffix=f'_{agent_type}.zip', delete=False
                ) as f:
                    model_path = f.name
                    model.save(model_path)

            return {
                'metrics': {
                    'val_score': val_score,
                    'sharpe_ratio': sharpe_ratio,
                    'total_timesteps': total_timesteps,
                    'final_balance': val_env.balance,
                    'total_trades': getattr(val_env, 'total_trades', 0)
                },
                'model_path': model_path,
                'model': model
            }

        except ImportError as e:
            warnings.warn(f"RL dependencies not available: {e}", stacklevel=2)
            return {
                'metrics': {
                    'val_score': 0.0,
                    'error': f'Missing RL dependencies: {e}'
                },
                'model_path': None
            }
        except Exception as e:
            warnings.warn(f"RL training failed: {e}", stacklevel=2)
            return {
                'metrics': {
                    'val_score': 0.0,
                    'error': f'RL training failed: {e}'
                },
                'model_path': None
            }

    def evaluate_ensemble(self, X: np.ndarray, y: np.ndarray,
                         training_results: dict[str, Any]) -> dict[str, Any]:
        """
        Evaluate ensemble performance.

        Args:
            X: Feature matrix
            y: Target vector
            training_results: Individual model training results

        Returns:
            Ensemble evaluation results
        """
        try:
            import numpy as np
            from sklearn.metrics import mean_squared_error

            # Extract trained models and their scores
            models = []
            model_names = []
            scores = []

            for model_name, result in training_results.items():
                if isinstance(result, dict) and 'model' in result:
                    model = result['model']
                    metrics = result.get('metrics', {})
                    score = metrics.get('val_score', 0)

                    # Only include models with valid scores and sklearn-compatible models
                    if score != 0 and hasattr(model, 'predict'):
                        models.append((model_name, model))
                        model_names.append(model_name)
                        scores.append(abs(score))  # Use absolute value for scoring

            if len(models) < 2:
                return {
                    'metrics': {
                        'ensemble_score': 0.0,
                        'improvement_over_best': 0.0,
                        'error': 'Insufficient models for ensemble'
                    },
                    'method': (self.config.ensemble_config.method
                              if self.config.ensemble_config else None)
                }

            # Create ensemble based on method
            method = (self.config.ensemble_config.method
                     if self.config.ensemble_config else 'weighted_average')

            # Make predictions with each model
            predictions = []
            for model_name, model in models:
                try:
                    pred = model.predict(X)
                    predictions.append(pred)
                except Exception as e:
                    warnings.warn(f"Model {model_name} prediction failed: {e}", stacklevel=2)
                    continue

            if not predictions:
                return {
                    'metrics': {
                        'ensemble_score': 0.0,
                        'improvement_over_best': 0.0,
                        'error': 'No valid predictions from models'
                    },
                    'method': method
                }

            # Calculate ensemble prediction
            if method == 'weighted_average':
                # Weight by performance scores
                weights = np.array(scores[:len(predictions)]) / np.sum(scores[:len(predictions)])
                ensemble_pred = np.average(predictions, axis=0, weights=weights)

            elif method == 'gating':
                # Use best performing model
                best_idx = np.argmax(scores[:len(predictions)])
                ensemble_pred = predictions[best_idx]

            elif method == 'stacking':
                # Simple averaging (true stacking would need meta-learner)
                ensemble_pred = np.mean(predictions, axis=0)

            else:
                # Default to simple averaging
                ensemble_pred = np.mean(predictions, axis=0)

            # Calculate ensemble performance
            ensemble_mse = mean_squared_error(y, ensemble_pred)
            ensemble_score = -ensemble_mse  # Higher is better for consistency

            # Calculate improvement over best individual model
            best_individual_score = max(scores[:len(predictions)]) if scores else 0
            improvement = (abs(ensemble_score) - best_individual_score) / max(best_individual_score, 1e-8)

            return {
                'metrics': {
                    'ensemble_score': ensemble_score,
                    'improvement_over_best': improvement
                },
                'method': method
            }

        except Exception as e:
            warnings.warn(f"Ensemble evaluation failed: {e}", stacklevel=2)
            return {
                'metrics': {
                    'ensemble_score': 0.0,
                    'improvement_over_best': 0.0,
                    'error': f'Ensemble evaluation failed: {e}'
                },
                'method': (self.config.ensemble_config.method
                          if self.config.ensemble_config else None)
            }

    def _save_model(self, model, model_type: str) -> str:
        """Save trained model and return path."""
        output_dir = Path(self.config.output_dir) / self.experiment_id
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / f"{model_type}_model.pkl"

        # Save model (implementation depends on model type)
        if hasattr(model, 'save_model'):
            model.save_model(str(model_path))
        else:
            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

        # Log artifact
        self.registry.log_artifact(
            self.experiment_id, model_type, str(model_path), "model"
        )

        return str(model_path)

    def _calculate_sharpe_ratio(self, returns: np.ndarray,
                               risk_free_rate: float = 0.0) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) <= 1:
            return 0.0

        excess_returns = returns - risk_free_rate

        if np.std(excess_returns) == 0:
            return 0.0

        # Annualize assuming daily returns
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return float(sharpe)

    def _log_final_results(self) -> None:
        """Log final experiment summary."""
        if not self.experiment_id:
            return

        # Calculate summary metrics
        summary_metrics = {
            'experiment_name': self.config.experiment_name,
            'n_models_trained': len(self.results.get('training', {})),
            'best_model': None,
            'best_score': 0.0
        }

        # Find best performing model
        training_results = self.results.get('training', {})
        for model_type, result in training_results.items():
            score = result.get('metrics', {}).get('val_score', 0.0)
            if score > summary_metrics['best_score']:
                summary_metrics['best_score'] = score
                summary_metrics['best_model'] = model_type

        # Log summary
        self.registry.log_results(
            self.experiment_id,
            "experiment_summary",
            summary_metrics
        )


    def get_experiment_summary(self) -> dict[str, Any]:
        """Get comprehensive experiment summary."""
        if not self.experiment_id:
            return {}

        return self.registry.get_experiment_summary(self.experiment_id)
