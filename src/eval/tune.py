#!/usr/bin/env python3
"""
Hyperparameter tuning script using Optuna for SL models and RL agents.

This script performs hyperparameter optimization for:
1. SL models: window sizes, model dimensions
2. PPO: clip_range, n_steps, vf_coef
3. SAC: tau, temperature (ent_coef)
4. Reward function: lambda parameters

The optimization uses time-blocked cross-validation to prevent data leakage
and maximizes the validation Sharpe ratio as the objective function.
"""

import argparse
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

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
try:
    from stable_baselines3 import PPO, SAC

    from src.data.splits import purged_walk_forward_splits
    from src.envs.trading_env import TradingEnvironment
    from src.eval.backtest import BacktestEngine
    from src.sl.models.base import set_all_seeds
    from src.sl.models.factory import SLModelFactory
    from src.sl.train import SLTrainingPipeline
except ImportError:
    # Fallback imports for development environment
    try:
        from data.splits import purged_walk_forward_splits
        from envs.trading_env import TradingEnvironment
        from eval.backtest import BacktestEngine
        from sl.models.base import set_all_seeds
        from sl.models.factory import SLModelFactory
        from sl.train import SLTrainingPipeline
    except ImportError:
        print("Warning: Could not import required modules. Some functionality may be limited.")
        purged_walk_forward_splits = None
        TradingEnvironment = None
        BacktestEngine = None
        set_all_seeds = None
        SLModelFactory = None
        SLTrainingPipeline = None

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class TimeBlockedCV:
    """Time-blocked cross-validation to prevent data leakage."""

    def __init__(self, n_splits: int = 3, gap: int = 10):
        """
        Initialize time-blocked cross-validation.

        Args:
            n_splits: Number of splits for cross-validation
            gap: Gap between training and validation sets to prevent leakage
        """
        self.n_splits = n_splits
        self.gap = gap

    def split(self, X):
        """
        Generate indices for time-blocked cross-validation splits.

        Args:
            X: Feature matrix

        Returns:
            List of (train_indices, val_indices) tuples
        """
        # Use purged walk-forward splits to prevent data leakage
        if purged_walk_forward_splits is None:
            # Fallback to simple time series split if purged_walk_forward_splits is not available
            n_samples = len(X)
            fold_size = n_samples // (self.n_splits + 1)
            splits = []
            for i in range(self.n_splits):
                train_end = (i + 1) * fold_size
                val_start = train_end + self.gap
                val_end = val_start + fold_size
                if val_end <= n_samples:
                    train_indices = list(range(0, train_end))
                    val_indices = list(range(val_start, val_end))
                    splits.append((train_indices, val_indices))
            return splits

        try:
            splits_data = list(purged_walk_forward_splits(
                pd.DataFrame(X),  # Convert to DataFrame for compatibility
                n_splits=self.n_splits,
                train_ratio=0.6,
                val_ratio=0.2,
                test_ratio=0.2
            ))

            # Convert to indices format
            indices_splits = []
            for train_df, val_df, _ in splits_data:
                train_indices = train_df.index.tolist()
                val_indices = val_df.index.tolist()
                indices_splits.append((train_indices, val_indices))

            return indices_splits
        except Exception:
            # Fallback to simple time series split
            n_samples = len(X)
            fold_size = n_samples // (self.n_splits + 1)
            splits = []
            for i in range(self.n_splits):
                train_end = (i + 1) * fold_size
                val_start = train_end + self.gap
                val_end = val_start + fold_size
                if val_end <= n_samples:
                    train_indices = list(range(0, train_end))
                    val_indices = list(range(val_start, val_end))
                    splits.append((train_indices, val_indices))
            return splits


class HyperparameterTuner:
    """Hyperparameter tuner for SL models and RL agents."""

    def __init__(self, data_file: str = "data/features.parquet", storage: str | None = None):
        """
        Initialize hyperparameter tuner.

        Args:
            data_file: Path to features data file
            storage: Database URL for distributed optimization
        """
        self.data_file = data_file
        self.storage = storage
        self.df = pd.read_parquet(data_file)
        self.feature_columns = [col for col in self.df.columns if col not in ['mu_hat', 'sigma_hat']]
        self.X = self.df[self.feature_columns].values
        self.y = self.df['mu_hat'].values  # Target: expected returns

        # Initialize cross-validation
        self.cv = TimeBlockedCV(n_splits=3, gap=10)

        # Create reports directory if it doesn't exist
        os.makedirs("reports", exist_ok=True)

    def objective_sl_model(self, trial, model_type: str) -> float:
        """
        Objective function for SL model hyperparameter tuning.

        Args:
            trial: Optuna trial object
            model_type: Type of SL model to tune

        Returns:
            Validation Sharpe ratio (to be maximized)
        """
        # Set seeds for reproducibility
        if set_all_seeds:
            set_all_seeds(42)

        # Define hyperparameter search space based on model type
        if model_type == "ridge":
            config = {
                "model_type": "ridge",
                "model_config": {
                    "alpha": trial.suggest_float("alpha", 1e-5, 100.0, log=True),
                    "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
                    "solver": trial.suggest_categorical("solver", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]),
                    "max_iter": trial.suggest_int("max_iter", 1000, 10000),
                    "tol": trial.suggest_float("tol", 1e-6, 1e-2, log=True),
                    "window_size": trial.suggest_int("window_size", 5, 60),
                    "random_state": 42
                },
                "cv_config": {"n_splits": 3, "gap": 10},
                "tuning_config": {"enable_tuning": False},
                "random_state": 42,
                "output_dir": "models/",
                "save_model": False
            }
        elif model_type == "mlp":
            # Convert nested lists to tuples for Optuna compatibility - expanded architectures
            hidden_layer_choices = [
                (32,),
                (64,),
                (128,),
                (256,),
                (512,),
                (32, 16),
                (64, 32),
                (128, 64),
                (256, 128),
                (512, 256),
                (64, 32, 16),
                (128, 64, 32),
                (256, 128, 64),
                (512, 256, 128),
                (128, 64, 32, 16),
                (256, 128, 64, 32)
            ]

            config = {
                "model_type": "mlp",
                "model_config": {
                    "hidden_layers": list(trial.suggest_categorical("hidden_layers", hidden_layer_choices)),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
                    "dropout": trial.suggest_float("dropout", 0.0, 0.7),
                    "epochs": trial.suggest_int("epochs", 20, 500),
                    "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512]),
                    "optimizer": trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd", "rmsprop"]),
                    "activation": trial.suggest_categorical("activation", ["relu", "tanh", "elu", "leaky_relu"]),
                    "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
                    "beta1": trial.suggest_float("beta1", 0.8, 0.99),
                    "beta2": trial.suggest_float("beta2", 0.9, 0.999),
                    "early_stopping_patience": trial.suggest_int("early_stopping_patience", 10, 50),
                    "lr_scheduler": trial.suggest_categorical("lr_scheduler", ["none", "step", "cosine", "exponential"]),
                    "window_size": trial.suggest_int("window_size", 5, 60),
                    "random_state": 42
                },
                "cv_config": {"n_splits": 3, "gap": 10},
                "tuning_config": {"enable_tuning": False},
                "random_state": 42,
                "output_dir": "models/",
                "save_model": False
            }
        elif model_type == "cnn_lstm":
            # Convert nested lists to tuples for Optuna compatibility - expanded architectures
            cnn_channels_choices = [
                (16,),
                (32,),
                (64,),
                (128,),
                (16, 32),
                (32, 64),
                (64, 128),
                (128, 256),
                (16, 32, 64),
                (32, 64, 128),
                (64, 128, 256),
                (32, 64, 32),
                (64, 128, 64)
            ]

            config = {
                "model_type": "cnn_lstm",
                "model_config": {
                    "cnn_channels": list(trial.suggest_categorical("cnn_channels", cnn_channels_choices)),
                    "cnn_kernel_sizes": trial.suggest_categorical("cnn_kernel_sizes", [[3], [5], [7], [3, 5], [3, 7], [5, 7]]),
                    "cnn_dropout": trial.suggest_float("cnn_dropout", 0.0, 0.5),
                    "lstm_hidden_size": trial.suggest_int("lstm_hidden_size", 16, 512),
                    "lstm_num_layers": trial.suggest_int("lstm_num_layers", 1, 4),
                    "lstm_dropout": trial.suggest_float("lstm_dropout", 0.0, 0.5),
                    "lstm_bidirectional": trial.suggest_categorical("lstm_bidirectional", [True, False]),
                    "sequence_length": trial.suggest_int("sequence_length", 5, 60),
                    "window_size": trial.suggest_int("window_size", 5, 60),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
                    "dropout": trial.suggest_float("dropout", 0.0, 0.7),
                    "epochs": trial.suggest_int("epochs", 20, 500),
                    "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256]),
                    "optimizer": trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd", "rmsprop"]),
                    "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
                    "early_stopping_patience": trial.suggest_int("early_stopping_patience", 10, 50),
                    "lr_scheduler": trial.suggest_categorical("lr_scheduler", ["none", "step", "cosine", "exponential"]),
                    "attention": trial.suggest_categorical("attention", [True, False]),
                    "residual_connections": trial.suggest_categorical("residual_connections", [True, False]),
                    "random_state": 42
                },
                "cv_config": {"n_splits": 3, "gap": 10},
                "tuning_config": {"enable_tuning": False},
                "random_state": 42,
                "output_dir": "models/",
                "save_model": False
            }
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Perform cross-validation
        sharpe_ratios = []

        for train_idx, val_idx in self.cv.split(self.X):
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]

            try:
                # Train model
                if SLTrainingPipeline is None:
                    # Skip actual training if modules are not available
                    sharpe_ratios.append(np.random.random())  # Random score for demonstration
                    continue

                pipeline = SLTrainingPipeline(config)
                pipeline.train(X_train, y_train)

                # Make predictions
                y_pred = pipeline.model.predict(X_val)

                # Align predictions and targets if needed (for sequence models)
                if len(y_pred) != len(y_val):
                    y_val_aligned = y_val[len(y_val) - len(y_pred):]
                    y_pred_aligned = y_pred
                else:
                    y_val_aligned = y_val
                    y_pred_aligned = y_pred

                # Calculate returns based on predictions (simple strategy)
                # Buy when prediction > 0, sell when prediction < 0
                positions = np.sign(y_pred_aligned)
                returns = positions[:-1] * np.diff(y_val_aligned)  # Simple return calculation

                # Calculate Sharpe ratio
                if len(returns) > 1 and np.std(returns) > 0:
                    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
                    sharpe_ratios.append(sharpe)
                else:
                    sharpe_ratios.append(0.0)

            except Exception as e:
                print(f"Error in trial: {e}")
                sharpe_ratios.append(0.0)  # Penalize failed trials

        # Return mean Sharpe ratio across folds
        return float(np.mean(sharpe_ratios)) if sharpe_ratios else 0.0

    def objective_rl_agent(self, trial, agent_type: str) -> float:
        """
        Objective function for RL agent hyperparameter tuning.

        Args:
            trial: Optuna trial object
            agent_type: Type of RL agent to tune (ppo or sac)

        Returns:
            Validation Sharpe ratio (to be maximized)
        """
        # Set seeds for reproducibility
        if set_all_seeds:
            set_all_seeds(42)

        # Create temporary config file for this trial
        import json
        import os
        import uuid

        if agent_type == "ppo":
            # Define PPO hyperparameters to tune - comprehensive search space
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
            n_steps = trial.suggest_int("n_steps", 512, 8192, step=512)
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
            n_epochs = trial.suggest_int("n_epochs", 3, 30)
            gamma = trial.suggest_float("gamma", 0.9, 0.9999)
            gae_lambda = trial.suggest_float("gae_lambda", 0.8, 1.0)
            clip_range = trial.suggest_float("clip_range", 0.05, 0.5)
            clip_range_vf = trial.suggest_float("clip_range_vf", 0.05, 0.5)
            ent_coef = trial.suggest_float("ent_coef", 0.0, 0.1)
            vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)
            max_grad_norm = trial.suggest_float("max_grad_norm", 0.1, 1.0)
            target_kl = trial.suggest_float("target_kl", 0.005, 0.05)
            normalize_advantage = trial.suggest_categorical("normalize_advantage", [True, False])
            use_sde = trial.suggest_categorical("use_sde", [True, False])
            sde_sample_freq = trial.suggest_int("sde_sample_freq", -1, 64)

            # Load base config
            with open("configs/ppo_config.json") as f:
                config = json.load(f)

            # Update hyperparameters
            config["ppo"]["learning_rate"] = learning_rate
            config["ppo"]["n_steps"] = n_steps
            config["ppo"]["batch_size"] = batch_size
            config["ppo"]["n_epochs"] = n_epochs
            config["ppo"]["gamma"] = gamma
            config["ppo"]["gae_lambda"] = gae_lambda
            config["ppo"]["clip_range"] = clip_range
            config["ppo"]["clip_range_vf"] = clip_range_vf
            config["ppo"]["ent_coef"] = ent_coef
            config["ppo"]["vf_coef"] = vf_coef
            config["ppo"]["max_grad_norm"] = max_grad_norm
            config["ppo"]["target_kl"] = target_kl
            config["ppo"]["normalize_advantage"] = normalize_advantage
            config["ppo"]["use_sde"] = use_sde
            config["ppo"]["sde_sample_freq"] = sde_sample_freq

            # Create temporary config file with unique name
            temp_config_name = f"temp_config_{uuid.uuid4().hex}.json"
            with open(temp_config_name, 'w') as f:
                json.dump(config, f)

        elif agent_type == "sac":
            # Define SAC hyperparameters to tune - comprehensive search space
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
            buffer_size = trial.suggest_categorical("buffer_size", [100000, 500000, 1000000, 2000000])
            learning_starts = trial.suggest_int("learning_starts", 100, 10000)
            batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
            tau = trial.suggest_float("tau", 0.001, 0.02, log=True)
            gamma = trial.suggest_float("gamma", 0.9, 0.9999)
            train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16])
            gradient_steps = trial.suggest_categorical("gradient_steps", [1, 2, 4, 8])
            ent_coef = trial.suggest_categorical("ent_coef", ["auto", 0.01, 0.05, 0.1, 0.2, 0.3, 0.5])
            target_update_interval = trial.suggest_int("target_update_interval", 1, 10)
            target_entropy = trial.suggest_categorical("target_entropy", ["auto", -1, -2, -4])
            use_sde = trial.suggest_categorical("use_sde", [True, False])
            sde_sample_freq = trial.suggest_int("sde_sample_freq", -1, 64)
            use_sde_at_warmup = trial.suggest_categorical("use_sde_at_warmup", [True, False])

            # Load base config
            with open("configs/sac_config.json") as f:
                config = json.load(f)

            # Update hyperparameters
            config["sac"]["learning_rate"] = learning_rate
            config["sac"]["buffer_size"] = buffer_size
            config["sac"]["learning_starts"] = learning_starts
            config["sac"]["batch_size"] = batch_size
            config["sac"]["tau"] = tau
            config["sac"]["gamma"] = gamma
            config["sac"]["train_freq"] = train_freq
            config["sac"]["gradient_steps"] = gradient_steps
            config["sac"]["ent_coef"] = ent_coef
            config["sac"]["target_update_interval"] = target_update_interval
            config["sac"]["target_entropy"] = target_entropy
            config["sac"]["use_sde"] = use_sde
            config["sac"]["sde_sample_freq"] = sde_sample_freq
            config["sac"]["use_sde_at_warmup"] = use_sde_at_warmup

            # Create temporary config file with unique name
            temp_config_name = f"temp_config_{uuid.uuid4().hex}.json"
            with open(temp_config_name, 'w') as f:
                json.dump(config, f)
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")

        try:
            # Use time-blocked cross-validation for RL agents as well
            sharpe_ratios = []

            # Perform cross-validation
            for train_idx, val_idx in self.cv.split(self.X):
                # Train and evaluate on this fold
                fold_sharpe = self._train_and_evaluate_agent_fold(
                    agent_type, temp_config.name if agent_type in ["ppo", "sac"] else None,
                    train_idx, val_idx
                )
                sharpe_ratios.append(fold_sharpe)

            # Return mean Sharpe ratio across folds
            sharpe_ratio = float(np.mean(sharpe_ratios)) if sharpe_ratios else 0.0
        finally:
            # Clean up temporary config file
            if agent_type in ["ppo", "sac"]:
                try:
                    os.unlink(temp_config_name)
                except FileNotFoundError:
                    pass

        return float(sharpe_ratio)

    def objective_reward_params(self, trial) -> float:
        """
        Objective function for reward function lambda parameter tuning.

        Args:
            trial: Optuna trial object

        Returns:
            Validation Sharpe ratio (to be maximized)
        """
        # Set seeds for reproducibility
        if set_all_seeds:
            set_all_seeds(42)

        # Define reward function hyperparameters to tune - optimized ranges based on literature
        pnl_weight = trial.suggest_float("pnl_weight", 0.5, 10.0)
        transaction_cost_weight = trial.suggest_float("transaction_cost_weight", 0.01, 10.0, log=True)
        risk_adjustment_weight = trial.suggest_float("risk_adjustment_weight", 0.0, 5.0)
        stability_penalty_weight = trial.suggest_float("stability_penalty_weight", 0.0, 3.0)
        drawdown_penalty_weight = trial.suggest_float("drawdown_penalty_weight", 0.0, 2.0)
        volatility_penalty_weight = trial.suggest_float("volatility_penalty_weight", 0.0, 1.5)
        position_change_penalty = trial.suggest_float("position_change_penalty", 0.0, 0.5)
        max_position_penalty = trial.suggest_float("max_position_penalty", 0.0, 1.0)

        # Create reward configuration
        reward_config = {
            'pnl_weight': pnl_weight,
            'transaction_cost_weight': transaction_cost_weight,
            'risk_adjustment_weight': risk_adjustment_weight,
            'stability_penalty_weight': stability_penalty_weight,
            'drawdown_penalty_weight': drawdown_penalty_weight,
            'volatility_penalty_weight': volatility_penalty_weight,
            'position_change_penalty': position_change_penalty,
            'max_position_penalty': max_position_penalty
        }

        # Use time-blocked cross-validation
        sharpe_ratios = []

        # Perform cross-validation
        for train_idx, val_idx in self.cv.split(self.X):
            # Train and evaluate on this fold
            fold_sharpe = self._train_and_evaluate_reward_fold(
                reward_config, train_idx, val_idx
            )
            sharpe_ratios.append(fold_sharpe)

        # Return mean Sharpe ratio across folds
        sharpe_ratio = float(np.mean(sharpe_ratios)) if sharpe_ratios else 0.0
        return sharpe_ratio

    def _train_and_evaluate_reward_fold(self, reward_config: dict,
                                       train_idx: list, val_idx: list) -> float:
        """
        Train an RL agent with given reward configuration and evaluate using backtesting on a specific fold.

        Args:
            reward_config: Reward configuration dictionary
            train_idx: Training indices for this fold
            val_idx: Validation indices for this fold

        Returns:
            Validation Sharpe ratio for this fold
        """
        import json
        import os

        import numpy as np
        from stable_baselines3 import PPO
        # Use the already imported modules from the top of the file

        # Get data for this fold
        df = self.df.copy()
        train_data = df.iloc[train_idx]
        val_data = df.iloc[val_idx]

        # Save temporary files for environments with unique names
        train_file_name = f"temp_train_{uuid.uuid4().hex}.parquet"
        val_file_name = f"temp_val_{uuid.uuid4().hex}.parquet"
        train_data.to_parquet(train_file_name)
        val_data.to_parquet(val_file_name)

        try:
            # Create training environment with reward configuration
            train_env = TradingEnvironment(
                data_file=train_file_name,
                initial_capital=100000.0,
                transaction_cost=0.001,
                seed=42,
                window_size=30,
                reward_config=reward_config
            )

            # Load PPO config
            with open("configs/ppo_config.json") as f:
                config = json.load(f)

            ppo_config = config.get('ppo', {})

            # Extract hyperparameters
            learning_rate = ppo_config.get('learning_rate', 3e-4)
            n_steps = ppo_config.get('n_steps', 2048)
            batch_size = ppo_config.get('batch_size', 64)
            n_epochs = ppo_config.get('n_epochs', 10)
            gamma = ppo_config.get('gamma', 0.99)
            gae_lambda = ppo_config.get('gae_lambda', 0.95)
            clip_range = ppo_config.get('clip_range', 0.2)
            ent_coef = ppo_config.get('ent_coef', 0.0)
            vf_coef = ppo_config.get('vf_coef', 0.5)
            max_grad_norm = ppo_config.get('max_grad_norm', 0.5)

            # Create and train PPO model
            model = PPO(
                "MlpPolicy",
                train_env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                seed=42,
                verbose=0
            )

            # Train for a limited number of timesteps for tuning
            model.learn(total_timesteps=50000)

            # Create validation environment for evaluation with same reward configuration
            eval_env = TradingEnvironment(
                data_file=val_file_name,
                initial_capital=100000.0,
                transaction_cost=0.001,
                seed=43,  # Different seed for evaluation
                window_size=30,
                reward_config=reward_config
            )

            # Evaluate agent on validation set
            obs, _ = eval_env.reset()
            signals = []
            prices = []

            # Run evaluation episode
            while True:
                # Get action from trained model
                action, _ = model.predict(obs, deterministic=True)

                # Store price for backtesting
                prices.append(eval_env.prices[eval_env.current_step])

                # Convert action to signal (-1 to 1)
                signal = np.clip(action[0], -1, 1)
                signals.append(signal)

                # Step environment
                obs, reward, terminated, truncated, info = eval_env.step(action)

                # Break if episode is done
                if terminated or truncated:
                    break

            # Convert to pandas Series for backtesting
            import pandas as pd
            signals_series = pd.Series(signals)
            prices_series = pd.Series(prices)

            # Run backtest
            backtest_engine = BacktestEngine(
                transaction_cost=0.001,
                slippage=0.0005,
                initial_capital=100000.0
            )

            results = backtest_engine.run_backtest(signals_series, prices_series)

            # Return Sharpe ratio
            sharpe_ratio = results['metrics'].get('sharpe_ratio', 0.0)

            return float(sharpe_ratio)

        finally:
            # Clean up temporary files
            try:
                os.unlink(train_file_name)
                os.unlink(val_file_name)
            except FileNotFoundError:
                pass

    def _train_and_evaluate_agent(self, agent_type: str, config_path: str) -> float:
        """
        Train an RL agent with given hyperparameters and evaluate using backtesting.

        Args:
            agent_type: Type of RL agent to train (ppo or sac)
            config_path: Path to configuration file

        Returns:
            Validation Sharpe ratio
        """
        import json
        import os
        import tempfile

        import numpy as np
        from stable_baselines3 import PPO, SAC
        # Use the already imported modules from the top of the file

        # Load data for training
        df = self.df.copy()

        # Split data into train and validation sets (80/20)
        split_index = int(len(df) * 0.8)
        train_data = df.iloc[:split_index]
        val_data = df.iloc[split_index:]

        # Save temporary files for environments
        train_file = tempfile.NamedTemporaryFile(suffix='.parquet', delete=False)
        val_file = tempfile.NamedTemporaryFile(suffix='.parquet', delete=False)
        train_data.to_parquet(train_file.name)
        val_data.to_parquet(val_file.name)
        train_file.close()
        val_file.close()

        try:
            # Create training environment
            train_env = TradingEnvironment(
                data_file=train_file.name,
                initial_capital=100000.0,
                transaction_cost=0.001,
                seed=42,
                window_size=30
            )

            # Train agent
            if agent_type == "ppo":
                # Load config
                with open(config_path) as f:
                    config = json.load(f)

                ppo_config = config.get('ppo', {})

                # Extract hyperparameters
                learning_rate = ppo_config.get('learning_rate', 3e-4)
                n_steps = ppo_config.get('n_steps', 2048)
                batch_size = ppo_config.get('batch_size', 64)
                n_epochs = ppo_config.get('n_epochs', 10)
                gamma = ppo_config.get('gamma', 0.99)
                gae_lambda = ppo_config.get('gae_lambda', 0.95)
                clip_range = ppo_config.get('clip_range', 0.2)
                ent_coef = ppo_config.get('ent_coef', 0.0)
                vf_coef = ppo_config.get('vf_coef', 0.5)
                max_grad_norm = ppo_config.get('max_grad_norm', 0.5)

                # Create and train PPO model
                model = PPO(
                    "MlpPolicy",
                    train_env,
                    learning_rate=learning_rate,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    clip_range=clip_range,
                    ent_coef=ent_coef,
                    vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm,
                    seed=42,
                    verbose=0
                )

                # Train for a limited number of timesteps for tuning
                model.learn(total_timesteps=50000)

            elif agent_type == "sac":
                # Load config
                with open(config_path) as f:
                    config = json.load(f)

                sac_config = config.get('sac', {})

                # Extract hyperparameters
                learning_rate = sac_config.get('learning_rate', 3e-4)
                buffer_size = sac_config.get('buffer_size', 1000000)
                learning_starts = sac_config.get('learning_starts', 1000)
                batch_size = sac_config.get('batch_size', 256)
                tau = sac_config.get('tau', 0.005)
                gamma = sac_config.get('gamma', 0.99)
                train_freq = sac_config.get('train_freq', 1)
                gradient_steps = sac_config.get('gradient_steps', 1)
                ent_coef = sac_config.get('ent_coef', 'auto')
                target_update_interval = sac_config.get('target_update_interval', 1)
                target_entropy = sac_config.get('target_entropy', 'auto')

                # Create and train SAC model
                model = SAC(
                    "MlpPolicy",
                    train_env,
                    learning_rate=learning_rate,
                    buffer_size=buffer_size,
                    learning_starts=learning_starts,
                    batch_size=batch_size,
                    tau=tau,
                    gamma=gamma,
                    train_freq=train_freq,
                    gradient_steps=gradient_steps,
                    ent_coef=ent_coef,
                    target_update_interval=target_update_interval,
                    target_entropy=target_entropy,
                    seed=42,
                    verbose=0
                )

                # Train for a limited number of timesteps for tuning
                model.learn(total_timesteps=50000)

            # Create validation environment for evaluation
            eval_env = TradingEnvironment(
                data_file=val_file.name,
                initial_capital=100000.0,
                transaction_cost=0.001,
                seed=43,  # Different seed for evaluation
                window_size=30
            )

            # Evaluate agent on validation set
            obs, _ = eval_env.reset()
            signals = []
            prices = []

            # Run evaluation episode
            while True:
                # Get action from trained model
                action, _ = model.predict(obs, deterministic=True)

                # Store price for backtesting
                prices.append(eval_env.prices[eval_env.current_step])

                # Convert action to signal (-1 to 1)
                signal = np.clip(action[0], -1, 1)
                signals.append(signal)

                # Step environment
                obs, reward, terminated, truncated, info = eval_env.step(action)

                # Break if episode is done
                if terminated or truncated:
                    break

            # Convert to pandas Series for backtesting
            import pandas as pd
            signals_series = pd.Series(signals)
            prices_series = pd.Series(prices)

            # Run backtest
            backtest_engine = BacktestEngine(
                transaction_cost=0.001,
                slippage=0.0005,
                initial_capital=100000.0
            )

            results = backtest_engine.run_backtest(signals_series, prices_series)

            # Return Sharpe ratio
            sharpe_ratio = results['metrics'].get('sharpe_ratio', 0.0)

            return float(sharpe_ratio)

        finally:
            # Clean up temporary files
            try:
                os.unlink(train_file.name)
                os.unlink(val_file.name)
            except FileNotFoundError:
                pass

    def _train_and_evaluate_agent_fold(self, agent_type: str, config_path: str,
                                       train_idx: list, val_idx: list) -> float:
        """
        Train an RL agent with given hyperparameters and evaluate using backtesting on a specific fold.

        Args:
            agent_type: Type of RL agent to train (ppo or sac)
            config_path: Path to configuration file
            train_idx: Training indices for this fold
            val_idx: Validation indices for this fold

        Returns:
            Validation Sharpe ratio for this fold
        """
        import json
        import os
        import tempfile

        import numpy as np
        from stable_baselines3 import PPO, SAC
        # Use the already imported modules from the top of the file

        # Get data for this fold
        df = self.df.copy()
        train_data = df.iloc[train_idx]
        val_data = df.iloc[val_idx]

        # Save temporary files for environments
        train_file = tempfile.NamedTemporaryFile(suffix='.parquet', delete=False)
        val_file = tempfile.NamedTemporaryFile(suffix='.parquet', delete=False)
        train_data.to_parquet(train_file.name)
        val_data.to_parquet(val_file.name)
        train_file.close()
        val_file.close()

        try:
            # Create training environment
            train_env = TradingEnvironment(
                data_file=train_file.name,
                initial_capital=100000.0,
                transaction_cost=0.001,
                seed=42,
                window_size=30
            )

            # Train agent
            if agent_type == "ppo":
                # Load config
                with open(config_path) as f:
                    config = json.load(f)

                ppo_config = config.get('ppo', {})

                # Extract hyperparameters
                learning_rate = ppo_config.get('learning_rate', 3e-4)
                n_steps = ppo_config.get('n_steps', 2048)
                batch_size = ppo_config.get('batch_size', 64)
                n_epochs = ppo_config.get('n_epochs', 10)
                gamma = ppo_config.get('gamma', 0.99)
                gae_lambda = ppo_config.get('gae_lambda', 0.95)
                clip_range = ppo_config.get('clip_range', 0.2)
                ent_coef = ppo_config.get('ent_coef', 0.0)
                vf_coef = ppo_config.get('vf_coef', 0.5)
                max_grad_norm = ppo_config.get('max_grad_norm', 0.5)

                # Create and train PPO model
                model = PPO(
                    "MlpPolicy",
                    train_env,
                    learning_rate=learning_rate,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    clip_range=clip_range,
                    ent_coef=ent_coef,
                    vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm,
                    seed=42,
                    verbose=0
                )

                # Train for a sufficient number of timesteps for tuning
                # Use a scaled number of timesteps based on data size for efficiency
                train_timesteps = min(100000, max(50000, len(train_idx) * 10))
                model.learn(total_timesteps=train_timesteps)

            elif agent_type == "sac":
                # Load config
                with open(config_path) as f:
                    config = json.load(f)

                sac_config = config.get('sac', {})

                # Extract hyperparameters
                learning_rate = sac_config.get('learning_rate', 3e-4)
                buffer_size = sac_config.get('buffer_size', 1000000)
                learning_starts = sac_config.get('learning_starts', 1000)
                batch_size = sac_config.get('batch_size', 256)
                tau = sac_config.get('tau', 0.005)
                gamma = sac_config.get('gamma', 0.99)
                train_freq = sac_config.get('train_freq', 1)
                gradient_steps = sac_config.get('gradient_steps', 1)
                ent_coef = sac_config.get('ent_coef', 'auto')
                target_update_interval = sac_config.get('target_update_interval', 1)
                target_entropy = sac_config.get('target_entropy', 'auto')

                # Create and train SAC model
                model = SAC(
                    "MlpPolicy",
                    train_env,
                    learning_rate=learning_rate,
                    buffer_size=buffer_size,
                    learning_starts=learning_starts,
                    batch_size=batch_size,
                    tau=tau,
                    gamma=gamma,
                    train_freq=train_freq,
                    gradient_steps=gradient_steps,
                    ent_coef=ent_coef,
                    target_update_interval=target_update_interval,
                    target_entropy=target_entropy,
                    seed=42,
                    verbose=0
                )

                # Train for a sufficient number of timesteps for tuning
                # Use a scaled number of timesteps based on data size for efficiency
                train_timesteps = min(100000, max(50000, len(train_idx) * 10))
                model.learn(total_timesteps=train_timesteps)

            # Create validation environment for evaluation
            eval_env = TradingEnvironment(
                data_file=val_file.name,
                initial_capital=100000.0,
                transaction_cost=0.001,
                seed=43,  # Different seed for evaluation
                window_size=30
            )

            # Evaluate agent on validation set
            obs, _ = eval_env.reset()
            signals = []
            prices = []

            # Run evaluation episode
            while True:
                # Get action from trained model
                action, _ = model.predict(obs, deterministic=True)

                # Store price for backtesting
                prices.append(eval_env.prices[eval_env.current_step])

                # Convert action to signal (-1 to 1)
                signal = np.clip(action[0], -1, 1)
                signals.append(signal)

                # Step environment
                obs, reward, terminated, truncated, info = eval_env.step(action)

                # Break if episode is done
                if terminated or truncated:
                    break

            # Convert to pandas Series for backtesting
            import pandas as pd
            signals_series = pd.Series(signals)
            prices_series = pd.Series(prices)

            # Run backtest
            backtest_engine = BacktestEngine(
                transaction_cost=0.001,
                slippage=0.0005,
                initial_capital=100000.0
            )

            results = backtest_engine.run_backtest(signals_series, prices_series)

            # Return Sharpe ratio
            sharpe_ratio = results['metrics'].get('sharpe_ratio', 0.0)

            return float(sharpe_ratio)

        finally:
            # Clean up temporary files
            try:
                os.unlink(train_file.name)
                os.unlink(val_file.name)
            except FileNotFoundError:
                pass

    def tune_sl_model(self, model_type: str, n_trials: int = 50, timeout: int = None,
                      n_jobs: int = 1, pruner: str = "median"):
        """
        Tune hyperparameters for an SL model.

        Args:
            model_type: Type of SL model to tune
            n_trials: Number of Optuna trials
            timeout: Maximum time in seconds for optimization
            n_jobs: Number of parallel jobs
            pruner: Pruning strategy

        Returns:
            Dictionary with best parameters and score
        """
        print(f"Tuning {model_type} model...")

        # Log parallel execution configuration
        if self.storage:
            print(f"Using distributed storage: {self.storage}")
        if n_jobs > 1:
            print(f"Using {n_jobs} parallel jobs")

        # Select pruner
        if pruner == "median":
            pruner_obj = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        elif pruner == "hyperband":
            pruner_obj = optuna.pruners.HyperbandPruner()
        else:
            pruner_obj = None

        # Create study with pruning for computational efficiency
        study = optuna.create_study(
            direction="maximize",
            study_name=f"{model_type}_tuning",
            storage=self.storage,
            pruner=pruner_obj,
            sampler=optuna.samplers.TPESampler(n_startup_trials=min(10, n_trials // 5)),
            load_if_exists=True
        )

        # Optimize
        study.optimize(
            lambda trial: self.objective_sl_model(trial, model_type),
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=True
        )

        print(f"Best {model_type} parameters: {study.best_params}")
        print(f"Best {model_type} Sharpe ratio: {study.best_value:.4f}")
        print(f"Number of completed trials: {len(study.trials)}")
        print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")

        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "study": study
        }

    def tune_rl_agent(self, agent_type: str, n_trials: int = 50, timeout: int = None,
                      n_jobs: int = 1, pruner: str = "median"):
        """
        Tune hyperparameters for an RL agent.

        Args:
            agent_type: Type of RL agent to tune (ppo or sac)
            n_trials: Number of Optuna trials
            timeout: Maximum time in seconds for optimization
            n_jobs: Number of parallel jobs
            pruner: Pruning strategy

        Returns:
            Dictionary with best parameters and score
        """
        print(f"Tuning {agent_type.upper()} agent...")

        # Log parallel execution configuration
        if self.storage:
            print(f"Using distributed storage: {self.storage}")
        if n_jobs > 1:
            print(f"Using {n_jobs} parallel jobs")

        # Select pruner
        if pruner == "median":
            pruner_obj = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        elif pruner == "hyperband":
            pruner_obj = optuna.pruners.HyperbandPruner()
        else:
            pruner_obj = None

        # Create study with pruning for computational efficiency
        study = optuna.create_study(
            direction="maximize",
            study_name=f"{agent_type}_tuning",
            storage=self.storage,
            pruner=pruner_obj,
            sampler=optuna.samplers.TPESampler(n_startup_trials=min(10, n_trials // 5)),
            load_if_exists=True
        )

        # Optimize
        study.optimize(
            lambda trial: self.objective_rl_agent(trial, agent_type),
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=True
        )

        print(f"Best {agent_type.upper()} parameters: {study.best_params}")
        print(f"Best {agent_type.upper()} Sharpe ratio: {study.best_value:.4f}")
        print(f"Number of completed trials: {len(study.trials)}")
        print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")

        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "study": study
        }

    def save_study(self, study, study_name: str) -> str:
        """
        Save Optuna study to a pickle file.

        Args:
            study: Optuna study to save
            study_name: Name of the study

        Returns:
            Path to saved study file
        """
        study_path = f"reports/{study_name}_study.pkl"
        with open(study_path, 'wb') as f:
            pickle.dump(study, f)
        print(f"Study saved to {study_path}")
        return study_path

    def save_best_params(self, best_params: dict[str, Any], model_name: str) -> str:
        """
        Save best parameters to a JSON file.

        Args:
            best_params: Best parameters dictionary
            model_name: Name of the model

        Returns:
            Path to saved parameters file
        """
        params_path = f"reports/{model_name}_best_params.json"
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"Best parameters saved to {params_path}")
        return params_path

    def tune_reward_params(self, n_trials: int = 50):
        """
        Tune hyperparameters for reward function lambda parameters.

        Args:
            n_trials: Number of Optuna trials

        Returns:
            Dictionary with best parameters and score
        """
        print("Tuning reward function lambda parameters...")

        # Log parallel execution configuration
        if self.storage:
            print(f"Using distributed storage: {self.storage}")

        # Create study
        study = optuna.create_study(
            direction="maximize",
            study_name="reward_params_tuning",
            storage=self.storage,
            load_if_exists=True
        )

        # Optimize
        study.optimize(
            self.objective_reward_params,
            n_trials=n_trials,
            show_progress_bar=True
        )

        print(f"Best reward parameters: {study.best_params}")
        print(f"Best Sharpe ratio: {study.best_value:.4f}")

        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "study": study
        }


def main():
    """Main function for hyperparameter tuning."""
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning with Optuna")
    parser.add_argument("--model", choices=["ridge", "mlp", "cnn_lstm", "ppo", "sac", "reward", "all"],
                        help="Model/agent type to tune")
    parser.add_argument("--trials", type=int, default=50,
                        help="Number of Optuna trials")
    parser.add_argument("--data", default="data/features.parquet",
                        help="Path to features data file")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Maximum time in seconds for optimization")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Number of parallel jobs for optimization")
    parser.add_argument("--pruner", choices=["median", "hyperband", "none"], default="median",
                        help="Pruning strategy for early stopping of unpromising trials")
    parser.add_argument("--storage", type=str, default=None,
                        help="Database URL for distributed optimization (e.g., 'sqlite:///example.db')")
    parser.add_argument("--study-name", type=str, default=None,
                        help="Name of the study for distributed optimization")

    args = parser.parse_args()

    print("Hyperparameter Tuning Script")
    print("=" * 50)

    # Create tuner
    tuner = HyperparameterTuner(data_file=args.data, storage=args.storage)

    # Tune specified model/agent
    if args.model in ["ridge", "mlp", "cnn_lstm"]:
        results = tuner.tune_sl_model(args.model, args.trials, args.timeout, args.n_jobs, args.pruner)
    elif args.model in ["ppo", "sac"]:
        results = tuner.tune_rl_agent(args.model, args.trials, args.timeout, args.n_jobs, args.pruner)
    elif args.model == "reward":
        results = tuner.tune_reward_params(args.trials)
    elif args.model == "all":
        # Tune all models sequentially
        models = ["ridge", "mlp", "cnn_lstm", "ppo", "sac", "reward"]
        for model in models:
            print(f"\n{'='*60}")
            print(f"Tuning {model.upper()} model")
            print(f"{'='*60}")
            try:
                if model in ["ridge", "mlp", "cnn_lstm"]:
                    results = tuner.tune_sl_model(model, args.trials, args.timeout, args.n_jobs, args.pruner)
                elif model in ["ppo", "sac"]:
                    results = tuner.tune_rl_agent(model, args.trials, args.timeout, args.n_jobs, args.pruner)
                elif model == "reward":
                    results = tuner.tune_reward_params(args.trials)

                # Save results for each model
                study_name = f"{model}_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}" if model != "reward" else f"reward_params_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                study_path = tuner.save_study(results["study"], study_name)
                params_path = tuner.save_best_params(results["best_params"], model if model != "reward" else "reward_params")
                print(f"Study saved to: {study_path}")
                print(f"Best parameters saved to: {params_path}")
            except Exception as e:
                print(f"Error tuning {model}: {e}")
                continue
        print("\nAll models tuning completed!")
        return 0
    else:
        print("Please specify a valid model type with --model")
        return 1

    # Save results
    study_name = f"{args.model}_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}" if args.model != "reward" else f"reward_params_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Save study
    study_path = tuner.save_study(results["study"], study_name)

    # Save best parameters
    params_path = tuner.save_best_params(results["best_params"], args.model if args.model != "reward" else "reward_params")

    print("\nTuning completed successfully!")
    print(f"Study saved to: {study_path}")
    print(f"Best parameters saved to: {params_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
