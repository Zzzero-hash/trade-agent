#!/usr/bin/env python3
from __future__ import annotations


"""Clean PPO training script (recovered after corruption).

Features:
* Deterministic seeding via utils.set_seed
* Parallel envs (SubprocVecEnv) + evaluation env (DummyVecEnv)
* Optional reward normalization (VecNormalize)
* Basic training loop with EvalCallback
"""

import argparse
import json
import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
)


sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    ),
)
import contextlib

from trade_agent.envs.trading_env import TradingEnvironment  # noqa: E402
from trade_agent.utils import set_seed  # noqa: E402


class EntropyCoefficientCallback(BaseCallback):
    """Linearly anneal entropy coefficient if configured."""

    def __init__(
        self,
        initial_coef: float,
        final_coef: float,
        total_timesteps: int,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.initial = initial_coef
        self.final = final_coef
        self.total = max(total_timesteps, 1)

    def _on_step(self) -> bool:  # noqa: D401
        try:
            frac = min(1.0, self.num_timesteps / self.total)
            current = self.initial + (self.final - self.initial) * frac
            if hasattr(self.model, 'ent_coef'):
                self.model.ent_coef = current  # type: ignore[attr-defined]
        except Exception:
            pass
        return True


class PPOTrainer:
    """Trainer wrapping Stable-Baselines3 PPO with deterministic seeding."""

    def __init__(
        self,
        config_path: str | None = "configs/ppo_config.json",
        *,
        config: dict[str, Any] | None = None,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            if config_path is None:
                raise ValueError(
                    "config_path required when config not provided"
                )
            with open(config_path) as f:
                self.config = json.load(f)
        self.ppo_config = self.config.get('ppo', {})
        self.training_config = self.config.get('training', {})
        self.seed = int(self.ppo_config.get('seed', 42))
        self.deterministic = bool(self.ppo_config.get('deterministic', True))
        set_seed(self.seed, deterministic=self.deterministic)
        os.makedirs("models/rl", exist_ok=True)
        os.makedirs("reports", exist_ok=True)

    # ------------------ Environment creation ------------------
    def create_envs(
        self,
        data_file: str,
        n_envs: int = 4,
        initial_capital: float = 100_000.0,
        transaction_cost: float = 0.001,
        window_size: int = 30,
        validation_split: float = 0.2,
    ) -> tuple[Any, Any, pd.DataFrame, pd.DataFrame]:
        df = pd.read_parquet(data_file)
        split_index = int(len(df) * (1 - validation_split))
        train_data = df.iloc[:split_index]
        val_data = df.iloc[split_index:]
        train_path = "data/train_temp.parquet"
        val_path = "data/val_temp.parquet"
        train_data.to_parquet(train_path)
        val_data.to_parquet(val_path)

        def make_env(rank: int, base_seed: int) -> Any:
            def _init() -> TradingEnvironment:
                return TradingEnvironment(
                    data_file=train_path,
                    initial_capital=initial_capital,
                    transaction_cost=transaction_cost,
                    seed=base_seed + rank,
                    window_size=window_size,
                )
            return _init

        train_env = SubprocVecEnv(
            [make_env(i, self.seed) for i in range(n_envs)]
        )
        set_seed(self.seed, env=train_env, deterministic=self.deterministic)
        reward_cfg = self.ppo_config.get('reward_scaling', {})
        clip_r = reward_cfg.get('clip_range', 10.0)
        train_env = VecNormalize(
            train_env,
            norm_obs=False,
            norm_reward=True,
            clip_reward=clip_r,
        )

        def make_eval_env() -> TradingEnvironment:
            return TradingEnvironment(
                data_file=val_path,
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
                seed=self.seed + 1000,
                window_size=window_size,
            )

        eval_env = DummyVecEnv([make_eval_env])
        set_seed(
            self.seed + 1000,
            env=eval_env,
            deterministic=self.deterministic,
        )
        eval_env = VecNormalize(
            eval_env,
            norm_obs=False,
            norm_reward=False,
            training=False,
        )
        return train_env, eval_env, train_data, val_data

    # ------------------ Model setup ------------------
    def setup_model(
        self, env: Any, hyperparameters: dict[str, Any] | None = None
    ) -> PPO:
        hp: dict[str, Any] = {**self.ppo_config, **(hyperparameters or {})}
        return PPO(
            'MlpPolicy',
            env,
            learning_rate=hp.get('learning_rate', 3e-4),
            n_steps=hp.get('n_steps', 2048),
            batch_size=hp.get('batch_size', 64),
            n_epochs=hp.get('n_epochs', 10),
            gamma=hp.get('gamma', 0.99),
            gae_lambda=hp.get('gae_lambda', 0.95),
            clip_range=hp.get('clip_range', 0.2),
            ent_coef=hp.get('ent_coef', 0.0),
            vf_coef=hp.get('vf_coef', 0.5),
            max_grad_norm=hp.get('max_grad_norm', 0.5),
            seed=self.seed,
            verbose=1,
            tensorboard_log='./logs/ppo/',
            device=hp.get('device', 'cpu'),
        )

    def setup_callbacks(
        self, eval_env: Any, total_timesteps: int
    ) -> list[Any]:
        eval_freq = int(self.training_config.get('eval_freq', 10_000))
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path='./models/rl/',
            log_path='./logs/ppo/',
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            verbose=1,
        )
        ent_cb = EntropyCoefficientCallback(
            initial_coef=self.ppo_config.get('ent_coef', 0.05),
            final_coef=self.ppo_config.get('ent_coef_final', 0.01),
            total_timesteps=total_timesteps,
        )
        return [eval_cb, ent_cb]

    # ------------------ Plotting ------------------
    def plot_learning_curves(
        self, save_path: str = 'reports/ppo_learning.png'
    ) -> None:
        # Placeholder synthetic plot (no real metrics captured here)
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(10)
        ax.plot(x, np.random.randn(10).cumsum())
        ax.set_title('PPO Training (placeholder)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Reward (synthetic)')
        fig.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

    # ------------------ Training ------------------
    def train(
        self,
        data_file: str = 'data/features.parquet',
        n_envs: int = 4,
        total_timesteps: int | None = None,
        initial_capital: float = 100_000.0,
        transaction_cost: float = 0.001,
        window_size: int = 30,
    ) -> PPO:
        train_env, eval_env, _, _ = self.create_envs(
            data_file,
            n_envs,
            initial_capital,
            transaction_cost,
            window_size,
        )
        if total_timesteps is None:
            total_timesteps = int(
                self.training_config.get('total_timesteps', 100_000)
            )
        callbacks = self.setup_callbacks(eval_env, total_timesteps)
        model = self.setup_model(train_env)
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=False,
        )
        model.save('models/rl/ppo_final.zip')
        self.plot_learning_curves()
        meta = {
            'training_config': self.training_config,
            'ppo_config': self.ppo_config,
        }
        with open('reports/ppo_training_metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)
        # cleanup tmp
        for fp in ('data/train_temp.parquet', 'data/val_temp.parquet'):
            with contextlib.suppress(FileNotFoundError):
                os.remove(fp)
        return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/ppo_config.json')
    parser.add_argument('--data', default='data/features.parquet')
    parser.add_argument('--n-envs', type=int, default=4)
    parser.add_argument('--timesteps', type=int, default=None)
    parser.add_argument('--capital', type=float, default=100_000.0)
    parser.add_argument('--cost', type=float, default=0.001)
    parser.add_argument('--window', type=int, default=30)
    parser.add_argument('--no-deterministic', action='store_true')
    args = parser.parse_args()
    override = None
    if args.no_deterministic:
        try:
            with open(args.config) as f:
                override = json.load(f)
            override.setdefault('ppo', {})['deterministic'] = False
        except Exception:
            override = None
    trainer = PPOTrainer(args.config, config=override)
    try:
        trainer.train(
            data_file=args.data,
            n_envs=args.n_envs,
            total_timesteps=args.timesteps,
            initial_capital=args.capital,
            transaction_cost=args.cost,
            window_size=args.window,
        )
    except Exception:
        # keep exit code non-zero for CI visibility
        sys.exit(1)


if __name__ == '__main__':  # pragma: no cover
    main()
