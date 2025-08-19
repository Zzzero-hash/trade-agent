"""SAC agent implementation providing unified interface.

The configuration dictionary should include at minimum keys compatible with
Stable-Baselines3 SAC initialisation; missing values fall back to sensible
defaults for unit tests (very small buffers / timesteps).
"""

from __future__ import annotations

# mypy: ignore-errors
import os
from collections.abc import Callable
from typing import Any

import numpy as np
from gym import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)

from agents.base import Agent


class SACAgent(Agent):
    """Soft Actor-Critic agent (Stable-Baselines3 backend)."""

    def __init__(
        self,
        config: dict[str, Any],
        env_creator: Callable[[dict[str, Any]], Any],
    ) -> None:
        self.config = config
        self.env_creator = env_creator
        self.env = env_creator(config)
        if not hasattr(self.env, "observation_space") or not hasattr(
            self.env, "action_space"
        ):
            raise AttributeError(
                "Environment missing observation_space or action_space"
            )
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        # Extract hyperparameters (small defaults for quick tests)
        g = self.config.get
        self.policy_kwargs = g("policy_kwargs", {})
        self.learning_rate = g("learning_rate", 3e-4)
        self.buffer_size = g("buffer_size", 5000)
        self.learning_starts = g("learning_starts", 10)
        self.batch_size = g("batch_size", 64)
        self.tau = g("tau", 0.005)
        self.gamma = g("gamma", 0.99)
        self.train_freq = g("train_freq", 1)
        self.gradient_steps = g("gradient_steps", 1)
        self.ent_coef = g("ent_coef", "auto")
        self.target_update_interval = g("target_update_interval", 1)
        self.target_entropy = g("target_entropy", "auto")
        self.use_sde = g("use_sde", False)
        self.sde_sample_freq = g("sde_sample_freq", -1)
        self.use_sde_at_warmup = g("use_sde_at_warmup", False)
        self.tensorboard_log = g("tensorboard_log", None)
        self.create_eval_env = g("create_eval_env", False)
        self.eval_freq = g("eval_freq", 200)
        self.n_eval_episodes = g("n_eval_episodes", 2)
        self.log_path = g("log_path", "./logs/")
        self.verbose = g("verbose", 0)
        self.seed = g("seed", None)
        self.device = g("device", "cpu")
        self.n_timesteps = g("n_timesteps", 100)
        self.model: SAC | None = None

    def fit(self, data: Any | None = None) -> None:  # type: ignore[override]
        if isinstance(self.action_space, spaces.Discrete):  # pragma: no cover
            raise ValueError("SAC requires continuous action space")
        self.model = SAC(
            "MlpPolicy",
            self.env,
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            learning_starts=self.learning_starts,
            batch_size=self.batch_size,
            tau=self.tau,
            gamma=self.gamma,
            train_freq=self.train_freq,
            gradient_steps=self.gradient_steps,
            ent_coef=self.ent_coef,
            target_update_interval=self.target_update_interval,
            target_entropy=self.target_entropy,
            use_sde=self.use_sde,
            sde_sample_freq=self.sde_sample_freq,
            use_sde_at_warmup=self.use_sde_at_warmup,
            tensorboard_log=self.tensorboard_log,
            policy_kwargs=self.policy_kwargs,
            verbose=self.verbose,
            seed=self.seed,
            device=self.device,
        )
        callbacks: list[Any] = []  # Ensure proper formatting
        if self.create_eval_env:
            eval_env = self.env_creator(self.config)
            stop_cb = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=1,
                min_evals=1,
                verbose=False,
            )
            callbacks.append(
                EvalCallback(
                    eval_env,
                    best_model_save_path=os.path.join(
                        self.log_path, "best_model"
                    ),
                    log_path=self.log_path,
                    eval_freq=self.eval_freq,
                    n_eval_episodes=self.n_eval_episodes,
                    deterministic=True,
                    render=False,
                    callback_after_eval=stop_cb,
                )
            )
        self.model.learn(  # type: ignore[attr-defined]
            total_timesteps=self.n_timesteps,
            callback=callbacks,
            log_interval=50,
        )

    def predict(self, obs: np.ndarray) -> np.ndarray:
        if self.model is None:  # pragma: no cover - defensive
            raise RuntimeError("Model not trained")
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    def save(self, path: str) -> None:
        if self.model is None:  # pragma: no cover - defensive
            raise RuntimeError("Nothing to save")
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model = SAC.load(path, env=self.env)  # type: ignore[attr-defined]
