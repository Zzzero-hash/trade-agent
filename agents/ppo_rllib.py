"""PPO agent (RLlib) unified interface with lazy trainer init."""

# mypy: ignore-errors

from collections.abc import Callable
from typing import Any

import numpy as np

from agents.base import Agent


class PPOAgent(Agent):
    def __init__(
        self,
        config: dict[str, Any],
        env_creator: Callable[..., Any],
    ) -> None:
        self.config = config
        self.env_creator = env_creator
        self.trainer: Any | None = None
        self.env_name = config.get("env_name", "UnifiedTradingEnv")

    def _ensure_trainer(self) -> None:
        if self.trainer is not None:
            return
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig  # type: ignore

        if not ray.is_initialized():  # pragma: no cover
            ray.init(
                ignore_reinit_error=True,
                include_dashboard=False,
                local_mode=True,
                num_cpus=1,
            )

        from ray.tune.registry import register_env

        register_env(
            self.env_name, lambda _cfg: self.env_creator(self.config)
        )
        cfg = (
            PPOConfig()
            .environment(self.env_name)
            .framework("torch")
            .training(
                train_batch_size=32,
                minibatch_size=16,
                num_epochs=1,
            )
            .resources(num_gpus=0)
        )
        self.trainer = cfg.build()

    def fit(self, data: Any | None = None) -> None:  # type: ignore[override]
        self._ensure_trainer()
        for _ in range(int(self.config.get("training_iterations", 1))):
            self.trainer.train()

    def predict(self, obs: np.ndarray) -> np.ndarray:
        if self.trainer is None:
            raise RuntimeError("Agent not trained")
        # RLlib may expect batch dimension; ensure flat obs
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim == 1:
            arr_in = arr
        else:  # pragma: no cover - defensive
            arr_in = arr.reshape(-1)
        try:
            act = self.trainer.compute_single_action(
                arr_in
            )
        except AttributeError:  # pragma: no cover - fallback
            act = self.trainer.compute_single_action(arr_in)
        return np.atleast_1d(act)

    def save(self, path: str) -> None:
        if self.trainer is None:
            raise RuntimeError("Nothing to save")
        self.trainer.save(path)

    def load(self, path: str) -> None:
        self._ensure_trainer()
        if self.trainer is None:  # pragma: no cover - defensive
            raise RuntimeError("Trainer not initialised")
        self.trainer.restore(path)
