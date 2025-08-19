"""Minimal continuous Box observation/action toy environment."""
from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import Env, spaces


class ToyTradingEnv(Env):  # type: ignore[type-arg]
    metadata = {"render.modes": []}

    def __init__(self, feature_dim: int = 6, seed: int | None = 42) -> None:
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.feature_dim = feature_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(feature_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.state = np.zeros(feature_dim, dtype=np.float32)
        self.t = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:  # type: ignore[override]
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.rng.normal(0, 0.01, size=self.feature_dim).astype(
            np.float32
        )
        self.t = 0
        return self.state.copy(), {}

    def step(
        self, action: np.ndarray | list[float] | float
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:  # type: ignore[override]
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        ret = self.rng.normal(0.0005, 0.01)
        reward = float(action[0] * ret)
        self.state += self.rng.normal(0, 0.01, size=self.feature_dim).astype(
            np.float32
        )
        self.t += 1
        terminated = self.t >= 50
        truncated = False
        return (
            self.state.copy(),
            float(reward),
            bool(terminated),
            bool(truncated),
            {},
        )

    def render(self) -> None:  # pragma: no cover
        pass
