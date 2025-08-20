"""Hybrid policy agent: supervised head + simple adjustment actor.

Compact module kept intentionally small for fast unit tests. Provides the
unified Agent interface.
"""

from __future__ import annotations

from collections.abc import Callable

# mypy: ignore-errors
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from trade_agent.agents.base import Agent


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.net(x)


class HybridPolicyAgent(Agent):
    def __init__(
        self,
        config: dict[str, Any],
        env_creator: Callable[[dict[str, Any]], Any],
    ) -> None:
        self.config = config
        self.env = env_creator(config)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        input_dim = int(np.prod(self.observation_space.shape))
        hidden_dim = config.get("hidden_dim", 32)
        self.lr = float(config.get("lr", 1e-3))
        self.sl_head = _MLP(input_dim, hidden_dim)
        self.actor = nn.Linear(input_dim, 1)
        params = list(self.sl_head.parameters()) + list(self.actor.parameters())
        self.optim = torch.optim.Adam(params, lr=self.lr)
        self.trained = False

    def fit(self, data: Any | None = None) -> None:  # type: ignore[override]
        obs = np.random.randn(
            8, int(np.prod(self.observation_space.shape))
        ).astype(np.float32)
        target = np.tanh(obs.sum(axis=1, keepdims=True))
        x = torch.from_numpy(obs)
        y = torch.from_numpy(target)
        loss = ((self.sl_head(x) - y) ** 2).mean()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.trained = True

    def _flatten_obs(self, obs: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(obs.reshape(1, -1).astype(np.float32))

    def predict(self, obs: np.ndarray) -> np.ndarray:
        if not self.trained:
            raise RuntimeError("Model not trained")
        with torch.no_grad():
            x = self._flatten_obs(obs)
            action = (
                torch.tanh(self.sl_head(x) + self.actor(x))
                .cpu()
                .numpy()
                .ravel()
            )
        return np.clip(action, self.action_space.low, self.action_space.high)

    def save(self, path: str) -> None:
        torch.save(
            {
                "sl_head": self.sl_head.state_dict(),
                "actor": self.actor.state_dict(),
                "config": self.config,
            },
            path,
        )

    def load(self, path: str) -> None:
        state = torch.load(path, map_location="cpu")
        self.sl_head.load_state_dict(
            state["sl_head"]
        )  # type: ignore[arg-type]
        self.actor.load_state_dict(state["actor"])  # type: ignore[arg-type]
        self.trained = True
