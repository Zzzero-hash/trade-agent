"""PortfolioTradingEnvironment scaffold supporting multiple symbols (vectorized).

This is an early scaffold; integrates later with bridge multi-symbol outputs.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class PortfolioTradingEnvironment(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, data: dict[str, pd.DataFrame], window_size: int = 30,
                 initial_capital: float = 100000.0):
        super().__init__()
        self.symbols = list(data.keys())
        self.window_size = window_size
        self.initial_capital = initial_capital
        # Align lengths
        min_len = min(len(df) for df in data.values())
        self.data = {s: df.iloc[-min_len:] for s, df in data.items()}
        self.features = np.stack([
            self.data[s][['Close']].values.squeeze() for s in self.symbols
        ], axis=1)
        self.n_assets = len(self.symbols)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)
        obs_dim = self.n_assets * window_size + self.n_assets  # window + positions
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.reset()

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            np.random.seed(seed)
        self.positions = np.zeros(self.n_assets, dtype=np.float32)
        self.cash = self.initial_capital
        self.current_step = self.window_size
        return self._get_obs(), {}

    def _get_obs(self):
        start = self.current_step - self.window_size
        window = self.features[start:self.current_step]
        window = window if window.shape[0] == self.window_size else np.pad(
            window, ((self.window_size - window.shape[0], 0), (0, 0))
        )
        return np.concatenate([window.flatten(), self.positions]).astype(np.float32)

    def step(self, action: np.ndarray):
        action = np.clip(action, -1, 1)
        target_pos = action  # proportional positions
        self.features[self.current_step - 1]
        # Simple PnL reward
        pnl = np.dot(self.positions, self.features[self.current_step - 1] - self.features[self.current_step - 2]) if self.current_step > 1 else 0.0
        self.positions = target_pos
        self.current_step += 1
        obs = self._get_obs()
        terminated = self.current_step >= len(self.features)
        truncated = False
        return obs, float(pnl), terminated, truncated, {}
