"""Trading environment (migrated from trade_agent.agents.envs.trading_env)."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from trade_agent.agents.sl.models.base import set_all_seeds  # type: ignore
from trade_agent.envs.observation_schema import (
    compute_observation_schema,
    save_observation_schema,
)


__all__ = ["TradingEnvironment"]


class TradingEnvironment(gym.Env):  # type: ignore[type-arg]
    """Single‑asset trading environment using engineered feature windows."""

    def __init__(
        self,
        data_file: str = "data/features.parquet",
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,
        seed: int = 42,
        window_size: int = 30,
        reward_config: Mapping[str, float] | None = None,
        include_targets: bool = True,
        schema_report_path: str | None = None,
    ) -> None:
        super().__init__()
        self.seed = seed
        set_all_seeds(seed)

        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.window_size = window_size

        self.reward_config: dict[str, float] = dict(reward_config or {})
        self.pnl_weight = self.reward_config.get("pnl_weight", 1.0)
        self.transaction_cost_weight = self.reward_config.get(
            "transaction_cost_weight", 1.0
        )
        self.risk_adjustment_weight = self.reward_config.get(
            "risk_adjustment_weight", 0.0
        )
        self.stability_penalty_weight = self.reward_config.get(
            "stability_penalty_weight", 0.0
        )

        self._load_data(data_file)

        try:  # Observation schema validation (non‑fatal)
            df_preview = pd.read_parquet(data_file)
            obs_schema = compute_observation_schema(
                df_preview,
                window_size=window_size,
                include_targets=include_targets,
            )
            if include_targets and obs_schema.missing_targets:
                raise ValueError(
                    f"Missing target columns {obs_schema.missing_targets}"
                )
            if schema_report_path:
                save_observation_schema(obs_schema, schema_report_path)
        except Exception:  # pragma: no cover
            pass

        self.current_step = 0
        self.position = 0.0
        self.cash = initial_capital
        self.equity = initial_capital
        self.last_price = 0.0

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        obs_dim = self.window_size * self.n_features + 4
        self.observation_space = spaces.Box(
            low=-1e6, high=1e6, shape=(obs_dim,), dtype=np.float32
        )

        self.accounting_errors: list[str] = []

    # ---------------- Core Helpers ---------------- #
    def _load_data(self, data_file: str) -> None:
        df = pd.read_parquet(data_file)
    # Reconstruct synthetic close price if missing (cumulative log_returns)
        if 'close' not in df.columns and 'Close' not in df.columns:
            if 'log_returns' in df.columns:
                base_price = 100.0
                lr = df['log_returns'].fillna(0.0)
                df['close'] = np.exp(np.log(base_price) + lr.cumsum())
            else:  # final fallback
                df['close'] = 100.0
        target_cols = ["mu_hat", "sigma_hat"]
        # Synthesize target columns if missing (zero placeholders) so that
        # lightweight smoke tests using bare OHLCV data still function.
        for col in target_cols:
            if col not in df.columns:  # pragma: no cover (edge fallback)
                df[col] = 0.0
        feature_cols = [c for c in df.columns if c not in target_cols]
        self.features = df[feature_cols].values
        self.targets = df[target_cols].values
        price_series = (
            df['close'] if 'close' in df.columns else df.get('Close')
        )
        self.prices = (
            price_series.values
            if price_series is not None
            else np.ones(len(df)) * 100
        )
        self.dates = df.index.tolist()
        self.n_features = len(feature_cols)
        if len(self.features) < self.window_size:
            msg = (
                f"Data length ({len(self.features)}) < window size "
                f"({self.window_size})"
            )
            raise ValueError(msg)

    # ---------------- Gym API ---------------- #
    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.position = 0.0
        self.cash = self.initial_capital
        self.equity = self.initial_capital
        self.last_price = self.prices[self.current_step - 1]
        return self._get_observation(), self._get_info()

    def _get_observation(self) -> np.ndarray:
        feature_window = self.features[
            self.current_step - self.window_size : self.current_step
        ].flatten()
        mu_hat = self.targets[self.current_step, 0]
        sigma_hat = self.targets[self.current_step, 1]
        position = self.position
        cash_equity = self.cash / self.equity if self.equity else 0.0
        obs = np.concatenate(
            [feature_window, [mu_hat], [sigma_hat], [position], [cash_equity]]
        )
        return np.clip(
            obs, self.observation_space.low, self.observation_space.high
        ).astype(np.float32)

    def _get_info(self) -> dict[str, Any]:
        return {
            "step": self.current_step,
            "date": self.dates[self.current_step]
            if self.current_step < len(self.dates)
            else None,
            "position": self.position,
            "cash": self.cash,
            "equity": self.equity,
            "last_price": self.last_price,
            "accounting_errors": self.accounting_errors.copy(),
        }

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        target_exposure = action[0]

        current_price = self.prices[self.current_step]
        next_price = (
            self.prices[self.current_step + 1]
            if self.current_step + 1 < len(self.prices)
            else current_price
        )
        return_t = np.log(next_price / current_price)

        target_position_value = target_exposure * self.equity
        target_position_shares = (
            target_position_value / current_price if current_price > 0 else 0.0
        )

        position_change = target_position_shares - self.position
        abs_position_change = abs(position_change)

        transaction_costs = (
            self.transaction_cost * abs_position_change * current_price
        )

        self.position = target_position_shares
        self.cash -= position_change * current_price
        self.cash -= transaction_costs

        self.equity = self.cash + self.position * next_price

        pnl_reward = self.position * return_t
        cost_reward = (
            transaction_costs / self.equity if self.equity != 0 else 0.0
        )

        weighted_pnl = self.pnl_weight * pnl_reward
        weighted_cost = self.transaction_cost_weight * cost_reward

        reward = weighted_pnl - weighted_cost

        if self.risk_adjustment_weight > 0:
            risk_adjustment = abs(position_change) / (
                abs(self.position) + 1e-8
            )
            reward -= self.risk_adjustment_weight * risk_adjustment

        if self.stability_penalty_weight > 0:
            stability_penalty = (abs(position_change) / self.equity) ** 2
            reward -= self.stability_penalty_weight * stability_penalty

        self._check_accounting_invariants()

        self.current_step += 1
        self.last_price = current_price

        terminated = self.current_step >= len(self.features) - 1
        truncated = False

        if not terminated:
            observation = self._get_observation()
        else:
            observation = np.zeros(
                self.observation_space.shape, dtype=np.float32
            )

        info = self._get_info()
        return (
            observation,
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def _check_accounting_invariants(self) -> None:
        current_price = self.prices[self.current_step]
        calculated_equity = self.cash + self.position * current_price
        equity_mismatch = abs(calculated_equity - self.equity)
        if equity_mismatch > 1e-6:
            self.accounting_errors.append(
                "Equity mismatch at step "
                f"{self.current_step}: diff={equity_mismatch}"
            )

    def render(self) -> None:  # pragma: no cover
        pass

    def close(self) -> None:  # pragma: no cover
        pass
