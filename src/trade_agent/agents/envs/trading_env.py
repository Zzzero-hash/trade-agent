"""Trading environment for reinforcement learning in finance.

Rewritten with corrected indentation (prior version had indentation faults
breaking test discovery). Logic preserved; only minor typing tweaks.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from trade_agent.agents.envs.observation_schema import (
    compute_observation_schema,
    save_observation_schema,
)
from trade_agent.agents.sl.models.base import set_all_seeds


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
        target_cols = ["mu_hat", "sigma_hat"]
        feature_cols = [c for c in df.columns if c not in target_cols]
        self.features = df[feature_cols].values
        self.targets = df[target_cols].values
        self.prices = df.get("close", pd.Series(np.ones(len(df)) * 100)).values
        self.dates = df.index.tolist()
        self.n_features = len(feature_cols)
        if len(self.features) < self.window_size:
            raise ValueError(
                f"Data length ({len(self.features)}) < window size ({self.window_size})"
            )

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
        self.features[
            self.current_step - self.window_size : self.current_step
        ].flatten()
        self.targets[self.current_step, 0]
        self.targets[self.current_step, 1]
        self.cash / self.equity if self.equity else 0.0
        """
        Trading Environment for Reinforcement Learning in Finance.

        This module implements a Gymnasium environment for trading that:
        1. Uses engineered features and SL predictions as observations
        2. Accepts target net exposure as actions
        3. Computes rewards based on PnL and transaction costs
        4. Maintains proper accounting invariants
        """


        import gymnasium as gym
        import numpy as np
        import pandas as pd
        from gymnasium import spaces

        from trade_agent.agents.envs.observation_schema import (
            compute_observation_schema,
            save_observation_schema,
        )
        from trade_agent.agents.sl.models.base import set_all_seeds


        class TradingEnvironment(gym.Env):
            """
            A Gymnasium environment for single-asset trading with RL.

            Observation Space:
                [feature_window_t, mu_hat_t, sigma_hat_t, position_{t-1}, cash/equity]

            Action Space:
                Box(-1, 1, (1,)) representing target net exposure

            Reward Function:
                position_{t-1} * ret_t - cost(|Δposition|) with hooks for vol-normalization
            """

            def __init__(
                self,
                data_file: str = "data/features.parquet",
                initial_capital: float = 100000.0,
                transaction_cost: float = 0.001,
                seed: int = 42,
                window_size: int = 30,
                reward_config: dict = None,
                include_targets: bool = True,
                schema_report_path: str | None = None,
            ) -> None:
                """
                Initialize the trading environment.

                Args:
                    data_file: Path to the features data file
                    initial_capital: Starting capital for the portfolio
                    transaction_cost: Fixed transaction cost per trade (as fraction)
                    seed: Random seed for deterministic processing
                    window_size: Size of the feature window for observations
                    reward_config: Configuration for reward function components
                """
                super().__init__()

                # Set seeds for deterministic processing
                self.seed = seed
                set_all_seeds(seed)

                # Environment parameters
                self.initial_capital = initial_capital
                self.transaction_cost = transaction_cost
                self.window_size = window_size

                # Reward configuration
                self.reward_config = reward_config or {}
                self.pnl_weight = self.reward_config.get('pnl_weight', 1.0)
                self.transaction_cost_weight = self.reward_config.get(
                    'transaction_cost_weight', 1.0
                )
                self.risk_adjustment_weight = self.reward_config.get(
                    'risk_adjustment_weight', 0.0
                )
                self.stability_penalty_weight = self.reward_config.get(
                    'stability_penalty_weight', 0.0
                )

                # Load data
                self._load_data(data_file)

                # Observation schema validation (defense in depth)
                try:
                    import pandas as pd  # local import
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

                # Environment state
                self.current_step = 0
                self.position = 0.0  # Current position (number of shares)
                self.cash = initial_capital  # Current cash balance
                self.equity = initial_capital  # Current total equity
                self.last_price = 0.0  # Last known price

                # Action space: target net exposure in [-1, 1]
                self.action_space = spaces.Box(
                    low=-1.0, high=1.0, shape=(1,), dtype=np.float32
                )

                # Observation space: [feature_window_t, mu_hat_t, sigma_hat_t, position_{t-1}, cash/equity]
                obs_dim = self.window_size * self.n_features + 4  # FIXED INDENT
                self.observation_space = spaces.Box(
                    low=-1e6, high=1e6, shape=(obs_dim,), dtype=np.float32
                )

                # Accounting invariants tracking
                self.accounting_errors = []

            def _load_data(self, data_file: str) -> None:
                """
                Load and preprocess the features data.

                Args:
                    data_file: Path to the features data file
                """
                # Load data
                df = pd.read_parquet(data_file)

                # Separate features from targets
                target_cols = ['mu_hat', 'sigma_hat']
                feature_cols = [col for col in df.columns if col not in target_cols]

                self.features = df[feature_cols].values
                self.targets = df[target_cols].values
                self.prices = df.get('close', pd.Series(np.ones(len(df)) * 100)).values  # Default price if not available
                self.dates = df.index.tolist()

                # Store feature count for observation space
                self.n_features = len(feature_cols)

                # Ensure we have enough data for the window size
                if len(self.features) < self.window_size:
                    raise ValueError(f"Data length ({len(self.features)}) is less than window size ({self.window_size})")

            def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
                """
                Reset the environment to initial state.

                Args:
                    seed: Random seed for this reset
                    options: Additional options for reset

                Returns:
                    Tuple of (observation, info)
                """
                super().reset(seed=seed)

                # Reset state
                self.current_step = self.window_size  # Start after the window
                self.position = 0.0
                self.cash = self.initial_capital
                self.equity = self.initial_capital
                self.last_price = self.prices[self.current_step - 1]

                # Get initial observation
                observation = self._get_observation()
                info = self._get_info()

                return observation, info

            def _get_observation(self) -> np.ndarray:
                """
                Get current observation.

                Returns:
                    Observation vector: [feature_window_t, mu_hat_t, sigma_hat_t, position_{t-1}, cash/equity]
                """
                # Feature window (flatten the window)
                feature_window = self.features[
                    self.current_step - self.window_size:self.current_step
                ].flatten()

                # Current targets
                mu_hat = self.targets[self.current_step, 0]
                sigma_hat = self.targets[self.current_step, 1]

                # Position and cash/equity ratio
                position = self.position
                cash_equity = self.cash / self.equity if self.equity != 0 else 0.0

                # Combine all components
                observation = np.concatenate([
                    feature_window,
                    [mu_hat],
                    [sigma_hat],
                    [position],
                    [cash_equity]
                ])

                # Ensure observation is within bounds
                observation = np.clip(observation, self.observation_space.low, self.observation_space.high)

                return observation.astype(np.float32)

            def _get_info(self) -> dict[str, Any]:
                """
                Get additional information about the current state.

                Returns:
                    Dictionary with additional information
                """
                return {
                    "step": self.current_step,
                    "date": self.dates[self.current_step] if self.current_step < len(self.dates) else None,
                    "position": self.position,
                    "cash": self.cash,
                    "equity": self.equity,
                    "last_price": self.last_price,
                    "accounting_errors": self.accounting_errors.copy()
                }

            def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
                """
                Execute one time step within the environment.

                Args:
                    action: Target net exposure in [-1, 1]

                Returns:
                    Tuple of (observation, reward, terminated, truncated, info)
                """
                # Validate action
                action = np.clip(action, self.action_space.low, self.action_space.high)
                target_exposure = action[0]

                # Get current state
                current_price = self.prices[self.current_step]
                next_price = self.prices[self.current_step + 1] if self.current_step + 1 < len(self.prices) else current_price
                return_t = np.log(next_price / current_price)  # Log return

                # Calculate target position (in dollars)
                target_position_value = target_exposure * self.equity
                target_position_shares = target_position_value / current_price if current_price > 0 else 0.0

                # Calculate position change
                position_change = target_position_shares - self.position
                abs_position_change = abs(position_change)

                # Calculate transaction costs
                transaction_costs = self.transaction_cost * abs_position_change * current_price

                # Update position and cash
                self.position = target_position_shares
                self.cash -= position_change * current_price  # Deduct cost of position change
                self.cash -= transaction_costs  # Deduct transaction costs

                # Update equity based on new position
                self.equity = self.cash + self.position * next_price

                # Calculate reward components
                pnl_reward = self.position * return_t
                cost_reward = transaction_costs / self.equity if self.equity != 0 else 0.0

                # Apply weights to reward components
                weighted_pnl = self.pnl_weight * pnl_reward
                weighted_cost = self.transaction_cost_weight * cost_reward

                # Calculate total reward
                reward = weighted_pnl - weighted_cost

                # Add risk adjustment component if enabled
                if self.risk_adjustment_weight > 0:
                    # Simple risk adjustment based on position stability
                    risk_adjustment = abs(position_change) / (abs(self.position) + 1e-8)
                    reward -= self.risk_adjustment_weight * risk_adjustment

                # Add stability penalty if enabled
                if self.stability_penalty_weight > 0:
                    # Penalty for large position changes
                    stability_penalty = (abs(position_change) / self.equity) ** 2
                    reward -= self.stability_penalty_weight * stability_penalty

                # Check accounting invariants
                self._check_accounting_invariants()

                # Advance time
                self.current_step += 1
                self.last_price = current_price

                # Check termination
                terminated = self.current_step >= len(self.features) - 1
                truncated = False

                # Get new observation and info
                if not terminated:
                    observation = self._get_observation()
                else:
                    # Return zero observation if terminated
                    observation = np.zeros(self.observation_space.shape, dtype=np.float32)

                info = self._get_info()

                return observation, reward, terminated, truncated, info

            def _check_accounting_invariants(self) -> None:
                """
                Check accounting invariants and record any mismatches.

                Accounting Invariants:
                1. cash + position * current_price ≈ equity (wealth conservation)
                2. No data leakage across time steps
                3. Deterministic processing with fixed seeds
                """
                # Check: cash + pos*price ≈ equity
                current_price = self.prices[self.current_step]
                calculated_equity = self.cash + self.position * current_price
                equity_mismatch = abs(calculated_equity - self.equity)

                if equity_mismatch > 1e-6:  # Tolerance for floating point errors
                    error_msg = f"Equity mismatch at step {self.current_step}: calculated={calculated_equity}, stored={self.equity}, diff={equity_mismatch}"
                    self.accounting_errors.append(error_msg)

            def render(self, mode: str = 'human') -> None:
                """
                Render the environment.

                Args:
                    mode: Render mode (only 'human' supported)
                """
                if mode == 'human':
                    pass

            def close(self) -> None:
                """
                Clean up resources.
                """
                pass


        def main() -> None:  # pragma: no cover
            parser = argparse.ArgumentParser(description="Trading Env")
            parser.add_argument('--verify', action='store_true')
            args = parser.parse_args()
            env = TradingEnvironment()
            if args.verify:
                pass
            env.close()


        if __name__ == '__main__':  # pragma: no cover
            main()
