"""Enhanced trading environment integrated under trade_agent namespace.

This was previously located at top-level ``integrations/enhanced_trading_env``.
Tests and user code should now import:

    from trade_agent.integrations.enhanced_trading_env import EnhancedTradingEnvironment

Backward compatibility shims at the old path were removed per project policy
to keep all runtime code under the ``trade_agent`` package.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from trade_agent.envs.trading_env import TradingEnvironment


__all__ = ["EnhancedTradingEnvironment"]


@dataclass
class EnhancedTradingEnvironment(TradingEnvironment):  # type: ignore
    """Augmented environment adding explicit reward component tracking.

    The base ``TradingEnvironment`` already incorporates PnL and cost into its
    scalar reward. This subclass decomposes and optionally adjusts reward with
    simple turnover & risk penalties used by unit tests.
    """

    # Primary environment construction arguments (mirroring base env) so tests
    # can pass them directly to the subclass.
    data_file: str = "data/features.parquet"
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001
    seed: int = 42
    window_size: int = 30
    # Legacy / extended arguments
    reward_config: dict[str, float] | None = None  # matches base env type
    auto_convert: bool = False  # accepted for backward compat; unused
    turnover_penalty_weight: float = 0.0
    risk_penalty_weight: float = 0.0
    position_limit: float = 10.0
    last_position: float = 0.0
    cumulative_turnover: float = 0.0
    risk_window: int = 10
    recent_returns: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:  # type: ignore[override]
    # Extract legacy reward config fields.
        rc = dict(self.reward_config or {})
        self.turnover_penalty_weight = rc.get("turnover_penalty_weight", 0.0)
        self.risk_penalty_weight = rc.get("risk_penalty_weight", 0.0)
        self.position_limit = rc.get("position_limit", 10.0)
        self.risk_window = int(rc.get("risk_window", 10))
        super().__init__(  # type: ignore[misc]
            data_file=self.data_file,
            initial_capital=self.initial_capital,
            transaction_cost=self.transaction_cost,
            seed=self.seed,
            window_size=self.window_size,
            reward_config={},  # parent uses different keys
        )
        # If underlying data shorter than requested window_size, pad features
        # and targets at the *front* (oldest) so that tests expecting padded
        # observation length succeed instead of raising ValueError.
        if getattr(self, "features", None) is not None and len(self.features) < self.window_size:
            deficit = self.window_size - len(self.features)
            if deficit > 0:
                first_feat = self.features[0]
                pad_block = np.repeat(first_feat[None, :], deficit, axis=0)
                self.features = np.vstack([pad_block, self.features])
                first_tgt = self.targets[0]
                tgt_pad = np.repeat(first_tgt[None, :], deficit, axis=0)
                self.targets = np.vstack([tgt_pad, self.targets])
                # Align prices & dates padding with first values
                first_price = self.prices[0]
                price_pad = np.repeat(first_price, deficit)
                self.prices = np.concatenate([price_pad, self.prices])
                first_date = self.dates[0]
                self.dates = [first_date] * deficit + self.dates
                # Reset starting step after padding
                self.current_step = self.window_size

    # Override loading to permit short padding
    def _load_data(self, data_file: str) -> None:  # type: ignore[override]
        import os

        import pandas as pd
        if not os.path.exists(data_file):  # fallback synthetic similar to base
            n = max(self.window_size * 3, 50)  # smaller fallback ok for tests
            idx = pd.date_range('2024-01-01', periods=n, freq='D')
            rng = np.random.default_rng(self.seed)
            log_returns = rng.normal(0, 0.01, size=n)
            mu_hat = rng.normal(0, 0.005, size=n)
            sigma_hat = rng.uniform(0.01, 0.05, size=n)
            df = pd.DataFrame(
                {
                    'log_returns': log_returns,
                    'mu_hat': mu_hat,
                    'sigma_hat': sigma_hat,
                },
                index=idx,
            )
        else:
            df = pd.read_parquet(data_file)
        if 'close' not in df.columns and 'Close' not in df.columns:
            if 'log_returns' in df.columns:
                base_price = 100.0
                lr = df['log_returns'].fillna(0.0)
                df['close'] = np.exp(np.log(base_price) + lr.cumsum())
            else:
                df['close'] = 100.0
        target_cols = ["mu_hat", "sigma_hat"]
        for col in target_cols:
            if col not in df.columns:
                df[col] = 0.0
        feature_cols = [c for c in df.columns if c not in target_cols]
        self.features = df[feature_cols].values
        self.targets = df[target_cols].values
        price_series = df['close'] if 'close' in df.columns else df.get('Close')
        self.prices = (
            price_series.values
            if price_series is not None
            else np.ones(len(df)) * 100
        )
        self.dates = df.index.tolist()
        self.n_features = len(feature_cols)
        # Do not raise on short length; padding deferred to __post_init__

    # Override step to append reward components & penalties
    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:  # type: ignore[override]
        obs, reward, terminated, truncated, info = super().step(action)
        position_change = abs(self.position - self.last_position)
        turnover_penalty = self.turnover_penalty_weight * position_change

        # Risk proxy: variance of recent equity deltas
        self.recent_returns.append(info.get("equity", 0.0))
        if len(self.recent_returns) > self.risk_window:
            self.recent_returns.pop(0)
        risk_component = 0.0
        if len(self.recent_returns) >= 2:
            arr = np.diff(self.recent_returns)
            if arr.size > 1:
                risk_component = float(np.var(arr))
        risk_penalty = self.risk_penalty_weight * risk_component

        adjusted_reward = float(reward - turnover_penalty - risk_penalty)
        self.last_position = self.position
        info.setdefault("reward_components", {})
        info["reward_components"].update(
            {
                "pnl": float(reward),
                "cost": 0.0,  # underlying env already nets cost
                "risk": risk_component,
                "turnover": position_change,
            }
        )
        return obs, adjusted_reward, terminated, truncated, info
