"""
Enhanced Trading Environment: Flexible data loading with bridge integration.

This enhanced version can work with both traditional features.parquet files
and our new data pipeline outputs via the bridge component.
"""

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


# Add proper path handling
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from trade_agent.agents.sl.models.base import set_all_seeds
except ImportError:
    # Fallback if sl module not available
    def set_all_seeds(seed) -> None:
        np.random.seed(seed)


class EnhancedTradingEnvironment(gym.Env):
    """
    Enhanced trading environment that can work with multiple data formats.

    Supports both:
    - Traditional features.parquet with mu_hat/sigma_hat
    - Bridge-converted data pipeline outputs
    """

    def __init__(
        self,
        data_file: str = "data/features.parquet",
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,
        seed: int = 42,
        window_size: int = 30,
        reward_config: dict = None,
        auto_convert: bool = True
    ) -> None:
        """
        Initialize enhanced trading environment.

        Args:
            data_file: Path to data file
            initial_capital: Starting capital
            transaction_cost: Transaction cost rate
            seed: Random seed
            window_size: Observation window size
            reward_config: Reward function configuration
            auto_convert: Automatically convert data if needed
        """
        super().__init__()

        self.data_file = data_file
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.window_size = window_size
        self.auto_convert = auto_convert

        # Set random seed
        set_all_seeds(seed)
        np.random.seed(seed)

        # Initialize reward configuration
        self.reward_config = reward_config or {}
        self.position_limit = self.reward_config.get('position_limit', 2.0)
        self.transaction_cost_weight = self.reward_config.get(
            'transaction_cost_weight', 1.0
        )
        self.pnl_weight = self.reward_config.get('pnl_weight', 1.0)
        self.risk_penalty_weight = self.reward_config.get(
            'risk_penalty_weight', 0.0
        )
        self.turnover_penalty_weight = self.reward_config.get(
            'turnover_penalty_weight', 0.0
        )
        # Rolling window for risk (price return volatility)
        self.risk_window = int(self.reward_config.get('risk_window', 20))

        # Load and prepare data
        self._load_and_prepare_data(data_file)

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Dynamic observation space based on available features
        obs_dim = self._calculate_observation_dimension()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Initialize state
        self.reset()

    def _load_and_prepare_data(self, data_file: str) -> None:
        """Load and prepare data, auto-converting if necessary."""

        # Load data
        df = pd.read_parquet(data_file)

        # Check if we have required SL predictions
        required_cols = ['mu_hat', 'sigma_hat']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols and self.auto_convert:
            df = self._auto_convert_data(df)
        elif missing_cols:
            raise ValueError(
                "Missing required columns: "
                f"{missing_cols}. Set auto_convert=True to fix."
            )

    # Prepare feature columns (exclude targets & non-numeric)
        target_cols = ['mu_hat', 'sigma_hat']
        exclude_cols = ['symbol', 'date', 'timestamp']
        feature_cols = [
            col for col in df.columns
            if col not in target_cols and col not in exclude_cols
        ]

        # Feature columns summary
        feature_count = len(feature_cols)
        feature_cols[:5]
        if feature_count > 5:
            pass
        else:
            pass
        [c for c in exclude_cols if c in df.columns]

        # Ensure numeric data types
        for col in feature_cols + target_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col], errors='coerce'
                ).astype(np.float32)
        df = df.fillna(0.0)

        # Store data
        self.features = df[feature_cols].values.astype(np.float32)
        self.targets = df[target_cols].values.astype(np.float32)

        # Handle price data
        if 'Close' in df.columns:
            self.prices = df['Close'].values.astype(np.float32)
        elif 'close' in df.columns:
            self.prices = df['close'].values.astype(np.float32)
        else:
            # Default prices if not available
            self.prices = np.ones(len(df), dtype=np.float32) * 100

        # Handle dates if available
        if hasattr(df.index, 'date') or isinstance(df.index, pd.DatetimeIndex):
            self.dates = df.index
        else:
            self.dates = pd.date_range('2024-01-01', periods=len(df), freq='D')


    def _auto_convert_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Auto-convert pipeline data to trading format."""
        from integrations.data_trading_bridge import DataTradingBridge

        # Save current data to temp file
        temp_file = "temp_pipeline_data.parquet"
        df.to_parquet(temp_file)

        # Convert using bridge
        bridge = DataTradingBridge()
        converted_file = bridge.convert_pipeline_output_to_trading_format(
            temp_file
        )

        # Load converted data
        converted_df = pd.read_parquet(converted_file)

        # Clean up temp file
        Path(temp_file).unlink(missing_ok=True)

        return converted_df

    def _calculate_observation_dimension(self) -> int:
        """Calculate observation dimension based on available features."""
        # Base features from window
        feature_dim = self.features.shape[1] * self.window_size if len(self.features) > 0 else 100

        # Add current predictions (mu_hat, sigma_hat)
        prediction_dim = 2

        # Add position and cash/equity ratio
        state_dim = 2

        return feature_dim + prediction_dim + state_dim


    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)

        # Reset financial state
        self.cash = self.initial_capital
        self.position = 0.0
        self.equity = self.initial_capital
        self.last_price = (
            float(self.prices[self.window_size])
            if len(self.prices) > self.window_size
            else float(self.prices[-1])
        )
        self._price_return_history = []  # rolling list for volatility
        self._last_position = 0.0

        # Reset time state
        self.current_step = self.window_size  # Start after window

        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        if self.current_step >= len(self.features):
            # Return zeros if we're past the data
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        # Get feature window
        start_idx = max(0, self.current_step - self.window_size)
        end_idx = self.current_step

        if start_idx < len(self.features):
            feature_window = self.features[start_idx:end_idx]

            # Pad if necessary
            if len(feature_window) < self.window_size:
                padding = np.zeros((self.window_size - len(feature_window), self.features.shape[1]))
                feature_window = np.vstack([padding, feature_window])
        else:
            feature_window = np.zeros((self.window_size, self.features.shape[1]))

        # Flatten feature window
        flattened_features = feature_window.flatten()

        # Get current predictions
        if self.current_step < len(self.targets):
            current_targets = self.targets[self.current_step]
        else:
            current_targets = np.array([0.0, 0.02])  # Default mu_hat, sigma_hat

        # Get current state
        cash_equity_ratio = self.cash / self.equity if self.equity != 0 else 1.0
        current_state = np.array([self.position, cash_equity_ratio])

        # Combine all components
        observation = np.concatenate([
            flattened_features,
            current_targets,
            current_state
        ])

        # Ensure correct dimension and clip to observation space
        target_dim = self.observation_space.shape[0]
        if len(observation) > target_dim:
            observation = observation[:target_dim]
        elif len(observation) < target_dim:
            # Pad with zeros
            padding = np.zeros(target_dim - len(observation))
            observation = np.concatenate([observation, padding])

        # Clip to valid range
        observation = np.clip(observation, -1e6, 1e6)  # Reasonable bounds

        return observation.astype(np.float32)

    def step(self, action: np.ndarray):
        """Execute one environment step (Gymnasium API)."""
        # 1. Action validation & scaling
        action = np.asarray(action, dtype=np.float32)
        if action.shape == ():  # scalar
            action = np.array([action], dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        target_position = float(action[0]) * self.position_limit

        # 2. Current price
        if self.current_step < len(self.prices):
            current_price = float(self.prices[self.current_step])
        else:
            current_price = float(self.prices[-1])

        # 3. Trade execution
        position_change = target_position - self.position
        transaction_costs = self.transaction_cost * abs(position_change) * current_price
        self.cash -= position_change * current_price + transaction_costs
        self.position = target_position
        turnover_metric = abs(position_change) / self.position_limit if self.position_limit > 0 else abs(position_change)

        # 4. Advance time
        self.current_step += 1

        # 5. Reward calculation
        reward = 0.0
        pnl_component = 0.0
        cost_component = transaction_costs / max(self.equity, 1e-9)
        risk_component = 0.0
        if self.current_step < len(self.prices):
            next_price = float(self.prices[self.current_step])
            price_return = (next_price - current_price) / current_price if current_price != 0 else 0.0
            # Update rolling price return history
            self._price_return_history.append(price_return)
            if len(self._price_return_history) > self.risk_window:
                self._price_return_history.pop(0)
            pnl_component = self.position * price_return
            # Risk as volatility * exposure (abs position)
            if len(self._price_return_history) >= 2:  # need at least 2 for std
                price_vol = float(np.std(self._price_return_history))
                risk_component = price_vol * abs(self.position)
            # Final reward composition
            reward = (
                self.pnl_weight * pnl_component
                - self.transaction_cost_weight * cost_component
                - self.risk_penalty_weight * risk_component
                - self.turnover_penalty_weight * turnover_metric
            )
        # Ensure reward is scalar float
        reward = float(reward)

        # 6. Equity update
        idx_equity = min(self.current_step, len(self.prices) - 1)
        current_price_for_equity = float(self.prices[idx_equity])
        self.equity = float(self.cash + self.position * current_price_for_equity)

        # 7. Termination conditions
        terminated = self.current_step >= (len(self.features) - 1)
        truncated = self.equity <= 0.1 * self.initial_capital

        # 8. Observation & info
        observation = self._get_observation()
        info = self._get_info()
        info['reward_components'] = {
            'pnl': float(pnl_component),
            'cost': float(cost_component),
            'risk': float(risk_component),
            'turnover': float(turnover_metric)
        }

        return observation, reward, terminated, truncated, info

    def _get_info(self) -> dict:
        """Get additional information."""
        return {
            "step": self.current_step,
            "position": self.position,
            "cash": self.cash,
            "equity": self.equity,
            "date": self.dates[min(self.current_step, len(self.dates) - 1)]
        }


def test_enhanced_environment() -> bool | None:
    """Test the enhanced environment with bridge integration."""

    # Test with bridge-converted data
    try:
        # Find a bridge-converted file
        bridge_files = list(Path("data/bridge_cache").glob("*trading_format.parquet"))

        if bridge_files:
            bridge_file = bridge_files[0]

            # Create environment
            env = EnhancedTradingEnvironment(
                data_file=str(bridge_file),
                auto_convert=False  # Data already converted
            )


            # Test functionality
            obs, info = env.reset()

            # Test multiple steps
            total_reward = 0
            for _i in range(5):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                if terminated or truncated:
                    break


            return True

        return False

    except Exception:
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_enhanced_environment()
