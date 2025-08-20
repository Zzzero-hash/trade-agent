"""
Ensemble combiner for RL models with optional gating model functionality.

This module implements ensemble action selection by combining actions from
multiple reinforcement learning models using weighted averaging. It also
provides an optional gating model that can dynamically determine the weight
for combining SAC and PPO actions based on regime features like volatility.

Example usage:
    # Using fixed weight
    action = ensemble_action(obs, w=0.7)

    # Using gating model for dynamic weights
    feature_names = ['rolling_vol_20', 'rolling_vol_60', 'realized_vol', 'atr']
    gating_model = GatingModel(feature_names, method="volatility_threshold")
    action = ensemble_action(obs, w=0.5, gating=gating_model)

    # Using risk governor for risk management
    risk_governor = RiskGovernor(max_exposure=0.5, max_steps_per_bar=1)
    action = ensemble_action(obs, w=0.5, risk_governor=risk_governor,
                                current_equity=100000.0)

    # Running backtest with CLI
    python src/ensemble/combine.py --w 0.5
    python src/ensemble/combine.py --w 0.5 --gating
    python src/ensemble/combine.py --w 0.5 --risk-governor
"""

import argparse
import contextlib
import os
import sys
from typing import Any

import numpy as np
import pandas as pd


try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.base_class import BaseAlgorithm
except ImportError:
    sys.exit(1)

# Add parent directory to path to import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from trade_agent.envs.trading_env import TradingEnvironment
except ImportError:
    TradingEnvironment = None


class RiskGovernor:
    """
    Risk governor that applies risk management constraints to ensemble actions.

    This class implements three key risk management features:
    1. Hard exposure caps - Limit the maximum position size that can be taken
    2. Max step per bar - Limit the number of actions per time bar to prevent
       overtrading
    3. Drawdown-based scaling - Reduce position sizing when drawdown exceeds
       certain thresholds

    The risk governor works alongside the ensemble action selection to ensure
    that the combined actions don't exceed specified risk limits.

    Risk Management Features:
    - Hard Exposure Caps: Limit maximum position size to prevent excessive
      exposure to any single asset or market condition
    - Max Step Per Bar: Limit trading frequency to prevent overtrading and
      reduce transaction costs
    - Drawdown-Based Scaling: Automatically reduce position sizing when the
      strategy experiences drawdowns to preserve capital

    Example usage:
        # Create risk governor with default parameters
        risk_gov = RiskGovernor()

        # Create risk governor with custom parameters
        risk_gov = RiskGovernor(
            max_exposure=0.5,  # Limit to 50% exposure
            max_steps_per_bar=2,  # Allow up to 2 actions per bar
            drawdown_thresholds=[0.05, 0.10, 0.15],  # 5%, 10%, 15% drawdowns
            drawdown_scalings=[0.8, 0.6, 0.4]  # Scale by 80%, 60%, 40%
        )

        # Apply risk constraints to an action
        constrained_action = risk_gov.apply_constraints(
            action, current_equity)

        # Reset counters at the start of each new bar
        risk_gov.reset_bar_counters()

        # Reset equity tracking for a new backtest
        risk_gov.reset_equity_tracking(initial_equity=100000.0)
    """

    def __init__(
        self,
        max_exposure: float = 1.0,
        max_steps_per_bar: int = 1,
        drawdown_thresholds: list[float] | None = None,
        drawdown_scalings: list[float] | None = None,
        initial_equity: float = 100000.0
    ) -> None:
        """
        Initialize the risk governor.

        Args:
            max_exposure: Maximum allowed exposure (default: 1.0 for 100%)
            max_steps_per_bar: Maximum number of actions per time bar
                (default: 1)
            drawdown_thresholds: List of drawdown thresholds as fractions
                (e.g., [0.05, 0.10])
            drawdown_scalings: List of scaling factors for each threshold
                (e.g., [0.7, 0.5])
            initial_equity: Initial equity for drawdown calculations
        """
        self.max_exposure = max_exposure
        self.max_steps_per_bar = max_steps_per_bar

        # Set default drawdown thresholds if not provided
        self.drawdown_thresholds = drawdown_thresholds or [0.05, 0.10, 0.15]
        self.drawdown_scalings = drawdown_scalings or [0.7, 0.5, 0.3]

        # Validate drawdown thresholds and scalings
        if len(self.drawdown_thresholds) != len(self.drawdown_scalings):
            raise ValueError(
                "Drawdown thresholds and scalings must have the same length")

        # Sort thresholds and scalings in descending order of thresholds
        sorted_pairs = sorted(
            zip(self.drawdown_thresholds, self.drawdown_scalings, strict=False),
            key=lambda x: x[0], reverse=True)
        self.drawdown_thresholds, self.drawdown_scalings = zip(*sorted_pairs, strict=False)

        self.initial_equity = initial_equity
        self.max_equity = initial_equity  # Track maximum equity achieved

        # Track steps per bar
        self.steps_in_current_bar = 0
        self.current_bar_id = None

    def apply_constraints(
        self,
        action: np.ndarray,
        current_equity: float,
        bar_id: Any = None
    ) -> np.ndarray:
        """
        Apply all risk constraints to the given action.

        This method applies all three risk management features in sequence:
        1. Checks max steps per bar to prevent overtrading
        2. Updates equity tracking for drawdown calculations
        3. Applies drawdown-based scaling to reduce position sizing
        4. Applies hard exposure cap to limit maximum position size

        Args:
            action: The action to constrain (target net exposure in [-1, 1])
            current_equity: Current equity value
            bar_id: Identifier for the current time bar (optional)

        Returns:
            Constrained action that respects all risk limits
        """
        # Update bar tracking
        self._update_bar_tracking(bar_id)

        # Check if we've exceeded max steps per bar
        if self.steps_in_current_bar > self.max_steps_per_bar:
            # Prevent trading by setting action to maintain current position
            return np.array([0.0])  # No change in position

        # Update equity tracking
        self.max_equity = max(self.max_equity, current_equity)

        # Apply drawdown-based scaling
        scaled_action = self._apply_drawdown_scaling(action, current_equity)

        # Apply hard exposure cap
        constrained_action = self._apply_exposure_cap(scaled_action)

        # Increment step counter
        self.steps_in_current_bar += 1

        return constrained_action

    def _update_bar_tracking(self, bar_id: Any) -> None:
        """
        Update bar tracking, resetting counter when bar changes.

        Args:
            bar_id: Identifier for the current time bar
        """
        if bar_id != self.current_bar_id:
            self.current_bar_id = bar_id
            self.steps_in_current_bar = 1  # This action counts as the first step
        # If bar_id is None, we don't track bars separately and count all steps
        elif bar_id is None:
            # If no bar_id provided, just increment the counter
            pass

    def _apply_drawdown_scaling(
        self,
        action: np.ndarray,
        current_equity: float
    ) -> np.ndarray:
        """
        Apply drawdown-based scaling to reduce position sizing during drawdowns.

        This method calculates the current drawdown and applies scaling factors
        to reduce position sizing when drawdown thresholds are exceeded. Higher
        drawdowns result in more aggressive position sizing reductions.

        Args:
            action: The action to scale
            current_equity: Current equity value

        Returns:
            Scaled action (reduced position sizing during drawdowns)
        """
        # Calculate current drawdown
        if self.max_equity <= 0:
            drawdown = 0.0
        else:
            drawdown = (self.max_equity - current_equity) / self.max_equity

        # Determine scaling factor based on drawdown thresholds
        scaling_factor = 1.0
        for threshold, scaling in zip(self.drawdown_thresholds,
                                       self.drawdown_scalings, strict=False):
            if drawdown >= threshold:
                scaling_factor = scaling
                break

        # Apply scaling if needed
        if scaling_factor < 1.0:
            return action * scaling_factor

        return action

    def _apply_exposure_cap(self, action: np.ndarray) -> np.ndarray:
        """
        Apply hard exposure cap to limit maximum position size.

        This method constrains the action to be within the specified exposure
        limits to prevent excessive position sizing. The exposure cap is
        symmetric, meaning both long and short positions are limited.

        Args:
            action: The action to constrain (target net exposure in [-1, 1])

        Returns:
            Constrained action within exposure limits
        """
        # Clip action to exposure cap
        constrained_action = np.clip(action, -self.max_exposure,
                                     self.max_exposure)

        # If action was constrained, print a message
        if not np.isclose(constrained_action[0], action[0], atol=1e-6):
            pass

        return constrained_action

    def reset_bar_counters(self) -> None:
        """
        Reset bar counters. Call this at the start of each new bar.

        This method resets the step counter for the current bar, allowing
        trading to resume for the new bar. It should be called at the
        beginning of each new time bar (e.g., new day, new hour) to
        reset the max steps per bar limit.
        """
        self.steps_in_current_bar = 0
        self.current_bar_id = None

    def reset_equity_tracking(
        self,
        initial_equity: float | None = None
    ) -> None:
        """
        Reset equity tracking. Call this when starting a new backtest or
        live session.

        This method resets the maximum equity tracking to the initial equity
        value, effectively resetting the drawdown calculation. It should be
        called at the beginning of each new backtest or live trading session.

        Args:
            initial_equity: New initial equity value (if None, uses existing
                value)
        """
        if initial_equity is not None:
            self.initial_equity = initial_equity
        self.max_equity = self.initial_equity


class GatingModel:
    """
    Gating model that determines the optimal weight for combining SAC and PPO actions
    based on regime features like volatility.

    The gating model analyzes regime features to determine when to use SAC vs PPO,
    allowing the ensemble to adapt to different market conditions.

    Example usage:
        feature_names = ['rolling_vol_20', 'rolling_vol_60', 'realized_vol', 'atr']
        gating_model = GatingModel(feature_names, method="volatility_threshold")
        weight = gating_model.get_weight(observation)

    Methods:
        volatility_threshold: High volatility -> prefer PPO, Low volatility -> prefer SAC
        volatility_inverse: Higher volatility -> lower SAC weight (smooth transition)
        fixed: Always return 0.5 (no gating)
    """

    def __init__(self, feature_names: list[str], method: str = "volatility_threshold") -> None:
        """
        Initialize the gating model.

        Args:
            feature_names: List of feature names available in the observation
            method: Method for determining weights ("volatility_threshold", "volatility_inverse", or "fixed")
        """
        self.feature_names = feature_names
        self.method = method

        # Map feature names to indices for quick lookup
        self.feature_indices = {name: i for i, name in enumerate(feature_names)}

        # Default parameters for volatility-based gating
        self.volatility_threshold = 0.02  # Threshold for high volatility
        self.volatility_scaling = 10.0    # Scaling factor for inverse volatility method

    def get_weight(self, obs: np.ndarray) -> float:
        """
        Determine the weight for SAC model based on regime features.

        Args:
            obs: Observation vector containing regime features

        Returns:
            Weight for SAC model (0.0 = only PPO, 1.0 = only SAC)
        """
        if self.method == "fixed":
            # Return fixed weight of 0.5
            return 0.5

        # Extract volatility features from observation
        vol_features = self._extract_volatility_features(obs)

        if not vol_features:
            # If no volatility features found, use fixed weight
            return 0.5

        # Calculate average volatility across available features
        avg_volatility = np.mean(list(vol_features.values()))

        if self.method == "volatility_threshold":
            # Use threshold-based approach: high vol -> more PPO, low vol -> more SAC
            # Rationale: In high volatility regimes, PPO's clipped objective function
            # is more stable than SAC's entropy maximization which can lead to
            # excessive exploration in volatile markets.
            if avg_volatility > self.volatility_threshold:
                # High volatility regime: prefer PPO (lower SAC weight)
                return 0.3
            # Low volatility regime: prefer SAC (higher SAC weight)
            # Rationale: In low volatility regimes, SAC's superior sample efficiency
            # and asymptotic optimality can be leveraged.
            return 0.7
        if self.method == "volatility_inverse":
            # Use inverse volatility: higher vol -> lower SAC weight
            # This provides a smooth transition between models based on volatility
            # Normalize volatility to [0, 1] range using exponential scaling
            normalized_vol = 1 - np.exp(-self.volatility_scaling * avg_volatility)
            # Invert so that high volatility gives low SAC weight
            return 1.0 - normalized_vol
        # Default to fixed weight if method not recognized
        return 0.5

    def _extract_volatility_features(self, obs: np.ndarray) -> dict[str, float]:
        """
        Extract volatility-related features from observation.

        Args:
            obs: Observation vector

        Returns:
            Dictionary of volatility feature names and values
        """
        vol_features = {}

        # Look for common volatility-related features
        volatility_feature_names = [
            "rolling_vol_20", "rolling_vol_60", "realized_vol", "atr"
        ]

        for feature_name in volatility_feature_names:
            if feature_name in self.feature_indices:
                idx = self.feature_indices[feature_name]
                vol_features[feature_name] = obs[idx]

        return vol_features


def load_model(model_path: str, model_type: str) -> BaseAlgorithm | None:
    """
    Load a trained model from disk.

    Args:
        model_path: Path to the model file
        model_type: Type of model ('ppo' or 'sac')

    Returns:
        Loaded model or None if loading failed
    """
    try:
        if not os.path.exists(model_path):
            return None

        if model_type.lower() == 'ppo':
            model = PPO.load(model_path)
        elif model_type.lower() == 'sac':
            model = SAC.load(model_path)
        else:
            return None

        return model
    except Exception:
        return None


def ensemble_action(
    obs: np.ndarray,
    w: float = 0.5,
    gating: GatingModel | None = None,
    risk_governor: RiskGovernor | None = None,
    current_equity: float = 100000.0
) -> np.ndarray:
    """
    Compute ensemble action by combining PPO and SAC model actions.

    Args:
        obs: Observation for which to compute action
        w: Weight for SAC model (0.0 = only PPO, 1.0 = only SAC)
        gating: Optional gating model to dynamically determine weights based on regime features
        risk_governor: Optional risk governor to apply risk constraints
        current_equity: Current equity value for risk calculations

    Returns:
        Combined action as weighted average of model actions,
        potentially constrained by risk governor
    """
    # Load models if not already loaded
    # In a production implementation, models would be loaded once and cached
    ppo_model = load_model("models/rl/ppo_final.zip", "ppo")
    sac_model = load_model("models/rl/sac.zip", "sac")

    if ppo_model is None or sac_model is None:
        raise RuntimeError("Failed to load one or both models")

    # Get actions from both models
    # For deterministic actions, we use the action directly
    # For stochastic policies, we might want to sample
    ppo_action, _ = ppo_model.predict(obs, deterministic=True)
    sac_action, _ = sac_model.predict(obs, deterministic=True)

    # Determine weight based on gating model or use fixed weight
    if gating is not None:
        # Use gating model to determine dynamic weight
        dynamic_w = gating.get_weight(obs)
    else:
        # Use fixed weight
        dynamic_w = w

    # Combine actions with weighted average
    # a = w * a_sac + (1-w) * a_ppo
    combined_action = dynamic_w * sac_action + (1 - dynamic_w) * ppo_action

    # Apply risk constraints if risk governor is provided
    if risk_governor is not None:
        combined_action = risk_governor.apply_constraints(
            combined_action, current_equity)

    return combined_action


def create_validation_environment(
    data_file: str = "data/features.parquet",
    initial_capital: float = 100000.0,
    transaction_cost: float = 0.001,
    window_size: int = 30,
    validation_split: float = 0.2,
    seed: int = 42
) -> Any:
    """
    Create validation environment with fixed seed for deterministic evaluation.

    Args:
        data_file: Path to features data file
        initial_capital: Starting capital for portfolio
        transaction_cost: Transaction cost per trade
        window_size: Feature window size
        validation_split: Proportion of data to use for validation
        seed: Random seed for deterministic evaluation

    Returns:
        Validation environment
    """
    if TradingEnvironment is None:
        raise RuntimeError("TradingEnvironment is not available")

    # Load data
    df = pd.read_parquet(data_file)

    # Split data to get validation set (same as in training)
    split_index = int(len(df) * (1 - validation_split))
    val_data = df.iloc[split_index:]

    # Save temporary validation file
    val_file = "data/val_temp.parquet"
    val_data.to_parquet(val_file)

    # Create validation environment with fixed seed
    return TradingEnvironment(
        data_file=val_file,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost,
        seed=seed + 1000,  # Different seed but fixed for evaluation
        window_size=window_size
    )



def backtest_ensemble(
    w: float = 0.5,
    use_gating: bool = False,
    use_risk_governor: bool = False
) -> None:
    """
    Run backtest using ensemble action selection on validation data.

    Args:
        w: Weight for SAC model in ensemble
        use_gating: Whether to use gating model for dynamic weight determination
        use_risk_governor: Whether to use risk governor for risk management
    """
    try:
        # Create validation environment
        env = create_validation_environment()

        # Initialize gating model if requested
        gating_model = None
        if use_gating:
            # Get feature names from the environment
            # In a real implementation, this would come from the environment's
            # feature metadata
            feature_names = [
                'log_returns', 'rolling_mean_20', 'rolling_vol_20',
                'rolling_mean_60', 'rolling_vol_60', 'atr', 'rsi',
                'price_z_score', 'volume_z_score', 'realized_vol',
                'day_of_week', 'month', 'day_of_month', 'is_monday',
                'is_friday', 'is_month_start', 'is_month_end'
            ]
            gating_model = GatingModel(feature_names,
                                          method="volatility_threshold")
        else:
            pass

        # Initialize risk governor if requested
        risk_governor = None
        if use_risk_governor:
            risk_governor = RiskGovernor()

        # Reset environment
        obs, info = env.reset()
        initial_equity = info['equity']
        done = False
        total_reward = 0
        step_count = 0

        # Track performance metrics
        equity_history = [initial_equity]
        returns_history = []
        positions_history = [info['position']]
        actions_taken = 0
        actions_prevented = 0

        if use_gating and use_risk_governor or use_gating or use_risk_governor:
            pass
        else:
            pass

        while not done:
            # Get ensemble action with risk constraints if applicable
            action = ensemble_action(
                obs,
                w=w,
                gating=gating_model,
                risk_governor=risk_governor,
                current_equity=info['equity']
            )

            # Check if action was prevented by risk governor
            if risk_governor is not None and action[0] == 0.0 and info['position'] != 0.0:
                actions_prevented += 1
            else:
                actions_taken += 1

            # Execute action in environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Track metrics
            total_reward += reward
            step_count += 1
            equity_history.append(info['equity'])
            returns_history.append(reward)
            positions_history.append(info['position'])

            if step_count % 100 == 0:  # Print every 100 steps
                pass

        # Calculate comprehensive performance metrics
        final_equity = info['equity']
        (final_equity - initial_equity) / initial_equity
        total_reward / step_count if step_count > 0 else 0

        # Calculate Sharpe ratio (assuming risk-free rate of 0 for simplicity)
        if len(returns_history) > 1:
            np.mean(returns_history) / (np.std(returns_history) + 1e-8) * np.sqrt(252)  # Annualized
        else:
            pass

        # Calculate maximum drawdown
        equity_array = np.array(equity_history)
        cumulative_returns = (equity_array - initial_equity) / initial_equity
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        np.min(drawdown) if len(drawdown) > 0 else 0

        # Calculate volatility
        np.std(returns_history) * np.sqrt(252) if len(returns_history) > 1 else 0  # Annualized

        # Calculate win rate
        positive_returns = sum(1 for r in returns_history if r > 0)
        positive_returns / len(returns_history) if len(returns_history) > 0 else 0

        # Calculate average position size
        np.mean(np.abs(positions_history)) if len(positions_history) > 0 else 0

        # Print comprehensive results

        if use_risk_governor:
            total_actions = actions_taken + actions_prevented
            if total_actions > 0:
                actions_prevented / total_actions


        # Clean up temporary file
        with contextlib.suppress(FileNotFoundError):
            os.remove("data/val_temp.parquet")

    except Exception:
        import traceback
        traceback.print_exc()


def main() -> None:
    """Main function for CLI execution."""
    parser = argparse.ArgumentParser(
        description="Ensemble combiner for RL models with comprehensive backtesting capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.ensemble.combine --w 0.5
  python -m src.ensemble.combine --w 0.7 --gating
  python -m src.ensemble.combine --w 0.3 --risk-governor
  python -m src.ensemble.combine --w 0.5 --gating --risk-governor

Features:
  - Weighted ensemble of PPO and SAC models
  - Dynamic weight determination with gating model
  - Risk management with risk governor
  - Comprehensive backtesting with performance metrics
"""
    )
    parser.add_argument(
        "--w",
        type=float,
        default=0.5,
        help="Weight for SAC model in ensemble (0.0 = only PPO, 1.0 = only SAC) (default: 0.5)"
    )
    parser.add_argument(
        "--gating",
        action="store_true",
        help="Use gating model for dynamic weight determination based on market regime features"
    )
    parser.add_argument(
        "--risk-governor",
        action="store_true",
        help="Use risk governor for risk management (position limits, drawdown protection, etc.)"
    )

    args = parser.parse_args()

    # Validate parameters
    if not 0.0 <= args.w <= 1.0:
        sys.exit(1)

    if args.gating and args.risk_governor:
        backtest_ensemble(w=args.w, use_gating=True, use_risk_governor=True)
    elif args.gating:
        backtest_ensemble(w=args.w, use_gating=True)
    elif args.risk_governor:
        backtest_ensemble(w=args.w, use_risk_governor=True)
    else:
        backtest_ensemble(w=args.w)


if __name__ == "__main__":
    main()
