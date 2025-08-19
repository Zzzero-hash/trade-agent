#!/usr/bin/env python3
"""
Comprehensive backtesting script that compares ensemble performance against individual PPO and SAC policies.

This script extends the existing evaluation methodology to provide a detailed comparison of:
1. Individual PPO policy performance
2. Individual SAC policy performance
3. Ensemble performance with fixed weights
4. Ensemble performance with dynamic gating
5. Risk governor validation and bounds enforcement

The script uses identical data and methodology as existing evaluation scripts to ensure fair comparison.
"""

import json
import os
import sys
import traceback
from datetime import datetime
from typing import Any

import numpy as np
from stable_baselines3 import PPO, SAC


# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import contextlib

from src.ensemble.combine import (  # noqa: E402
    GatingModel,
    RiskGovernor,
    create_validation_environment,
)
from trade_agent.agents.envs.trading_env import TradingEnvironment  # noqa: E402
from trade_agent.agents.sl.models.base import set_all_seeds  # noqa: E402


class ValidationUtils:
    """Utility class for comprehensive validation checks."""

    @staticmethod
    def check_for_nan_inf(data: np.ndarray, data_name: str) -> bool:
        """Check for NaN or Inf values in data array."""
        has_nan = np.isnan(data).any()
        has_inf = np.isinf(data).any()

        if has_nan:
            pass
        if has_inf:
            pass

        return not (has_nan or has_inf)

    @staticmethod
    def validate_action_bounds(action: np.ndarray, min_bound: float = -1.0, max_bound: float = 1.0) -> bool:
        """Validate that actions are within expected bounds."""
        action_val = action[0] if isinstance(action, np.ndarray) else action
        within_bounds = min_bound <= action_val <= max_bound

        if not within_bounds:
            pass

        return within_bounds

    @staticmethod
    def validate_governor_enforcement(original_action: np.ndarray, constrained_action: np.ndarray,
                                    risk_governor: RiskGovernor) -> bool:
        """Validate that risk governor properly enforced constraints."""
        original_val = original_action[0] if isinstance(original_action, np.ndarray) else original_action
        constrained_val = constrained_action[0] if isinstance(constrained_action, np.ndarray) else constrained_action

        # Check exposure cap enforcement
        if abs(constrained_val) > risk_governor.max_exposure + 1e-6:
            return False

        # Check if constraint was applied appropriately
        if abs(original_val) > risk_governor.max_exposure and abs(constrained_val) <= risk_governor.max_exposure:
            pass

        return True


def load_model_safe(model_path: str, model_type: str) -> Any | None:
    """Load model with comprehensive error handling."""
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


def calculate_sharpe_ratio(returns: list[float], risk_free_rate: float = 0.0) -> float:
    """Calculate annualized Sharpe ratio using the pattern from existing scripts."""
    if len(returns) <= 1:
        return 0.0

    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate

    if np.std(excess_returns) == 0:
        return 0.0

    # Annualize assuming daily returns (252 trading days)
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)


def calculate_performance_metrics(returns: list[float], equity_history: list[float],
                                initial_capital: float) -> dict[str, Any]:
    """Calculate comprehensive performance metrics following existing patterns."""
    if not returns or not equity_history:
        return {}

    returns_array = np.array(returns)
    equity_array = np.array(equity_history)

    # Basic metrics
    mean_reward = np.mean(returns_array)
    mean_return = (equity_history[-1] - initial_capital)
    std_return = np.std(equity_array[1:] - equity_array[:-1]) if len(equity_array) > 1 else 0.0

    # Sharpe ratio
    sharpe_ratio = calculate_sharpe_ratio(returns)

    # Maximum drawdown
    cumulative_returns = (equity_array - initial_capital) / initial_capital
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = cumulative_returns - running_max
    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0

    # Volatility (annualized)
    volatility = np.std(returns_array) * np.sqrt(252) if len(returns_array) > 1 else 0.0

    # Win rate
    positive_returns = sum(1 for r in returns if r > 0)
    win_rate = positive_returns / len(returns) if returns else 0.0

    # Value at Risk (95%)
    var_95 = np.percentile(returns_array, 5) if len(returns_array) > 0 else 0.0

    return {
        'mean_reward': mean_reward,
        'mean_return': mean_return,
        'std_return': std_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'volatility': volatility,
        'win_rate': win_rate,
        'var_95': var_95,
        'total_steps': len(returns),
        'final_equity': equity_history[-1] if equity_history else initial_capital
    }


def evaluate_individual_policy(model: Any, env: TradingEnvironment, model_name: str,
                             n_episodes: int = 1) -> dict[str, Any]:
    """Evaluate individual policy using the same methodology as existing evaluation scripts."""

    episode_returns = []
    episode_rewards = []
    all_equity_history = []
    validation_results = {'nan_inf_checks': [], 'bounds_checks': []}

    for _episode in range(n_episodes):

        obs, info = env.reset()
        total_reward = 0.0
        step_count = 0
        episode_equity_history = [info['equity']]
        episode_reward_history = []

        while True:
            # Get deterministic action
            action, _ = model.predict(obs, deterministic=True)

            # Validation checks
            validation_results['nan_inf_checks'].append(
                ValidationUtils.check_for_nan_inf(action, f"{model_name}_action_step_{step_count}")
            )
            validation_results['bounds_checks'].append(
                ValidationUtils.validate_action_bounds(action)
            )

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            episode_equity_history.append(info['equity'])
            episode_reward_history.append(reward)

            # Validation checks on rewards and equity
            ValidationUtils.check_for_nan_inf(np.array([reward]), f"{model_name}_reward_step_{step_count}")
            ValidationUtils.check_for_nan_inf(np.array([info['equity']]), f"{model_name}_equity_step_{step_count}")

            if terminated or truncated:
                break

        # Calculate episode return
        episode_return = info['equity'] - env.initial_capital
        episode_rewards.append(total_reward)
        episode_returns.append(episode_return)
        all_equity_history.extend(episode_equity_history)


    # Calculate comprehensive metrics
    metrics = calculate_performance_metrics(episode_rewards, all_equity_history, env.initial_capital)
    metrics['episode_returns'] = episode_returns
    metrics['episode_rewards'] = episode_rewards
    metrics['n_episodes'] = n_episodes
    metrics['validation_results'] = validation_results

    return metrics


def evaluate_ensemble_fixed_weight(ppo_model: Any, sac_model: Any, env: TradingEnvironment,
                                  weight: float, n_episodes: int = 1) -> dict[str, Any]:
    """Evaluate ensemble with fixed weight."""

    episode_returns = []
    episode_rewards = []
    all_equity_history = []
    validation_results = {'nan_inf_checks': [], 'bounds_checks': [], 'action_combinations': []}

    for _episode in range(n_episodes):

        obs, info = env.reset()
        total_reward = 0.0
        step_count = 0
        episode_equity_history = [info['equity']]
        episode_reward_history = []

        while True:
            # Get individual actions for logging
            ppo_action, _ = ppo_model.predict(obs, deterministic=True)
            sac_action, _ = sac_model.predict(obs, deterministic=True)

            # Combine actions manually (without risk governor for this test)
            combined_action = weight * sac_action + (1 - weight) * ppo_action

            # Validation checks
            validation_results['nan_inf_checks'].append(
                ValidationUtils.check_for_nan_inf(combined_action, f"ensemble_action_step_{step_count}")
            )
            validation_results['bounds_checks'].append(
                ValidationUtils.validate_action_bounds(combined_action)
            )
            validation_results['action_combinations'].append({
                'step': step_count,
                'ppo_action': ppo_action[0] if isinstance(ppo_action, np.ndarray) else ppo_action,
                'sac_action': sac_action[0] if isinstance(sac_action, np.ndarray) else sac_action,
                'combined_action': combined_action[0] if isinstance(combined_action, np.ndarray) else combined_action,
                'weight': weight
            })

            # Step environment
            obs, reward, terminated, truncated, info = env.step(combined_action)
            total_reward += reward
            step_count += 1

            episode_equity_history.append(info['equity'])
            episode_reward_history.append(reward)

            if terminated or truncated:
                break

        # Calculate episode return
        episode_return = info['equity'] - env.initial_capital
        episode_rewards.append(total_reward)
        episode_returns.append(episode_return)
        all_equity_history.extend(episode_equity_history)


    # Calculate comprehensive metrics
    metrics = calculate_performance_metrics(episode_rewards, all_equity_history, env.initial_capital)
    metrics['ensemble_weight'] = weight
    metrics['episode_returns'] = episode_returns
    metrics['episode_rewards'] = episode_rewards
    metrics['n_episodes'] = n_episodes
    metrics['validation_results'] = validation_results

    return metrics


def evaluate_ensemble_with_gating(ppo_model: Any, sac_model: Any, env: TradingEnvironment,
                                base_weight: float, n_episodes: int = 1) -> dict[str, Any]:
    """Evaluate ensemble with dynamic gating model."""

    # Create gating model
    feature_names = [
        'log_returns', 'rolling_mean_20', 'rolling_vol_20',
        'rolling_mean_60', 'rolling_vol_60', 'atr', 'rsi',
        'price_z_score', 'volume_z_score', 'realized_vol',
        'day_of_week', 'month', 'day_of_month', 'is_monday',
        'is_friday', 'is_month_start', 'is_month_end'
    ]
    gating_model = GatingModel(feature_names, method="volatility_threshold")

    episode_returns = []
    episode_rewards = []
    all_equity_history = []
    validation_results = {'nan_inf_checks': [], 'bounds_checks': [], 'gating_decisions': []}

    for _episode in range(n_episodes):

        obs, info = env.reset()
        total_reward = 0.0
        step_count = 0
        episode_equity_history = [info['equity']]
        episode_reward_history = []

        while True:
            # Get individual actions
            ppo_action, _ = ppo_model.predict(obs, deterministic=True)
            sac_action, _ = sac_model.predict(obs, deterministic=True)

            # Get dynamic weight from gating model
            dynamic_weight = gating_model.get_weight(obs)
            combined_action = dynamic_weight * sac_action + (1 - dynamic_weight) * ppo_action

            # Validation checks
            validation_results['nan_inf_checks'].append(
                ValidationUtils.check_for_nan_inf(combined_action, f"gated_ensemble_action_step_{step_count}")
            )
            validation_results['bounds_checks'].append(
                ValidationUtils.validate_action_bounds(combined_action)
            )
            validation_results['gating_decisions'].append({
                'step': step_count,
                'dynamic_weight': dynamic_weight,
                'base_weight': base_weight,
                'ppo_action': ppo_action[0] if isinstance(ppo_action, np.ndarray) else ppo_action,
                'sac_action': sac_action[0] if isinstance(sac_action, np.ndarray) else sac_action,
                'combined_action': combined_action[0] if isinstance(combined_action, np.ndarray) else combined_action
            })

            # Step environment
            obs, reward, terminated, truncated, info = env.step(combined_action)
            total_reward += reward
            step_count += 1

            episode_equity_history.append(info['equity'])
            episode_reward_history.append(reward)

            if terminated or truncated:
                break

        # Calculate episode return
        episode_return = info['equity'] - env.initial_capital
        episode_rewards.append(total_reward)
        episode_returns.append(episode_return)
        all_equity_history.extend(episode_equity_history)


    # Calculate comprehensive metrics
    metrics = calculate_performance_metrics(episode_rewards, all_equity_history, env.initial_capital)
    metrics['base_weight'] = base_weight
    metrics['gating_method'] = "volatility_threshold"
    metrics['episode_returns'] = episode_returns
    metrics['episode_rewards'] = episode_rewards
    metrics['n_episodes'] = n_episodes
    metrics['validation_results'] = validation_results

    return metrics


def evaluate_ensemble_with_risk_governor(ppo_model: Any, sac_model: Any, env: TradingEnvironment,
                                        weight: float, n_episodes: int = 1) -> dict[str, Any]:
    """Evaluate ensemble with risk governor."""

    # Create risk governor with conservative settings
    risk_governor = RiskGovernor(
        max_exposure=0.5,  # Limit to 50% exposure
        max_steps_per_bar=1,  # Single action per bar
        drawdown_thresholds=[0.05, 0.10, 0.15],  # 5%, 10%, 15%
        drawdown_scalings=[0.7, 0.5, 0.3],  # Scale down positions
        initial_equity=env.initial_capital
    )

    episode_returns = []
    episode_rewards = []
    all_equity_history = []
    validation_results = {
        'nan_inf_checks': [],
        'bounds_checks': [],
        'governor_enforcements': [],
        'actions_prevented': 0,
        'actions_taken': 0
    }

    for _episode in range(n_episodes):

        obs, info = env.reset()
        risk_governor.reset_equity_tracking(env.initial_capital)
        total_reward = 0.0
        step_count = 0
        episode_equity_history = [info['equity']]
        episode_reward_history = []

        while True:
            # Get individual actions
            ppo_action, _ = ppo_model.predict(obs, deterministic=True)
            sac_action, _ = sac_model.predict(obs, deterministic=True)

            # Combine actions
            combined_action = weight * sac_action + (1 - weight) * ppo_action

            # Apply risk governor
            original_action = combined_action.copy()
            constrained_action = risk_governor.apply_constraints(
                combined_action,
                current_equity=info['equity'],
                bar_id=step_count  # Use step as bar ID for testing
            )

            # Validation checks
            validation_results['nan_inf_checks'].append(
                ValidationUtils.check_for_nan_inf(constrained_action, f"governed_action_step_{step_count}")
            )
            validation_results['bounds_checks'].append(
                ValidationUtils.validate_action_bounds(constrained_action)
            )
            validation_results['governor_enforcements'].append(
                ValidationUtils.validate_governor_enforcement(original_action, constrained_action, risk_governor)
            )

            # Check if action was prevented
            if np.allclose(constrained_action, [0.0]) and not np.allclose(original_action, [0.0]):
                validation_results['actions_prevented'] += 1
            else:
                validation_results['actions_taken'] += 1

            # Step environment
            obs, reward, terminated, truncated, info = env.step(constrained_action)
            total_reward += reward
            step_count += 1

            episode_equity_history.append(info['equity'])
            episode_reward_history.append(reward)

            if terminated or truncated:
                break

        # Calculate episode return
        episode_return = info['equity'] - env.initial_capital
        episode_rewards.append(total_reward)
        episode_returns.append(episode_return)
        all_equity_history.extend(episode_equity_history)


    # Calculate comprehensive metrics
    metrics = calculate_performance_metrics(episode_rewards, all_equity_history, env.initial_capital)
    metrics['ensemble_weight'] = weight
    metrics['risk_governor_config'] = {
        'max_exposure': risk_governor.max_exposure,
        'max_steps_per_bar': risk_governor.max_steps_per_bar,
        'drawdown_thresholds': list(risk_governor.drawdown_thresholds),
        'drawdown_scalings': list(risk_governor.drawdown_scalings)
    }
    metrics['episode_returns'] = episode_returns
    metrics['episode_rewards'] = episode_rewards
    metrics['n_episodes'] = n_episodes
    metrics['validation_results'] = validation_results

    return metrics


def generate_comparative_analysis(ppo_metrics: dict[str, Any], sac_metrics: dict[str, Any],
                                ensemble_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate comprehensive comparative analysis and recommendations."""

    # Individual policy performance
    ppo_return = ppo_metrics['mean_return']
    sac_return = sac_metrics['mean_return']
    min_individual_return = min(ppo_return, sac_return)
    max_individual_return = max(ppo_return, sac_return)


    # Analyze ensemble results
    analysis = {
        'individual_performance': {
            'ppo': ppo_metrics,
            'sac': sac_metrics,
            'min_return': min_individual_return,
            'max_return': max_individual_return
        },
        'ensemble_performance': {},
        'performance_comparison': {},
        'validation_summary': {},
        'recommendations': []
    }

    best_ensemble_return = float('-inf')

    for i, ensemble_result in enumerate(ensemble_results):
        ensemble_return = ensemble_result['mean_return']
        ensemble_type = ensemble_result.get('ensemble_type', f'ensemble_{i}')


        # Track best ensemble
        if ensemble_return > best_ensemble_return:
            best_ensemble_return = ensemble_return

        # Store ensemble performance
        analysis['ensemble_performance'][ensemble_type] = ensemble_result

        # Performance comparison
        beats_min = ensemble_return >= min_individual_return
        beats_max = ensemble_return >= max_individual_return
        improvement_over_min = ((ensemble_return - min_individual_return) / abs(min_individual_return)) * 100

        analysis['performance_comparison'][ensemble_type] = {
            'beats_min_individual': beats_min,
            'beats_max_individual': beats_max,
            'improvement_over_min_pct': improvement_over_min,
            'return': ensemble_return
        }

        if beats_min:
            pass
        else:
            pass

    # Validation summary
    validation_summary = {
        'total_nan_inf_failures': 0,
        'total_bounds_failures': 0,
        'total_governor_failures': 0,
        'by_method': {}
    }

    for ensemble_result in ensemble_results:
        method = ensemble_result.get('ensemble_type', 'unknown')
        validation_results = ensemble_result.get('validation_results', {})

        nan_inf_failures = sum(1 for check in validation_results.get('nan_inf_checks', []) if not check)
        bounds_failures = sum(1 for check in validation_results.get('bounds_checks', []) if not check)
        governor_failures = sum(1 for check in validation_results.get('governor_enforcements', []) if not check)

        validation_summary['total_nan_inf_failures'] += nan_inf_failures
        validation_summary['total_bounds_failures'] += bounds_failures
        validation_summary['total_governor_failures'] += governor_failures

        validation_summary['by_method'][method] = {
            'nan_inf_failures': nan_inf_failures,
            'bounds_failures': bounds_failures,
            'governor_failures': governor_failures
        }

    analysis['validation_summary'] = validation_summary

    # Generate recommendations
    recommendations = []

    # Performance-based recommendations
    if best_ensemble_return < min_individual_return:
        recommendations.append({
            'type': 'performance',
            'priority': 'high',
            'issue': 'Ensemble underperforms individual policies',
            'recommendation': 'Review ensemble weights and gating strategies. Consider regime-specific feature engineering.',
            'details': f'Best ensemble return ${best_ensemble_return:.2f} < minimum individual return ${min_individual_return:.2f}'
        })
    elif best_ensemble_return > max_individual_return:
        recommendations.append({
            'type': 'performance',
            'priority': 'positive',
            'issue': 'Ensemble outperforms individual policies',
            'recommendation': 'Current ensemble strategy is effective. Consider further optimization.',
            'details': f'Best ensemble return ${best_ensemble_return:.2f} > maximum individual return ${max_individual_return:.2f}'
        })

    # Validation-based recommendations
    if validation_summary['total_nan_inf_failures'] > 0:
        recommendations.append({
            'type': 'validation',
            'priority': 'critical',
            'issue': 'NaN/Inf values detected in actions or returns',
            'recommendation': 'Add robust input validation and fallback mechanisms for NaN/Inf handling.',
            'details': f'Total NaN/Inf failures: {validation_summary["total_nan_inf_failures"]}'
        })

    if validation_summary['total_bounds_failures'] > 0:
        recommendations.append({
            'type': 'validation',
            'priority': 'high',
            'issue': 'Actions outside expected bounds detected',
            'recommendation': 'Review action space definition and add action clipping.',
            'details': f'Total bounds failures: {validation_summary["total_bounds_failures"]}'
        })

    if validation_summary['total_governor_failures'] > 0:
        recommendations.append({
            'type': 'validation',
            'priority': 'medium',
            'issue': 'Risk governor enforcement issues detected',
            'recommendation': 'Review risk governor implementation and constraint logic.',
            'details': f'Total governor failures: {validation_summary["total_governor_failures"]}'
        })

    # Strategy-specific recommendations
    if all(result['mean_return'] < 0 for result in [ppo_metrics, sac_metrics] + ensemble_results):
        recommendations.append({
            'type': 'strategy',
            'priority': 'high',
            'issue': 'All strategies showing negative returns',
            'recommendation': 'Consider regime feature engineering, alternative reward functions, or longer evaluation periods.',
            'details': 'All individual and ensemble strategies have negative mean returns'
        })

    analysis['recommendations'] = recommendations

    # Print summary

    for i, _rec in enumerate(recommendations, 1):
        pass

    return analysis


def save_comprehensive_report(analysis: dict[str, Any], output_file: str = "reports/ensemble_vs_individual_backtest.json") -> None:
    """Save comprehensive analysis report to JSON."""
    # Convert numpy arrays and other non-serializable objects
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer | np.floating | np.bool_):
            return obj.item()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        return obj

    # Make analysis serializable
    serializable_analysis = make_serializable(analysis)

    # Add metadata
    serializable_analysis['metadata'] = {
        'generated_at': datetime.now().isoformat(),
        'script_version': '1.0',
        'methodology': 'Deterministic evaluation on validation split (80/20)',
        'data_source': 'data/features.parquet',
        'environment_config': {
            'initial_capital': 100000.0,
            'transaction_cost': 0.001,
            'window_size': 30,
            'validation_split': 0.2
        }
    }

    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(serializable_analysis, f, indent=2)



def main() -> int | None:
    """Main function to run comprehensive ensemble vs individual backtesting."""

    # Set seeds for reproducibility
    seed = 42
    set_all_seeds(seed)

    # Create validation environment using the same methodology as existing scripts
    try:
        env = create_validation_environment(
            data_file="data/features.parquet",
            initial_capital=100000.0,
            transaction_cost=0.001,
            window_size=30,
            validation_split=0.2,
            seed=seed
        )
    except Exception:
        traceback.print_exc()
        return 1

    # Load models
    ppo_paths = ["models/rl/ppo_final.zip", "models/rl/best_model.zip"]
    sac_paths = ["models/rl/sac.zip", "models/rl/sac_final.zip"]

    ppo_model = None
    sac_model = None

    for path in ppo_paths:
        ppo_model = load_model_safe(path, "ppo")
        if ppo_model is not None:
            break

    for path in sac_paths:
        sac_model = load_model_safe(path, "sac")
        if sac_model is not None:
            break

    if ppo_model is None or sac_model is None:
        return 1


    # Evaluation parameters
    n_episodes = 1  # Use single episode for deterministic comparison (same as existing scripts)

    try:
        # Evaluate individual policies

        ppo_metrics = evaluate_individual_policy(ppo_model, env, "PPO", n_episodes)
        env.reset()  # Reset environment between evaluations
        sac_metrics = evaluate_individual_policy(sac_model, env, "SAC", n_episodes)

        # Evaluate ensemble configurations

        ensemble_results = []

        # 1. Fixed weight ensemble (multiple weights)
        weights_to_test = [0.3, 0.5, 0.7]
        for weight in weights_to_test:
            env.reset()
            result = evaluate_ensemble_fixed_weight(ppo_model, sac_model, env, weight, n_episodes)
            result['ensemble_type'] = f'fixed_weight_{weight:.1f}'
            ensemble_results.append(result)

        # 2. Dynamic gating ensemble
        env.reset()
        gating_result = evaluate_ensemble_with_gating(ppo_model, sac_model, env, 0.5, n_episodes)
        gating_result['ensemble_type'] = 'dynamic_gating'
        ensemble_results.append(gating_result)

        # 3. Risk governor ensemble
        env.reset()
        governor_result = evaluate_ensemble_with_risk_governor(ppo_model, sac_model, env, 0.5, n_episodes)
        governor_result['ensemble_type'] = 'risk_governor'
        ensemble_results.append(governor_result)

        # Generate comprehensive analysis

        analysis = generate_comparative_analysis(ppo_metrics, sac_metrics, ensemble_results)

        # Save comprehensive report
        save_comprehensive_report(analysis)

        # Print final summary

        min_individual = min(ppo_metrics['mean_return'], sac_metrics['mean_return'])
        best_ensemble = max(ensemble_results, key=lambda x: x['mean_return'])


        if best_ensemble['mean_return'] >= min_individual:
            pass
        else:
            pass


        # Clean up temporary files
        with contextlib.suppress(FileNotFoundError):
            os.remove("data/val_temp.parquet")

        return 0

    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
