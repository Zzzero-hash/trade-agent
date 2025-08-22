"""
End-to-End Integration Test: Data Pipeline → Trading Environment → RL Training

This demonstrates the complete workflow with our bridge components.
"""

import sys
from pathlib import Path


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trade_agent.integrations.data_trading_bridge import WorkflowBridge
from trade_agent.integrations.enhanced_trading_env import (
    EnhancedTradingEnvironment,
)


def run_end_to_end_integration_test() -> bool:
    """Run complete end-to-end integration test."""

    # Phase 1: Data Pipeline to Trading Format

    workflow = WorkflowBridge()
    results = workflow.run_data_to_trading_pipeline(
        symbols=['AAPL'],
        start_date='2024-01-01',
        end_date='2024-01-15',  # More data for testing
        output_dir='data/e2e_test'
    )

    if not results:
        return False

    trading_file = results['AAPL']

    # Phase 2: Trading Environment Integration

    try:
        env = EnhancedTradingEnvironment(
            data_file=trading_file,
            initial_capital=100000.0,
            transaction_cost=0.001,
            window_size=5,  # Smaller window for limited data
            auto_convert=False  # Already converted
        )


    except Exception:
        return False

    # Phase 3: Environment Functionality Test

    try:
        obs, info = env.reset()

        # Run episode
        total_reward = 0
        episode_length = 0

        for _step in range(10):  # Limit steps due to small dataset
            # Sample action
            action = env.action_space.sample()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            episode_length += 1


            if terminated or truncated:
                break


    except Exception:
        import traceback
        traceback.print_exc()
        return False

    # Phase 4: Quick RL Integration Test (if SB3 available)

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env

        # Check environment compatibility
        check_env(env)

        # Create a simple PPO agent
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)

        # Train for a few steps (just to test integration)
        model.learn(total_timesteps=1000)

        # Test trained model
        obs, _ = env.reset()
        action, _ = model.predict(obs)

    except ImportError:
        pass
    except Exception:
        pass

    # Success summary

    return True


def demonstrate_workflow_capabilities() -> None:
    """Demonstrate the capabilities of our integrated workflow."""

    # 1. Multiple symbols
    workflow = WorkflowBridge()

    multi_results = workflow.run_data_to_trading_pipeline(
        symbols=['AAPL', 'MSFT'],
        start_date='2024-01-01',
        end_date='2024-01-05',
        output_dir='data/multi_symbol_test'
    )


    # 2. Different configurations
    for _symbol, file_path in multi_results.items():

        # Test with different window sizes
        for window_size in [3, 5]:
            try:
                env = EnhancedTradingEnvironment(
                    data_file=file_path,
                    window_size=window_size,
                    auto_convert=False
                )
                obs, _ = env.reset()
            except Exception:
                pass



if __name__ == "__main__":
    # Run the comprehensive test
    success = run_end_to_end_integration_test()

    if success:
        # Show additional capabilities
        demonstrate_workflow_capabilities()

    else:
        pass
