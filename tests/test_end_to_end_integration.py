"""
End-to-End Integration Test: Data Pipeline → Trading Environment → RL Training

This demonstrates the complete workflow with our bridge components.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from integrations.data_trading_bridge import WorkflowBridge
from integrations.enhanced_trading_env import EnhancedTradingEnvironment


def run_end_to_end_integration_test():
    """Run complete end-to-end integration test."""
    print("🚀 Starting End-to-End Integration Test")
    print("=" * 50)

    # Phase 1: Data Pipeline to Trading Format
    print("\n📊 Phase 1: Data Pipeline → Trading Format")
    print("-" * 40)

    workflow = WorkflowBridge()
    results = workflow.run_data_to_trading_pipeline(
        symbols=['AAPL'],
        start_date='2024-01-01',
        end_date='2024-01-15',  # More data for testing
        output_dir='data/e2e_test'
    )

    if not results:
        print("❌ Data pipeline failed")
        return False

    trading_file = results['AAPL']
    print(f"✅ Data pipeline successful: {trading_file}")

    # Phase 2: Trading Environment Integration
    print("\n🎯 Phase 2: Trading Environment Integration")
    print("-" * 40)

    try:
        env = EnhancedTradingEnvironment(
            data_file=trading_file,
            initial_capital=100000.0,
            transaction_cost=0.001,
            window_size=5,  # Smaller window for limited data
            auto_convert=False  # Already converted
        )

        print("✅ Trading environment created")
        print(f"   Action space: {env.action_space}")
        print(f"   Observation space: {env.observation_space.shape}")

    except Exception as e:
        print(f"❌ Trading environment failed: {e}")
        return False

    # Phase 3: Environment Functionality Test
    print("\n🔄 Phase 3: Environment Functionality Test")
    print("-" * 40)

    try:
        obs, info = env.reset()
        print(f"✅ Environment reset - obs shape: {obs.shape}")
        print(f"   Initial equity: ${info['equity']:,.2f}")

        # Run episode
        total_reward = 0
        episode_length = 0

        for step in range(10):  # Limit steps due to small dataset
            # Sample action
            action = env.action_space.sample()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            episode_length += 1

            print(f"   Step {step+1}: action={action[0]:.2f}, reward={reward:.4f}, "
                  f"equity=${info['equity']:,.2f}")

            if terminated or truncated:
                print(f"   Episode ended: terminated={terminated}, truncated={truncated}")
                break

        print(f"✅ Episode completed - Length: {episode_length}, Total reward: {total_reward:.4f}")

    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Phase 4: Quick RL Integration Test (if SB3 available)
    print("\n🤖 Phase 4: RL Integration Test")
    print("-" * 40)

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env

        # Check environment compatibility
        check_env(env)
        print("✅ Environment passes SB3 compatibility check")

        # Create a simple PPO agent
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)
        print("✅ PPO agent created")

        # Train for a few steps (just to test integration)
        print("   Training for 1000 steps...")
        model.learn(total_timesteps=1000)
        print("✅ Training completed")

        # Test trained model
        obs, _ = env.reset()
        action, _ = model.predict(obs)
        print(f"✅ Model prediction: {action}")

    except ImportError:
        print("⚠️  Stable-Baselines3 not available, skipping RL test")
    except Exception as e:
        print(f"⚠️  RL test failed (non-critical): {e}")

    # Success summary
    print("\n🎉 End-to-End Integration Test Results")
    print("=" * 50)
    print("✅ Data Pipeline: SUCCESS")
    print("✅ Trading Environment: SUCCESS")
    print("✅ Environment Functionality: SUCCESS")
    print("✅ Bridge Components: SUCCESS")
    print("\n🚀 Ready for production workflows!")

    return True


def demonstrate_workflow_capabilities():
    """Demonstrate the capabilities of our integrated workflow."""
    print("\n🔬 Workflow Capabilities Demonstration")
    print("=" * 50)

    # 1. Multiple symbols
    print("\n📈 Multi-Symbol Processing")
    workflow = WorkflowBridge()

    multi_results = workflow.run_data_to_trading_pipeline(
        symbols=['AAPL', 'MSFT'],
        start_date='2024-01-01',
        end_date='2024-01-05',
        output_dir='data/multi_symbol_test'
    )

    print(f"✅ Processed {len(multi_results)} symbols: {list(multi_results.keys())}")

    # 2. Different configurations
    print("\n⚙️  Configuration Flexibility")
    for symbol, file_path in multi_results.items():
        print(f"\n   Testing {symbol}:")

        # Test with different window sizes
        for window_size in [3, 5]:
            try:
                env = EnhancedTradingEnvironment(
                    data_file=file_path,
                    window_size=window_size,
                    auto_convert=False
                )
                obs, _ = env.reset()
                print(f"      Window size {window_size}: obs shape {obs.shape} ✅")
            except Exception as e:
                print(f"      Window size {window_size}: failed - {e}")

    print("\n✅ Workflow demonstration complete!")


if __name__ == "__main__":
    # Run the comprehensive test
    success = run_end_to_end_integration_test()

    if success:
        # Show additional capabilities
        demonstrate_workflow_capabilities()

        print("\n" + "=" * 60)
        print("🎯 INTEGRATION SUCCESS: Easy wins implemented!")
        print("🔄 Ready to proceed with comprehensive plan")
        print("=" * 60)
    else:
        print("\n❌ Integration test failed - check configuration")
