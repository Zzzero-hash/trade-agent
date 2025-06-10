"""
Integration tests for TD3 Agent with Trading Environment.
Tests the complete pipeline from config to training.
"""

import pytest
import numpy as np
import torch

pytestmark = pytest.mark.integration
from src.agents.td3_agent import TD3Agent
from src.agents.configs import TD3Config
from src.envs.trading_env import TradingEnv


class TestTD3Integration:
    """Integration tests for TD3 agent with trading environment."""
    
    @pytest.fixture
    def trading_env(self):
        """Create a trading environment for testing."""
        env_cfg = {
            "dataset_paths": ["data/sample_training_data_simple_20250607_192034.csv"],
            "window_size": 10,
            "initial_balance": 10000,
            "transaction_cost": 0.001,
            "include_features": False
        }
        return TradingEnv(env_cfg)
    
    @pytest.fixture
    def td3_config(self):
        """Create TD3 config optimized for integration testing."""
        return TD3Config(
            learning_rate=1e-3,
            gamma=0.99,
            tau=0.01,  # Faster updates for testing
            batch_size=16,  # Smaller batch for faster testing
            buffer_capacity=1000,
            hidden_dims=[32, 32],  # Smaller networks for speed
            policy_delay=2,
            target_noise=0.1,
            noise_clip=0.3,
            exploration_noise=0.1
        )
    
    def test_td3_with_trading_env_initialization(self, trading_env, td3_config):
        """Test TD3 agent initializes correctly with trading environment."""
        state_dim = trading_env.observation_space.shape[0]
        action_dim = trading_env.action_space.shape[0]
        
        agent = TD3Agent(
            config=td3_config,
            state_dim=state_dim,
            action_dim=action_dim
        )
        
        # Verify dimensions match
        assert agent.state_dim == state_dim
        assert agent.action_dim == action_dim
        
        # Verify networks have correct input/output dimensions
        dummy_state = torch.randn(1, state_dim)
        dummy_action = torch.randn(1, action_dim)
        
        # Test actor output shape
        with torch.no_grad():
            action_output = agent.actor(dummy_state)
            assert action_output.shape == (1, action_dim)
        
        # Test critic output shape
        with torch.no_grad():
            q1_output = agent.critic_1(dummy_state, dummy_action)
            q2_output = agent.critic_2(dummy_state, dummy_action)
            assert q1_output.shape == (1, 1)
            assert q2_output.shape == (1, 1)
    
    def test_td3_environment_interaction(self, trading_env, td3_config):
        """Test TD3 agent can interact with trading environment."""
        state_dim = trading_env.observation_space.shape[0]
        action_dim = trading_env.action_space.shape[0]
        
        agent = TD3Agent(
            config=td3_config,
            state_dim=state_dim,
            action_dim=action_dim
        )
        
        # Test environment reset and action selection
        state = trading_env.reset()
        assert len(state) == state_dim
        
        # Test action selection
        action = agent.select_action(state, add_noise=False)
        assert len(action) == action_dim
        assert all(-1.0 <= a <= 1.0 for a in action)  # Actions should be bounded
        
        # Test environment step
        next_state, reward, done, info = trading_env.step(action)
        assert len(next_state) == state_dim
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
    
    def test_td3_training_episode(self, trading_env, td3_config):
        """Test TD3 agent can complete a training episode."""
        state_dim = trading_env.observation_space.shape[0]
        action_dim = trading_env.action_space.shape[0]
        
        agent = TD3Agent(
            config=td3_config,
            state_dim=state_dim,
            action_dim=action_dim
        )
        
        # Collect experiences for one episode
        state = trading_env.reset()
        episode_length = 0
        max_steps = 100  # Limit episode length for testing
        
        while episode_length < max_steps:
            action = agent.select_action(state, add_noise=True)
            next_state, reward, done, info = trading_env.step(action)
            
            # Store experience
            agent.store_experience(state, action, reward, next_state, done)
            
            state = next_state
            episode_length += 1
            
            if done:
                break
        
        # Verify experiences were stored
        assert len(agent.replay_buffer) == episode_length
        
        # Test training if we have enough experiences
        if len(agent.replay_buffer) >= agent.batch_size:
            initial_total_it = agent.total_it
            metrics = agent.train()
            
            # Verify training occurred
            assert agent.total_it == initial_total_it + 1
            assert isinstance(metrics, dict)
            assert "critic_1_loss" in metrics
            assert "critic_2_loss" in metrics
    
    def test_td3_config_dataclass_integration(self, trading_env):
        """Test TD3 works with dataclass config in realistic scenario."""
        # Test various config scenarios
        configs_to_test = [
            TD3Config(),  # Default config
            TD3Config(learning_rate=5e-4, batch_size=32),  # Custom params
            TD3Config(hidden_dims=[128, 128], policy_delay=3)  # Different architecture
        ]
        
        state_dim = trading_env.observation_space.shape[0]
        action_dim = trading_env.action_space.shape[0]
        
        for config in configs_to_test:
            agent = TD3Agent(
                config=config,
                state_dim=state_dim,
                action_dim=action_dim
            )
            
            # Verify config was applied correctly
            assert agent.lr == config.learning_rate
            assert agent.batch_size == config.batch_size
            assert agent.hidden_dims == config.hidden_dims
            assert agent.policy_delay == config.policy_delay
            
            # Test basic functionality
            state = trading_env.reset()
            action = agent.select_action(state)
            assert len(action) == action_dim
    
    def test_td3_save_load_with_training_state(self, trading_env, td3_config, tmp_path):
        """Test TD3 agent save/load preserves training state."""
        state_dim = trading_env.observation_space.shape[0]
        action_dim = trading_env.action_space.shape[0]
        
        # Create and train agent
        agent1 = TD3Agent(
            config=td3_config,
            state_dim=state_dim,
            action_dim=action_dim
        )
        
        # Add some experiences and train
        for _ in range(20):
            state = np.random.randn(state_dim).astype(np.float32)
            action = np.random.uniform(-1, 1, action_dim).astype(np.float32)
            reward = np.random.randn()
            next_state = np.random.randn(state_dim).astype(np.float32)
            done = False
            agent1.store_experience(state, action, reward, next_state, done)
        
        # Train a few steps
        for _ in range(3):
            agent1.train()
        
        original_total_it = agent1.total_it
        
        # Save agent
        save_path = tmp_path / "td3_agent.pth"
        agent1.save(str(save_path))
        
        # Create new agent and load
        agent2 = TD3Agent(
            config=td3_config,
            state_dim=state_dim,
            action_dim=action_dim
        )
        agent2.load(str(save_path))
        
        # Verify training state was preserved
        assert agent2.total_it == original_total_it
        
        # Verify agents produce similar outputs
        test_state = np.random.randn(state_dim)
        action1 = agent1.select_action(test_state, add_noise=False)
        action2 = agent2.select_action(test_state, add_noise=False)
        
        # Actions should be very similar (allowing for small numerical differences)
        np.testing.assert_allclose(action1, action2, rtol=1e-5)


if __name__ == "__main__":
    # Quick integration test
    import sys
    sys.path.append('/workspaces/trading-rl-agent')
    
    from src.envs.trading_env import TradingEnv
    from src.agents.configs import TD3Config
    
    print("🧪 Running TD3 Integration Test...")
    
    # Initialize environment and agent
    env_cfg = {
        "dataset_paths": ["data/sample_training_data_simple_20250607_192034.csv"],
        "window_size": 10,
        "initial_balance": 10000,
        "transaction_cost": 0.001,
        "include_features": False
    }
    env = TradingEnv(env_cfg)
    config = TD3Config(batch_size=16, buffer_capacity=100)
    
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]  # Flatten
    action_dim = 1  # TD3 needs continuous actions, but trading env has discrete
    
    agent = TD3Agent(config=config, state_dim=state_dim, action_dim=action_dim)
    
    print(f"✅ Environment: {state_dim} states, {action_dim} actions")
    print(f"✅ Agent initialized with {sum(p.numel() for p in agent.actor.parameters())} actor parameters")
    
    # Test interaction (simplified)
    state, info = env.reset()
    if len(state.shape) > 1:
        state = state.flatten()  # Flatten for TD3
        
    action_continuous = agent.select_action(state)
    # Convert continuous action to discrete for environment
    action_discrete = 1 if action_continuous[0] > 0.33 else (2 if action_continuous[0] < -0.33 else 0)
    
    result = env.step(action_discrete)
    next_state, reward, terminated, truncated, info = result
    done = terminated or truncated
    
    print(f"✅ Environment interaction successful")
    print(f"   Continuous Action: {action_continuous}")
    print(f"   Discrete Action: {action_discrete}")
    print(f"   Reward: {reward:.4f}")
    print(f"   Done: {done}")
    
    print("🎉 TD3 Integration Test Passed!")
