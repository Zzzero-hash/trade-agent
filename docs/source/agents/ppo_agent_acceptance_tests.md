# PPO Agent Acceptance Tests

## Overview

This document outlines the acceptance tests for the PPO (Proximal Policy Optimization) agent implementation, ensuring it meets all requirements for integration with the trading environment and provides appropriate performance for financial trading.

## Test Environment

### Prerequisites

- Trading environment implementation complete
- PPO agent implementation complete
- Required dependencies installed
- Test data available

### Test Configuration

```python
TEST_CONFIG = {
    "seed": 42,
    "data_file": "data/sample_data.parquet",
    "initial_capital": 100000.0,
    "transaction_cost": 0.001,
    "window_size": 30,
    "total_timesteps": 10000,
    "eval_freq": 1000
}
```

## Acceptance Tests

### Test 1: PPO Agent Trains Successfully with Deterministic Results

#### Objective

Verify that the PPO agent can train successfully and produce deterministic results when using fixed seeds.

#### Test Steps

1. Initialize trading environment with fixed seed
2. Create PPO agent with the same fixed seed
3. Train the agent for a specified number of timesteps
4. Record final performance metrics
5. Repeat steps 1-4 with the same seeds
6. Compare results from both runs

#### Expected Results

- Training completes without errors
- Both runs produce identical results (same rewards, policies, etc.)
- No random variations in performance metrics

#### Implementation

```python
def test_deterministic_training():
    """Verify that PPO agent trains with deterministic results"""
    # First run
    env1 = TradingEnvironment(
        seed=TEST_CONFIG["seed"],
        data_file=TEST_CONFIG["data_file"],
        initial_capital=TEST_CONFIG["initial_capital"],
        transaction_cost=TEST_CONFIG["transaction_cost"],
        window_size=TEST_CONFIG["window_size"]
    )

    agent1 = PPOAgent(env1, seed=TEST_CONFIG["seed"])
    agent1.learn(total_timesteps=TEST_CONFIG["total_timesteps"])

    # Second run with same seed
    env2 = TradingEnvironment(
        seed=TEST_CONFIG["seed"],
        data_file=TEST_CONFIG["data_file"],
        initial_capital=TEST_CONFIG["initial_capital"],
        transaction_cost=TEST_CONFIG["transaction_cost"],
        window_size=TEST_CONFIG["window_size"]
    )

    agent2 = PPOAgent(env2, seed=TEST_CONFIG["seed"])
    agent2.learn(total_timesteps=TEST_CONFIG["total_timesteps"])

    # Compare results
    assert agent1.final_reward == agent2.final_reward
    assert agent1.policy_weights == agent2.policy_weights
```

### Test 2: Agent Integrates Correctly with Trading Environment

#### Objective

Verify that the PPO agent integrates correctly with the trading environment and can interact with it properly.

#### Test Steps

1. Initialize trading environment
2. Create PPO agent with the environment
3. Reset environment
4. Get observation from environment
5. Get action prediction from agent
6. Step environment with agent's action
7. Verify correct interaction at each step

#### Expected Results

- Agent can successfully get observations from environment
- Agent can produce valid actions within action space
- Environment can process agent's actions correctly
- No interface mismatch errors

#### Implementation

```python
def test_environment_integration():
    """Verify PPO agent integrates correctly with trading environment"""
    env = TradingEnvironment(
        seed=TEST_CONFIG["seed"],
        data_file=TEST_CONFIG["data_file"]
    )

    agent = PPOAgent(env)

    # Test reset
    obs, info = env.reset()
    assert obs.shape == (514,)  # Expected observation shape
    assert isinstance(info, dict)

    # Test prediction
    action, _ = agent.predict(obs)
    assert action.shape == (1,)  # Expected action shape
    assert -1.0 <= action[0] <= 1.0  # Action within bounds

    # Test step
    new_obs, reward, terminated, truncated, new_info = env.step(action)
    assert new_obs.shape == (514,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(new_info, dict)
```

### Test 3: Training Hyperparameters Are Appropriate for Financial Domain

#### Objective

Verify that the PPO agent uses hyperparameters appropriate for the financial trading domain.

#### Test Steps

1. Initialize PPO agent with default financial hyperparameters
2. Verify hyperparameter values against expected financial defaults
3. Train agent for a short period
4. Monitor training stability
5. Check that hyperparameters produce reasonable learning behavior

#### Expected Results

- Learning rate is appropriate for financial applications (3e-4)
- Batch size balances stability and efficiency (64)
- Clipping range prevents excessive policy updates (0.2)
- Training remains stable without gradient explosions
- Value function and policy losses behave reasonably

#### Implementation

```python
def test_financial_hyperparameters():
    """Verify PPO agent uses appropriate financial hyperparameters"""
    env = TradingEnvironment(seed=TEST_CONFIG["seed"])

    agent = PPOAgent(
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5
    )

    # Verify hyperparameters
    assert agent.learning_rate == 3e-4
    assert agent.batch_size == 64
    assert agent.clip_range == 0.2
    assert agent.gamma == 0.99

    # Short training to verify stability
    agent.learn(total_timesteps=1000)

    # Check that losses are reasonable
    assert agent.policy_loss < 10.0  # Reasonable upper bound
    assert agent.value_loss < 10.0   # Reasonable upper bound
```

### Test 4: Performance Meets Financial Objectives

#### Objective

Verify that the trained PPO agent meets basic financial performance objectives.

#### Test Steps

1. Train PPO agent on sample data
2. Evaluate performance metrics
3. Compare against baseline performance
4. Verify risk-adjusted returns are positive
5. Check that transaction costs are reasonable

#### Expected Results

- Sharpe ratio > 0.5 (positive risk-adjusted returns)
- Maximum drawdown < 20% (reasonable risk control)
- Win rate > 50% (more winning than losing trades)
- Profit factor > 1.1 (more profit than loss)
- Transaction costs < 5% of total returns

#### Implementation

```python
def test_financial_performance():
    """Verify PPO agent meets financial performance objectives"""
    env = TradingEnvironment(
        seed=TEST_CONFIG["seed"],
        data_file=TEST_CONFIG["data_file"]
    )

    agent = PPOAgent(env)
    agent.learn(total_timesteps=TEST_CONFIG["total_timesteps"])

    # Evaluate performance
    eval_env = TradingEnvironment(
        seed=TEST_CONFIG["seed"] + 1,  # Different seed for evaluation
        data_file=TEST_CONFIG["data_file"]
    )

    performance = evaluate_agent(agent, eval_env, n_episodes=10)

    # Check financial objectives
    assert performance["sharpe_ratio"] > 0.5
    assert performance["max_drawdown"] < 0.20
    assert performance["win_rate"] > 0.50
    assert performance["profit_factor"] > 1.1
    assert performance["transaction_costs_ratio"] < 0.05
```

### Test 5: Model Save and Load Functionality

#### Objective

Verify that the PPO agent can save and load models correctly.

#### Test Steps

1. Train PPO agent
2. Save the trained model
3. Load the model into a new agent instance
4. Compare predictions between original and loaded agents
5. Verify that performance is identical

#### Expected Results

- Model saves without errors
- Model loads without errors
- Loaded model produces identical predictions
- Loaded model has identical performance

#### Implementation

```python
def test_save_load_functionality():
    """Verify PPO agent save and load functionality"""
    env = TradingEnvironment(seed=TEST_CONFIG["seed"])
    agent = PPOAgent(env)
    agent.learn(total_timesteps=1000)

    # Save model
    save_path = "test_ppo_model.pkl"
    agent.save(save_path)

    # Load model into new agent
    new_env = TradingEnvironment(seed=TEST_CONFIG["seed"])
    new_agent = PPOAgent(new_env)
    new_agent.load(save_path)

    # Compare predictions
    obs, _ = new_env.reset()
    action1, _ = agent.predict(obs)
    action2, _ = new_agent.predict(obs)

    assert np.allclose(action1, action2)
```

## Test Execution

### Running All Tests

```bash
# Run all PPO acceptance tests
python -m pytest tests/rl/ppo/test_ppo_acceptance.py -v

# Run specific test
python -m pytest tests/rl/ppo/test_ppo_acceptance.py::test_deterministic_training -v
```

### Test Reporting

- Generate detailed test reports
- Log any failures with full traceback
- Save performance metrics for analysis
- Create summary report of all test results

## Success Criteria

### Pass Criteria

All acceptance tests must pass with:

- No errors or exceptions
- All assertions satisfied
- Performance within expected ranges
- Deterministic behavior verified

### Failure Handling

If any test fails:

1. Log detailed error information
2. Identify root cause
3. Fix implementation issues
4. Re-run failed test
5. Verify all other tests still pass

## Performance Benchmarks

### Expected Training Performance

- Training time for 10,000 steps: < 30 minutes
- Memory usage: < 2GB
- CPU utilization: < 80%
- GPU utilization (if available): < 90%

### Expected Evaluation Performance

- Inference time per step: < 10ms
- Episode completion time: < 1 minute
- Evaluation accuracy: > 99.9%

## Validation Metrics

### Financial Metrics

- Sharpe Ratio: > 0.5
- Maximum Drawdown: < 20%
- Win Rate: > 50%
- Profit Factor: > 1.1
- Calmar Ratio: > 1.0

### Technical Metrics

- Training Stability: No gradient explosions
- Deterministic Behavior: 100% reproducibility
- Interface Compliance: Full Gymnasium compatibility
- Resource Usage: Within system constraints

## Rollback Plan

If acceptance tests fail:

1. Identify which tests are failing
2. Review implementation against requirements
3. Fix specific issues
4. Re-run failed tests
5. If issues persist, rollback to previous working version
6. Re-implement with corrected approach
