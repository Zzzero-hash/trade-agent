# SAC Agent Configuration

## Overview

This document describes the configuration parameters for the SAC (Soft Actor-Critic) agent implementation. These parameters have been specifically tuned for the financial trading domain.

## Configuration Parameters

### SAC Algorithm Parameters

```json
{
  "sac": {
    "algorithm": "SAC",
    "learning_rate": 3e-4,
    "buffer_size": 1000000,
    "learning_starts": 1000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto",
    "target_update_interval": 1,
    "target_entropy": "auto",
    "seed": 42,
    "verbose": 1
  }
}
```

### Parameter Descriptions

| Parameter                | Description                    | Value   | Rationale                                                         |
| ------------------------ | ------------------------------ | ------- | ----------------------------------------------------------------- |
| `algorithm`              | Algorithm name                 | "SAC"   | Identifies the algorithm being used                               |
| `learning_rate`          | Learning rate for optimizer    | 3e-4    | Standard learning rate that works well for financial applications |
| `buffer_size`            | Size of replay buffer          | 1000000 | Large enough to store diverse experiences for financial data      |
| `learning_starts`        | Steps before learning starts   | 1000    | Allow sufficient experience collection before learning            |
| `batch_size`             | Minibatch size                 | 256     | Balance between stability and computational efficiency            |
| `tau`                    | Target network update rate     | 0.005   | Standard value for smooth target network updates                  |
| `gamma`                  | Discount factor                | 0.99    | Appropriate for financial time horizons                           |
| `train_freq`             | Training frequency             | 1       | Train every step for responsive learning                          |
| `gradient_steps`         | Gradient steps per update      | 1       | One gradient step per training update                             |
| `ent_coef`               | Entropy coefficient            | "auto"  | Automatic entropy tuning for optimal exploration                  |
| `target_update_interval` | Target network update interval | 1       | Update target networks every step                                 |
| `target_entropy`         | Target entropy                 | "auto"  | Automatic target entropy calculation                              |
| `seed`                   | Random seed                    | 42      | Ensures deterministic training for reproducibility                |
| `verbose`                | Verbosity level                | 1       | Basic training progress output                                    |

### Training Configuration

```json
{
  "training": {
    "total_timesteps": 1000000,
    "eval_freq": 10000,
    "checkpoint_freq": 50000
  }
}
```

### Training Parameter Descriptions

| Parameter         | Description              | Value   | Rationale                                        |
| ----------------- | ------------------------ | ------- | ------------------------------------------------ |
| `total_timesteps` | Total training timesteps | 1000000 | Sufficient for SAC to converge on financial data |
| `eval_freq`       | Evaluation frequency     | 10000   | Regular evaluation without excessive overhead    |
| `checkpoint_freq` | Checkpoint frequency     | 50000   | Save progress regularly for recovery             |

## Financial Domain Considerations

### Risk Sensitivity

The configuration uses a high discount factor (0.99) to account for the long-term nature of trading strategies while maintaining responsiveness to market changes.

### Transaction Cost Awareness

The reward structure in the trading environment already accounts for transaction costs, so the SAC hyperparameters are tuned to work with this cost-aware reward signal.

### Market Regime Adaptation

The automatic entropy tuning helps the agent adapt to different market regimes by balancing exploration and exploitation dynamically.

## Hyperparameter Tuning Guidelines

### Learning Rate

- Start with 3e-4 as a baseline
- If training is unstable, reduce to 1e-4
- If learning is too slow, increase to 1e-3

### Buffer Size

- For smaller datasets, use 100000-500000
- For larger datasets, use 1000000+
- Financial data typically works well with 1000000

### Batch Size

- For smaller datasets, use 128-256
- For larger datasets, use 256-512
- Financial data typically works well with 256

### Entropy Coefficient

- Use "auto" for automatic tuning
- If manual tuning needed, start with 0.1-0.5
- Higher values encourage more exploration

## Best Practices

### Deterministic Training

Always set a fixed seed (42) to ensure reproducible results across training runs.

### Evaluation Frequency

Evaluate the agent every 10,000 steps to monitor progress without excessive computational overhead.

### Checkpointing

Save checkpoints every 50,000 steps to enable recovery from failures and analysis of training progress.

## Performance Monitoring

### Key Metrics to Track

1. Episode rewards
2. Policy loss
3. Q-function loss
4. Entropy measures
5. Gradient norms

### Financial Metrics

1. Sharpe ratio
2. Maximum drawdown
3. Win rate
4. Profit factor
5. Calmar ratio

## Configuration File Location

The SAC configuration should be stored in `configs/sac_config.json` with the following structure:

```json
{
  "sac": {
    "algorithm": "SAC",
    "learning_rate": 3e-4,
    "buffer_size": 1000000,
    "learning_starts": 1000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto",
    "target_update_interval": 1,
    "target_entropy": "auto",
    "seed": 42,
    "verbose": 1
  },
  "training": {
    "total_timesteps": 1000000,
    "eval_freq": 10000,
    "checkpoint_freq": 50000
  }
}
```
