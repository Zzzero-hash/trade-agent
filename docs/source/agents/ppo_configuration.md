# PPO Agent Configuration

## Overview

This document describes the configuration parameters for the PPO (Proximal Policy Optimization) agent implementation. These parameters have been specifically tuned for the financial trading domain.

## Configuration Parameters

### PPO Algorithm Parameters

```json
{
  "ppo": {
    "algorithm": "PPO",
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "seed": 42
  }
}
```

### Parameter Descriptions

| Parameter       | Description                            | Value | Rationale                                                               |
| --------------- | -------------------------------------- | ----- | ----------------------------------------------------------------------- |
| `algorithm`     | Algorithm name                         | "PPO" | Identifies the algorithm being used                                     |
| `learning_rate` | Learning rate for optimizer            | 3e-4  | Standard learning rate that works well for financial applications       |
| `n_steps`       | Number of steps per environment update | 2048  | Large enough to get good estimates but not too large for financial data |
| `batch_size`    | Minibatch size                         | 64    | Balance between stability and computational efficiency                  |
| `n_epochs`      | Number of epochs per update            | 10    | Sufficient epochs for policy improvement without overfitting            |
| `gamma`         | Discount factor                        | 0.99  | Appropriate for financial time horizons                                 |
| `gae_lambda`    | GAE lambda parameter                   | 0.95  | Standard value for bias-variance tradeoff                               |
| `clip_range`    | Clipping parameter                     | 0.2   | PPO's key clipping parameter to prevent large updates                   |
| `ent_coef`      | Entropy coefficient                    | 0.0   | No explicit entropy regularization needed for this application          |
| `vf_coef`       | Value function coefficient             | 0.5   | Balance between policy and value function optimization                  |
| `max_grad_norm` | Gradient clipping                      | 0.5   | Prevents gradient explosion during training                             |
| `seed`          | Random seed                            | 42    | Ensures deterministic training for reproducibility                      |

### MLP Features Extractor Parameters

```json
{
  "mlp_features": {
    "input_dim": 514,
    "hidden_layers": [256, 128, 64],
    "output_dim": 64,
    "activation": "ReLU"
  }
}
```

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

## Financial Domain Considerations

### Risk Sensitivity

The configuration uses a high discount factor (0.99) to account for the long-term nature of trading strategies while maintaining responsiveness to market changes.

### Transaction Cost Awareness

The reward structure in the trading environment already accounts for transaction costs, so the PPO hyperparameters are tuned to work with this cost-aware reward signal.

### Market Regime Adaptation

The GAE lambda parameter (0.95) provides a good balance between bias and variance, allowing the agent to adapt to different market regimes while maintaining stable learning.

## Hyperparameter Tuning Guidelines

### Learning Rate

- Start with 3e-4 as a baseline
- If training is unstable, reduce to 1e-4
- If learning is too slow, increase to 1e-3

### Batch Size

- For smaller datasets, use 32-64
- For larger datasets, use 128-256
- Financial data typically works well with 64

### Clipping Range

- Standard value is 0.2
- For more conservative updates, reduce to 0.1
- For more aggressive learning, increase to 0.3

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
3. Value function loss
4. Entropy measures
5. Gradient norms

### Financial Metrics

1. Sharpe ratio
2. Maximum drawdown
3. Win rate
4. Profit factor
5. Calmar ratio
