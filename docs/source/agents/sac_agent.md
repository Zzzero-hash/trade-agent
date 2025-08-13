# SAC Agent with CNN+LSTM Model

## Overview

The SAC (Soft Actor-Critic) agent implementation integrates the CNN+LSTM model as a feature extractor within the Ray RLlib framework. This implementation is specifically designed for processing multi-asset financial time series data in reinforcement learning-based trading applications.

## Architecture

The SAC agent consists of three main components:

1. **CNNLSTMFeatureExtractor**: Wraps the CNN+LSTM model to make it compatible with Ray RLlib's model interface
2. **CNNLSTMSACModel**: Custom SAC model that uses the CNN+LSTM feature extractor with separate policy and Q-networks
3. **SACAgent**: High-level interface for creating and using a SAC agent with the CNN+LSTM feature extractor

## Key Components

### CNNLSTMFeatureExtractor

This class serves as a bridge between the CNN+LSTM model and Ray RLlib's model system. It processes multi-asset financial time series data and produces a fixed-size feature vector for the policy and value networks.

#### Parameters

- `obs_space`: The observation space of the environment
- `action_space`: The action space of the environment
- `num_outputs`: Number of outputs for the model
- `model_config`: Model configuration dictionary
- `name`: Name of the model
- `time_steps`: Number of time steps in the input sequence (default: 30)
- `assets`: Number of assets in the input (default: 100)
- `features`: Number of features per asset (default: 15)
- `cnn_filters`: Number of filters for each CNN layer (default: [64, 32, 16])
- `lstm_units`: Number of units for each LSTM layer (default: [128, 64])
- `dropout_rate`: Dropout rate for regularization (default: 0.2)
- `output_dim`: Dimension of the output features (default: 64)

### CNNLSTMSACModel

This custom SAC model integrates the CNN+LSTM feature extractor with separate policy and Q-networks, following the Ray RLlib model interface.

#### Parameters

All parameters from `CNNLSTMFeatureExtractor` plus:

- `q_hiddens`: Hidden layers for Q-networks (default: [256, 256])
- `policy_hiddens`: Hidden layers for policy network (default: [256, 256])

### SACAgent

This high-level interface provides a simplified way to create and use a SAC agent with the CNN+LSTM feature extractor for financial time series data.

#### Parameters

- `env`: The trading environment (name, class, or instance)
- `time_steps`: Number of time steps in the input sequence (default: 30)
- `assets`: Number of assets in the input (default: 100)
- `features`: Number of features per asset (default: 15)
- `cnn_filters`: Number of filters for each CNN layer (default: [64, 32, 16])
- `lstm_units`: Number of units for each LSTM layer (default: [128, 64])
- `dropout_rate`: Dropout rate for regularization (default: 0.2)
- `output_dim`: Dimension of the output features (default: 64)
- `learning_rate`: Learning rate for the optimizer (default: 3e-4)
- `tau`: Target network update rate (default: 0.005)
- `target_entropy`: Target entropy for entropy regularization (default: "auto")
- `q_hiddens`: Hidden layers for Q-networks (default: [256, 256])
- `policy_hiddens`: Hidden layers for policy network (default: [256, 256])
- `buffer_size`: Size of the replay buffer (default: 1000000)
- `batch_size`: Size of a batch sampled from replay buffer for training (default: 256)
- `train_batch_size`: Training batch size (default: 256)
- `gamma`: Discount factor (default: 0.99)
- `n_step`: N-step learning (default: 1)
- `num_workers`: Number of rollout workers (default: 0)
- `num_gpus`: Number of GPUs to use (default: 0)
- `framework`: Deep learning framework to use (default: "torch")

## Usage Example

```python
from src.agents.sac_agent import SACAgent
import gymnasium as gym

# Create your trading environment
# env = YourTradingEnvironment()

# Define CNN+LSTM parameters
TIME_STEPS = 30
ASSETS = 100
FEATURES = 15

# Create the SAC agent
agent = SACAgent(
    env=env,
    time_steps=TIME_STEPS,
    assets=ASSETS,
    features=FEATURES,
    output_dim=64,
    learning_rate=3e-4,
    buffer_size=100000,
    batch_size=256,
    train_batch_size=256,
    gamma=0.99,
    tau=0.005,
    num_workers=0,
    num_gpus=0,
    framework="torch",
)

# Train the agent
agent.train(num_iterations=1000)

# Compute action from the agent
observation = env.reset()
action = agent.compute_action(observation)

# Save the agent
agent.save("sac_agent_checkpoint")

# Restore the agent
agent.restore("sac_agent_checkpoint")
```

## Configuration Options

### CNN+LSTM Model Configuration

The CNN+LSTM model can be configured with the following parameters:

- `time_steps`: Number of time steps in the input sequence (default: 30)
- `assets`: Number of assets in the input (default: 100)
- `features`: Number of features per asset (default: 15)
- `cnn_filters`: Number of filters for each CNN layer (default: [64, 32, 16])
- `lstm_units`: Number of units for each LSTM layer (default: [128, 64])
- `dropout_rate`: Dropout rate for regularization (default: 0.2)
- `output_dim`: Dimension of the output features (default: 64)

### SAC Algorithm Configuration

The SAC algorithm can be configured with the following parameters:

- `learning_rate`: Learning rate for the optimizer (default: 3e-4)
- `tau`: Target network update rate (default: 0.005)
- `target_entropy`: Target entropy for entropy regularization (default: "auto")
- `buffer_size`: Size of the replay buffer (default: 1000000)
- `batch_size`: Size of a batch sampled from replay buffer for training (default: 256)
- `train_batch_size`: Training batch size (default: 256)
- `gamma`: Discount factor (default: 0.99)
- `n_step`: N-step learning (default: 1)
- `num_workers`: Number of rollout workers (default: 0)
- `num_gpus`: Number of GPUs to use (default: 0)

## Integration with CNN+LSTM Model

The SAC agent seamlessly integrates with the CNN+LSTM model architecture:

1. The `CNNLSTMFeatureExtractor` wraps the CNN+LSTM model to make it compatible with Ray RLlib
2. The feature extractor processes multi-asset financial time series data through:
   - CNN layers for spatial feature extraction from cross-sectional market data
   - LSTM layers for temporal sequence modeling across time steps
   - Dense layers for producing final output features
3. The output features are then used by the policy and Q-networks in the SAC algorithm

This integration allows the SAC agent to effectively capture both cross-sectional and temporal patterns in financial data while leveraging the proven SAC algorithm for policy optimization with maximum entropy reinforcement learning.
