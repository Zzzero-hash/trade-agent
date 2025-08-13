# PPO Agent with CNN+LSTM Model

## Overview

The PPO (Proximal Policy Optimization) agent implementation integrates the CNN+LSTM model as a feature extractor within the Stable-Baselines3 framework. This implementation is specifically designed for processing multi-asset financial time series data in reinforcement learning-based trading applications.

## Architecture

The PPO agent consists of three main components:

1. **CNNLSTMFeaturesExtractor**: Wraps the CNN+LSTM model to make it compatible with Stable-Baselines3's feature extractor interface
2. **CNNLSTMPolicy**: Custom policy that uses the CNN+LSTM feature extractor with separate policy and value networks
3. **PPOAgent**: High-level interface for creating and using a PPO agent with the CNN+LSTM feature extractor

## Key Components

### CNNLSTMFeaturesExtractor

This class serves as a bridge between the CNN+LSTM model and Stable-Baselines3's feature extraction system. It processes multi-asset financial time series data and produces a fixed-size feature vector for the policy network.

#### Parameters

- `observation_space`: The observation space of the environment
- `time_steps`: Number of time steps in the input sequence (default: 30)
- `assets`: Number of assets in the input (default: 100)
- `features`: Number of features per asset (default: 15)
- `cnn_filters`: Number of filters for each CNN layer (default: [64, 32, 16])
- `lstm_units`: Number of units for each LSTM layer (default: [128, 64])
- `dropout_rate`: Dropout rate for regularization (default: 0.2)
- `output_dim`: Dimension of the output features (default: 64)

### CNNLSTMPolicy

This custom policy integrates the CNN+LSTM feature extractor with separate policy and value networks, following the Stable-Baselines3 policy interface.

#### Parameters

All parameters from `CNNLSTMFeaturesExtractor` plus:

- `net_arch`: Specification of the policy and value networks
- `activation_fn`: Activation function (default: nn.Tanh)
- `ortho_init`: Whether to use orthogonal initialization (default: True)
- `use_sde`: Whether to use generalized State Dependent Exploration (default: False)
- `log_std_init`: Initial value for the log standard deviation (default: 0.0)
- And other standard Stable-Baselines3 policy parameters

### PPOAgent

This high-level interface provides a simplified way to create and use a PPO agent with the CNN+LSTM feature extractor for financial time series data.

#### Parameters

All parameters from `CNNLSTMPolicy` plus PPO-specific parameters:

- `env`: The trading environment
- `learning_rate`: Learning rate for the optimizer (default: 3e-4)
- `n_steps`: Number of steps to run for each environment per update (default: 2048)
- `batch_size`: Minibatch size (default: 64)
- `n_epochs`: Number of epochs when optimizing the surrogate loss (default: 10)
- `gamma`: Discount factor (default: 0.99)
- `gae_lambda`: Factor for trade-off of bias vs variance for Generalized Advantage Estimator (default: 0.95)
- `clip_range`: Clipping parameter (default: 0.2)
- And other standard PPO parameters

## Usage Example

```python
from src.agents.ppo_agent import PPOAgent
import gymnasium as gym

# Create your trading environment
# env = YourTradingEnvironment()

# Define CNN+LSTM parameters
TIME_STEPS = 30
ASSETS = 100
FEATURES = 15

# Create the PPO agent
agent = PPOAgent(
    env=env,
    time_steps=TIME_STEPS,
    assets=ASSETS,
    features=FEATURES,
    output_dim=64,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
)

# Train the agent
agent.learn(total_timesteps=100000)

# Get action from the agent
observation, _ = env.reset()
action, _ = agent.predict(observation)

# Save the agent
agent.save("ppo_agent_model")

# Load the agent
agent.load("ppo_agent_model")
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

### PPO Algorithm Configuration

The PPO algorithm can be configured with the following parameters:

- `learning_rate`: Learning rate for the optimizer (default: 3e-4)
- `n_steps`: Number of steps to run for each environment per update (default: 2048)
- `batch_size`: Minibatch size (default: 64)
- `n_epochs`: Number of epochs when optimizing the surrogate loss (default: 10)
- `gamma`: Discount factor (default: 0.99)
- `gae_lambda`: Factor for trade-off of bias vs variance for Generalized Advantage Estimator (default: 0.95)
- `clip_range`: Clipping parameter (default: 0.2)
- `ent_coef`: Entropy coefficient for the loss calculation (default: 0.0)
- `vf_coef`: Value function coefficient for the loss calculation (default: 0.5)
- `max_grad_norm`: The maximum value for the gradient clipping (default: 0.5)

## Integration with CNN+LSTM Model

The PPO agent seamlessly integrates with the CNN+LSTM model architecture:

1. The `CNNLSTMFeaturesExtractor` wraps the CNN+LSTM model to make it compatible with Stable-Baselines3
2. The feature extractor processes multi-asset financial time series data through:
   - CNN layers for spatial feature extraction from cross-sectional market data
   - LSTM layers for temporal sequence modeling across time steps
   - Dense layers for producing final output features
3. The output features are then used by the policy and value networks in the PPO algorithm

This integration allows the PPO agent to effectively capture both cross-sectional and temporal patterns in financial data while leveraging the proven PPO algorithm for policy optimization.
