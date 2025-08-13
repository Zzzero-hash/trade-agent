# Gymnasium Trading Environment

The Gymnasium Trading Environment component implements a custom trading environment that follows the Gymnasium interface standard. It combines features and supervised learning predictions to create observations, defines actions as target positions, and computes rewards based on PnL, risk, and transaction costs.

## Overview

The Trading Environment component handles:

1. Environment state management
2. Observation space definition
3. Action space definition
4. Reward function design
5. Transaction cost modeling
6. Risk management integration
7. Episode management and termination conditions

## Environment Design

### Observation Space

The observation at time t consists of:

- `features_t`: Engineered features from the Data/Features component
- `SL_t`: Supervised learning predictions from the SL Forecasters component

```python
observation_space = spaces.Dict({
    'features': spaces.Box(low=-np.inf, high=np.inf, shape=(n_features,)),
    'predictions': spaces.Box(low=-np.inf, high=np.inf, shape=(n_predictions,))
})
```

### Action Space

Actions represent target portfolio positions:

- Continuous action space: `action_t ∈ [-1, 1]` where:
  - -1: Maximum short position
  - 0: Flat position
  - 1: Maximum long position

```python
action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_assets,))
```

### Reward Function

The reward function balances multiple objectives:

- Profit and Loss (PnL)
- Risk-adjusted returns
- Transaction costs
- Portfolio stability

```python
reward = sharpe_ratio * risk_aversion - transaction_costs - penalty
```

## Environment State

### Portfolio State

- Current positions for each asset
- Cash balance
- Total portfolio value
- Historical performance metrics

### Market State

- Current market prices
- Market volatility estimates
- Liquidity conditions
- Market regime indicators

### Time State

- Current time step
- Episode start and end times
- Trading calendar information

## Reward Function Components

### Profit and Loss (PnL)

- Realized PnL from position changes
- Unrealized PnL from current positions
- Dividend and corporate action adjustments

### Risk Measures

- Portfolio volatility
- Value-at-Risk (VaR)
- Maximum drawdown
- Beta to market factors

### Transaction Costs

- Fixed costs per trade
- Variable costs based on trade size
- Market impact costs
- Slippage modeling

### Penalties

- Position concentration penalties
- Leverage constraints
- Short sale restrictions
- Holding period penalties

## Environment Pipeline

```{mermaid}
graph TD
    A[Reset Environment] --> B[Initial State]
    B --> C[Get Observation]
    C --> D[Agent Action]
    D --> E[Execute Trade]
    E --> F[Update State]
    F --> G[Calculate Reward]
    G --> H[Check Termination]
    H -->|Continue| C
    H -->|Terminate| I[Episode End]

    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#e8f5e8
    style D fill:#fff3e0
    style E fill:#fce4ec
    style F fill:#f1f8e9
    style G fill:#e0f2f1
    style H fill:#d7ccc8
    style I fill:#fafafa
```

## Transaction Cost Modeling

### Fixed Costs

- Brokerage fees
- Exchange fees
- Regulatory fees

### Variable Costs

- Bid-ask spread
- Market impact
- Slippage

### Market Impact Model

- Linear impact model
- Square-root impact model
- Temporary vs. permanent impact

## Risk Management Integration

### Position Limits

- Individual asset exposure limits
- Sector concentration limits
- Country/region exposure limits

### Leverage Constraints

- Gross exposure limits
- Net exposure limits
- Margin requirements

### VaR Constraints

- Portfolio VaR limits
- Component VaR monitoring
- Stress testing

## Module Structure

```
src/envs/
├── __init__.py
├── trading_env.py
├── reward_functions/
│   ├── __init__.py
│   ├── base_reward.py
│   ├── sharpe_reward.py
│   └── risk_adjusted.py
├── transaction_costs/
│   ├── __init__.py
│   ├── fixed_costs.py
│   ├── variable_costs.py
│   └── market_impact.py
├── risk_management/
│   ├── __init__.py
│   ├── position_limits.py
│   ├── leverage_constraints.py
│   └── var_models.py
└── utils/
    ├── __init__.py
    ├── observation_builder.py
    ├── action_processor.py
    └── state_tracker.py
```

## Interfaces

### Environment Interface

```python
class TradingEnvironment(gym.Env):
    def __init__(self, config):
        """Initialize the trading environment"""
        pass

    def reset(self):
        """Reset the environment to initial state"""
        pass

    def step(self, action):
        """Execute one time step within the environment"""
        pass

    def render(self, mode='human'):
        """Render the environment"""
        pass

    def close(self):
        """Clean up resources"""
        pass
```

### Reward Function Interface

```python
class RewardFunction:
    def calculate_reward(self, state, action, next_state):
        """Calculate reward for transition"""
        pass

    def reset(self):
        """Reset reward function state"""
        pass
```

## Configuration

The Trading Environment can be configured through configuration files:

```yaml
env:
  observation:
    features_window: 30
    prediction_horizon: 5

  action:
    max_position: 1.0
    min_position: -1.0
    position_step: 0.1

  reward:
    function: "sharpe_ratio"
    risk_aversion: 0.5
    transaction_cost_weight: 0.1
    penalty_weight: 0.05

  transaction_costs:
    fixed_cost: 0.001
    variable_cost: 0.0001
    market_impact: 0.1

  risk_management:
    max_position_size: 0.1
    max_sector_exposure: 0.3
    max_leverage: 2.0
```

## Episode Management

### Episode Start

- Initialize portfolio with starting capital
- Set initial positions
- Load historical data for the episode

### Episode Progression

- Process one time step per call to step()
- Update portfolio based on market movements
- Calculate and return reward

### Episode Termination

- End of data
- Portfolio bankruptcy
- Maximum drawdown exceeded
- Manual termination

## Performance Considerations

- Vectorized calculations for efficiency
- Memory-efficient state storage
- Fast market data access
- Parallel environment instances for training
- Just-in-time compilation for critical functions
