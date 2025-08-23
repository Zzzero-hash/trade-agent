# Trading System Architecture Overview

This document outlines the complete trading system architecture, which follows a modular design that separates concerns into distinct components. The system is designed for training reinforcement learning agents for algorithmic trading with proper risk management and ensemble methods.

## System Architecture Diagram

````{mermaid}
graph TD
    A[Market Data<br/>OHLCV, ticks] --> B[Data/Features<br/>Clean, engineer]
    B --> C[SL Forecasters<br/>E[r], σ, probs]
    C --> D[Gymnasium Trading Env<br/>obs_t = [features_t, SL_t]<br/>action_t ∈ [-1,1] target pos<br/>reward_t = f(PnL, risk, costs)]
    D --> E[RL<br/>Train PPO and SAC]
    E --> F[Ensemble Combiner<br/>a = w* a_SAC + (1-w)*a_PPO<br/>+ risk caps / governors]
    F --> G[Execution / Backtest / Live]
    ```{mermaid}
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#e8f5e8
    style D fill:#fff3e0
    style E fill:#fce4ec
    style F fill:#f1f8e9
    style G fill:#fafafa
````

## Detailed Module Structure

````{mermaid}
graph TD
    A[trade-agent Project]
    A --> B[data Package]
    A --> C[features Package]
    A --> D[sl Package]
    A --> E[envs Package]
    A --> F[rl Package]
    A --> G[ensemble Package]
    ```{mermaid}
    A --> I[docs Directory]

    B --> B1[collectors]
    B --> B2[storage]
    B --> B3[validators]
    B --> B4[pipelines]

    C --> C1[preprocessing]
    C --> C2[extraction]
    C --> C3[transformation]
    C --> C4[selection]
    C --> C5[pipelines]

    D --> D1[models]
    D --> D2[training]
    D --> D3[evaluation]
    D --> D4[leakage]
    D --> D5[pipelines]

    E --> E1[reward_functions]
    E --> E2[transaction_costs]
    E --> E3[risk_management]
    E --> E4[utils]

    F --> F1[ppo]
    F --> F2[sac]
    F --> F3[training]
    F --> F4[hyperparameter]
    F --> F5[utils]

    G --> G1[weighting]
    G --> G2[risk_management]
    G --> G3[validation]
    G --> G4[utils]

    H --> H1[backtesting]
    H --> H2[execution]
    H --> H3[live_trading]
    H --> H4[metrics]
    H --> H5[reporting]

    I --> I1[agents]
    I --> I2[data]
    I --> I3[features]
    I --> I4[sl]
    I --> I5[envs]
    I --> I6[rl]
    I --> I7[ensemble]
    I --> I8[eval]

    style A fill:#e0e0e0
    style B fill:#e1f5fe
    style C fill:#f3e5f5
    style D fill:#e8f5e8
    style E fill:#fff3e0
    style F fill:#fce4ec
    style G fill:#f1f8e9
    style H fill:#e0f2f1
    style I fill:#d7ccc8
````

## Component Overview

### 1. Market Data

Responsible for collecting and preprocessing raw market data including OHLCV (Open, High, Low, Close, Volume) data and tick data from various sources.

### 2. Data/Features

Processes raw market data into engineered features suitable for machine learning models. This includes data cleaning, normalization, and feature engineering.

### 3. SL Forecasters

Implements supervised learning models to forecast expected returns, volatility, and probability distributions. Designed with careful attention to prevent data leakage.

### 4. Gymnasium Trading Environment

A custom trading environment that follows the Gymnasium interface standard. It combines features and supervised learning predictions to create observations, defines actions as target positions, and computes rewards based on PnL, risk, and transaction costs.

### 5. Reinforcement Learning

Implements two RL algorithms:

- PPO (Proximal Policy Optimization) using Stable-Baselines3
- SAC (Soft Actor-Critic) using Ray RLlib

### 6. Ensemble Combiner

Combines predictions from both RL agents using weighted averaging, with additional risk management caps and governors.

### 7. Execution / Backtest / Live

Handles order execution, backtesting infrastructure, and live trading capabilities.

## Data Flow

The system follows a linear data flow where each component processes data and passes it to the next component. The reinforcement learning component trains on historical data, and the ensemble combiner makes final decisions for execution.

## Design Principles

1. **Modularity**: Each component is designed to be independent and replaceable
2. **No Data Leakage**: Special care is taken in the SL Forecasters to prevent future information from leaking into predictions
3. **Risk Management**: Built-in risk caps and governors in the ensemble layer
4. **Extensibility**: The architecture allows for easy addition of new components or replacement of existing ones
