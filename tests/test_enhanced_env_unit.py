from pathlib import Path

import numpy as np
import pandas as pd

from trade_agent.integrations.enhanced_trading_env import (
    EnhancedTradingEnvironment,
)


def _make_minimal_data(rows=15):
    idx = pd.date_range('2024-01-01', periods=rows, freq='D')
    prices = np.linspace(100, 101, rows)
    df = pd.DataFrame({
        'Open': prices + 0.1,
        'High': prices + 0.2,
        'Low': prices - 0.2,
        'Close': prices,
        'Volume': np.random.randint(1000, 2000, size=rows),
        'mu_hat': np.random.normal(0, 0.001, size=rows),
        'sigma_hat': np.random.uniform(0.01, 0.03, size=rows)
    }, index=idx)
    p = Path('data/test_env')
    p.mkdir(parents=True, exist_ok=True)
    file = p / 'test_env.parquet'
    df.to_parquet(file)
    return str(file)

def test_reward_is_float() -> None:
    path = _make_minimal_data()
    env = EnhancedTradingEnvironment(data_file=path, window_size=5, auto_convert=False)
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, float)


def test_window_padding() -> None:
    path = _make_minimal_data(rows=15)
    env = EnhancedTradingEnvironment(data_file=path, window_size=10, auto_convert=False)
    obs, info = env.reset()
    # Observation length must match observation_space
    assert obs.shape[0] == env.observation_space.shape[0]
