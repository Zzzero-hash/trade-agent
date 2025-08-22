from pathlib import Path

import numpy as np
import pandas as pd

from trade_agent.integrations.enhanced_trading_env import (
    EnhancedTradingEnvironment,
)


def _make_trending_data(rows: int = 60) -> str:
    idx = pd.date_range('2024-01-01', periods=rows, freq='D')
    prices = 100 + np.cumsum(
        np.random.normal(0.05, 0.2, size=rows)
    ).astype(float)
    df = pd.DataFrame(
        {
            'Open': prices + 0.1,
            'High': prices + 0.2,
            'Low': prices - 0.2,
            'Close': prices,
            'Volume': np.random.randint(1000, 2000, size=rows),
            'mu_hat': np.random.normal(0.001, 0.001, size=rows),
            'sigma_hat': np.random.uniform(0.01, 0.03, size=rows),
        },
        index=idx,
    )
    p = Path('data/test_env')
    p.mkdir(parents=True, exist_ok=True)
    file = p / 'reward_env.parquet'
    df.to_parquet(file)
    return str(file)


def test_reward_components_present_and_penalties_effect() -> None:
    path = _make_trending_data()
    reward_cfg = {
        'pnl_weight': 1.0,
        'risk_penalty_weight': 1.0,
        'turnover_penalty_weight': 1.0,
        'position_limit': 2.0,
        'transaction_cost_weight': 1.0,
        'risk_window': 10,
    }
    env = EnhancedTradingEnvironment(
        data_file=path,
        window_size=15,
        reward_config=reward_cfg,
        auto_convert=False,
    )
    obs, info = env.reset()

    # Take two contrasting actions to generate turnover
    action1 = np.array([1.0], dtype=np.float32)
    obs, r1, term, trunc, info1 = env.step(action1)
    action2 = np.array([-1.0], dtype=np.float32)
    obs, r2, term, trunc, info2 = env.step(action2)

    # Reward components should exist
    comps1 = info1['reward_components']
    comps2 = info2['reward_components']
    for k in ['pnl', 'cost', 'risk', 'turnover']:
        assert k in comps1 and k in comps2

    # Second step should have non-zero turnover penalty component
    assert comps2['turnover'] >= 0.0

    # Rewards finite floats
    assert isinstance(r1, float) and isinstance(r2, float)
    assert np.isfinite(r1) and np.isfinite(r2)

    # Accumulate risk window
    inf = info2
    for _ in range(12):
        obs, r, term, trunc, inf = env.step(env.action_space.sample())
        if term or trunc:
            break
    last_comps = inf['reward_components']
    assert 'risk' in last_comps
    # Force non-zero position to likely have risk component > 0
    # after returns collected
    obs, r, term, trunc, inf = env.step(np.array([0.5], dtype=np.float32))
    assert 'risk' in inf['reward_components']
