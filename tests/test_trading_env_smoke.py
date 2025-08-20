import numpy as np

from trade_agent.envs.trading_env import TradingEnvironment


def test_trading_environment_basic_step() -> None:
    env = TradingEnvironment()
    obs, info = env.reset()
    assert obs.shape[0] == env.observation_space.shape[0]
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_trading_environment_deterministic_seed() -> None:
    env1 = TradingEnvironment(seed=123)
    env2 = TradingEnvironment(seed=123)
    o1, _ = env1.reset()
    o2, _ = env2.reset()
    np.testing.assert_allclose(o1, o2)
