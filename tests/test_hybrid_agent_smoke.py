import numpy as np

from trade_agent.agents.hybrid_policy import HybridPolicyAgent
from trade_agent.envs.trading_env import TradingEnvironment


def _env_creator(cfg):
    return TradingEnvironment()


def test_hybrid_agent_predict() -> None:
    agent = HybridPolicyAgent({"hidden_dim": 8, "lr": 1e-3}, _env_creator)
    agent.fit()
    obs, _ = agent.env.reset()
    action = agent.predict(obs)
    assert action.shape[0] == agent.action_space.shape[0]
    assert np.all(action <= agent.action_space.high)
    assert np.all(action >= agent.action_space.low)
