import sys
import types
from pathlib import Path

import pytest

from trade_agent.portfolio.manager import PortfolioConfig, PortfolioManager

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

if "src.envs.finrl_trading_env" not in sys.modules:
    sys.modules["src.envs.finrl_trading_env"] = types.SimpleNamespace(
        register_env=lambda: None,
    )

if "trading_rl_agent" not in sys.modules:
    pkg = types.ModuleType("trading_rl_agent")
    pkg.__path__ = [
        str(Path(__file__).resolve().parents[2] / "src" / "trading_rl_agent"),
    ]
    sys.modules["trading_rl_agent"] = pkg

if "nltk.sentiment.vader" not in sys.modules:
    dummy = types.ModuleType("nltk.sentiment.vader")

    class DummySIA:
        def polarity_scores(self, _):
            """
            Return a neutral sentiment score for the given text.

            Parameters:
                text (str): The input text to analyze.

            Returns:
                dict: A dictionary with a single key 'compound' set to 0.0, indicating neutral sentiment.
            """
            return {"compound": 0.0}

    dummy.SentimentIntensityAnalyzer = DummySIA
    sys.modules["nltk.sentiment.vader"] = dummy

base = Path(__file__).resolve().parents[2] / "src" / "trading_rl_agent"
for pkg in ["features", "portfolio", "risk"]:
    key = f"trading_rl_agent.{pkg}"
    if key not in sys.modules:
        mod = types.ModuleType(key)
        mod.__path__ = [str(base / pkg)]
        sys.modules[key] = mod

pytestmark = pytest.mark.unit


def test_execute_trade_and_update_prices():
    """
    Test that executing a trade and updating prices correctly updates positions, unrealized PnL, and performance history in the PortfolioManager.
    """
    cfg = PortfolioConfig(max_position_size=0.5, max_leverage=2.0)
    pm = PortfolioManager(1000.0, cfg)

    assert pm.execute_trade("AAPL", 10, 10.0)
    assert "AAPL" in pm.positions

    pm.update_prices({"AAPL": 12.0})
    pos = pm.positions["AAPL"]
    assert pos.unrealized_pnl == pytest.approx(20.0)
    assert pm.performance_history[-1]["total_value"] == pm.total_value


def test_trade_rejected_by_risk():
    """
    Test that a trade exceeding the maximum position size is rejected by the PortfolioManager.

    Verifies that when attempting to execute a trade that violates risk constraints, the trade is not executed, no positions are created, and cash remains unchanged.
    """
    cfg = PortfolioConfig(max_position_size=0.05)
    pm = PortfolioManager(1000.0, cfg)

    result = pm.execute_trade("AAPL", 10, 10.0)
    assert result is False
    assert pm.positions == {}
    assert pm.cash == 1000.0
