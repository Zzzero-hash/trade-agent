import sys
import types
from pathlib import Path

import pytest

import trade_agent.features.technical_indicators as ti

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
            Return a dictionary with a neutral compound sentiment score for the given text.

            Parameters:
                text (str): The input text to analyze.

            Returns:
                dict: A dictionary containing a single key 'compound' with a value of 0.0.
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


def test_get_feature_names():
    cfg = ti.IndicatorConfig(
        sma_periods=[3],
        ema_periods=[5],
        obv_enabled=False,
        vwap_enabled=False,
    )
    indicator = ti.TechnicalIndicators(cfg)
    names = indicator.get_feature_names()
    assert names == [
        "sma_3",
        "ema_5",
        "rsi",
        "macd",
        "macd_signal",
        "macd_histogram",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "atr",
        "doji",
        "hammer",
        "bullish_engulfing",
        "bearish_engulfing",
        "shooting_star",
        "morning_star",
        "evening_star",
    ]
