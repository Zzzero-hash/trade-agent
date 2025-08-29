import pandas as pd
import pytest

from trade_agent.engine.nodes.crossover import (
    CrossoverConfig,
    CrossoverNode,
    compute_crossover_signals,
)


def test_compute_crossover_mode_events_only() -> None:
    fast = pd.Series([1, 2, 3, 2, 1, 2, 3], name="fast")
    slow = pd.Series([2, 2, 2, 2, 2, 2, 2], name="slow")
    sig = compute_crossover_signals(fast, slow, mode="cross")
    # Cross up at index 1 (fast 2 > slow 2? equal -> no; index 2: 3>2 -> +1)
    # First positive cross at idx 2, cross down at idx 4 (1<2 -> -1), cross up at idx 6
    expected = [0, 0, 1, 0, -1, 0, 1]
    assert sig.tolist() == expected


def test_compute_crossover_state_mode_position_held() -> None:
    fast = pd.Series([1, 2, 3, 2, 1, 2, 3], name="fast")
    slow = pd.Series([2, 2, 2, 2, 2, 2, 2], name="slow")
    sig = compute_crossover_signals(fast, slow, mode="state")
    # After first cross up at idx 2 stay long until cross down idx 4, then flat? index4 diff negative -> position short until cross up again idx6
    expected = [0, 0, 1, 1, -1, -1, 1]
    assert sig.tolist() == expected


def test_node_integration_cross_mode() -> None:
    df = pd.DataFrame(
        {
            "sma_fast": [1, 2, 3, 2, 1, 2, 3],
            "sma_slow": [2, 2, 2, 2, 2, 2, 2],
        }
    )
    node = CrossoverNode(CrossoverConfig(fast_col="sma_fast", slow_col="sma_slow", emit="cross", copy=False))
    out = node.run(df)
    assert out is df
    assert "signal" in df.columns
    assert out["signal"].tolist() == [0, 0, 1, 0, -1, 0, 1]


def test_missing_columns() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [2, 3]})
    node = CrossoverNode(CrossoverConfig(fast_col="x", slow_col="y"))
    with pytest.raises(KeyError):
        node.run(df)


def test_invalid_config_same_cols() -> None:
    with pytest.raises(ValueError):
        CrossoverConfig(fast_col="a", slow_col="a").validate()


def test_mismatched_length_raises() -> None:
    fast = pd.Series([1, 2, 3])
    slow = pd.Series([1, 2])
    with pytest.raises(ValueError):
        compute_crossover_signals(fast, slow)
