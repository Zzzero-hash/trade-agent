import pandas as pd
import pytest

from trade_agent.engine.nodes.sma import SMAConfig, SMANode, compute_sma


def test_compute_sma_basic_defaults() -> None:
    prices = pd.Series([1, 2, 3, 4, 5], name="close")
    sma = compute_sma(prices, window=3)
    # First two NaN (min_periods defaults to window)
    assert sma.isna().sum() == 2
    assert sma.iloc[-1] == pytest.approx((3 + 4 + 5) / 3)


def test_compute_sma_min_periods_growth() -> None:
    prices = pd.Series([1, 2, 3,], name="close")
    sma = compute_sma(prices, window=3, min_periods=1)
    assert sma.isna().sum() == 0
    assert list(sma) == [1.0, (1 + 2) / 2, 2.0]


def test_compute_sma_dataframe_input_custome_names() -> None:
    df = pd.DataFrame({"close": [10, 11, 12, 13]})
    sma = compute_sma(df, window=2, output_col="fast_sma")
    assert sma.name == "fast_sma"
    assert list(sma[-2:]) == [11.5, 12.5]


def test_invalid_window_raises() -> None:
    with pytest.raises(ValueError):
        compute_sma(pd.Series([1, 2, 3], name="close"), window=0)


def test_missing_price_column_raises() -> None:
    df = pd.DataFrame({"open": [1, 2, 3]})
    with pytest.raises(KeyError):
        compute_sma(df, window=2, price_col="close")


def test_sma_config_finalize_defaults() -> None:
    cfg = SMAConfig(window=5)
    cfg.finalize()
    assert cfg.output_col == "sma_5"
    assert cfg.min_periods == 5


def test_sma_node_adds_column_copy() -> None:
    df = pd.DataFrame({"close": [10, 11, 12, 13, 14]})
    node = SMANode(SMAConfig(window=2))
    out = node.run(df)
    assert "sma_2" in out.columns
    assert out["sma_2"].iloc[-1] == pytest.approx((13 + 14) / 2)
    # Original not mutated
    assert "sma_2" not in df.columns


def test_sma_node_inplace() -> None:
    df = pd.DataFrame({"close": [1, 2, 3]})
    node = SMANode(SMAConfig(window=2, copy=False))
    out = node.run(df)
    assert out is df
    assert "sma_2" in df.columns


def test_sma_node_rejects_non_df() -> None:
    node = SMANode(SMAConfig(window=2))
    with pytest.raises(TypeError):
        node.run([1, 2, 3])  # type: ignore
