import numpy as np
import pandas as pd

from trade_agent.features.build import build_features


def _make_df(n: int = 50) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(0)
    price = 100 + rng.standard_normal(n).cumsum()
    high = price + rng.random(n)
    low = price - rng.random(n)
    open_ = price + rng.normal(0, 0.1, n)
    volume = rng.integers(100, 1000, n)  # type: ignore[arg-type]
    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": price,
        "volume": volume,
    }, index=idx)


def test_build_features_includes_raw_ohlcv() -> None:
    df = _make_df()
    feats = build_features(df, include_raw_ohlcv=True)
    for col in ["open", "high", "low", "close", "volume"]:
        assert col in feats.columns, f"missing raw column {col}"


def test_build_features_excludes_raw_when_flag_false() -> None:
    df = _make_df()
    feats = build_features(df, include_raw_ohlcv=False)
    for col in ["open", "high", "low", "close", "volume"]:
        assert col not in feats.columns, "raw column present despite flag"
