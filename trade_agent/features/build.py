"""Feature engineering for financial OHLCV data.

Generates leakage-aware predictive features plus forward targets (``mu_hat``
and ``sigma_hat``). Optionally appends the original raw OHLCV columns so RL
environments can access actual prices/volumes in addition to engineered
features.
"""
from __future__ import annotations

# mypy: ignore-errors
import argparse
import sys
from collections.abc import Sequence

import numpy as np
import pandas as pd

from trade_agent.data.loaders import load_ohlcv_data


__all__ = [
    "compute_log_returns",
    "compute_rolling_stats",
    "compute_atr",
    "compute_rsi",
    "compute_z_scores",
    "compute_realized_volatility",
    "compute_calendar_flags",
    "define_targets",
    "build_features",
]


# ---------------------------------------------------------------------------
# Primitive feature computations
# ---------------------------------------------------------------------------
def compute_log_returns(df: pd.DataFrame) -> pd.Series:
    """Log returns of the close price."""
    return np.log(df["close"] / df["close"].shift(1))


def compute_rolling_stats(
    df: pd.DataFrame, windows: Sequence[int] | None = None
) -> pd.DataFrame:
    """Rolling mean and std (vol) of log returns; statistics shifted 1 step."""
    if windows is None:
        windows = (20, 60)
    lr = compute_log_returns(df)
    out = pd.DataFrame(index=df.index)
    for w in windows:
        out[f"rolling_mean_{w}"] = lr.rolling(w).mean().shift(1)
        out[f"rolling_vol_{w}"] = lr.rolling(w).std().shift(1)
    return out


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range (shifted)."""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean().shift(1)


def compute_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Relative Strength Index (shifted)."""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window).mean().shift(1)
    avg_loss = loss.rolling(window).mean().shift(1)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_z_scores(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Z-scores for close and volume (using shifted stats)."""
    out = pd.DataFrame(index=df.index)
    price_mean = df["close"].rolling(window).mean().shift(1)
    price_std = df["close"].rolling(window).std().shift(1)
    out["price_z_score"] = (df["close"] - price_mean) / price_std
    vol_mean = df["volume"].rolling(window).mean().shift(1)
    vol_std = df["volume"].rolling(window).std().shift(1)
    out["volume_z_score"] = (df["volume"] - vol_mean) / vol_std
    return out


def compute_realized_volatility(
    df: pd.DataFrame, window: int = 20
) -> pd.Series:
    """Rolling standard deviation of log returns (shifted)."""
    lr = compute_log_returns(df)
    return lr.rolling(window).std().shift(1)


def compute_calendar_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar based categorical flags (all shifted to avoid leakage)."""
    feats = pd.DataFrame(index=df.index)
    feats["day_of_week"] = df.index.dayofweek  # type: ignore[attr-defined]
    feats["month"] = df.index.month  # type: ignore[attr-defined]
    feats["day_of_month"] = df.index.day  # type: ignore[attr-defined]
    feats["is_monday"] = (
        df.index.dayofweek == 0  # type: ignore[attr-defined]
    ).astype(int)
    feats["is_friday"] = (
        df.index.dayofweek == 4  # type: ignore[attr-defined]
    ).astype(int)
    feats["is_month_start"] = (
        df.index.is_month_start.astype(int)  # type: ignore[attr-defined]
    )
    feats["is_month_end"] = (
        df.index.is_month_end.astype(int)  # type: ignore[attr-defined]
    )
    return feats.shift(1)


def define_targets(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Forward mean and std of future log returns over ``horizon`` steps."""
    lr = compute_log_returns(df)
    targets = pd.DataFrame(index=df.index)
    targets["mu_hat"] = lr.shift(-horizon).rolling(horizon).mean()
    targets["sigma_hat"] = lr.shift(-horizon).rolling(horizon).std()
    return targets


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def build_features(
    df: pd.DataFrame,
    horizon: int = 5,
    include_raw_ohlcv: bool = True,
    raw_prefix: str | None = None,
) -> pd.DataFrame:
    """Assemble engineered feature matrix + forward targets.

    All rolling/statistical features are shifted to keep the row at *t*
    dependent only on information available up to (not after) *t*. Targets
    look forward and therefore are not shifted backward.
    """
    np.random.seed(42)
    feats = pd.DataFrame(index=df.index)
    feats["log_returns"] = compute_log_returns(df).shift(1)
    feats = pd.concat([feats, compute_rolling_stats(df)], axis=1)
    feats["atr"] = compute_atr(df).shift(1)
    feats["rsi"] = compute_rsi(df).shift(1)
    feats = pd.concat([feats, compute_z_scores(df)], axis=1)
    feats["realized_vol"] = compute_realized_volatility(df).shift(1)
    feats = pd.concat([feats, compute_calendar_flags(df)], axis=1)
    feats = pd.concat([feats, define_targets(df, horizon)], axis=1)

    if include_raw_ohlcv:
        raw_cols = [
            c
            for c in ["open", "high", "low", "close", "volume"]
            if c in df.columns
        ]
        raw = df[raw_cols].copy()
        if raw_prefix:
            raw = raw.rename(columns={c: f"{raw_prefix}{c}" for c in raw_cols})
        feats = pd.concat([feats, raw], axis=1)

    return feats.dropna()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate engineered features from OHLCV data"
    )
    p.add_argument(
        "--in", dest="input_file", required=True, help="Input parquet/csv path"
    )
    p.add_argument("--out", required=True, help="Output parquet path")
    p.add_argument(
        "--horizon", type=int, default=5, help="Forward horizon for targets"
    )
    p.add_argument(
        "--no-raw", action="store_true", help="Exclude raw OHLCV columns"
    )
    p.add_argument(
        "--raw-prefix", default=None, help="Optional prefix for raw OHLCV cols"
    )
    return p.parse_args()


def main() -> None:  # pragma: no cover
    args = _parse_args()
    try:
        if args.input_file.endswith(".csv"):
            df = load_ohlcv_data(args.input_file)
        else:
            df = pd.read_parquet(args.input_file)
        out = build_features(
            df,
            horizon=args.horizon,
            include_raw_ohlcv=not args.no_raw,
            raw_prefix=args.raw_prefix,
        )
        out.to_parquet(args.out)
    except Exception:
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
