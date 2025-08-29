from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class SMAConfig:
    """
    Configuration for a Simple Moving Average (SMA) indicator.

    window: Positive lookback window length.
    price_col: Input price column name in the DataFrame (default 'close').
    output_col: Output column name (default 'sma_{window}').
    min_periods: Minimum periods required to produce a value (default = window).
                 Set to 1 for a "growing" SMA from the first row.
    copy: If True (default) return a copy of the DataFrame with the SMA column.
          If False mutate the passed DataFrame (in-place).
    """

    window: int
    price_col: str = "close"
    output_col: str | None = None
    min_periods: int | None = None
    copy: bool = True

    def finalize(self) -> None:
        if self.window <= 0:
            raise ValueError("Window must be > 0")
        if self.output_col is None:
            self.output_col = f"sma_{self.window}"
        if self.min_periods is None:
            self.min_periods = self.window
        if self.min_periods <= 0:
            raise ValueError("min_periods must be > 0")


def compute_sma(
    data: pd.Series | pd.DataFrame,
    window: int,
    *,
    price_col: str = "close",
    min_periods: int | None = None,
    output_col: str | None = None,
) -> pd.Series:
    """
    Pure SMA computation (stateless, side-effect free).

    If 'data' is:
    - pd.Series: use it directly (name used if set, else price_col).
    - pd.DataFrame: use data[price_col].

    Returns a Series named output_col (or 'sma_{window}')
    """
    if window <= 0:
        raise ValueError("window must be > 0")

    if isinstance(data, pd.Series):
        price_series = data
        if price_series.name is None:
            price_series = price_series.rename(price_col)
    else:
        if price_col not in data.columns:
            raise KeyError(f"price_col '{price_col}' not found in columns {list(data.columns)}")
        price_series = data[price_col]

    if min_periods is None:
        min_periods = window
    if min_periods <= 0:
        raise ValueError("min_periods must be > 0")

    if output_col is None:
        output_col = f"sma_{window}"

    sma = price_series.rolling(window=window, min_periods=min_periods).mean()
    sma.name = output_col
    return sma


class SMANode:
    """
    Lightweight node wrapper (no current shared BaseNode in repo).

    Usage:
        cfg = SMAConfig(window=14)
        node = SMANode(cfg)
        out_df = node.run(df)

    Adjust later if a formal pipeline/BaseNode abstraction is introduced.
    """
    def __init__(self, config: SMAConfig) -> None:
        config.finalize()
        self.config = config

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("SMANode expects a pandas DataFrame input")
        if self.config.price_col not in df.columns:
            raise KeyError(f"Missing price column '{self.config.price_col}' in DataFrame")

        sma_series = compute_sma(
            df,
            self.config.window,
            price_col=self.config.price_col,
            min_periods=self.config.min_periods,
            output_col=self.config.output_col,
        )
        target = df.copy() if self.config.copy else df
        target[sma_series.name] = sma_series
        return target


__all__: Sequence[str] = ["SMAConfig", "compute_sma", "SMANode"]
