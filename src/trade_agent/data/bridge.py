"""Data -> trading environment bridge (refactored from legacy integrations).

Provides minimal, testable functions to convert pipeline outputs into the
format expected by `TradingEnvironment`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def convert_pipeline_output_to_trading_format(
    input_path: str,
    output_path: str | None = None,
    *,
    price_col: str = "close",
) -> str:
    """Convert a pipeline parquet file to trading parquet.

    Adds placeholder `mu_hat` / `sigma_hat` columns if absent.
    """
    df = pd.read_parquet(input_path)
    if "mu_hat" not in df.columns:
        df["mu_hat"] = 0.0
    if "sigma_hat" not in df.columns:
        df["sigma_hat"] = (
            df[price_col]
            .pct_change()
            .rolling(20)
            .std()
            .fillna(0.0)
        )
    out = (
        output_path
        or str(
            Path(input_path).with_suffix("").as_posix()
            + "_trading.parquet"
        )
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    return out


__all__ = ["convert_pipeline_output_to_trading_format"]
