from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import pandas as pd


EmitMode = Literal["cross", "state"]


@dataclass(slots=True)
class CrossoverConfig:
    """
    Configuration for SMA (or any two series) crossover signal generation.

    fast_col: Name of the fast (short lookback) series column.
    slow_col: Name of the slow (long lookback) series column.
    output_col: Signal output column (default 'signal').
    emit: 'cross' -> only emit +-1 at crossover rows (0 elsewhere)
          'state' -> maintain position state after cross (long=1, short=-1)
    copy: If True (default) operate on a copy; else mutate the passed DataFrame.
    """
    fast_col: str
    slow_col: str
    output_col: str = "signal"
    emit: EmitMode = "cross"
    copy: bool = True

    def validate(self) -> None:
        if self.fast_col == self.slow_col:
            raise ValueError("fast_col and slow_col must differ.")
        if self.emit not in ("cross", "state"):
            raise ValueError("emit must be 'cross' or 'state'")


def _raw_sign(series: pd.Series) -> pd.Series:
    # map positive -> 1, negative -> -1, zero -> 0
    return series.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)).astype("int8")


def compute_crossover_signals(
    fast: pd.Series,
    slow: pd.Series,
    *,
    mode: EmitMode = "cross",
    name: str = "signal",
) -> pd.Series:
    """
    Compute crossover signals given two aligned Series.

    Semantics (aligned to project tests):
        - We ignore any initial 'short' (fast < slow) region; we stay flat (0)
          until the first positive (fast > slow) cross appears.
        - After first long entry:
            mode='cross': emit +1 / -1 only on rows where non-zero sign changes.
            mode='state': maintain position state each row (zeros retain prior state).
        - Zero diff (fast == slow) retains prior state (state mode) and emits 0 (cross mode).

    If the series starts already long (fast > slow), that opening bar is treated as a cross.
    """
    if len(fast) != len(slow):
        raise ValueError("fast and slow series must be same length")
    if not fast.index.equals(slow.index):
        # Align if indexes differ (outer join not desired; strict requirement)
        raise ValueError("fast and slow indexes must match")

    sign = _raw_sign(fast - slow)  # values in {-1,0,1}

    out_vals: list[int] = []
    state = 0  # current held position (for state mode after armed)
    armed = False  # becomes True after first long (+1) cross

    for s in sign:
        if not armed:
            if s == 1:
                #  first long cross
                armed = True
                state = 1
                out_vals.append(1 if mode == "cross" else 1)
            else:
                #  remain flat ignoring initial shorts / zeros
                out_vals.append(0)
        else:
            if mode == "cross":
                if s != 0 and s != state:
                    #  position flip event
                    state = s
                    out_vals.append(s)
                else:
                    out_vals.append(0)
            else:  # state
                if s != 0 and s != state:
                    state = s
                # retain state even if s == 0
                out_vals.append(state)

    return pd.Series(out_vals, index=fast.index, name=name, dtype="int8")


class CrossoverNode:
    """
    Node that produces crossover signals from two SMA (or arbitrary) columns.

    Example:
        cfg = CrossoverConfig(fast_col="sma_10", slow_col="sma_30", emit="state")
        node = CrossoverNode(cfg)
        df = node.run(df_with_smas)
    """
    def __init__(self, config: CrossoverConfig) -> None:
        config.validate()
        self.config = config

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("CrossoverNode expects a pandas DataFrame input.")
        missing = [c for c in (self.config.fast_col, self.config.slow_col) if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required column(s): {missing}")

        signals = compute_crossover_signals(
            df[self.config.fast_col],
            df[self.config.slow_col],
            mode=self.config.emit,
            name=self.config.output_col,
        )
        target = df.copy() if self.config.copy else df
        target[self.config.output_col] = signals
        return target


__all__: Sequence[str] = [
    "CrossoverConfig",
    "compute_crossover_signals",
    "CrossoverNode",
]
