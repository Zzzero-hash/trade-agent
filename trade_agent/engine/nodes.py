from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, runtime_checkable, cast


class ExecutionContext(Dict[str, Any]):
    """Simple execution context passed between nodes."""


@runtime_checkable
class Node(Protocol):
    id: str

    def run(self, ctx: ExecutionContext) -> ExecutionContext:  # pragma: no cover
        ...


@dataclass
class DataSourceNode:
    id: str
    symbol: str

    def run(self, ctx: ExecutionContext) -> ExecutionContext:
        import math
        prices = [100 + 5 * math.sin(i / 3) for i in range(50)]
        ctx[f"prices:{self.symbol}"] = prices
        return ctx


@dataclass
class SmaTransformNode:
    id: str
    symbol: str
    window: int

    def run(self, ctx: ExecutionContext) -> ExecutionContext:
        prices_any = ctx.get(f"prices:{self.symbol}")
        if not isinstance(prices_any, list):
            raise ValueError("Missing prices in context")
        prices = cast(List[float], prices_any)
        sma: List[float] = []
        for i in range(len(prices)):
            start = max(0, i - self.window + 1)
            sma.append(sum(prices[start:i + 1]) / (i - start + 1))
        ctx[f"sma:{self.symbol}:{self.window}"] = sma
        return ctx


@dataclass
class SmaCrossoverSignalNode:
    id: str
    symbol: str
    fast: int
    slow: int

    def run(self, ctx: ExecutionContext) -> ExecutionContext:
        fast_series = ctx.get(f"sma:{self.symbol}:{self.fast}")
        slow_series = ctx.get(f"sma:{self.symbol}:{self.slow}")
        if fast_series is None or slow_series is None:
            raise ValueError("Missing SMA series")
        signals: List[Dict[str, Any]] = []
        prev_state = None
        for f, s in zip(fast_series, slow_series):
            state = f > s
            if prev_state is not None and state != prev_state:
                signals.append({
                    "type": "buy" if state else "sell",
                    "price": f,
                })
            prev_state = state
        ctx[f"signals:{self.symbol}:sma_xover"] = signals
        return ctx


def execute_pipeline(nodes: List[Node]) -> ExecutionContext:
    ctx: ExecutionContext = ExecutionContext()
    for n in nodes:
        ctx = n.run(ctx)
    return ctx
