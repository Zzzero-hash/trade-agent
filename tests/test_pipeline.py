import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trade_agent.engine.nodes import (  # noqa: E402
    DataSourceNode,
    SmaTransformNode,
    SmaCrossoverSignalNode,
    execute_pipeline,
)


def test_sma_crossover_pipeline():
    nodes = [
        DataSourceNode(id="data", symbol="XYZ"),
        SmaTransformNode(id="fast", symbol="XYZ", window=3),
        SmaTransformNode(id="slow", symbol="XYZ", window=8),
        SmaCrossoverSignalNode(id="xover", symbol="XYZ", fast=3, slow=8),
    ]
    ctx = execute_pipeline(nodes)
    signals = ctx.get("signals:XYZ:sma_xover")
    assert isinstance(signals, list)
    # Should have at least one signal flip
    assert any(sig["type"] in {"buy", "sell"} for sig in signals)
