# Architecture (Phase 0 Scaffold)

High-level components:

- backend: FastAPI app exposing health + (soon) pipeline CRUD & execution.
- engine: In-memory node execution (data -> transform -> signal) with simple SMA crossover example.
- ingest: Placeholder for data adapters.
- shared: Cross-cutting utilities (feature flags, config) - TBD.
- premium: Boundary for closed-source or licensed extensions (empty stubs now).
- frontend: (Not yet added) React/TypeScript canvas using React Flow for graph editing.

Execution Flow Example:
1. DataSourceNode produces synthetic price series.
2. Two SmaTransformNode instances compute fast & slow averages.
3. SmaCrossoverSignalNode emits mock buy/sell events when fast/slow relationship flips.

Future layers: persistence abstraction, backtest runner with metrics, auth, plugin discovery, real data adapters, RL environments.
