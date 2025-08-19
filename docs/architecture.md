# Architecture Overview

This document provides a concise technical view of the `trade-agent` framework—its data & model flow, modular boundaries, and extension points. Documentation builds are already enforced in _strict_ mode (warnings fail CI) so this page follows the same standards (Mermaid + MyST flavoured Markdown).

## High‑Level Data / Control Flow

```mermaid
flowchart LR
    subgraph INGEST[Data Layer]
        A[Raw Market Data\n(csv/parquet/APIs)] --> VAL[Validation & Cleaning]
        VAL --> FE[Feature Engineering\n(pipelines, leakage guards)]
    end

    subgraph SL[Supervised Learning Forecasters]
        FE --> SLF[SL Models\n(Ridge/MLP/CNN+LSTM/Transformer/GARCH...)]
        SLF -->|predictions, risk stats| SIG[Signal Frame]
    end

    subgraph RL[Reinforcement Learning]
        SIG --> ENV[Gymnasium Env Wrapper\n(reward, costs, risk)]
        ENV --> AGENTS[PPO / SAC Policies]
        AGENTS --> ACT[Action Proposals\n(position adjustments)]
    end

    subgraph ENSEMBLE[Ensemble & Risk]
        SLF --> ENS[Ensemble Layer\n(weighting, governors)]
        ACT --> ENS
        ENS --> EXEC[Execution / Backtest / Live Adapter]
    end

    EXEC --> METRICS[Metrics & Reports]
    METRICS --> REG[Artifact & Experiment Registry]

    subgraph OPT[Optimization]
        OPTUNA[Hydra + Optuna Sweeps\n(search spaces under conf/search_space/)] --> SLF
        OPTUNA --> AGENTS
    end

    subgraph OBS[Observability]
        LOGS[(Structured Logs)]
        TB[(TensorBoard)]
        METRICS --> LOGS
        METRICS --> TB
    end

    subgraph CONF[Configuration]
        HYDRA[(Hydra YAML\nconf/ hierarchy)] --> ALL[(All Components)]
    end
```

### Legend

- Rectangles = processing stages / modules.
- Parallelograms = data artifacts (features, signals, actions, metrics).
- Subgraphs = cohesive subsystems with clear extension seams.

## Subsystem Responsibilities

| Subsystem               | Responsibilities                                                                 | Key Extension Points                                                   |
| ----------------------- | -------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| Data Layer              | Load, validate, align, and persist market & derived data                         | New loaders, feature calculators, schema validators                    |
| SL Forecasters          | Train predictive models producing expected return / risk estimates               | Add model group in `conf/model/` + implementation module               |
| RL Environment / Agents | Convert state (features + SL signals) to policy actions under constraints        | New reward functions, custom SB3 policies, additional RLlib algorithms |
| Ensemble & Risk         | Combine SL & RL outputs, apply position capping, volatility / drawdown governors | Alternate weighting schemes, Bayesian / meta-learners                  |
| Optimization            | Hyperparameter sweeps (Optuna), temporal CV objectives, pruning                  | Custom search spaces, multi-objective strategies                       |
| Evaluation              | Backtests, financial & regression metrics, temporal CV, walk-forward             | Additional metrics, scenario / stress modules                          |
| Reporting & Registry    | Persist artifacts, metrics, configuration snapshots                              | MLflow / W&B adapter, richer HTML reports                              |

## Temporal Cross‑Validation

Temporal CV (purged, embargoed splits) reduces leakage when features use rolling windows. The new module `trade_agent.evaluation.temporal_cv` provides:

- `PurgedTimeSeriesSplit` – generator implementing gap + embargo.
- `temporal_cv_scores` – compute per‑fold + aggregate metrics (e.g. MSE, Sharpe) for SL model selection.
- Optuna helper to integrate early pruning through intermediate fold reports.

Use it during optimization by setting `optimization.metric=cv_mse` and including a search space (see **Search Spaces** below).

## Search Spaces (Hydra + Optuna)

Reusable search space YAMLs live under `conf/search_space/`. Add one via:

```yaml
hydra:
  sweeper:
    search_space:
      model.learning_rate:
        type: float
        low: 1e-5
        high: 1e-3
        log: true
```

Launch: `python scripts/train_sl_hydra.py -m hydra/sweeper=optuna +search_space=transformer_core model=transformer`.

## Strict Documentation Policy

Sphinx builds run with warnings-as-errors (see `Makefile` target `docs` and tooling in `pyproject.toml`). This enforces:

- Valid cross references & autodoc imports
- Clean type hints
- Mermaid diagram parsing (via `sphinxcontrib-mermaid`)

Run locally:

```bash
make docs        # strict one-off build
```

## Extension Workflow (Example)

1. Add model YAML: `conf/model/new_model.yaml`.
2. Implement estimator under `src/trade_agent/agents/sl/` or analogous package.
3. (Optional) Create `conf/search_space/new_model.yaml` for Optuna.
4. Train: `python scripts/train_sl_hydra.py model=new_model`.
5. Optimize: `python scripts/train_sl_hydra.py -m hydra/sweeper=optuna +search_space=new_model optuna.n_trials=50`.
6. Evaluate via backtest + metrics; inspect artifacts in `models/` or experiment outputs.

## Future Enhancements

- Multi-objective sweeps (risk-adjusted return + drawdown)
- Live trading adapters (broker API abstraction layer)
- Rich HTML dashboards summarizing temporal CV fold dispersion
- Registry backend pluggable (MLflow, Weights & Biases)

---

For deeper module-level details refer to the Sphinx docs in `docs/source/`.
