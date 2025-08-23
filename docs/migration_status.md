# Migration Status

Canonical tracking of module migration from legacy `src/` layout to consolidated `trade_agent/` package.

Legend:

| Status | Meaning                                           |
| ------ | ------------------------------------------------- |
| ✅     | Fully migrated; legacy shim (if needed) in place  |
| 🟨     | Partial: exists in both locations; work remaining |
| 🛠️     | In progress (actively being migrated/refactored)  |
| ❌     | Not yet migrated / legacy only                    |
| 🔄     | Deprecated legacy path pending removal window     |

## High-Level Summary

| Category                         | Legacy Path(s)                 | New Path(s)                                          | Status | Notes                                                   |
| -------------------------------- | ------------------------------ | ---------------------------------------------------- | ------ | ------------------------------------------------------- |
| Core Eval Metrics                | `src/eval/`, `src/evaluation/` | `trade_agent/evaluation/`                            | ✅     | Financial metrics present; tests reference new path     |
| Backtesting & Walk-forward       | `src/eval/`                    | `trade_agent/evaluation/` (planned `backtesting.py`) | ❌     | Not yet migrated; ensure leakage-safe alignment         |
| Data Loaders & Splits            | `src/data/`                    | `trade_agent/data/`                                  | 🛠️     | Directory exists; confirm split utilities & add shims   |
| Feature Engineering              | `src/features/`                | `trade_agent/features/`                              | 🟨     | Partial: feature schema present; hashing/versioning TBD |
| Supervised Learning (SL)         | `src/sl/`, `src/agents/sl/`    | `trade_agent/agents/sl/`                             | 🟨     | Core evaluation integrated; all trainers not verified   |
| Reinforcement Learning (RL)      | `src/rl/`, `src/envs/`         | `trade_agent/envs/`, `trade_agent/agents/`           | ❌     | Standardized env interface pending (M4)                 |
| Ensemble & Risk                  | (none / scattered)             | `trade_agent/execution/ensemble/` (planned)          | ❌     | To be created (M5)                                      |
| Orchestration / Training Scripts | `scripts/`, `src/workflows/`   | `trade_agent/training/`                              | 🟨     | Hydra configs present; unify entrypoints                |
| Integrations / Plugins           | `src/integrations/`            | `trade_agent/integrations/`                          | 🟨     | Partial; audit external deps                            |
| Common Utilities                 | `src/common/`, `src/utils/`    | `trade_agent/common/`, `trade_agent/utils/`          | 🟨     | Needs type hint pass & consolidation                    |
| Legacy Shims                     | various                        | top-level thin re-exports                            | 🛠️     | Audit required; add import matrix test                  |

## Detailed Module Inventory

### Evaluation

- New: `trade_agent/evaluation/financial_metrics.py` (confirmed by tests)
- Pending: backtester, walk-forward harness consolidation.

### Data & Features

- Validate: loaders & purged splits present under `trade_agent/data/` (TBD).
- Feature schema: `data/features_schema.json` (consider relocating or referencing).

### Supervised Learning

- Ridge training & evaluation integrated (artifacts in `models/`).
- Need confirmation for MLP, CNN+LSTM, transformer trainers consolidation.

### Reinforcement Learning

- Legacy code under `src/rl/` & `src/envs/` awaiting standardization.

### Ensemble & Risk

- Not implemented; planned risk governors (max exposure, volatility targeting, drawdown circuit breaker).

### Orchestration

- Hydra configs in `conf/` active; legacy JSON configs deprecated but still available.

## Action Queue (Next)

1. Inventory exact files in `src/data/` vs `trade_agent/data/`; populate table row details (determinism, schema validation).
2. Identify / create backtester module under `trade_agent/evaluation/` with shim in legacy path.
3. Add smoke test for Sharpe & Sortino (TRA-10 requirement) referencing synthetic sample.
4. Expand SL trainers registry & confirm deprecation warnings for legacy JSON configs.
5. Draft RL env interface spec (keys, shapes) before migration (pre-work for M4).
6. Prepare import matrix test to verify old paths still resolve.

## Update Process

- Update status icons as PRs merge; ensure each ✅ has associated test coverage.
- When moving from 🟨 to ✅, note commit hash & test name ensuring coverage.
- Keep this file concise; move extended rationale to dedicated docs if needed.

---

Last Updated: (initial scaffold)

# Migration Status

Tracking progress of moving legacy modules into canonical `trade_agent` package (Hydra-config driven architecture).

## Legend

- ✅ Migrated & shim in place
- 🚧 In progress / partial
- ⏳ Not started

## Modules

| Area                  | Legacy Path Examples            | New Path                                    | Status | Notes                                                            |
| --------------------- | ------------------------------- | ------------------------------------------- | ------ | ---------------------------------------------------------------- |
| Data Loaders & Splits | `src/data/`                     | `trade_agent.data`                          | ⏳     | To migrate loaders, temporal split utilities, schema validation. |
| Features Build        | `src/features/build.py`         | `trade_agent.data.features`                 | ⏳     | Consolidate feature funcs; ensure deterministic hashing.         |
| Evaluation Metrics    | `src/eval/financial_metrics.py` | `trade_agent.evaluation.financial_metrics`  | ⏳     | Add Calmar, Omega, tail risk metrics.                            |
| Backtester            | `src/evaluation/backtest.py`    | `trade_agent.evaluation.backtest`           | ⏳     | Refactor for deterministic alignment; add tests.                 |
| SL Trainers           | `src/sl/`                       | `trade_agent.agents.sl`                     | ⏳     | Migrate trainers + Optuna search spaces.                         |
| RL Envs & Agents      | `src/rl/`, `src/envs/`          | `trade_agent.envs`, `trade_agent.agents.rl` | ⏳     | Standardize obs schema & reward shaping.                         |
| Ensemble & Risk       | `src/ensemble/`                 | `trade_agent.execution.ensemble`            | ⏳     | Implement ensemble builder + governors.                          |
| Plugins System        | `src/plugins/`                  | `trade_agent.plugins`                       | ⏳     | Ensure entry points registered.                                  |

## Immediate Next (M0)

1. Enable imports for existing code (fix failing tests due to shadow empty root package).
2. Add smoke test for financial metrics (Sharpe, Sortino) once metrics module migrated.
3. Type hint high-churn modules as they migrate.

This file will be updated as milestones progress (see Linear issue TRA-17).
