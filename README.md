## trade-agent

Modular research and experimentation framework for building, training, evaluating, and deploying algorithmic trading agents that combine supervised learning (SL) forecasters with reinforcement learning (RL) policies and risk-aware ensemble execution.

### Key Capabilities

- Feature + data engineering pipelines (Parquet-based, leakage-aware)
- Supervised learning forecasters (linear, ridge, GARCH, MLP, CNN+LSTM, transformer, etc.)
- Gymnasium-compatible trading environments with reward, transaction costs, and risk management
- RL agents (PPO via Stable-Baselines3, SAC via Ray RLlib) with configurable hyperparameters
- Ensemble combiner to merge agent actions with weighting & governors
- Evaluation / backtesting hooks (extensible)
- Sphinx documentation (multi-version) published to GitHub Pages
- Mermaid diagrams, API autodoc, strict docs build (warnings fail CI)

### High-Level Architecture

```
Market Data -> Feature Engineering -> SL Forecasters -> RL Environment -> RL Agents
      \------------------------------------------------------------------/
                                 Ensemble -> Execution / Backtest / Live
```

See the full documentation (diagrams & detailed module descriptions):
https://zzzzero-hash.github.io/trade-agent/

### Repository Layout

```
├── configs/             # JSON configs for models & agents
├── data/                # Sample / cached feature & label data (parquet / csv)
├── docs/                # Sphinx documentation source (built to GitHub Pages)
├── models/              # Serialized trained model artifacts + metadata
├── src/                 # (Pending) Core library/package code (add modules here)
├── tests/               # Tests (unit / integration) – extend as modules mature
├── logs/                # Training / evaluation logs (RL + SL)
├── main.py              # Entry point (e.g., orchestration / CLI placeholder)
├── run_analysis.py      # Script for exploratory analysis / validation
├── extract_validation.py# Symbol / data extraction validation utility
├── pyproject.toml       # Project & dependency metadata
├── Makefile             # Common dev & docs tasks
└── README.md            # This file
```

### Getting Started

Prerequisites:

- Python 3.10+ (confirm via `python --version`)
- (Optional) Make for easier task execution

Install dependencies:

```
pip install -e .
```

Verify environment (run a quick docs build & lint step if defined):

```
make docs
```

### Data & Features

Sample parquet / csv files are included under `data/` for experimentation. In production, integrate real collectors & validation pipelines (see documentation sections: Data Handling Pipeline, Symbol Validation Framework).

### Supervised Models

Models are configured via JSON in `configs/` and saved to `models/` with paired `*_metadata.json` for reproducibility. Extend training scripts to add new model families (tree-based, transformers, probabilistic, etc.).

### Reinforcement Learning

Two RL approaches co-exist:

- PPO (Stable-Baselines3) – on-policy, sample efficient in structured environments
- SAC (Ray RLlib) – off-policy, continuous action robustness

Both consume observations composed of engineered features + SL outputs, and output target position adjustments (normalized in [-1, 1] or similar domain-specific scaling).

### Ensemble Layer

Combines multiple policy outputs (e.g., SAC vs PPO) with weighting and risk governors (position caps, volatility scaling, etc.). Extend this to meta-learn weights or integrate Bayesian model averaging.

### Documentation

All docs are written in MyST Markdown (with Mermaid) under `docs/source/`. To work locally:

```
make docs          # One-off strict build (warnings -> error)
make docs-live     # Live rebuild (if configured with sphinx-autobuild)
make docs-clean    # Remove build artifacts
```

Published automatically to GitHub Pages via CI (multi-version enabled if branches/tags present).

### Configuration

Central configs are JSON under `configs/`. Favor explicit, version-controlled configuration objects for reproducibility. Consider layering (base + override) or Hydra integration as complexity grows.

### Logging & Outputs

- `logs/ppo`, `logs/sac` capture RL training traces (tensorboard / custom metrics – integrate as needed)
- `models/` persists binary model states
- `reports/` (future) for generated evaluation summaries

### Development Workflow

1. Add / modify core modules in `src/`
2. Write or update tests in `tests/`
3. Regenerate / extend docs (API autodoc will pick up properly exported symbols)
4. Commit with meaningful messages referencing tasks/issues
5. Let CI build docs + (future) run tests / lint / publish

### Testing

Add fast unit tests near each subsystem (data, features, sl, rl, ensemble). For stochastic RL components, fix seeds and assert distributional properties / shape / range rather than exact equality.

### Extensibility Roadmap (Suggested)

- [ ] Package formalization (e.g., `src/trade_agent/` with **init**.py)
- [ ] CLI entrypoints (typer / click) for training & evaluation
- [ ] Enhanced backtester integration
- [ ] Live trading adapter abstraction (broker APIs)
- [ ] Advanced risk module (VaR / CVaR / scenario stress)
- [ ] Model registry + experiment tracking (Weights & Biases or MLflow)

### Contributing

Contributions welcome. Please:

1. Open an issue / discussion for major changes
2. Keep PRs focused and documented
3. Add / update tests & docs for new behavior

### License

Specify a license (e.g., MIT, Apache 2.0) by adding a `LICENSE` file if distribution is intended.

### Security / Disclaimer

This codebase is for research & educational purposes. No guarantees of performance or suitability for live financial trading. Use at your own risk.

---

For deeper architectural details, diagrams, and module-level guides visit the full documentation:
https://zzzzero-hash.github.io/trade-agent/

Happy experimenting.
