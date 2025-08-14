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

This section outlines the steps to set up and run the `trade-agent` project.

#### Prerequisites

Ensure you have the following installed on your system:

- **Python**: Version 3.10 or higher. You can verify your Python version by running:
  ```bash
  python --version
  ```
- **Poetry**: For dependency management. Install it using pip:
  ```bash
  pip install poetry
  ```
- **Make** (Optional): For easier execution of common development and documentation tasks.

#### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/zzzzero-hash/trade-agent.git
    cd trade-agent
    ```
2.  **Install dependencies using Poetry**:
    ```bash
    poetry install
    ```
    This command will create a virtual environment and install all project dependencies.
3.  **Activate the virtual environment**:
    ```bash
    poetry shell
    ```

#### Verification

To ensure your environment is set up correctly, you can run the documentation build and lint steps:

```bash
make docs
```

This command will build the Sphinx documentation and perform linting checks. Any warnings will result in a build failure, ensuring documentation quality.

### Usage Examples

This section provides quick examples to get started with key functionalities. For more detailed examples and advanced configurations, please refer to the comprehensive documentation in the `docs/source/` directory.

#### 1. Training a Supervised Learning Model

To train a supervised learning model (e.g., Ridge Regression) using a predefined configuration:

```bash
poetry run python main.py train-sl --config configs/ridge_config.json
```

This command will:

- Load data from `data/sample_data.parquet`.
- Apply feature engineering.
- Train a Ridge Regression model based on `configs/ridge_config.json`.
- Save the trained model and its metadata to the `models/` directory.

#### 2. Running a Backtest

To run a backtest with a trained model:

```bash
poetry run python scripts/run_backtest.py --model-path models/sl_model_ridge_20250811_164323.pkl --config configs/ridge_config.json
```

_Note: Replace `sl_model_ridge_20250811_164323.pkl` with the actual path to your trained model._

This command will:

- Load the specified trained model.
- Execute a backtest using historical data.
- Generate a backtest report in the `reports/` directory.

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

## How We Evaluate

Our trading agent's performance is rigorously evaluated against a set of key metrics and acceptance thresholds to ensure robustness and profitability. We consider the following criteria:

- **Sharpe Ratio**: A measure of risk-adjusted return. We aim for a Sharpe Ratio of **1.5 or higher** on out-of-sample data.
- **Turnover**: The rate at which positions are opened and closed. High turnover can lead to increased transaction costs. We target a maximum daily turnover of **20%** of the portfolio value.
- **Drawdown Brakes**: Mechanisms to limit potential losses.
  - **Maximum Drawdown**: The largest peak-to-trough decline in the portfolio value. We require a maximum drawdown of **no more than 15%**.
  - **Daily Drawdown Limit**: A threshold for daily losses. If the daily drawdown exceeds **5%**, trading activity is paused or reduced.

These thresholds are subject to adjustment based on market conditions and strategic objectives.
