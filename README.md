# Trading RL Agent

A production-grade hybrid reinforcement learning trading system that combines CNN+LSTM supervised learning with deep RL optimization for algorithmic trading.

## 🚀 Features

### Core Components

- **CNN+LSTM Models**: Hybrid neural networks for market pattern recognition
- **Reinforcement Learning**: SAC, TD3, PPO agents for trading decision optimization
- **Feature Engineering**: 150+ technical indicators with robust implementation
- **Data Pipeline**: Multi-source data ingestion with parallel processing (10-50x speedup)
- **Risk Management**: VaR, CVaR, position sizing, and portfolio optimization
- **Real-time Processing**: Live data feeds and sentiment analysis

### Infrastructure

- **Configuration Management**: YAML-based configuration with validation
- **CLI Interface**: Unified command-line interface using Typer
- **Logging & Monitoring**: Structured logging with MLflow/TensorBoard integration
- **Testing**: Comprehensive test suite with pytest
- **Code Quality**: Black, isort, ruff, mypy integration
- **Docker Support**: Containerized deployment ready

### Performance Optimizations

- **Parallel Data Fetching**: Ray-based parallel processing (10-50x speedup)
- **Mixed Precision Training**: 2-3x faster training with 30-50% memory reduction
- **Memory-Mapped Datasets**: 60-80% memory reduction for large datasets
- **Advanced LR Scheduling**: 1.5-2x faster convergence
- **Gradient Checkpointing**: Train larger models with same memory

## 📦 Installation

### Prerequisites

- Python 3.9+ (3.12 recommended)
- Git
- Docker (optional, for containerized deployment)

### Quick Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/trading-rl-agent.git
   cd trading-rl-agent
   ```

2. **Set up development environment**

   ```bash
   # Core dependencies only (fast setup)
   ./setup-env.sh core

   # Add ML dependencies
   ./setup-env.sh ml

   # Full production setup
   ./setup-env.sh full
   ```

3. **Install pre-commit hooks**

   ```bash
   pre-commit install
   ```

4. **Verify installation**
   ```bash
   python -c "import trading_rl_agent; print('✅ Package imported successfully')"
   ```

## 🎯 Quick Start

### 1. Train a CNN+LSTM Model

```bash
# Optimized training with all performance improvements (RECOMMENDED)
python train.py --epochs 100 --gpu

# With hyperparameter optimization for best results
python train.py --optimize-hyperparams --n-trials 50 --epochs 30 --gpu

# Forex-focused training (recommended for forex strategies)
python train.py --forex-focused --epochs 150 --gpu

# Force rebuild dataset
python train.py --force-rebuild --epochs 100 --gpu
```

### 2. Run Reinforcement Learning Training

```bash
# Train RL agents
python -m trading_rl_agent.agents.trainer

# Run hyperparameter tuning
python -m trading_rl_agent.agents.tune
```

### 3. Evaluate Models

```bash
# Evaluate trained models
python evaluate.py

# Generate performance reports
python -m trading_rl_agent.evaluation
```

### 4. Use the CLI

```bash
# View available commands
python cli.py --help

# Run data pipeline
python cli.py data process

# Train models
python cli.py train cnn-lstm
```

## 📚 Documentation

- **[Project Status](PROJECT_STATUS.md)**: Current development status and roadmap
- **[Development Roadmap](TODO.md)**: Detailed task list and priorities
- **[Contributing Guidelines](CONTRIBUTING.md)**: How to contribute to the project

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  Feature Eng.   │    │  CNN+LSTM Model │
│                 │    │                 │    │                 │
│ • yfinance      │───▶│ • Technical     │───▶│ • Pattern       │
│ • Alpha Vantage │    │   Indicators    │    │   Recognition   │
│ • Professional  │    │ • Alternative   │    │ • Uncertainty   │
│   Feeds         │    │   Data          │    │   Estimation    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  RL Environment │    │  RL Agents      │
                       │                 │    │                 │
                       │ • State Space   │───▶│ • SAC           │
                       │ • Action Space  │    │ • TD3           │
                       │ • Reward Func   │    │ • PPO           │
                       └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Risk Management │    │ Portfolio Mgmt  │
                       │                 │    │                 │
                       │ • VaR/CVaR      │───▶│ • Multi-asset   │
                       │ • Position Size │    │ • Rebalancing   │
                       │ • Monitoring    │    │ • Analytics     │
                       └─────────────────┘    └─────────────────┘
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest

# Run specific test categories
python -m pytest tests/unit/
python -m pytest tests/integration/
python -m pytest tests/smoke/

# Run with coverage
python -m pytest --cov=trading_rl_agent
```

## 🔧 Development

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
ruff check src/ tests/
mypy src/

# Run all quality checks
python run_comprehensive_tests.py --quality-only
```

### Adding Features

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Implement your changes following the [contributing guidelines](CONTRIBUTING.md)
3. Add tests for new functionality
4. Run the test suite: `python -m pytest`
5. Submit a pull request

## 📊 Current Status

- **Core Infrastructure**: ✅ Complete
- **Data Pipeline**: ✅ Complete with parallel processing
- **CNN+LSTM Training**: ✅ Complete with optimizations
- **RL Agents**: ✅ Complete
- **Risk Management**: ✅ Complete
- **Production Deployment**: ✅ Complete
- **Monitoring & Logging**: ✅ Complete
- **Testing**: ✅ Complete
- **Documentation**: ✅ Complete

## 🚀 Performance Benchmarks

| Component       | Before Optimization | After Optimization | Improvement            |
| --------------- | ------------------- | ------------------ | ---------------------- |
| Data Fetching   | Sequential          | Parallel (Ray)     | **10-50x faster**      |
| Training Speed  | Standard            | Mixed Precision    | **2-3x faster**        |
| Memory Usage    | Standard            | Optimized          | **30-50% less**        |
| Dataset Loading | Standard            | Memory-mapped      | **60-80% less memory** |
| Convergence     | Standard            | Advanced LR        | **1.5-2x faster**      |

## 🤝 Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Ray team for parallel processing capabilities
- yfinance for market data access
- Alpha Vantage for professional data feeds
