# Evaluation Framework Usage Guide

## Overview

This guide provides comprehensive instructions for using the Evaluation and Backtesting Framework to assess trading system components including supervised learning models, reinforcement learning agents, and ensemble strategies.

## Getting Started

### Prerequisites

1. Python 3.8+
2. All required dependencies installed (see `pyproject.toml`)
3. Trained models available in `models/` directory
4. Historical data available in `data/` directory
5. Configuration files in `configs/eval/` directory

### Installation

The evaluation framework is part of the main trading system and does not require separate installation. Ensure all dependencies are installed:

```bash
pip install -e .
```

### Basic Usage

To use the evaluation framework, import the main components:

```python
from src.eval.framework import EvaluationFramework
from src.eval.config import EvaluationConfig

# Initialize framework
config = EvaluationConfig()
framework = EvaluationFramework(config)

# Run evaluation
results = framework.evaluate_component(component, data)
```

## Framework Components

### 1. Evaluation Framework Core

#### Initialization

```python
from src.eval.framework import EvaluationFramework
from src.eval.config import EvaluationConfig

# Load configuration
config = EvaluationConfig.from_file('configs/eval/framework_config.json')

# Initialize framework
framework = EvaluationFramework(config)
```

#### Configuration Options

```json
{
  "deterministic": true,
  "seed": 42,
  "currency": "USD",
  "reporting": {
    "formats": ["html", "pdf", "csv"],
    "storage_path": "./reports/"
  }
}
```

### 2. Backtesting Pipeline

#### Event-driven Backtesting

```python
from src.eval.backtesting.event_driven import EventDrivenEngine

# Initialize event-driven engine
engine = EventDrivenEngine(config)

# Add strategy
engine.add_strategy(my_strategy)

# Run backtest
results = engine.run(historical_data)

# Generate report
report = engine.generate_report()
```

#### Vectorized Backtesting

```python
from src.eval.backtesting.vectorized import VectorizedEngine

# Initialize vectorized engine
engine = VectorizedEngine(config)

# Run backtest
results = engine.run(strategy_signals, price_data)

# Calculate metrics
metrics = engine.calculate_metrics(results['returns'])
```

#### Walk-forward Analysis

```python
from src.eval.backtesting.walk_forward import WalkForwardEngine

# Initialize walk-forward engine
engine = WalkForwardEngine(config)

# Run walk-forward analysis
results = engine.run(strategy, data)
```

### 3. Metrics Calculation

#### Return Metrics

```python
from src.eval.metrics.returns import ReturnsMetrics

# Initialize metrics calculator
returns_metrics = ReturnsMetrics()

# Calculate metrics
total_return = returns_metrics.calculate_total_return(returns)
sharpe_ratio = returns_metrics.calculate_sharpe_ratio(returns)
sortino_ratio = returns_metrics.calculate_sortino_ratio(returns)
```

#### Risk Metrics

```python
from src.eval.metrics.risk import RiskMetrics

# Initialize risk metrics calculator
risk_metrics = RiskMetrics()

# Calculate metrics
var_95 = risk_metrics.calculate_var(returns, confidence_level=0.95)
max_drawdown = risk_metrics.calculate_max_drawdown(returns)
beta = risk_metrics.calculate_beta(returns, benchmark_returns)
```

#### Transaction Metrics

```python
from src.eval.metrics.transactions import TransactionMetrics

# Initialize transaction metrics calculator
txn_metrics = TransactionMetrics()

# Calculate metrics
turnover = txn_metrics.calculate_turnover(positions, portfolio_value)
transaction_costs = txn_metrics.calculate_transaction_costs(trades)
```

### 4. Risk Analysis

#### Market Risk Analysis

```python
from src.eval.risk_analysis.market_risk import MarketRiskAnalyzer

# Initialize analyzer
market_risk = MarketRiskAnalyzer()

# Calculate VaR
var_95 = market_risk.calculate_var(returns, confidence_level=0.95)

# Calculate CVaR
cvar_95 = market_risk.calculate_cvar(returns, confidence_level=0.95)
```

#### Stress Testing

```python
from src.eval.risk_analysis.stress_testing import StressTester

# Initialize stress tester
stress_tester = StressTester(config)

# Define scenario
scenario = {
    "name": "market_crash",
    "shocks": {"equity": -0.2, "bonds": 0.05}
}

# Run stress test
results = stress_tester.run_hypothetical_scenario(portfolio, scenario)
```

#### Portfolio Risk Decomposition

```python
from src.eval.risk_analysis.portfolio_risk import PortfolioRiskAnalyzer

# Initialize analyzer
portfolio_risk = PortfolioRiskAnalyzer()

# Decompose risk
risk_contribution = portfolio_risk.calculate_risk_contribution(
    weights, covariance_matrix
)
```

### 5. Reporting and Visualization

#### Performance Reports

```python
from src.eval.reporting.performance import PerformanceReportGenerator

# Initialize report generator
report_gen = PerformanceReportGenerator()

# Generate HTML report
html_report = report_gen.generate_report(metrics, format='html')

# Generate PDF report
pdf_report = report_gen.generate_report(metrics, format='pdf')
```

#### Interactive Dashboards

```python
from src.eval.reporting.visualization import DashboardGenerator

# Initialize dashboard generator
dashboard = DashboardGenerator()

# Create dashboard
app = dashboard.create_dashboard(evaluation_data)

# Run dashboard
app.run_server(debug=True, port=8050)
```

## Component Evaluation

### 1. Supervised Learning Models

#### Evaluation Process

```python
from src.eval.integration.sl_evaluation import SLEvaluation

# Initialize SL evaluation
sl_eval = SLEvaluation(sl_model)

# Prepare test data
X_test, y_test = prepare_test_data()

# Run evaluation
results = sl_eval.evaluate(X_test, y_test)

# Generate report
report = sl_eval.generate_report(results)
```

#### Cross-validation Assessment

```python
# Perform temporal cross-validation
cv_results = sl_eval.cross_validate(X, y, cv_strategy='temporal')

# Analyze stability
stability_metrics = sl_eval.analyze_stability(cv_results)
```

### 2. Reinforcement Learning Agents

#### PPO Agent Evaluation

```python
from src.eval.integration.rl_evaluation import RLEvaluation

# Initialize RL evaluation
rl_eval = RLEvaluation()

# Evaluate PPO agent
ppo_results = rl_eval.evaluate_agent(ppo_agent, test_env)

# Compare with benchmark
benchmark_results = rl_eval.evaluate_benchmark(buy_and_hold_strategy, test_env)

# Generate comparison report
comparison = rl_eval.compare_strategies(ppo_results, benchmark_results)
```

#### SAC Agent Evaluation

```python
# Evaluate SAC agent
sac_results = rl_eval.evaluate_agent(sac_agent, test_env)

# Compare agents
agent_comparison = rl_eval.compare_agents(ppo_results, sac_results)
```

### 3. Ensemble Combiner

#### Ensemble Evaluation

```python
from src.eval.integration.ensemble_evaluation import EnsembleEvaluation

# Initialize ensemble evaluation
ensemble_eval = EnsembleEvaluation()

# Evaluate ensemble strategy
ensemble_results = ensemble_eval.evaluate(ensemble_combiner, test_data)

# Analyze weight effectiveness
weight_analysis = ensemble_eval.analyze_weights(ensemble_results)
```

## Deterministic Processing

### Ensuring Reproducibility

```python
from src.eval.utils.deterministic import set_all_seeds

# Set fixed seeds for reproducible results
set_all_seeds(42)

# Run evaluation
results = framework.evaluate_component(component, data)

# Same seeds will produce identical results
set_all_seeds(42)
results2 = framework.evaluate_component(component, data)

# Results should be identical
assert results == results2
```

### Configuration for Deterministic Processing

```json
{
  "eval": {
    "framework": {
      "deterministic": true,
      "seed": 42
    },
    "backtesting": {
      "random_seed": 42
    }
  }
}
```

## Command-line Usage

### Running Backtests

```bash
# Run backtest with default configuration
python scripts/eval/run_backtest.py --strategy my_strategy --data data/sample_data.csv

# Run backtest with custom configuration
python scripts/eval/run_backtest.py --config configs/eval/backtesting_config.json

# Run walk-forward analysis
python scripts/eval/run_backtest.py --mode walk_forward --strategy my_strategy
```

### Evaluating Components

```bash
# Evaluate SL model
python scripts/eval/evaluate_component.py --type sl --model models/sl_model.pkl

# Evaluate RL agent
python scripts/eval/evaluate_component.py --type rl --agent ppo --model models/ppo_agent.pkl

# Evaluate ensemble
python scripts/eval/evaluate_component.py --type ensemble --model models/ensemble.pkl
```

### Generating Reports

```bash
# Generate performance report
python scripts/eval/generate_report.py --type performance --format html

# Generate risk report
python scripts/eval/generate_report.py --type risk --format pdf

# Generate all reports
python scripts/eval/generate_report.py --all --output_dir reports/latest/
```

### Stress Testing

```bash
# Run stress test with historical scenario
python scripts/eval/stress_test.py --scenario 2008_crisis

# Run stress test with hypothetical scenario
python scripts/eval/stress_test.py --scenario equity_crash --config configs/eval/stress_config.json
```

## Configuration Files

### Framework Configuration

```json
{
  "eval": {
    "framework": {
      "deterministic": true,
      "seed": 42,
      "currency": "USD"
    },
    "metrics": {
      "risk_free_rate": 0.02,
      "benchmark_symbol": "SPY"
    },
    "reporting": {
      "formats": ["html", "pdf", "csv"],
      "storage_path": "./reports/",
      "dashboard_port": 8050
    }
  }
}
```

### Backtesting Configuration

```json
{
  "backtesting": {
    "event_driven": {
      "slippage_model": "fixed",
      "slippage_basis_points": 5,
      "transaction_cost_bps": 1
    },
    "vectorized": {
      "frequency": "daily",
      "compounding": true
    },
    "walk_forward": {
      "window_size": 252,
      "step_size": 63
    }
  }
}
```

### Risk Configuration

```json
{
  "risk_analysis": {
    "market_risk": {
      "var_method": "historical",
      "var_confidence_level": 0.95
    },
    "stress_testing": {
      "scenarios": [
        {
          "name": "equity_crash",
          "shocks": { "equity": -0.2 }
        }
      ]
    }
  }
}
```

## Best Practices

### 1. Data Preparation

- Ensure data quality and consistency
- Handle missing values appropriately
- Align timestamps correctly
- Validate data ranges

### 2. Configuration Management

- Use version-controlled configuration files
- Document configuration changes
- Test configuration updates
- Maintain backup configurations

### 3. Performance Optimization

- Use vectorized operations where possible
- Cache frequently computed results
- Monitor memory usage
- Optimize data structures

### 4. Deterministic Processing

- Always set fixed seeds for reproducibility
- Document seed values used
- Validate deterministic behavior
- Use consistent data splits

### 5. Reporting

- Generate reports regularly
- Store reports with timestamps
- Include metadata in reports
- Validate report accuracy

## Troubleshooting

### Common Issues

#### Issue 1: Framework Initialization Failure

**Symptom**: ImportError when importing evaluation modules
**Solution**:

1. Verify all dependencies are installed
2. Check Python path configuration
3. Ensure src directory is in PYTHONPATH

#### Issue 2: Backtesting Performance Issues

**Symptom**: Long execution times for backtesting
**Solution**:

1. Use vectorized engine for simple strategies
2. Optimize data structures
3. Reduce data size for testing
4. Profile code for bottlenecks

#### Issue 3: Metrics Calculation Errors

**Symptom**: Incorrect or unexpected metric values
**Solution**:

1. Verify input data format
2. Check for NaN or infinite values
3. Validate calculation formulas
4. Compare with external libraries

#### Issue 4: Reporting Generation Failures

**Symptom**: Reports not generated or malformed
**Solution**:

1. Check report template files
2. Verify output directory permissions
3. Validate data passed to report generator
4. Check disk space availability

### Debugging Tips

#### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Validate Data

```python
# Check data shape and types
print(data.shape)
print(data.dtypes)
print(data.isnull().sum())
```

#### Profile Performance

```python
import cProfile
cProfile.run('framework.evaluate_component(component, data)')
```

## Advanced Usage

### Custom Metrics Implementation

```python
from src.eval.metrics.base import MetricsCalculator

class CustomMetrics(MetricsCalculator):
    def calculate_custom_metric(self, data):
        # Implement custom metric calculation
        pass

# Register custom metric
framework.register_metric('custom', CustomMetrics())
```

### Custom Backtesting Strategy

```python
from src.eval.backtesting.base import BacktestingEngine

class CustomBacktestingEngine(BacktestingEngine):
    def custom_logic(self):
        # Implement custom backtesting logic
        pass

# Use custom engine
custom_engine = CustomBacktestingEngine(config)
```

### Extending Risk Analysis

```python
from src.eval.risk_analysis.base import RiskAnalyzer

class CustomRiskAnalyzer(RiskAnalyzer):
    def calculate_custom_risk(self, data):
        # Implement custom risk calculation
        pass

# Register custom analyzer
framework.register_risk_analyzer('custom', CustomRiskAnalyzer())
```

## Integration Examples

### Integration with Feature Pipeline

```python
from src.features.build import FeaturePipeline
from src.eval.integration.sl_evaluation import SLEvaluation

# Initialize feature pipeline
feature_pipeline = FeaturePipeline()

# Prepare data
features = feature_pipeline.transform(raw_data)

# Evaluate SL model with features
sl_eval = SLEvaluation(sl_model)
results = sl_eval.evaluate(features, targets)
```

### Integration with Trading Environment

```python
from src.envs.trading_env import TradingEnvironment
from src.eval.integration.rl_evaluation import RLEvaluation

# Initialize trading environment
env = TradingEnvironment(config)

# Evaluate RL agent in environment
rl_eval = RLEvaluation()
results = rl_eval.evaluate_agent(agent, env)
```

### Integration with Ensemble Combiner

```python
from src.ensemble.combiner import EnsembleCombiner
from src.eval.integration.ensemble_evaluation import EnsembleEvaluation

# Initialize ensemble combiner
ensemble = EnsembleCombiner(ppo_agent, sac_agent, config)

# Evaluate ensemble strategy
ensemble_eval = EnsembleEvaluation()
results = ensemble_eval.evaluate(ensemble, test_data)
```

## Performance Monitoring

### Real-time Metrics

```python
# Monitor evaluation performance
framework.enable_monitoring()

# Get real-time metrics
metrics = framework.get_monitoring_metrics()

# Disable monitoring
framework.disable_monitoring()
```

### Resource Usage Tracking

```python
# Track resource usage
framework.start_resource_tracking()

# Run evaluation
results = framework.evaluate_component(component, data)

# Get resource usage report
resource_report = framework.get_resource_usage()
```

## Security Considerations

### Data Protection

- Encrypt sensitive configuration files
- Protect access to model files
- Secure report storage
- Validate input data

### Access Control

- Implement user authentication for dashboards
- Restrict access to evaluation results
- Audit evaluation activities
- Secure API endpoints

## Maintenance

### Regular Updates

- Update dependencies regularly
- Review and update documentation
- Test with new data versions
- Validate configuration files

### Performance Tuning

- Monitor system performance
- Optimize slow components
- Update hardware resources
- Profile and refactor code

### Backup and Recovery

- Regular backup of configuration files
- Backup of generated reports
- Version control for all files
- Test recovery procedures
