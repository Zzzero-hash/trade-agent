# Evaluation Framework File Structure

## Overview

This document specifies the complete file structure for the Evaluation and Backtesting Framework, including all modules, components, and their respective file paths.

## Complete Directory Structure

```
src/eval/
├── __init__.py
├── framework.py
├── config.py
├── base.py
├── backtesting/
│   ├── __init__.py
│   ├── base.py
│   ├── event_driven.py
│   ├── vectorized.py
│   ├── walk_forward.py
│   ├── events.py
│   ├── order_management.py
│   ├── execution.py
│   ├── portfolio.py
│   └── utils.py
├── metrics/
│   ├── __init__.py
│   ├── base.py
│   ├── returns.py
│   ├── risk.py
│   ├── transactions.py
│   ├── performance_attribution.py
│   └── utils.py
├── risk_analysis/
│   ├── __init__.py
│   ├── base.py
│   ├── market_risk.py
│   ├── credit_liquidity_risk.py
│   ├── model_risk.py
│   ├── operational_risk.py
│   ├── stress_testing.py
│   ├── portfolio_risk.py
│   ├── tail_risk.py
│   └── utils.py
├── reporting/
│   ├── __init__.py
│   ├── base.py
│   ├── performance.py
│   ├── risk.py
│   ├── transactions.py
│   ├── comparative.py
│   ├── visualization.py
│   └── templates/
│       ├── performance_report.html
│       ├── risk_report.html
│       ├── transaction_report.html
│       └── dashboard_template.html
├── integration/
│   ├── __init__.py
│   ├── sl_evaluation.py
│   ├── rl_evaluation.py
│   ├── ensemble_evaluation.py
│   └── component_interface.py
└── utils/
    ├── __init__.py
    ├── deterministic.py
    ├── data_handling.py
    └── validation.py

docs/eval/
├── evaluation_framework_design.md
├── backtesting_pipeline_design.md
├── performance_metrics_design.md
├── risk_metrics_design.md
├── file_structure.md
├── implementation_plan.md
├── acceptance_tests.md
├── rollback_plan.md
└── usage_guide.md

tests/eval/
├── __init__.py
├── test_framework.py
├── test_backtesting/
│   ├── __init__.py
│   ├── test_event_driven.py
│   ├── test_vectorized.py
│   └── test_walk_forward.py
├── test_metrics/
│   ├── __init__.py
│   ├── test_returns.py
│   ├── test_risk.py
│   └── test_transactions.py
├── test_risk_analysis/
│   ├── __init__.py
│   ├── test_market_risk.py
│   ├── test_stress_testing.py
│   └── test_portfolio_risk.py
├── test_reporting/
│   ├── __init__.py
│   ├── test_performance_reports.py
│   └── test_visualization.py
└── test_integration/
    ├── __init__.py
    ├── test_sl_integration.py
    ├── test_rl_integration.py
    └── test_ensemble_integration.py

scripts/eval/
├── run_backtest.py
├── evaluate_component.py
├── generate_report.py
├── stress_test.py
└── compare_strategies.py

configs/eval/
├── framework_config.json
├── backtesting_config.json
├── metrics_config.json
└── risk_config.json

reports/eval/
├── performance/
│   ├── daily/
│   ├── monthly/
│   └── quarterly/
├── risk/
│   ├── var_reports/
│   ├── stress_tests/
│   └── concentration/
└── transactions/
    ├── trade_blotters/
    └── cost_analysis/
```

## Core Framework Files

### Main Framework Module

- `src/eval/framework.py`: Main evaluation framework class and interface
- `src/eval/config.py`: Configuration management for the evaluation framework
- `src/eval/base.py`: Base classes for evaluation components
- `src/eval/__init__.py`: Package initialization

### Utility Modules

- `src/eval/utils/deterministic.py`: Deterministic processing utilities
- `src/eval/utils/data_handling.py`: Data handling and preprocessing utilities
- `src/eval/utils/validation.py`: Validation and verification utilities
- `src/eval/utils/__init__.py`: Package initialization

## Backtesting Pipeline Files

### Core Backtesting Modules

- `src/eval/backtesting/base.py`: Base classes for backtesting engines
- `src/eval/backtesting/event_driven.py`: Event-driven backtesting engine
- `src/eval/backtesting/vectorized.py`: Vectorized backtesting engine
- `src/eval/backtesting/walk_forward.py`: Walk-forward analysis module
- `src/eval/backtesting/__init__.py`: Package initialization

### Event-driven Components

- `src/eval/backtesting/events.py`: Event classes and event queue management
- `src/eval/backtesting/order_management.py`: Order lifecycle management
- `src/eval/backtesting/execution.py`: Execution simulation and modeling
- `src/eval/backtesting/portfolio.py`: Portfolio tracking and management
- `src/eval/backtesting/utils.py`: Backtesting utilities

## Metrics Calculation Files

### Metrics Modules

- `src/eval/metrics/base.py`: Base classes for metrics calculation
- `src/eval/metrics/returns.py`: Return-based metrics calculation
- `src/eval/metrics/risk.py`: Risk metrics calculation
- `src/eval/metrics/transactions.py`: Transaction metrics calculation
- `src/eval/metrics/performance_attribution.py`: Performance attribution analysis
- `src/eval/metrics/utils.py`: Metrics utilities
- `src/eval/metrics/__init__.py`: Package initialization

## Risk Analysis Files

### Risk Analysis Modules

- `src/eval/risk_analysis/base.py`: Base classes for risk analysis
- `src/eval/risk_analysis/market_risk.py`: Market risk metrics and analysis
- `src/eval/risk_analysis/credit_liquidity_risk.py`: Credit and liquidity risk analysis
- `src/eval/risk_analysis/model_risk.py`: Model risk assessment
- `src/eval/risk_analysis/operational_risk.py`: Operational risk metrics
- `src/eval/risk_analysis/stress_testing.py`: Stress testing framework
- `src/eval/risk_analysis/portfolio_risk.py`: Portfolio-level risk analysis
- `src/eval/risk_analysis/tail_risk.py`: Tail risk analysis and extreme value theory
- `src/eval/risk_analysis/utils.py`: Risk analysis utilities
- `src/eval/risk_analysis/__init__.py`: Package initialization

## Reporting and Visualization Files

### Reporting Modules

- `src/eval/reporting/base.py`: Base classes for reporting
- `src/eval/reporting/performance.py`: Performance report generation
- `src/eval/reporting/risk.py`: Risk report generation
- `src/eval/reporting/transactions.py`: Transaction report generation
- `src/eval/reporting/comparative.py`: Comparative analysis reports
- `src/eval/reporting/visualization.py`: Data visualization components
- `src/eval/reporting/__init__.py`: Package initialization

### Report Templates

- `src/eval/reporting/templates/performance_report.html`: Performance report template
- `src/eval/reporting/templates/risk_report.html`: Risk report template
- `src/eval/reporting/templates/transaction_report.html`: Transaction report template
- `src/eval/reporting/templates/dashboard_template.html`: Interactive dashboard template

## Integration Files

### Component Integration Modules

- `src/eval/integration/sl_evaluation.py`: Supervised learning model evaluation
- `src/eval/integration/rl_evaluation.py`: Reinforcement learning agent evaluation
- `src/eval/integration/ensemble_evaluation.py`: Ensemble combiner evaluation
- `src/eval/integration/component_interface.py`: Standard interface for component evaluation
- `src/eval/integration/__init__.py`: Package initialization

## Documentation Files

### Design Documentation

- `docs/eval/evaluation_framework_design.md`: Overall framework architecture
- `docs/eval/backtesting_pipeline_design.md`: Backtesting pipeline design
- `docs/eval/performance_metrics_design.md`: Performance metrics design
- `docs/eval/risk_metrics_design.md`: Risk metrics design
- `docs/eval/file_structure.md`: File structure specification (this document)
- `docs/eval/implementation_plan.md`: Implementation roadmap
- `docs/eval/acceptance_tests.md`: Acceptance test specifications
- `docs/eval/rollback_plan.md`: Rollback procedures
- `docs/eval/usage_guide.md`: User guide and examples

## Test Files

### Framework Tests

- `tests/eval/test_framework.py`: Tests for main framework functionality

### Backtesting Tests

- `tests/eval/test_backtesting/test_event_driven.py`: Event-driven engine tests
- `tests/eval/test_backtesting/test_vectorized.py`: Vectorized engine tests
- `tests/eval/test_backtesting/test_walk_forward.py`: Walk-forward analysis tests

### Metrics Tests

- `tests/eval/test_metrics/test_returns.py`: Return metrics tests
- `tests/eval/test_metrics/test_risk.py`: Risk metrics tests
- `tests/eval/test_metrics/test_transactions.py`: Transaction metrics tests

### Risk Analysis Tests

- `tests/eval/test_risk_analysis/test_market_risk.py`: Market risk tests
- `tests/eval/test_risk_analysis/test_stress_testing.py`: Stress testing tests
- `tests/eval/test_risk_analysis/test_portfolio_risk.py`: Portfolio risk tests

### Integration Tests

- `tests/eval/test_integration/test_sl_integration.py`: SL model integration tests
- `tests/eval/test_integration/test_rl_integration.py`: RL agent integration tests
- `tests/eval/test_integration/test_ensemble_integration.py`: Ensemble integration tests

## Script Files

### Evaluation Scripts

- `scripts/eval/run_backtest.py`: Command-line backtesting script
- `scripts/eval/evaluate_component.py`: Component evaluation script
- `scripts/eval/generate_report.py`: Report generation script
- `scripts/eval/stress_test.py`: Stress testing script
- `scripts/eval/compare_strategies.py`: Strategy comparison script

## Configuration Files

### Framework Configuration

- `configs/eval/framework_config.json`: Main framework configuration
- `configs/eval/backtesting_config.json`: Backtesting configuration
- `configs/eval/metrics_config.json`: Metrics calculation configuration
- `configs/eval/risk_config.json`: Risk analysis configuration

## Report Storage

### Report Directories

- `reports/eval/performance/`: Performance reports storage
- `reports/eval/risk/`: Risk reports storage
- `reports/eval/transactions/`: Transaction reports storage

## Dependencies and Requirements

The evaluation framework requires the following file-based dependencies:

1. Existing trading environment files in `src/envs/`
2. Supervised learning model files in `src/sl/`
3. Reinforcement learning agent files in `src/rl/`
4. Ensemble combiner files in `src/ensemble/`
5. Data files in `data/`
6. Model files in `models/`

## Version Control

All evaluation framework files should be included in version control with appropriate `.gitignore` rules for:

- Generated reports in `reports/eval/`
- Temporary files and caches
- Local configuration overrides
