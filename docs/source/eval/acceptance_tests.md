# Evaluation Framework Acceptance Tests

## Overview

This document specifies the acceptance tests for the Evaluation and Backtesting Framework to ensure all components function correctly and meet the required specifications.

## Test Environment Setup

### Prerequisites

1. Python 3.8+
2. All required dependencies installed
3. Sample data available in `data/` directory
4. Trained models available in `models/` directory
5. Configuration files in `configs/eval/` directory

### Test Data Requirements

1. Historical market data (sample_data.csv, sample_data.parquet)
2. Trained SL models (sl*model*\*.pkl)
3. Trained RL models (ppo*\*.pkl, sac*\*.pkl)
4. Feature data (features.parquet, fe.parquet)

## Acceptance Tests

### Test 1: Framework Initialization

#### Objective

Verify that the evaluation framework initializes correctly with proper configuration.

#### Test Steps

1. Import evaluation framework modules
2. Initialize EvaluationFramework with default configuration
3. Verify framework components are properly instantiated
4. Check configuration loading from file

#### Expected Results

- Framework initializes without errors
- All components are properly instantiated
- Configuration is loaded correctly
- Deterministic processing is enabled

#### Test Code

```python
def test_framework_initialization():
    """Verify that evaluation framework initializes correctly"""
    from src.eval.framework import EvaluationFramework

    # Initialize framework
    framework = EvaluationFramework()

    # Verify components
    assert framework.metrics_calculator is not None
    assert framework.backtesting_engine is not None
    assert framework.risk_analyzer is not None
    assert framework.report_generator is not None

    # Verify deterministic processing
    assert framework.config.get('deterministic', False) == True
```

### Test 2: Metrics Accuracy

#### Objective

Verify that performance and risk metrics are calculated accurately.

#### Test Steps

1. Generate synthetic return series with known properties
2. Calculate metrics using evaluation framework
3. Compare with expected values
4. Validate against established libraries (e.g., pandas, scipy)

#### Expected Results

- Return metrics match expected values within tolerance
- Risk metrics are calculated correctly
- Transaction metrics are accurate
- Performance attribution is correct

#### Test Code

```python
def test_metrics_accuracy():
    """Verify that metrics are calculated accurately"""
    import numpy as np
    import pandas as pd
    from src.eval.metrics.returns import ReturnsMetrics
    from src.eval.metrics.risk import RiskMetrics

    # Generate synthetic data
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 1000))

    # Calculate metrics
    returns_metrics = ReturnsMetrics()
    risk_metrics = RiskMetrics()

    # Test return metrics
    total_return = returns_metrics.calculate_total_return(returns)
    sharpe_ratio = returns_metrics.calculate_sharpe_ratio(returns)

    # Validate against expected values
    expected_return = (1 + 0.001) ** 1000 - 1
    assert abs(total_return - expected_return) < 1e-10

    expected_sharpe = 0.001 / 0.02
    assert abs(sharpe_ratio - expected_sharpe) < 0.1
```

### Test 3: Backtesting Correctness

#### Objective

Verify that backtesting produces correct results and handles edge cases.

#### Test Steps

1. Create simple trading strategy with known behavior
2. Run backtest using event-driven engine
3. Run backtest using vectorized engine
4. Compare results between engines
5. Validate against known benchmarks

#### Expected Results

- Both backtesting engines produce consistent results
- Portfolio values match expected calculations
- Transaction costs are applied correctly
- Performance metrics are accurate

#### Test Code

```python
def test_backtesting_correctness():
    """Verify that backtesting produces correct results"""
    import pandas as pd
    import numpy as np
    from src.eval.backtesting.event_driven import EventDrivenEngine
    from src.eval.backtesting.vectorized import VectorizedEngine

    # Create simple strategy
    class SimpleStrategy:
        def __init__(self):
            self.position = 0

        def get_signal(self, data):
            # Simple buy-and-hold strategy
            return 1.0 if self.position == 0 else 0.0

    # Generate test data
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = pd.Series(np.cumprod(1 + np.random.normal(0.001, 0.02, 100)), index=dates)
    test_data = pd.DataFrame({'price': prices})

    # Test event-driven engine
    event_engine = EventDrivenEngine()
    event_engine.add_strategy(SimpleStrategy())
    event_results = event_engine.run(test_data)

    # Test vectorized engine
    vectorized_engine = VectorizedEngine()
    vectorized_engine.add_strategy(SimpleStrategy())
    vectorized_results = vectorized_engine.run(test_data)

    # Compare results
    assert abs(event_results['total_return'] - vectorized_results['total_return']) < 1e-10
```

### Test 4: Deterministic Processing

#### Objective

Verify that deterministic processing works correctly and produces reproducible results.

#### Test Steps

1. Run evaluation with fixed seed
2. Record results
3. Run evaluation again with same seed
4. Compare results
5. Run evaluation with different seed
6. Verify results differ

#### Expected Results

- Identical results with same seed
- Different results with different seeds
- All stochastic processes are properly controlled
- Reproducible backtesting results

#### Test Code

```python
def test_deterministic_processing():
    """Verify that deterministic processing works correctly"""
    import numpy as np
    from src.eval.utils.deterministic import set_all_seeds

    # Set seed and generate random numbers
    set_all_seeds(42)
    first_run = [np.random.random() for _ in range(10)]

    # Set same seed and generate random numbers
    set_all_seeds(42)
    second_run = [np.random.random() for _ in range(10)]

    # Set different seed and generate random numbers
    set_all_seeds(123)
    third_run = [np.random.random() for _ in range(10)]

    # Verify identical results with same seed
    assert first_run == second_run

    # Verify different results with different seeds
    assert first_run != third_run
```

### Test 5: SL Model Evaluation Integration

#### Objective

Verify that supervised learning models can be evaluated correctly.

#### Test Steps

1. Load trained SL model
2. Prepare test data
3. Run evaluation using framework
4. Validate metrics calculation
5. Check prediction accuracy

#### Expected Results

- SL model loads successfully
- Evaluation runs without errors
- Metrics are calculated correctly
- Prediction accuracy is within expected range

#### Test Code

```python
def test_sl_model_evaluation():
    """Verify that SL models can be evaluated correctly"""
    from src.eval.integration.sl_evaluation import SLEvaluation
    from src.sl.models.factory import SLModelFactory

    # Load SL model
    model_factory = SLModelFactory()
    model = model_factory.create_model("ridge", {"alpha": 1.0})

    # Create evaluation
    sl_eval = SLEvaluation(model)

    # Run evaluation (assuming test data is available)
    # results = sl_eval.evaluate(test_data, test_targets)

    # Verify evaluation structure
    # assert 'mse' in results
    # assert 'mae' in results
    # assert 'sharpe_ratio' in results
```

### Test 6: RL Agent Evaluation Integration

#### Objective

Verify that reinforcement learning agents can be evaluated correctly.

#### Test Steps

1. Load trained PPO/SAC agent
2. Prepare test environment
3. Run evaluation using framework
4. Validate metrics calculation
5. Check policy performance

#### Expected Results

- RL agents load successfully
- Evaluation runs without errors
- Metrics are calculated correctly
- Policy performance is within expected range

#### Test Code

```python
def test_rl_agent_evaluation():
    """Verify that RL agents can be evaluated correctly"""
    from src.eval.integration.rl_evaluation import RLEvaluation

    # Create evaluation (assuming agents are available)
    rl_eval = RLEvaluation()

    # Run evaluation (assuming test environment is available)
    # results = rl_eval.evaluate(ppo_agent, sac_agent, test_env)

    # Verify evaluation structure
    # assert 'sharpe_ratio' in results
    # assert 'max_drawdown' in results
    # assert 'win_rate' in results
```

### Test 7: Ensemble Combiner Evaluation

#### Objective

Verify that ensemble combiner can be evaluated correctly.

#### Test Steps

1. Load trained ensemble combiner
2. Prepare test data
3. Run evaluation using framework
4. Validate combination effectiveness
5. Check risk management validation

#### Expected Results

- Ensemble combiner loads successfully
- Evaluation runs without errors
- Combination weights are applied correctly
- Risk management is validated

#### Test Code

```python
def test_ensemble_combiner_evaluation():
    """Verify that ensemble combiner can be evaluated correctly"""
    from src.eval.integration.ensemble_evaluation import EnsembleEvaluation

    # Create evaluation (assuming ensemble is available)
    ensemble_eval = EnsembleEvaluation()

    # Run evaluation (assuming test data is available)
    # results = ensemble_eval.evaluate(ensemble_combiner, test_data)

    # Verify evaluation structure
    # assert 'combination_accuracy' in results
    # assert 'risk_adjusted_return' in results
```

### Test 8: Risk Analysis Validation

#### Objective

Verify that risk analysis components work correctly.

#### Test Steps

1. Generate test portfolio data
2. Run risk analysis using framework
3. Validate VaR calculation
4. Check stress testing results
5. Verify portfolio risk decomposition

#### Expected Results

- Risk analysis runs without errors
- VaR is calculated correctly
- Stress testing produces valid results
- Portfolio risk is decomposed properly

#### Test Code

```python
def test_risk_analysis_validation():
    """Verify that risk analysis components work correctly"""
    import pandas as pd
    import numpy as np
    from src.eval.risk_analysis.market_risk import MarketRiskAnalyzer

    # Generate test data
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 1000))

    # Run risk analysis
    risk_analyzer = MarketRiskAnalyzer()
    var_95 = risk_analyzer.calculate_var(returns, confidence_level=0.95)

    # Validate VaR (should be around -3% for normal distribution)
    assert var_95 < 0  # VaR should be negative
    assert var_95 > -0.1  # Shouldn't be too extreme
```

### Test 9: Reporting Generation

#### Objective

Verify that reports are generated in expected formats.

#### Test Steps

1. Run evaluation to generate results
2. Generate performance report
3. Generate risk report
4. Generate transaction report
5. Validate report formats

#### Expected Results

- All reports are generated without errors
- Reports are in correct formats (HTML, PDF, CSV)
- Report content is accurate
- Visualization components work correctly

#### Test Code

```python
def test_reporting_generation():
    """Verify that reports are generated in expected formats"""
    from src.eval.reporting.performance import PerformanceReportGenerator

    # Create report generator
    report_gen = PerformanceReportGenerator()

    # Generate sample report data
    sample_data = {
        'total_return': 0.15,
        'sharpe_ratio': 1.2,
        'max_drawdown': -0.08
    }

    # Generate HTML report
    html_report = report_gen.generate_report(sample_data, format='html')
    assert '<html>' in html_report

    # Generate CSV report
    csv_report = report_gen.generate_report(sample_data, format='csv')
    assert 'total_return' in csv_report
```

### Test 10: Configuration Management

#### Objective

Verify that configuration management works correctly.

#### Test Steps

1. Load default configuration
2. Modify configuration parameters
3. Save configuration
4. Reload configuration
5. Validate changes

#### Expected Results

- Configuration loads correctly
- Parameters can be modified
- Configuration saves successfully
- Changes persist after reload

#### Test Code

```python
def test_configuration_management():
    """Verify that configuration management works correctly"""
    from src.eval.config import EvaluationConfig

    # Load configuration
    config = EvaluationConfig()

    # Modify parameter
    original_seed = config.get('seed', 42)
    config.set('seed', 123)

    # Verify change
    assert config.get('seed') == 123

    # Reset to original
    config.set('seed', original_seed)
    assert config.get('seed') == original_seed
```

## Performance Requirements

### Test 11: Backtesting Performance

#### Objective

Verify that backtesting performance meets requirements.

#### Test Steps

1. Run backtest on large dataset
2. Measure execution time
3. Compare with performance benchmarks
4. Validate memory usage

#### Expected Results

- Backtesting completes within acceptable time limits
- Memory usage is optimized
- Performance scales appropriately with data size

#### Test Code

```python
def test_backtesting_performance():
    """Verify that backtesting performance meets requirements"""
    import time
    import pandas as pd
    import numpy as np
    from src.eval.backtesting.vectorized import VectorizedEngine

    # Generate large test dataset
    dates = pd.date_range('2010-01-01', periods=10000, freq='D')
    prices = pd.Series(np.cumprod(1 + np.random.normal(0.0005, 0.01, 10000)), index=dates)
    large_data = pd.DataFrame({'price': prices})

    # Measure execution time
    engine = VectorizedEngine()
    start_time = time.time()
    # results = engine.run(large_data)
    end_time = time.time()

    # Validate performance (should complete in reasonable time)
    execution_time = end_time - start_time
    assert execution_time < 300  # Should complete in under 5 minutes
```

## Integration Requirements

### Test 12: Component Integration

#### Objective

Verify that all components integrate correctly.

#### Test Steps

1. Initialize complete evaluation framework
2. Load all components (SL, RL, Ensemble)
3. Run end-to-end evaluation
4. Validate data flow between components
5. Check result consistency

#### Expected Results

- All components load successfully
- Data flows correctly between components
- End-to-end evaluation completes without errors
- Results are consistent across components

#### Test Code

```python
def test_component_integration():
    """Verify that all components integrate correctly"""
    from src.eval.framework import EvaluationFramework

    # Initialize framework
    framework = EvaluationFramework()

    # Verify component integration
    assert hasattr(framework, 'sl_evaluator')
    assert hasattr(framework, 'rl_evaluator')
    assert hasattr(framework, 'ensemble_evaluator')

    # Verify data flow integration
    assert framework.sl_evaluator is not None
    assert framework.rl_evaluator is not None
    assert framework.ensemble_evaluator is not None
```

## Validation Criteria

### Pass Criteria

- All acceptance tests pass without errors
- Metrics accuracy within 0.1% tolerance
- Backtesting results consistent between engines
- Deterministic processing verified
- Reports generated in all required formats
- Performance requirements met
- Integration successful

### Fail Criteria

- Any test fails to execute
- Metrics accuracy outside tolerance
- Backtesting engines produce inconsistent results
- Deterministic processing fails
- Report generation fails
- Performance requirements not met
- Integration failures

## Test Execution

### Automated Testing

1. Run all tests using pytest framework
2. Generate test coverage report
3. Validate coverage > 80%
4. Document any test failures

### Manual Testing

1. Review generated reports for visual correctness
2. Validate interactive dashboards
3. Check configuration file handling
4. Verify error handling and logging

## Test Maintenance

### Update Procedures

1. Update tests when requirements change
2. Add new tests for new features
3. Remove deprecated tests
4. Validate test suite regularly

### Version Control

1. All tests stored in version control
2. Test changes tracked with code changes
3. Test documentation maintained
4. Regression testing performed
