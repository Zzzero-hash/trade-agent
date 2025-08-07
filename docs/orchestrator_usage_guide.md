# Orchestrator Usage Guide

This guide demonstrates how to use our orchestrator to govern the complete data pipeline with validation feedback loops and storage integration.

## Pipeline Architecture

The orchestrator implements a robust two-stage pipeline:

### Stage 1: Validation Feedback Loop

```
Ingestion → Symbol Validation → Symbol Correction → Back to Ingestion
```

This stage continues until:

- All symbols pass validation, OR
- Maximum iterations reached

### Stage 2: Data Processing Pipeline

```
Cleaning → Processing → Evaluation → TimescaleDB Storage
```

## Quick Start Examples

### 1. Basic Validation Loop

```python
from src.data.pipeline_examples import ValidationFeedbackLoop

# Create validation loop
validation_loop = ValidationFeedbackLoop(max_iterations=3, auto_correct=True)

# Test symbols (including potentially invalid ones)
test_symbols = ["AAPL", "MSFT", "INVALID_SYMBOL", "GOOGL"]

# Execute validation loop
validated_data = validation_loop.execute(test_symbols)

print(f"Validated {len(validated_data)} symbols")
for symbol, data in validated_data.items():
    print(f"  {symbol}: {len(data)} records")
```

### 2. Complete Pipeline

```python
from src.data.pipeline_examples import CompleteDataPipeline

# Create complete pipeline
pipeline = CompleteDataPipeline()

# Execute with default S&P 500 symbols
result = pipeline.execute()

if result['success']:
    print(f"Pipeline completed successfully!")
    print(f"Metrics: {result['metrics']}")
    print(f"Data shape: {result['data'].shape}")
else:
    print(f"Pipeline failed: {result['error']}")
```

### 3. Custom Symbols Pipeline

```python
from src.data.pipeline_examples import CompleteDataPipeline

# Define custom symbols
custom_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

# Execute pipeline
pipeline = CompleteDataPipeline()
result = pipeline.execute(custom_symbols)

print(f"Success: {result['success']}")
print(f"Final records: {result['metrics']['final_records']}")
```

## Advanced Usage

### Using the Enhanced Orchestrator Components

#### 1. Validation Feedback Loop Configuration

```python
from src.data.pipeline_examples import ValidationFeedbackLoop

# Configure validation loop
validation_loop = ValidationFeedbackLoop(
    max_iterations=5,        # Maximum correction attempts
    auto_correct=True        # Enable automatic symbol correction
)

# Execute with custom symbols
symbols = ["AAPL", "SOME_INVALID_SYMBOL", "MSFT"]
validated_data = validation_loop.execute(symbols)
```

#### 2. Pipeline Configuration

```python
from src.data.orchestrator import PipelineConfig, DataPipeline

# Create custom pipeline configuration
config = PipelineConfig(
    validate_symbols=True,
    apply_corrections=True,
    enable_evaluation=True,
    parallel_batch_size=20
)

# Build pipeline with configuration
pipeline = (DataPipeline(config)
           .add_cleaning_step()
           .add_processing_step()
           .add_evaluation_step())

# Execute pipeline
result = pipeline.execute(your_data)
```

## How the Validation Loop Works

### Step-by-Step Process

1. **Initial Ingestion**: Attempt to fetch data for all symbols
2. **Symbol Validation**: Check each symbol against validation rules
3. **Result Processing**: Separate valid and invalid symbols
4. **Symbol Correction**: For invalid symbols:
   - Generate correction suggestions
   - Apply corrections automatically (if enabled)
   - Update symbol list for next iteration
5. **Iteration**: Repeat until all symbols valid or max iterations reached

### Example Validation Flow

```
Iteration 1:
Input:  ["AAPL", "INVALID_SYMBOL", "MSFT"]
Valid:  ["AAPL", "MSFT"] → Keep these
Invalid: ["INVALID_SYMBOL"] → Correct to "CORRECTED_SYMBOL"

Iteration 2:
Input:  ["CORRECTED_SYMBOL"]
Valid:  ["CORRECTED_SYMBOL"] → Success!
Invalid: [] → Loop complete
```

## Integration with Main Application

### Updating main.py

```python
from src.data.pipeline_examples import run_complete_pipeline

def main():
    """Enhanced main with orchestrator integration."""
    # Initialize Ray
    ray.init()

    try:
        # Run complete pipeline with validation loop
        result = run_complete_pipeline()

        if result['success']:
            logger.info("✅ Pipeline completed successfully")
            logger.info(f"Processed data shape: {result['data'].shape}")
            logger.info(f"Metrics: {result['metrics']}")
        else:
            logger.error(f"❌ Pipeline failed: {result['error']}")

    finally:
        ray.shutdown()
```

## Error Handling and Monitoring

### Pipeline Status Monitoring

```python
from src.data.pipeline_examples import CompleteDataPipeline

pipeline = CompleteDataPipeline()

# Execute and monitor
result = pipeline.execute(symbols)

# Check detailed metrics
if result['success']:
    metrics = result['metrics']
    print(f"Execution time: {metrics['execution_time']:.2f}s")
    print(f"Validated symbols: {metrics['validated_symbols']}")
    print(f"Final records: {metrics['final_records']}")
    print(f"Storage success: {metrics['storage_success']}")
```

### Handling Validation Failures

```python
from src.data.pipeline_examples import ValidationFeedbackLoop

validation_loop = ValidationFeedbackLoop(max_iterations=3)
validated_data = validation_loop.execute(symbols)

if not validated_data:
    print("⚠️ No symbols passed validation")
    print(f"Completed {validation_loop.iteration_count} iterations")
    # Handle fallback logic here
```

## TimescaleDB Integration

The pipeline includes mock TimescaleDB storage integration. To implement real storage:

### 1. Database Setup

```sql
-- Create database
CREATE DATABASE trading_db;

-- Create table
CREATE TABLE ohlcv_processed (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume BIGINT,
    -- Add your processed features here
    PRIMARY KEY (timestamp, symbol)
);

-- Create hypertable
SELECT create_hypertable('ohlcv_processed', 'timestamp');
```

### 2. Real Storage Implementation

Replace the mock storage in `pipeline_examples.py`:

```python
import psycopg2
import pandas as pd

def _execute_storage(self, data: pd.DataFrame) -> bool:
    """Real TimescaleDB storage implementation."""
    try:
        conn = psycopg2.connect(
            "postgresql://user:pass@localhost:5432/trading_db"
        )

        # Store data using pandas
        data.to_sql(
            'ohlcv_processed',
            conn,
            if_exists='append',
            index=False,
            method='multi'
        )

        conn.close()
        return True

    except Exception as e:
        logger.error(f"Storage failed: {e}")
        return False
```

## Running the Demonstration

Execute the complete demonstration:

```bash
# Run the demonstration script
python demo_orchestrator.py
```

This will show:

1. Validation feedback loop in action
2. Complete pipeline execution
3. Custom symbols processing
4. Error handling and recovery

## Best Practices

1. **Symbol Management**: Always validate symbols before processing
2. **Error Handling**: Implement proper error handling for data ingestion
3. **Resource Management**: Use Ray context managers for cleanup
4. **Monitoring**: Log pipeline metrics for performance tracking
5. **Configuration**: Use configuration objects for flexible pipeline setup

## Troubleshooting

### Common Issues

1. **Ray Initialization**: Ensure Ray is properly initialized before pipeline execution
2. **Symbol Validation**: Check symbol format and exchange requirements
3. **Data Quality**: Monitor validation results and correction effectiveness
4. **Memory Usage**: Use appropriate batch sizes for large symbol sets

### Debug Mode

Enable debug logging for detailed pipeline information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now run your pipeline with detailed logs
result = run_complete_pipeline(symbols)
```

## Summary

The orchestrator provides a robust, configurable pipeline system that:

- ✅ Handles symbol validation and correction automatically
- ✅ Implements feedback loops for data quality assurance
- ✅ Supports parallel processing for performance
- ✅ Integrates with TimescaleDB for time-series storage
- ✅ Provides comprehensive error handling and monitoring
- ✅ Maintains backward compatibility with existing code

Use the examples above to integrate the orchestrator into your trading system!
