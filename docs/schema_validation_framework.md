# Schema Validation and Column Consistency Framework

## Overview

This framework provides a comprehensive solution for ensuring consistent column counts and structure throughout your data pipeline. It addresses the common problem of variable column counts that can break downstream processing and model training.

## Key Components

### 1. `schema_validator.py` - Core Validation Engine

The main validation engine that provides:

- **ColumnSchema**: Dataclass defining expected schema for each pipeline stage
- **ValidationResult**: Detailed validation results with errors and warnings
- **SchemaValidator**: Main validator class with configurable validation rules
- **Global Functions**: Convenience functions for easy integration

**Key Features:**

- Validates required vs optional columns
- Enforces column count constraints (min/max)
- Checks data types compatibility
- Provides detailed error reporting
- Supports automatic column mapping and fixing

### 2. `schema_aware_processing.py` - Enhanced Data Pipeline

An enhanced data processor that integrates schema validation at every stage:

- **SchemaAwareProcessor**: Main processor class with validation at each step
- **Automatic Issue Fixing**: Attempts to fix common schema issues automatically
- **Feature Consistency**: Maintains consistent feature sets across runs
- **Batch Processing**: Ensures consistency across multiple data batches

### 3. `schema_config.py` - Configuration Management

Provides configurable validation rules and pipeline settings:

- **SchemaConfig**: Global configuration for validation behavior
- **PipelineStageConfig**: Stage-specific configuration options
- **Predefined Templates**: Common column sets for financial data

## Usage Examples

### Basic Schema Validation

```python
from src.data.schema_validator import validate_pipeline_stage

# Validate a DataFrame at a specific pipeline stage
result = validate_pipeline_stage(df, 'raw_input')

if result.is_valid:
    print("Schema validation passed!")
else:
    print(f"Validation errors: {result.errors}")
    print(f"Missing columns: {result.missing_columns}")
```

### Column Consistency Enforcement

```python
from src.data.schema_validator import enforce_consistent_columns

# Ensure DataFrame matches expected column structure
reference_columns = ['timestamp', 'asset', 'value', 'volume']
standardized_df = enforce_consistent_columns(df, reference_columns, 'cleaning')
```

### Schema-Aware Processing Pipeline

```python
from src.data.schema_aware_processing import process_data_with_validation

# Process data with automatic schema validation at each stage
processed_df = process_data_with_validation(raw_df)
```

### Batch Consistency

```python
from src.data.schema_aware_processing import ensure_column_consistency_across_batches

# Ensure multiple DataFrames have consistent columns
batches = [df1, df2, df3]  # DataFrames with potentially different columns
consistent_batches = ensure_column_consistency_across_batches(batches)
```

## Pipeline Stage Definitions

The framework defines validation schemas for each stage of your pipeline:

### 1. Raw Input Stage (`raw_input`)

- **Required**: `timestamp`, `value`, `asset`
- **Optional**: `Symbol`, `Date`, `Close`, `Open`, `High`, `Low`, `Volume`
- **Purpose**: Validate incoming data from various sources

### 2. Cleaned Stage (`cleaned`)

- **Required**: `timestamp`, `value`, `asset`
- **Purpose**: Ensure data cleaning preserved required structure

### 3. Transformed Stage (`transformed`)

- **Required**: `timestamp`, `value`, `asset` (with specific dtypes)
- **Purpose**: Validate data transformation and encoding

### 4. Feature Engineered Stage (`feature_engineered`)

- **Min columns**: 5
- **Max columns**: 100
- **Purpose**: Prevent feature explosion while ensuring minimum feature set

### 5. Final Output Stage (`final_output`)

- **Min columns**: 5
- **Max columns**: 50
- **Purpose**: Ensure model-ready data with appropriate feature count

## Configuration Options

### Global Settings

```python
from src.data.schema_config import SchemaConfig

config = SchemaConfig(
    strict_validation=False,    # Fail on any validation error
    auto_fix_issues=True,       # Attempt automatic fixes
    max_features=50,            # Maximum features for model
    min_features=5,             # Minimum features for model
    preserve_feature_columns=True  # Maintain consistent feature sets
)
```

### Stage-Specific Configuration

```python
from src.data.schema_config import PipelineStageConfig

stage_config = PipelineStageConfig(
    stage_name='custom_stage',
    required_columns={'col1', 'col2'},
    optional_columns={'col3', 'col4'},
    expected_dtypes={'col1': 'float64', 'col2': 'object'},
    max_null_percentage=0.1
)
```

## Benefits

### 1. **Consistent Column Counts**

- Ensures the same number of columns across pipeline runs
- Prevents model training failures due to feature count mismatches
- Maintains reproducible pipeline behavior

### 2. **Automatic Issue Detection**

- Identifies missing required columns
- Detects unexpected extra columns
- Validates data types and formats

### 3. **Intelligent Issue Fixing**

- Automatically maps common column name variations
- Adds missing columns with appropriate default values
- Removes or handles extra columns based on configuration

### 4. **Comprehensive Logging**

- Detailed validation reports at each stage
- Warning and error messages with specific issues
- Validation history tracking and analysis

### 5. **Flexible Configuration**

- Customizable validation rules per pipeline stage
- Configurable column count constraints
- Optional vs required column specifications

## Integration with Existing Pipeline

### Minimal Integration

Add validation checks to your existing pipeline:

```python
from src.data.schema_validator import validate_pipeline_stage

def your_existing_process_function(df):
    # Add validation at key points
    validate_pipeline_stage(df, 'raw_input')

    # Your existing processing...
    cleaned_df = your_clean_function(df)
    validate_pipeline_stage(cleaned_df, 'cleaned')

    # More processing...
    return final_df
```

### Full Integration

Replace your processing pipeline with the schema-aware version:

```python
from src.data.schema_aware_processing import process_data_with_validation

# Replace your existing pipeline
def process_data(df):
    return process_data_with_validation(df)
```

## Common Use Cases

### 1. **Model Training Consistency**

Ensure training and inference data have the same column structure:

```python
# During training
training_df = process_data_with_validation(raw_training_data)
model.fit(training_df)

# During inference
inference_df = process_data_with_validation(raw_inference_data)
predictions = model.predict(inference_df)  # Same column structure guaranteed
```

### 2. **Multi-Source Data Integration**

Handle data from different sources with varying column structures:

```python
# Data from different APIs/sources
yahoo_data = fetch_yahoo_data()
alpha_vantage_data = fetch_alpha_vantage_data()
iex_data = fetch_iex_data()

# Ensure all have consistent structure
all_data = ensure_column_consistency_across_batches([
    yahoo_data, alpha_vantage_data, iex_data
])
```

### 3. **Feature Engineering Stability**

Maintain consistent feature sets across different data periods:

```python
# The processor remembers feature columns from previous runs
processor = SchemaAwareProcessor()

# First run establishes feature set
jan_features = processor.process_data_with_schema_validation(jan_data)

# Subsequent runs maintain the same feature structure
feb_features = processor.process_data_with_schema_validation(feb_data)
# feb_features will have the same columns as jan_features
```

## Testing

Run the provided test script to see the framework in action:

```bash
python test_schema_validation_demo.py
```

This will demonstrate:

- Basic schema validation
- Column consistency enforcement
- Schema-aware processing pipeline
- Validation summary reporting
- Configuration options

## Best Practices

1. **Define Clear Schemas**: Set up appropriate schemas for each pipeline stage
2. **Use Auto-Fix Sparingly**: Enable auto-fix for development, consider disabling for production
3. **Monitor Validation Reports**: Regularly review validation summaries to identify data quality issues
4. **Test with Real Data**: Validate your schemas with actual data from your sources
5. **Version Control Schemas**: Track changes to your validation schemas over time

## Extension Points

The framework is designed to be extensible:

- **Custom Validation Rules**: Add domain-specific validation logic
- **New Pipeline Stages**: Define schemas for additional processing stages
- **Custom Fix Strategies**: Implement specialized automatic fixing logic
- **Integration Hooks**: Add callbacks for custom validation actions

This framework provides a robust foundation for maintaining data consistency throughout your pipeline, ensuring reliable and reproducible results for your trading agent system.
