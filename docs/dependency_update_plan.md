# Dependency Update Plan for trade-agent

## Current Status Analysis

After analyzing the existing `pyproject.toml` file, I've identified the following:

### Already Present Dependencies (Meeting Requirements)

- **PyTorch**: `torch>=2.0.0` (line 25)
- **pandas**: `pandas>=2.0.0` (line 23)
- **numpy**: `numpy<2.0.0,>=1.24.0` (line 22)
- **gymnasium**: `gymnasium>=0.29.0` (line 26)
- **stable-baselines3[extra]**: `stable-baselines3[extra]>=2.0.0,<3.0.0` (line 29)
- **scikit-learn**: `scikit-learn>=1.3.0` (line 24)
- **optuna**: `optuna>=3.0.0` (line 37)

### Missing Dependencies (Need to be Added)

1. **ta** (or **talib**): Technical analysis library
2. **pyarrow**: For efficient data processing and Parquet support
3. **fastapi**: For API serving capabilities

## Proposed Changes to pyproject.toml

The following dependencies should be added to the dependencies section in `pyproject.toml`:

```toml
# Trading-specific dependencies
"ta>=0.10.0",  # Technical analysis library
"pyarrow>=10.0.0",  # Efficient data processing
"fastapi>=0.100.0",  # API framework
```

These should be added after line 40 (following the "Scientific computing and analysis" section) to maintain logical grouping.

## Version Justification

- **ta>=0.10.0**: Latest stable version that provides comprehensive technical analysis indicators
- **pyarrow>=10.0.0**: Recent version with good performance and Parquet support
- **fastapi>=0.100.0**: Recent version with modern features and good performance

## Integration Considerations

1. **ta vs talib**: The `ta` library is a pure Python implementation and easier to install, while `talib` requires C compilation. For initial setup, `ta` is recommended.
2. **pyarrow**: Will enhance data processing capabilities, especially for large datasets
3. **fastapi**: Will enable serving of trained models through REST APIs

## Implementation Steps

1. Add the missing dependencies to the pyproject.toml file
2. Run `pip install -e .` to install the updated dependencies
3. Verify all dependencies can be installed without conflicts
4. Test the installation with the smoke test
