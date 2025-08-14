# Repository Cleanup Summary

## Overview
This document summarizes the cleanup performed on the trade-agent repository to manage large, binary, and ephemeral files properly.

## Files Removed from Git Tracking

### Large Binary Model Files (~18MB total)
- `models/rl/best_model.zip` (8.8MB) - RL model checkpoint
- `models/rl/sac.zip` (8.8MB) - SAC agent checkpoint  
- `models/rl/ppo_final.zip` (955KB) - PPO agent checkpoint
- `models/sl_model_*.pkl` - Various supervised learning models (20KB-296KB each)

### Generated Data Files (~500KB total)
- `data/features.parquet` (125KB) - Processed features
- `data/large_fe.parquet` (125KB) - Large feature engineering output
- `data/large_sample_data.parquet` (59KB) - Large sample dataset
- `data/fe.parquet` (10KB) - Feature engineering results
- `data/sample_data.parquet` (5KB) - Parquet sample data

### Reports and Logs (~2MB total)
- All files in `reports/` directory (CSV, HTML, PNG, JSON files)
- All files in `logs/` directory (TensorBoard logs, evaluations)

### Temporary/Test Files
- Various pytest temporary files with policy state pickles (1.9MB each)
- HTML dashboard files (4.7MB each)

## What's Kept in Git

### Essential Files
- `data/sample_data.csv` - Small CSV sample for testing
- Directory structure with `.gitkeep` files
- Comprehensive README files for each directory

### Documentation Added
- `data/README.md` - Data management guide
- `models/README.md` - Model training and storage guide  
- `reports/README.md` - Report generation guide
- `logs/README.md` - Logging and monitoring guide

## Git Configuration Changes

### Updated .gitignore
Added patterns to exclude:
```
# Generated files
logs/
reports/
models/
data/*.parquet
data/*.npz
data/*.pkl

# Large temporary files
**/test_*/*.pkl
**/pytest-*/
attribution_dashboard.html
locust-report.html
pattern_comparison.png
```

### Git LFS Configuration (.gitattributes)
Configured LFS tracking for:
- Model files: `*.pkl`, `*.h5`, `*.hdf5`
- Archives: `*.zip`
- Data files: `*.parquet`, `*.npz`, `*.csv`
- Images: `*.png`, `*.jpg`, `*.jpeg`
- Logs: `*.tfevents.*`
- Reports: `*.html`

## Repository Size Impact

### Before Cleanup
- Repository size: ~59MB
- Many large binary files tracked in Git history
- Slow clone/fetch operations

### After Cleanup  
- Active files removed from tracking
- Git LFS configured for future large files
- Comprehensive documentation added

## How to Regenerate Removed Files

### Data Files
```bash
# Run feature engineering
python src/data/processing.py

# Generate parquet files
python src/data/unified_orchestrator.py
```

### Model Files
```bash
# Train supervised learning models
python src/sl/train.py --config configs/

# Train RL models
python src/rl/train.py --algorithm ppo
python src/rl/train.py --algorithm sac
```

### Reports
```bash
# Generate backtesting reports
python src/evaluation/backtest.py --output-dir reports/

# Create visualizations
python src/visualization/equity_plots.py
```

### Logs
Training scripts automatically generate logs in the `logs/` directory when running model training.

## Future Recommendations

1. **Use Git LFS** for any files > 100MB
2. **Store models externally** using MLflow or similar model registry
3. **Implement DVC** for large dataset versioning
4. **Regular cleanup** of generated files
5. **Cloud storage** for production models and data

## Migration Notes

If you need the removed files:
1. Check the Git history before the cleanup commit
2. Regenerate using the provided scripts
3. For production models, use external storage solutions

## Git Commands Used

```bash
# Remove tracked files but keep locally
git rm --cached -r models/ logs/ reports/ data/*.parquet

# Add documentation
git add -f data/README.md models/README.md reports/README.md logs/README.md

# Initialize Git LFS
git lfs install
```

This cleanup significantly improves repository performance and establishes better practices for managing large files in a machine learning project.
