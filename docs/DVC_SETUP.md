# DVC Configuration for Large Data Management

## Overview

DVC (Data Version Control) is recommended for datasets larger than 100MB and for reproducible ML pipelines.

## Setup Instructions

### Install DVC

```bash
pip install dvc[s3]  # For S3 storage
# or
pip install dvc[gcs]  # For Google Cloud Storage
# or
pip install dvc[azure]  # For Azure Storage
```

### Initialize DVC

```bash
dvc init
git add .dvc .dvcignore
git commit -m "Initialize DVC"
```

### Add Remote Storage

```bash
# For S3
dvc remote add -d myremote s3://mybucket/dvcstore

# For Google Cloud Storage
dvc remote add -d myremote gs://mybucket/dvcstore

# For local remote (shared network storage)
dvc remote add -d myremote /shared/dvcstore
```

## Usage Examples

### Track Large Datasets

```bash
# Add large data file to DVC
dvc add data/large_dataset.parquet
git add data/large_dataset.parquet.dvc .gitignore
git commit -m "Add large dataset"

# Push to remote storage
dvc push
```

### Track Model Files

```bash
# Add trained models
dvc add models/large_model.pkl
git add models/large_model.pkl.dvc
git commit -m "Add trained model"
```

### Create Data Pipeline

```bash
# Create pipeline stages
dvc stage add -n prepare \
  -d src/data/processing.py \
  -d data/raw_data.csv \
  -o data/processed_data.parquet \
  python src/data/processing.py

dvc stage add -n train \
  -d src/models/train.py \
  -d data/processed_data.parquet \
  -o models/trained_model.pkl \
  python src/models/train.py
```

### Reproduce Pipeline

```bash
dvc repro
```

## Recommended Structure

```
data/
├── raw/           # DVC tracked raw data
├── processed/     # DVC tracked processed data
└── sample/        # Git tracked sample data

models/
├── checkpoints/   # DVC tracked model checkpoints
├── final/         # DVC tracked final models
└── experiments/   # DVC tracked experiment models

reports/
├── figures/       # DVC tracked large images
└── summaries/     # Git tracked text reports
```

## Benefits of DVC

1. **Version Control**: Track large files and datasets
2. **Reproducibility**: Recreate ML pipelines exactly
3. **Collaboration**: Share data without bloating Git
4. **Storage Flexibility**: Use cloud or network storage
5. **Pipeline Management**: Define and track ML workflows

## When to Use DVC vs Git LFS

### Use DVC for:

- Datasets > 100MB
- ML pipelines and experiments
- Frequently changing large files
- Team collaboration on data

### Use Git LFS for:

- Binary assets < 100MB
- Infrequently changed files
- Simple large file tracking
- When DVC setup is overkill

## Integration with Current Project

To integrate DVC with this project:

1. Install and initialize DVC
2. Move large files to DVC tracking
3. Update .gitignore to exclude DVC-tracked files
4. Create pipeline stages for training/evaluation
5. Configure remote storage for team sharing

## Example .dvc/config

```ini
[core]
    remote = myremote
    autostage = true

['remote "myremote"']
    url = s3://trade-agent-data/dvc
    access_key_id = your_access_key
    secret_access_key = your_secret_key
```

This configuration provides enterprise-grade data management for ML projects.
