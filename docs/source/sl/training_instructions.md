# Supervised Learning Model Training Instructions

This document provides instructions on how to train the supervised learning models for the trading agent.

## Prerequisites

Make sure you have:

1. Generated features using the feature engineering pipeline
2. All required dependencies installed

## Training Individual Models

To train a single model, use the `train_single_model.py` script:

```bash
python scripts/train_single_model.py --config configs/[model_config].json --data data/features.parquet --target mu_hat
```

Replace `[model_config]` with one of:

- `ridge_config` for Ridge regression
- `linear_config` for Linear regression
- `garch_config` for GARCH model
- `mlp_config` for MLP model
- `cnn_lstm_config` for CNN-LSTM model
- `transformer_config` for Transformer model

Example:

```bash
python scripts/train_single_model.py --config configs/ridge_config.json --data data/features.parquet --target mu_hat
```

## Training All Models

To train all models at once, use the `train_all_models.py` script:

```bash
python scripts/train_all_models.py --data data/features.parquet --target mu_hat
```

## Model Configurations

Each model has its own configuration file in the `configs/` directory:

- `ridge_config.json`: Ridge regression with hyperparameter tuning
- `linear_config.json`: Linear regression (no hyperparameter tuning)
- `garch_config.json`: GARCH model with hyperparameter tuning
- `mlp_config.json`: MLP model (hyperparameter tuning disabled for faster training)
- `cnn_lstm_config.json`: CNN-LSTM model (hyperparameter tuning disabled for faster training)
- `transformer_config.json`: Transformer model (hyperparameter tuning disabled for faster training)

## Output

Trained models and their metadata are saved in the `models/` directory:

- Model files: `sl_model_[model_type]_[timestamp].pkl`
- Metadata files: `sl_model_[model_type]_[timestamp]_metadata.json`

## Model Performance

The training script outputs performance metrics including:

- Training MSE (Mean Squared Error)
- Training MAE (Mean Absolute Error)
- Training R² (Coefficient of Determination)
- Cross-validation MSE (if enabled)

## Notes

1. For faster training, hyperparameter tuning is disabled for deep learning models by default
2. You can enable hyperparameter tuning by setting `"enable_tuning": true` in the config files
3. The sequence-based models (CNN-LSTM, Transformer) require sufficient data for sequence creation
4. The target variable `mu_hat` represents the expected return forecast
