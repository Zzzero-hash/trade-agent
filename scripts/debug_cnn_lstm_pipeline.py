#!/usr/bin/env python3
"""
Debug script for CNN-LSTM model training pipeline.
"""
import json
import os
import sys

import pandas as pd

from trade_agent.agents.sl.train import SLTrainingPipeline


# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    # Load configuration
    config_path = 'configs/cnn_lstm_config.json'
    with open(config_path) as f:
        config = json.load(f)


    # Load data
    df = pd.read_parquet('data/features.parquet')
    y = df['mu_hat'].values
    X = df.drop(columns=['mu_hat', 'sigma_hat']).values


    # Create and run training pipeline
    pipeline = SLTrainingPipeline(config)

    # Try to train the model
    try:
        pipeline.train(X, y)

    except Exception:
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
