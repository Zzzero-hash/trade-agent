#!/usr/bin/env python3
"""
Debug script for CNN-LSTM model training.
"""
import os
import sys

import pandas as pd


# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trade_agent.agents.sl.models.deep_learning import CNNLSTMModel


def main() -> None:
    # Load data
    df = pd.read_parquet('data/features.parquet')
    y = df['mu_hat'].values
    X = df.drop(columns=['mu_hat', 'sigma_hat']).values


    # Create CNN-LSTM model
    config = {
        "input_size": 17,
        "cnn_channels": [32, 64],
        "cnn_kernel_sizes": [3, 3],
        "lstm_hidden_size": 64,
        "lstm_num_layers": 2,
        "output_size": 1,
        "dropout": 0.2,
        "sequence_length": 10,
        "epochs": 5,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "random_state": 42
    }

    model = CNNLSTMModel(config)

    # Try to fit the model
    try:
        model.fit(X, y)

        # Try to make predictions
        model.predict(X[:20])

    except Exception:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
