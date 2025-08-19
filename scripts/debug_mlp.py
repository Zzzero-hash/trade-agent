#!/usr/bin/env python3
"""
Debug script for MLP model training.
"""
import os
import sys

import pandas as pd


# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trade_agent.agents.sl.models.deep_learning import MLPModel


def main() -> None:
    # Load data
    df = pd.read_parquet('data/features.parquet')
    y = df['mu_hat'].values
    X = df.drop(columns=['mu_hat', 'sigma_hat']).values


    # Create MLP model
    config = {
        "input_size": 17,
        "hidden_sizes": [64, 32, 16],
        "output_size": 1,
        "dropout": 0.2,
        "epochs": 5,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "random_state": 42
    }

    model = MLPModel(config)

    # Try to fit the model
    try:
        model.fit(X, y)

        # Try to make predictions
        model.predict(X[:10])

    except Exception:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
