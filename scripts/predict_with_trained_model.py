#!/usr/bin/env python3
"""
Script to demonstrate how to load and use a trained model for predictions.
"""
import os
import sys

import numpy as np
import pandas as pd


# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trade_agent.agents.sl.models.factory import SLModelFactory


def main() -> None:
    # Load features data
    df = pd.read_parquet('data/features.parquet')

    # Prepare features (drop target columns)
    X = df.drop(columns=['mu_hat', 'sigma_hat']).values

    # Load a trained model (example with Ridge model)
    # You can replace 'ridge' with other model types: 'linear', 'garch', 'mlp', 'cnn_lstm', 'transformer'
    model_type = 'ridge'

    # Find the latest trained model of the specified type
    import glob
    from datetime import datetime

    model_files = glob.glob(f'models/sl_model_{model_type}_*.pkl')
    if not model_files:
        return

    # Sort by timestamp to get the latest model
    def extract_timestamp(filename):
        # Extract timestamp from filename
        parts = filename.split('_')
        timestamp_str = parts[-1].split('.')[0]
        try:
            return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
        except ValueError:
            # If parsing fails, return a default old timestamp
            return datetime(2000, 1, 1)

    model_files.sort(key=extract_timestamp)
    latest_model_path = model_files[-1]

    # Load the model
    model = SLModelFactory.create_model(model_type, {})
    model = model.load_model(latest_model_path)


    # Make predictions
    predictions = model.predict(X[:10])  # Predict on first 10 samples


    # Compare with actual values
    actual_values = df['mu_hat'].values[:10]

    # Calculate simple metrics
    np.mean((predictions - actual_values) ** 2)
    np.mean(np.abs(predictions - actual_values))



if __name__ == "__main__":
    main()
