#!/usr/bin/env python3
"""
Debug script for train_model_from_config function.
"""
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sl.train import train_model_from_config


def main():
    # Try to train the model using the train_model_from_config function
    try:
        print("Training model using train_model_from_config...")
        results = train_model_from_config(
            config_path='configs/mlp_config.json',
            data_path='data/features.parquet',
            target_column='mu_hat'
        )
        print("Model trained successfully!")
        print(f"Results: {results}")

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
