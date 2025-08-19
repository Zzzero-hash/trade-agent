#!/usr/bin/env python3
"""
Debug script for train_model_from_config function.
"""
import os
import sys


# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trade_agent.agents.sl.train import train_model_from_config


def main() -> None:
    # Try to train the model using the train_model_from_config function
    try:
        train_model_from_config(
            config_path='configs/mlp_config.json',
            data_path='data/features.parquet',
            target_column='mu_hat'
        )

    except Exception:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
