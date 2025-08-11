#!/usr/bin/env python3
"""
Script to train all supervised learning models.
"""
import argparse
import json
import os
import sys
from typing import Any

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sl.train import train_model_from_config


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Train all supervised learning models")
    parser.add_argument("--data", required=True, help="Path to training data")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--config-dir", default="configs", help="Directory containing configuration files")

    args = parser.parse_args()

    # List of models to train with their config files
    models = [
        ("ridge", "ridge_config.json"),
        ("linear", "linear_config.json"),
        ("garch", "garch_config.json"),
        ("mlp", "mlp_config.json"),
        ("cnn_lstm", "cnn_lstm_config.json"),
        ("transformer", "transformer_config.json")
    ]

    results = {}

    for model_name, config_file in models:
        try:
            config_path = os.path.join(args.config_dir, config_file)
            print(f"\n{'='*50}")
            print(f"Training {model_name.upper()} model")
            print(f"{'='*50}")

            if not os.path.exists(config_path):
                print(f"Configuration file not found: {config_path}")
                continue

            print(f"Using config: {config_path}")
            print(f"Using data: {args.data}")
            print(f"Target column: {args.target}")

            model_results = train_model_from_config(config_path, args.data, args.target)

            results[model_name] = model_results
            print(f"{model_name.upper()} training completed successfully!")
            print("Results:")
            for key, value in model_results.items():
                print(f"  {key}: {value:.6f}")

        except Exception as e:
            print(f"Error training {model_name}: {e}")
            results[model_name] = {"error": str(e)}

    # Print summary
    print(f"\n{'='*50}")
    print("TRAINING SUMMARY")
    print(f"{'='*50}")

    for model_name, model_results in results.items():
        if "error" in model_results:
            print(f"{model_name.upper()}: FAILED - {model_results['error']}")
        else:
            print(f"{model_name.upper()}: SUCCESS")
            print(f"  Train MSE: {model_results.get('train_mse', 'N/A'):.6f}")
            print(f"  CV MSE: {model_results.get('cv_mse_mean', 'N/A'):.6f}")


if __name__ == "__main__":
    main()
