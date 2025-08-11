#!/usr/bin/env python3
"""
Script to train a single supervised learning model.
"""
import argparse
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sl.train import train_model_from_config


def main():
    parser = argparse.ArgumentParser(description="Train a single supervised learning model")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--data", required=True, help="Path to training data")
    parser.add_argument("--target", required=True, help="Target column name")

    args = parser.parse_args()

    try:
        print(f"Training model with config: {args.config}")
        print(f"Using data: {args.data}")
        print(f"Target column: {args.target}")

        results = train_model_from_config(args.config, args.data, args.target)

        print("Training completed successfully!")
        print("Results:")
        for key, value in results.items():
            print(f"  {key}: {value:.6f}")

    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
