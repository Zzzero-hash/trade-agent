#!/usr/bin/env python3
"""
Example script demonstrating the new unified experimentation framework.

This script shows how to create and run experiments programmatically
using the enhanced modular design.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import (
    CrossValidationConfig,
    DataConfig,
    EnsembleConfig,
    ExperimentConfig,
    ExperimentRegistry,
    OptimizationConfig,
    TrainingOrchestrator,
    create_model_config,
)


def example_simple_experiment():
    """Example: Simple single-model experiment."""
    print("=== Example 1: Simple Ridge Regression Experiment ===")

    # Create experiment configuration
    config = ExperimentConfig.create_default("simple_ridge_example")

    # Run experiment
    orchestrator = TrainingOrchestrator(config)
    orchestrator.run_full_pipeline()

    print(f"Experiment completed: {orchestrator.experiment_id}")
    return orchestrator.experiment_id


def example_optimization_experiment():
    """Example: Multi-model experiment with hyperparameter optimization."""
    print("\n=== Example 2: Multi-Model with Optimization ===")

    # Create custom configuration
    config = ExperimentConfig(
        experiment_name="multi_model_optimization",
        data_config=DataConfig(
            data_path="data/sample_data.parquet",
            target_column="mu_hat"
        ),
        model_configs=[
            create_model_config("ridge"),
            create_model_config("mlp"),
            create_model_config("cnn_lstm")
        ],
        cv_config=CrossValidationConfig(
            n_splits=3,  # Fewer splits for faster demo
            embargo_days=1
        ),
        optimization_config=OptimizationConfig(
            enabled=True,
            n_trials=10,  # Fewer trials for demo
            metric="sharpe"
        )
    )

    # Run experiment
    orchestrator = TrainingOrchestrator(config)
    orchestrator.run_full_pipeline()

    print(f"Optimization experiment completed: {orchestrator.experiment_id}")
    return orchestrator.experiment_id


def example_ensemble_experiment():
    """Example: Ensemble experiment with multiple models."""
    print("\n=== Example 3: Ensemble Experiment ===")

    # Create ensemble configuration
    config = ExperimentConfig(
        experiment_name="ensemble_example",
        data_config=DataConfig(
            data_path="data/sample_data.parquet",
            target_column="mu_hat"
        ),
        model_configs=[
            create_model_config("ridge"),
            create_model_config("mlp")
        ],
        cv_config=CrossValidationConfig(n_splits=3),
        optimization_config=OptimizationConfig(
            enabled=False  # Disable for faster demo
        ),
        ensemble_config=EnsembleConfig(
            enabled=True,
            method="weighted_average"
        )
    )

    # Run experiment
    orchestrator = TrainingOrchestrator(config)
    orchestrator.run_full_pipeline()

    print(f"Ensemble experiment completed: {orchestrator.experiment_id}")
    return orchestrator.experiment_id


def example_experiment_management():
    """Example: Experiment registry and management."""
    print("\n=== Example 4: Experiment Management ===")

    # Create registry
    registry = ExperimentRegistry()

    # List all experiments
    experiments_df = registry.list_experiments()
    print("All experiments:")
    print(experiments_df.to_string(index=False))

    # Show details for the latest experiment
    if not experiments_df.empty:
        latest_experiment_id = experiments_df.iloc[0]['experiment_id']
        print(f"\nDetails for experiment {latest_experiment_id}:")

        summary = registry.get_experiment_summary(latest_experiment_id)
        if summary:
            print(f"Name: {summary['experiment']['experiment_name']}")
            print(f"Created: {summary['experiment']['created_at']}")
            print(f"Number of models: {summary.get('n_models', 0)}")
            print(f"Best model: {summary.get('best_model', 'N/A')}")


def example_config_serialization():
    """Example: Configuration serialization and loading."""
    print("\n=== Example 5: Configuration Management ===")

    # Create a complex configuration
    config = ExperimentConfig(
        experiment_name="serialization_example",
        data_config=DataConfig(
            data_path="data/sample_data.parquet",
            target_column="mu_hat",
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1
        ),
        model_configs=[
            create_model_config("ridge", alpha=0.1),
            create_model_config("mlp", hidden_layer_sizes=(100, 50))
        ],
        cv_config=CrossValidationConfig(
            strategy="purged_time_series",
            n_splits=5,
            embargo_days=2
        ),
        optimization_config=OptimizationConfig(
            enabled=True,
            n_trials=50,
            metric="sharpe"
        ),
        ensemble_config=EnsembleConfig(
            enabled=True,
            method="gating",
            gating_features=["rolling_vol_20", "atr"]
        )
    )

    # Save configuration
    config_path = "examples/example_config.yaml"
    config.save(config_path)
    print(f"Configuration saved to: {config_path}")

    # Load configuration
    loaded_config = ExperimentConfig.load(config_path)
    print(f"Configuration loaded: {loaded_config.experiment_name}")

    # Validate configuration
    try:
        loaded_config.validate()
        print("Configuration validation: PASSED")
    except ValueError as e:
        print(f"Configuration validation: FAILED - {e}")


def main():
    """Run all examples."""
    print("Unified Experimentation Framework - Examples")
    print("=" * 50)

    try:
        # Create examples directory
        Path("examples").mkdir(exist_ok=True)

        # Run examples
        experiment_ids = []

        # Example 1: Simple experiment
        exp_id = example_simple_experiment()
        experiment_ids.append(exp_id)

        # Example 2: Optimization experiment
        exp_id = example_optimization_experiment()
        experiment_ids.append(exp_id)

        # Example 3: Ensemble experiment
        exp_id = example_ensemble_experiment()
        experiment_ids.append(exp_id)

        # Example 4: Experiment management
        example_experiment_management()

        # Example 5: Configuration management
        example_config_serialization()

        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print(f"Created {len(experiment_ids)} experiments:")
        for i, exp_id in enumerate(experiment_ids, 1):
            print(f"  {i}. {exp_id}")

    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
