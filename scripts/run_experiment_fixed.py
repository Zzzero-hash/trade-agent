#!/usr/bin/env python3
"""
Unified Trading Agent Experimentation Framework CLI

This script provides a command-line interface for running experiments
with the unified framework, supporting supervised learning models,
reinforcement learning agents (PPO, SAC), and ensemble methods.
"""

import argparse
import os
import sys

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from experiments import (
    CrossValidationConfig,
    DataConfig,
    EnsembleConfig,
    ExperimentConfig,
    ExperimentRegistry,
    OptimizationConfig,
    TrainingOrchestrator,
    create_model_config,
)


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Unified Trading Agent Experimentation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Basic experiment configuration
    parser.add_argument('--name', default='cli_experiment',
                       help='Experiment name (default: cli_experiment)')
    parser.add_argument('--config',
                       help='Path to experiment configuration YAML file')

    # Data configuration
    parser.add_argument('--data', default='data/sample_data.parquet',
                       help='Path to training data (default: data/sample_data.parquet)')
    parser.add_argument('--target', default='mu_hat',
                       help='Target column name (default: mu_hat)')

    # Model selection - INCLUDING PPO and SAC
    parser.add_argument('--models',
                       choices=['ridge', 'linear', 'mlp', 'cnn_lstm',
                               'transformer', 'garch', 'ppo', 'sac'],
                       nargs='*',
                       default=['ridge'],
                       help='Models to train (default: ridge)')

    # Cross-validation
    parser.add_argument('--cv-splits', type=int, default=5,
                       help='Number of CV splits (default: 5)')
    parser.add_argument('--cv-gap', type=int, default=0,
                       help='Gap between train/validation sets (default: 0)')

    # Hyperparameter optimization
    parser.add_argument('--optimize', action='store_true',
                       help='Enable hyperparameter optimization')
    parser.add_argument('--n-trials', type=int, default=100,
                       help='Number of optimization trials (default: 100)')
    parser.add_argument('--metric', choices=['sharpe', 'mse', 'mae'],
                       default='sharpe',
                       help='Optimization metric (default: sharpe)')

    # Ensemble methods
    parser.add_argument('--ensemble', action='store_true',
                       help='Enable ensemble methods')
    parser.add_argument('--ensemble-method',
                       choices=['weighted_average', 'gating', 'stacking'],
                       default='weighted_average',
                       help='Ensemble method (default: weighted_average)')

    # General options
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output-dir', default='experiments',
                       help='Output directory (default: experiments)')
    parser.add_argument('--no-save-models', action='store_true',
                       help='Do not save trained models')

    # Registry and reporting
    parser.add_argument('--list-experiments', action='store_true',
                       help='List all experiments')
    parser.add_argument('--show-experiment',
                       help='Show details for specific experiment ID')
    parser.add_argument('--export-results',
                       help='Export results to CSV/JSON file')

    # Debugging
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show configuration without running experiment')

    parser.epilog = """
Examples:
  # Simple ridge regression experiment
  python scripts/run_experiment.py --name ridge_test --models ridge --data data/sample_data.parquet

  # Multi-model experiment with optimization
  python scripts/run_experiment.py --name multi_model_test \\
    --models ridge mlp cnn_lstm --optimize --n-trials 50

  # RL agents experiment
  python scripts/run_experiment.py --name rl_test \\
    --models ppo sac --optimize --n-trials 20

  # Ensemble experiment
  python scripts/run_experiment.py --name ensemble_test \\
    --models ridge mlp --ensemble --ensemble-method weighted_average

  # Load experiment from config file
  python scripts/run_experiment.py --config experiments/my_experiment.yaml

  # List past experiments
  python scripts/run_experiment.py --list-experiments

  # Show experiment details
  python scripts/run_experiment.py --show-experiment EXPERIMENT_ID
"""

    return parser


def create_experiment_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Create ExperimentConfig from command-line arguments."""

    # Data configuration
    data_config = DataConfig(
        data_path=args.data,
        target_column=args.target,
        feature_columns=None,  # Auto-detect
        test_size=0.2,
        random_state=args.random_state
    )

    # Model configurations
    model_configs = []
    for model_name in args.models:
        model_config = create_model_config(
            model_type=model_name,
            hyperparameters={}  # Default hyperparameters
        )
        model_configs.append(model_config)

    # Cross-validation configuration
    cv_config = CrossValidationConfig(
        n_splits=args.cv_splits,
        gap=args.cv_gap,
        test_size=0.2,
        random_state=args.random_state
    )

    # Optimization configuration
    optimization_config = None
    if args.optimize:
        optimization_config = OptimizationConfig(
            enabled=True,
            study_name=f"{args.name}_optimization",
            n_trials=args.n_trials,
            direction='maximize',
            metric=args.metric,
            random_state=args.random_state
        )

    # Ensemble configuration
    ensemble_config = None
    if args.ensemble:
        ensemble_config = EnsembleConfig(
            enabled=True,
            method=args.ensemble_method,
            weights=None,  # Auto-determine
            gating_features=None
        )

    return ExperimentConfig(
        name=args.name,
        data_config=data_config,
        model_configs=model_configs,
        cv_config=cv_config,
        optimization_config=optimization_config,
        ensemble_config=ensemble_config,
        random_state=args.random_state,
        output_dir=args.output_dir,
        save_models=not args.no_save_models
    )


def run_experiment(config: ExperimentConfig, verbose: bool = False) -> str:
    """Run experiment with given configuration."""

    if verbose:
        print("=== Experiment Configuration ===")
        print(f"Name: {config.name}")
        print(f"Models: {[mc.model_type for mc in config.model_configs]}")
        print(f"Data: {config.data_config.data_path}")
        print(f"Target: {config.data_config.target_column}")
        print(f"Optimization: {config.optimization_config is not None and config.optimization_config.enabled}")
        print()

    # Initialize registry and orchestrator
    registry = ExperimentRegistry()
    orchestrator = TrainingOrchestrator(config, registry)

    # Register experiment
    experiment_id = registry.register_experiment(config)
    print(f"Registered experiment: {experiment_id}")

    try:
        # Run the full pipeline
        results = orchestrator.run_full_pipeline()

        # Print results summary
        if verbose:
            print("=== Experiment Results ===")
            print("Model Performance:")
            for model_name, score in results.items():
                if isinstance(score, (int, float)):
                    print(f"  {model_name}: {score}")
                elif isinstance(score, dict) and 'score' in score:
                    print(f"  {model_name}: {score['score']}")

        print(f"\nExperiment ID: {experiment_id}")
        config_path = os.path.join(config.output_dir, experiment_id, "experiment_config.yaml")
        print(f"Configuration saved to: {config_path}")

        return experiment_id

    except Exception as e:
        print(f"Pipeline failed for experiment {experiment_id}: {e}")
        raise


def list_experiments(verbose: bool = False):
    """List all experiments in the registry."""
    registry = ExperimentRegistry()
    experiments = registry.list_experiments()

    if experiments.empty:
        print("No experiments found.")
        return

    print("=== Experiments ===")
    if verbose:
        print(experiments.to_string(index=False))
    else:
        print(experiments[['experiment_id', 'experiment_name', 'created_at', 'status']].to_string(index=False))


def show_experiment_details(experiment_id: str):
    """Show detailed information about a specific experiment."""
    registry = ExperimentRegistry()
    details = registry.get_experiment_details(experiment_id)

    if not details:
        print(f"Experiment {experiment_id} not found.")
        return

    print(f"=== Experiment {experiment_id} ===")
    print(f"Name: {details.get('experiment_name', 'N/A')}")
    print(f"Created: {details.get('created_at', 'N/A')}")
    print(f"Status: {details.get('status', 'N/A')}")

    # Show model performance
    results = registry.get_experiment_results(experiment_id)
    if results:
        print("\nModel Performance:")
        for model, score in results.items():
            if isinstance(score, (int, float)):
                print(f"  {model}: {score:.4f}")
            elif isinstance(score, dict):
                main_score = score.get('score', score.get('validation_score', 'N/A'))
                print(f"  {model}: {main_score}")

    # Show best model
    best_model = details.get('best_model')
    if best_model:
        print(f"\nBest model: {best_model}")

    model_count = len(results) if results else 0
    print(f"Number of models: {model_count}")


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Handle registry operations
    if args.list_experiments:
        list_experiments(verbose=args.verbose)
        return

    if args.show_experiment:
        show_experiment_details(args.show_experiment)
        return

    # Handle experiment export
    if args.export_results:
        registry = ExperimentRegistry()
        registry.export_results(args.export_results)
        print(f"Results exported to: {args.export_results}")
        return

    # Create experiment configuration
    if args.config:
        # Load from YAML file
        config = ExperimentConfig.from_yaml(args.config)
    else:
        # Create from command-line arguments
        config = create_experiment_config_from_args(args)

    # Handle dry run
    if args.dry_run:
        print("=== Dry Run - Configuration Only ===")
        print(f"Experiment: {config.name}")
        print(f"Models: {[mc.model_type for mc in config.model_configs]}")
        print(f"Data: {config.data_config.data_path}")
        print(f"Optimization: {config.optimization_config is not None and config.optimization_config.enabled}")
        return

    # Run the experiment
    try:
        experiment_id = run_experiment(config, verbose=args.verbose)
        print(f"\nExperiment completed successfully: {experiment_id}")
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
