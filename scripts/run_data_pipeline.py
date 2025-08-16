#!/usr/bin/env python3
"""
Unified Data Pipeline CLI

This script provides a command-line interface for running data pipelines
with the unified framework, supporting data ingestion, validation, cleaning,
feature engineering, and quality monitoring.
"""

import argparse
import os
import sys

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data import (
    CleaningConfig,
    DataOrchestrator,
    DataPipelineConfig,
    DataRegistry,
    FeatureConfig,
    QualityConfig,
    StorageConfig,
    ValidationConfig,
    create_data_source_config,
)


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Unified Data Pipeline Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Basic pipeline configuration
    parser.add_argument('--name', default='cli_pipeline',
                       help='Pipeline name (default: cli_pipeline)')
    parser.add_argument('--config',
                       help='Path to pipeline configuration YAML file')

    # Data source configuration
    parser.add_argument('--sources', choices=['yahoo_finance', 'alpaca', 'csv', 'parquet'],
                       nargs='*', default=['yahoo_finance'],
                       help='Data sources to use (default: yahoo_finance)')
    parser.add_argument('--symbols', nargs='*',
                       default=['AAPL', 'GOOGL'],
                       help='Stock symbols to fetch (default: AAPL GOOGL)')
    parser.add_argument('--start-date', type=str,
                       default='2020-01-01',
                       help='Start date for data (default: 2020-01-01)')
    parser.add_argument('--end-date', type=str,
                       default='2024-01-01',
                       help='End date for data (default: 2024-01-01)')
    parser.add_argument('--interval', default='1d',
                       choices=['1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo'],
                       help='Data interval (default: 1d)')

    # Processing configuration
    parser.add_argument('--validate', action='store_true',
                       help='Enable data validation')
    parser.add_argument('--clean', action='store_true',
                       help='Enable data cleaning')
    parser.add_argument('--features', action='store_true',
                       help='Enable feature engineering')
    parser.add_argument('--quality', action='store_true',
                       help='Enable quality monitoring')

    # Storage configuration
    parser.add_argument('--format', choices=['parquet', 'csv'],
                       default='parquet',
                       help='Output format (default: parquet)')
    parser.add_argument('--output-dir', default='data/pipelines',
                       help='Output directory (default: data/pipelines)')

    # Processing options
    parser.add_argument('--parallel', action='store_true',
                       help='Enable parallel processing')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of workers for parallel processing (default: 4)')

    # Registry and reporting
    parser.add_argument('--list-runs', action='store_true',
                       help='List recent pipeline runs')
    parser.add_argument('--show-run',
                       help='Show details for specific run ID')
    parser.add_argument('--export-report',
                       help='Export run report to JSON file')
    parser.add_argument('--cleanup-runs', type=int,
                       help='Clean up runs older than N days')

    # General options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show configuration without running pipeline')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='Logging level (default: INFO)')

    parser.epilog = """
Examples:
  # Basic pipeline with Yahoo Finance data
  python scripts/run_data_pipeline.py --name basic_test --symbols AAPL MSFT

  # Full pipeline with all processing stages
  python scripts/run_data_pipeline.py --name full_pipeline \\
    --symbols AAPL GOOGL MSFT --validate --clean --features --quality

  # Multi-source pipeline
  python scripts/run_data_pipeline.py --name multi_source \\
    --sources yahoo_finance csv --parallel --workers 8

  # Load pipeline from config file
  python scripts/run_data_pipeline.py --config pipelines/my_pipeline.yaml

  # List recent runs
  python scripts/run_data_pipeline.py --list-runs

  # Show run details
  python scripts/run_data_pipeline.py --show-run RUN_ID
"""

    return parser


def create_pipeline_config_from_args(args: argparse.Namespace) -> DataPipelineConfig:
    """Create DataPipelineConfig from command-line arguments."""

    # Create data sources
    data_sources = []
    for source_type in args.sources:
        if source_type in ['yahoo_finance', 'alpaca']:
            source = create_data_source_config(
                source_type,
                name=f"{source_type}_source",
                symbols=args.symbols,
                start_date=args.start_date,
                end_date=args.end_date,
                interval=args.interval
            )
        else:
            # For CSV/Parquet, would need file paths
            source = create_data_source_config(
                source_type,
                name=f"{source_type}_source"
            )
        data_sources.append(source)

    # Create configuration objects
    validation_config = ValidationConfig(enabled=args.validate)
    cleaning_config = CleaningConfig(enabled=args.clean)
    feature_config = FeatureConfig(enabled=args.features)
    storage_config = StorageConfig(
        format=args.format,
        processed_data_dir=f"{args.output_dir}/processed",
        raw_data_dir=f"{args.output_dir}/raw"
    )
    quality_config = QualityConfig(enabled=args.quality)

    return DataPipelineConfig(
        pipeline_name=args.name,
        data_sources=data_sources,
        validation_config=validation_config,
        cleaning_config=cleaning_config,
        feature_config=feature_config,
        storage_config=storage_config,
        quality_config=quality_config,
        parallel_processing=args.parallel,
        n_workers=args.workers,
        output_dir=args.output_dir,
        log_level=args.log_level
    )


def run_pipeline(config: DataPipelineConfig, verbose: bool = False) -> str:
    """Run data pipeline with given configuration."""

    if verbose:
        print("=== Pipeline Configuration ===")
        print(f"Name: {config.pipeline_name}")
        print(f"Sources: {[s.name for s in config.data_sources]}")
        print(f"Validation: {config.validation_config.enabled}")
        print(f"Cleaning: {config.cleaning_config.enabled}")
        print(f"Features: {config.feature_config.enabled}")
        print(f"Quality: {config.quality_config.enabled}")
        print(f"Parallel: {config.parallel_processing}")
        print()

    # Initialize registry and orchestrator
    registry = DataRegistry()
    orchestrator = DataOrchestrator(config, registry)

    try:
        # Run the full pipeline
        results = orchestrator.run_full_pipeline()

        # Print results summary
        if verbose:
            print("=== Pipeline Results ===")
            for stage, stage_results in results.items():
                print(f"\n{stage.title()}:")
                if isinstance(stage_results, dict):
                    for source, source_results in stage_results.items():
                        if isinstance(source_results, dict) and 'data' in source_results:
                            records = len(source_results['data'])
                            columns = len(source_results['data'].columns)
                            print(f"  {source}: {records} records, {columns} columns")
                        else:
                            print(f"  {source}: {source_results}")

        print(f"\nPipeline completed successfully: {orchestrator.run_id}")
        return orchestrator.run_id

    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise


def list_pipeline_runs(verbose: bool = False, limit: int = 20):
    """List recent pipeline runs."""
    registry = DataRegistry()
    runs = registry.list_pipeline_runs(limit=limit)

    if runs.empty:
        print("No pipeline runs found.")
        return

    print("=== Recent Pipeline Runs ===")
    if verbose:
        print(runs.to_string(index=False))
    else:
        display_cols = ['run_id', 'pipeline_name', 'status', 'created_at']
        available_cols = [col for col in display_cols if col in runs.columns]
        print(runs[available_cols].to_string(index=False))


def show_run_details(run_id: str):
    """Show detailed information about a specific run."""
    registry = DataRegistry()
    details = registry.get_run_details(run_id)

    if not details:
        print(f"Run {run_id} not found.")
        return

    print(f"=== Run Details: {run_id} ===")
    run_info = details['run_info']
    print(f"Pipeline: {run_info.get('pipeline_name', 'N/A')}")
    print(f"Status: {run_info.get('status', 'N/A')}")
    print(f"Created: {run_info.get('created_at', 'N/A')}")
    print(f"Started: {run_info.get('started_at', 'N/A')}")
    print(f"Completed: {run_info.get('completed_at', 'N/A')}")

    if run_info.get('error_message'):
        print(f"Error: {run_info['error_message']}")

    # Show data sources
    sources = details['data_sources']
    if sources:
        print(f"\nData Sources ({len(sources)}):")
        for source in sources:
            symbols = source[4] if source[4] else "N/A"  # symbols column
            records = source[6] if source[6] else 0  # records_count column
            print(f"  {source[2]} ({source[3]}): {symbols}, {records} records")

    # Show validation results
    validations = details['validation_results']
    if validations:
        print(f"\nValidation Results ({len(validations)}):")
        for validation in validations:
            passed = "✓" if validation[4] else "✗"  # passed column
            score = f" (score: {validation[5]:.3f})" if validation[5] else ""
            print(f"  {passed} {validation[3]}: {validation[2]}{score}")

    # Show quality metrics
    metrics = details['quality_metrics']
    if metrics:
        print(f"\nQuality Metrics ({len(metrics)}):")
        for metric in metrics:
            passed = "✓" if metric[5] else "✗"  # passed column
            threshold = f" (threshold: {metric[4]:.3f})" if metric[4] else ""
            print(f"  {passed} {metric[2]}: {metric[3]:.3f}{threshold}")


def cleanup_old_runs(days: int):
    """Clean up runs older than specified days."""
    registry = DataRegistry()
    deleted_count = registry.cleanup_old_runs(days)
    print(f"Cleaned up {deleted_count} runs older than {days} days.")


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Handle registry operations
    if args.list_runs:
        list_pipeline_runs(verbose=args.verbose)
        return

    if args.show_run:
        show_run_details(args.show_run)
        return

    if args.cleanup_runs:
        cleanup_old_runs(args.cleanup_runs)
        return

    if args.export_report:
        if not args.show_run:
            print("Error: --show-run required with --export-report")
            return
        registry = DataRegistry()
        registry.export_run_report(args.show_run, args.export_report)
        print(f"Report exported to: {args.export_report}")
        return

    # Create pipeline configuration
    if args.config:
        # Load from YAML file
        config = DataPipelineConfig.load(args.config)
    else:
        # Create from command-line arguments
        config = create_pipeline_config_from_args(args)

    # Handle dry run
    if args.dry_run:
        print("=== Dry Run - Configuration Only ===")
        print(f"Pipeline: {config.pipeline_name}")
        print(f"Sources: {[s.name for s in config.data_sources]}")
        print(f"Symbols: {config.data_sources[0].symbols if config.data_sources else []}")
        print(f"Processing: validation={config.validation_config.enabled}, "
              f"cleaning={config.cleaning_config.enabled}, "
              f"features={config.feature_config.enabled}")
        return

    # Run the pipeline
    try:
        run_id = run_pipeline(config, verbose=args.verbose)
        print(f"\nPipeline completed successfully: {run_id}")

        # Show registry statistics
        if args.verbose:
            registry = DataRegistry()
            stats = registry.get_run_statistics()
            print("\n=== Registry Statistics ===")
            print(f"Total runs: {stats['total_runs']}")
            print(f"Recent runs (7 days): {stats['recent_runs']}")
            if stats['avg_duration_minutes']:
                print(f"Average duration: {stats['avg_duration_minutes']:.1f} minutes")

    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
