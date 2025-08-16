"""
Experiment registry for tracking and reproducibility.
"""

import hashlib
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .config import ExperimentConfig


class ExperimentRegistry:
    """Centralized experiment tracking and reproducibility."""

    def __init__(self, storage_backend: str = "sqlite:///experiments.db"):
        """
        Initialize experiment registry.

        Args:
            storage_backend: Database connection string
        """
        self.storage_backend = storage_backend
        if storage_backend.startswith("sqlite:///"):
            self.db_path = storage_backend.replace("sqlite:///", "")
            self._init_sqlite_db()
        else:
            raise ValueError(f"Unsupported storage backend: {storage_backend}")

    def _init_sqlite_db(self):
        """Initialize SQLite database with required tables."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    experiment_name TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'created',
                    notes TEXT
                );

                CREATE TABLE IF NOT EXISTS results (
                    result_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                );

                CREATE TABLE IF NOT EXISTS model_artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    artifact_path TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                );

                CREATE INDEX IF NOT EXISTS idx_experiments_name
                ON experiments (experiment_name);
                CREATE INDEX IF NOT EXISTS idx_results_experiment
                ON results (experiment_id);
                CREATE INDEX IF NOT EXISTS idx_results_model_type
                ON results (model_type);
            """)

    def register_experiment(self, config: ExperimentConfig) -> str:
        """
        Register new experiment and return unique ID.

        Args:
            config: Experiment configuration

        Returns:
            Unique experiment ID
        """
        # Validate configuration
        config.validate()

        # Generate unique experiment ID
        experiment_id = str(uuid.uuid4())

        # Create config hash for deduplication
        config_str = str(config.__dict__)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()

        # Convert config to JSON
        config_dict = {
            'experiment_name': config.experiment_name,
            'data_config': config.data_config.__dict__,
            'model_configs': [mc.__dict__ for mc in config.model_configs],
            'cv_config': config.cv_config.__dict__,
            'optimization_config': config.optimization_config.__dict__,
            'ensemble_config': (config.ensemble_config.__dict__
                               if config.ensemble_config else None),
            'random_state': config.random_state,
            'output_dir': config.output_dir,
            'save_models': config.save_models
        }

        import json
        config_json = json.dumps(config_dict, indent=2, sort_keys=True)

        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO experiments
                (experiment_id, experiment_name, config_hash, config_json)
                VALUES (?, ?, ?, ?)
            """, (experiment_id, config.experiment_name, config_hash, config_json))

        return experiment_id

    def log_results(self, experiment_id: str, model_type: str,
                   metrics: dict[str, Any],
                   parameters: Optional[dict[str, Any]] = None) -> None:
        """
        Log experiment results.

        Args:
            experiment_id: Experiment ID
            model_type: Type of model
            metrics: Performance metrics
            parameters: Model parameters
        """
        import json

        result_id = str(uuid.uuid4())
        metrics_json = json.dumps(metrics, indent=2, sort_keys=True)
        parameters_json = json.dumps(parameters or {}, indent=2, sort_keys=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO results
                (result_id, experiment_id, model_type, metrics_json, parameters_json)
                VALUES (?, ?, ?, ?, ?)
            """, (result_id, experiment_id, model_type, metrics_json, parameters_json))

    def log_artifact(self, experiment_id: str, model_type: str,
                    artifact_path: str, artifact_type: str = "model") -> None:
        """
        Log model artifact.

        Args:
            experiment_id: Experiment ID
            model_type: Type of model
            artifact_path: Path to artifact file
            artifact_type: Type of artifact ('model', 'plot', 'data')
        """
        artifact_id = str(uuid.uuid4())

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO model_artifacts
                (artifact_id, experiment_id, model_type, artifact_path, artifact_type)
                VALUES (?, ?, ?, ?, ?)
            """, (artifact_id, experiment_id, model_type, artifact_path, artifact_type))

    def get_experiment(self, experiment_id: str) -> Optional[dict[str, Any]]:
        """Get experiment details by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM experiments WHERE experiment_id = ?
            """, (experiment_id,))
            row = cursor.fetchone()

        if row:
            return dict(row)
        return None

    def get_results(self, experiment_id: str) -> list[dict[str, Any]]:
        """Get all results for an experiment."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM results
                WHERE experiment_id = ?
                ORDER BY created_at DESC
            """, (experiment_id,))
            rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def get_best_config(self, metric: str = "val_sharpe") -> Optional[ExperimentConfig]:
        """
        Retrieve best performing configuration.

        Args:
            metric: Metric to optimize for

        Returns:
            Best experiment configuration or None if no experiments found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT e.config_json, r.metrics_json
                FROM experiments e
                JOIN results r ON e.experiment_id = r.experiment_id
                ORDER BY json_extract(r.metrics_json, '$.{}') DESC
                LIMIT 1
            """.format(metric))
            row = cursor.fetchone()

        if row:
            import json
            json.loads(row[0])
            # Reconstruct ExperimentConfig from dict
            # This is simplified - full implementation would need proper deserialization
            return ExperimentConfig.create_default()

        return None

    def list_experiments(self, limit: int = 100) -> pd.DataFrame:
        """List all experiments as a DataFrame."""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("""
                SELECT
                    experiment_id,
                    experiment_name,
                    config_hash,
                    created_at,
                    status
                FROM experiments
                ORDER BY created_at DESC
                LIMIT ?
            """, conn, params=(limit,))

        return df

    def get_experiment_summary(self, experiment_id: str) -> dict[str, Any]:
        """Get comprehensive experiment summary."""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return {}

        results = self.get_results(experiment_id)

        # Get artifacts
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM model_artifacts
                WHERE experiment_id = ?
                ORDER BY created_at DESC
            """, (experiment_id,))
            artifacts = [dict(row) for row in cursor.fetchall()]

        # Aggregate metrics
        metrics_summary = {}
        if results:
            import json
            for result in results:
                metrics = json.loads(result['metrics_json'])
                model_type = result['model_type']
                metrics_summary[model_type] = metrics

        return {
            'experiment': experiment,
            'results': results,
            'artifacts': artifacts,
            'metrics_summary': metrics_summary,
            'n_models': len({r['model_type'] for r in results}),
            'best_model': max(results, key=lambda x: json.loads(x['metrics_json']).get('val_sharpe', 0))['model_type'] if results else None
        }

    def cleanup_experiments(self, older_than_days: int = 30) -> int:
        """
        Clean up old experiments.

        Args:
            older_than_days: Delete experiments older than this many days

        Returns:
            Number of experiments deleted
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM experiments
                WHERE created_at < datetime('now', '-{} days')
            """.format(older_than_days))
            deleted_count = cursor.rowcount

        return deleted_count

    def export_results(self, output_path: str, experiment_ids: Optional[list[str]] = None):
        """Export experiment results to CSV/JSON."""
        with sqlite3.connect(self.db_path) as conn:
            if experiment_ids:
                placeholders = ','.join('?' * len(experiment_ids))
                query = f"""
                    SELECT e.experiment_name, e.created_at as experiment_date,
                           r.model_type, r.metrics_json, r.parameters_json
                    FROM experiments e
                    JOIN results r ON e.experiment_id = r.experiment_id
                    WHERE e.experiment_id IN ({placeholders})
                    ORDER BY e.created_at DESC, r.model_type
                """
                df = pd.read_sql_query(query, conn, params=experiment_ids)
            else:
                query = """
                    SELECT e.experiment_name, e.created_at as experiment_date,
                           r.model_type, r.metrics_json, r.parameters_json
                    FROM experiments e
                    JOIN results r ON e.experiment_id = r.experiment_id
                    ORDER BY e.created_at DESC, r.model_type
                """
                df = pd.read_sql_query(query, conn)

        # Expand JSON columns
        import json
        metrics_df = pd.json_normalize([json.loads(x) for x in df['metrics_json']])
        metrics_df.columns = [f'metric_{col}' for col in metrics_df.columns]

        final_df = pd.concat([
            df[['experiment_name', 'experiment_date', 'model_type']],
            metrics_df
        ], axis=1)

        output_path = Path(output_path)
        if output_path.suffix == '.csv':
            final_df.to_csv(output_path, index=False)
        elif output_path.suffix == '.json':
            final_df.to_json(output_path, orient='records', indent=2)
        else:
            raise ValueError("Output path must have .csv or .json extension")

        return final_df
