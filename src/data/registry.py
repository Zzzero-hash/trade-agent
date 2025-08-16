"""
Data pipeline registry for tracking and managing data processing runs.
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """Safely serialize objects to JSON, handling special types."""
    def json_serializer(o):
        if isinstance(o, (pd.Timestamp, datetime)):
            return o.isoformat()
        elif isinstance(o, bool):
            return bool(o)  # Explicitly convert to Python bool
        elif hasattr(o, 'item'):  # NumPy types
            return o.item()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    return json.dumps(obj, default=json_serializer, **kwargs)

from .config import DataPipelineConfig


class DataRegistry:
    """Registry for tracking data pipeline executions and results."""

    def __init__(self, db_path: str = "data/registry.db"):
        """Initialize data registry with SQLite database."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self) -> None:
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Pipeline runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    run_id TEXT PRIMARY KEY,
                    pipeline_name TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_message TEXT,
                    input_data_hash TEXT,
                    output_data_hash TEXT,
                    metadata TEXT
                )
            """)

            # Data sources table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_sources (
                    source_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    symbols TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    records_count INTEGER,
                    file_path TEXT,
                    file_size INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES pipeline_runs (run_id)
                )
            """)

            # Validation results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS validation_results (
                    validation_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    validation_type TEXT NOT NULL,
                    passed BOOLEAN NOT NULL,
                    score REAL,
                    details TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES pipeline_runs (run_id)
                )
            """)

            # Quality metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quality_metrics (
                    metric_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    threshold REAL,
                    passed BOOLEAN,
                    details TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES pipeline_runs (run_id)
                )
            """)

            # Data lineage table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_lineage (
                    lineage_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    input_path TEXT,
                    output_path TEXT,
                    transformation TEXT NOT NULL,
                    parameters TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES pipeline_runs (run_id)
                )
            """)

            conn.commit()

    def register_pipeline_run(self, config: DataPipelineConfig) -> str:
        """Register a new pipeline run and return run ID."""
        run_id = str(uuid.uuid4())
        config_hash = self._compute_config_hash(config)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO pipeline_runs
                (run_id, pipeline_name, config_hash, status, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                run_id,
                config.pipeline_name,
                config_hash,
                'created',
                safe_json_dumps({
                    'n_workers': config.n_workers,
                    'parallel_processing': config.parallel_processing,
                    'memory_limit': config.memory_limit,
                    'random_state': config.random_state,
                    'log_level': config.log_level
                })
            ))
            conn.commit()

        return run_id

    def update_run_status(self, run_id: str, status: str,
                         error_message: Optional[str] = None) -> None:
        """Update pipeline run status."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            if status == 'running':
                cursor.execute("""
                    UPDATE pipeline_runs
                    SET status = ?, started_at = CURRENT_TIMESTAMP
                    WHERE run_id = ?
                """, (status, run_id))
            elif status in ['completed', 'failed']:
                cursor.execute("""
                    UPDATE pipeline_runs
                    SET status = ?, completed_at = CURRENT_TIMESTAMP,
                        error_message = ?
                    WHERE run_id = ?
                """, (status, error_message, run_id))
            else:
                cursor.execute("""
                    UPDATE pipeline_runs SET status = ? WHERE run_id = ?
                """, (status, run_id))

            conn.commit()

    def log_data_source(self, run_id: str, source_name: str,
                       source_type: str, symbols: list[str],
                       start_date: str, end_date: str,
                       records_count: int, file_path: str,
                       file_size: int) -> str:
        """Log data source information."""
        source_id = str(uuid.uuid4())

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO data_sources
                (source_id, run_id, source_name, source_type, symbols,
                 start_date, end_date, records_count, file_path, file_size)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                source_id, run_id, source_name, source_type,
                safe_json_dumps(symbols), start_date, end_date,
                records_count, file_path, file_size
            ))
            conn.commit()

        return source_id

    def log_validation_result(self, run_id: str, stage: str,
                             validation_type: str, passed: bool,
                             score: Optional[float] = None,
                             details: Optional[dict[str, Any]] = None) -> str:
        """Log validation result."""
        validation_id = str(uuid.uuid4())

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO validation_results
                (validation_id, run_id, stage, validation_type, passed,
                 score, details)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                validation_id, run_id, stage, validation_type, passed,
                score, safe_json_dumps(details) if details else None
            ))
            conn.commit()

        return validation_id

    def log_quality_metric(self, run_id: str, metric_name: str,
                          metric_value: float, threshold: Optional[float] = None,
                          passed: Optional[bool] = None,
                          details: Optional[dict[str, Any]] = None) -> str:
        """Log quality metric."""
        metric_id = str(uuid.uuid4())

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO quality_metrics
                (metric_id, run_id, metric_name, metric_value, threshold,
                 passed, details)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                metric_id, run_id, metric_name, metric_value, threshold,
                passed, safe_json_dumps(details) if details else None
            ))
            conn.commit()

        return metric_id

    def log_data_lineage(self, run_id: str, transformation: str,
                        input_path: Optional[str] = None,
                        output_path: Optional[str] = None,
                        parameters: Optional[dict[str, Any]] = None) -> str:
        """Log data transformation lineage."""
        lineage_id = str(uuid.uuid4())

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO data_lineage
                (lineage_id, run_id, input_path, output_path,
                 transformation, parameters)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                lineage_id, run_id, input_path, output_path,
                transformation, safe_json_dumps(parameters) if parameters else None
            ))
            conn.commit()

        return lineage_id

    def list_pipeline_runs(self, limit: int = 50) -> pd.DataFrame:
        """List recent pipeline runs."""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT run_id, pipeline_name, status, created_at,
                       started_at, completed_at, error_message
                FROM pipeline_runs
                ORDER BY created_at DESC
                LIMIT ?
            """
            return pd.read_sql_query(query, conn, params=(limit,))

    def get_run_details(self, run_id: str) -> Optional[dict[str, Any]]:
        """Get detailed information about a specific run."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get run details
            cursor.execute("""
                SELECT * FROM pipeline_runs WHERE run_id = ?
            """, (run_id,))
            run_data = cursor.fetchone()

            if not run_data:
                return None

            # Get column names
            columns = [desc[0] for desc in cursor.description]
            run_info = dict(zip(columns, run_data))

            # Get data sources
            cursor.execute("""
                SELECT * FROM data_sources WHERE run_id = ?
            """, (run_id,))
            sources = cursor.fetchall()

            # Get validation results
            cursor.execute("""
                SELECT * FROM validation_results WHERE run_id = ?
            """, (run_id,))
            validations = cursor.fetchall()

            # Get quality metrics
            cursor.execute("""
                SELECT * FROM quality_metrics WHERE run_id = ?
            """, (run_id,))
            metrics = cursor.fetchall()

            return {
                'run_info': run_info,
                'data_sources': sources,
                'validation_results': validations,
                'quality_metrics': metrics
            }

    def get_data_lineage(self, run_id: str) -> pd.DataFrame:
        """Get data lineage for a specific run."""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT lineage_id, input_path, output_path,
                       transformation, parameters, created_at
                FROM data_lineage
                WHERE run_id = ?
                ORDER BY created_at
            """
            return pd.read_sql_query(query, conn, params=(run_id,))

    def export_run_report(self, run_id: str, output_path: str) -> None:
        """Export comprehensive run report."""
        details = self.get_run_details(run_id)
        if not details:
            raise ValueError(f"Run {run_id} not found")

        lineage = self.get_data_lineage(run_id)

        report = {
            'run_id': run_id,
            'run_info': details['run_info'],
            'data_sources': details['data_sources'],
            'validation_results': details['validation_results'],
            'quality_metrics': details['quality_metrics'],
            'data_lineage': lineage.to_dict('records') if not lineage.empty else []
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

    def cleanup_old_runs(self, days: int = 30) -> int:
        """Clean up runs older than specified days."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM pipeline_runs
                WHERE created_at < datetime('now', '-{} days')
            """.format(days))
            deleted_count = cursor.rowcount
            conn.commit()

        return deleted_count

    def _compute_config_hash(self, config: DataPipelineConfig) -> str:
        """Compute hash of configuration for tracking changes."""
        import hashlib

        config_str = safe_json_dumps({
            'pipeline_name': config.pipeline_name,
            'data_sources': len(config.data_sources),
            'validation_enabled': config.validation_config.enabled,
            'cleaning_enabled': config.cleaning_config.enabled,
            'features_enabled': config.feature_config.enabled,
            'random_state': config.random_state
        }, sort_keys=True)

        return hashlib.md5(config_str.encode()).hexdigest()

    def get_run_statistics(self) -> dict[str, Any]:
        """Get overall registry statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Total runs
            cursor.execute("SELECT COUNT(*) FROM pipeline_runs")
            total_runs = cursor.fetchone()[0]

            # Runs by status
            cursor.execute("""
                SELECT status, COUNT(*) FROM pipeline_runs
                GROUP BY status
            """)
            status_counts = dict(cursor.fetchall())

            # Recent activity (last 7 days)
            cursor.execute("""
                SELECT COUNT(*) FROM pipeline_runs
                WHERE created_at >= datetime('now', '-7 days')
            """)
            recent_runs = cursor.fetchone()[0]

            # Average run duration for completed runs
            cursor.execute("""
                SELECT AVG(
                    (julianday(completed_at) - julianday(started_at)) * 24 * 60
                ) as avg_duration_minutes
                FROM pipeline_runs
                WHERE status = 'completed'
                AND started_at IS NOT NULL
                AND completed_at IS NOT NULL
            """)
            avg_duration = cursor.fetchone()[0]

            return {
                'total_runs': total_runs,
                'status_counts': status_counts,
                'recent_runs': recent_runs,
                'avg_duration_minutes': avg_duration
            }
