"""Feature schema extraction utilities.

Computes:
- ordered column schema with dtypes
- schema_hash (sha256 of canonical JSON spec)
- data_hash (sha256 aggregate of per-column hashes)

Design goals:
- Deterministic output irrespective of physical Parquet column order
- Simple implementation (streaming optional later)
- Extensible for stats (null counts, min/max, versioning)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


@dataclass
class ColumnSpec:
    name: str
    dtype: str


@dataclass
class SchemaResult:
    columns: list[ColumnSpec]
    schema_hash: str
    data_hash: str
    n_rows: int
    n_cols: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": [asdict(c) for c in self.columns],
            "schema_hash": self.schema_hash,
            "data_hash": self.data_hash,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
        }


def _hash_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def extract_schema(path: str, sample_rows: int | None = None) -> SchemaResult:
    """Extract feature schema & hashes from a parquet/CSV file.

    Args:
        path: path to .parquet or .csv file.
        sample_rows: if set, only use first N rows for data hash.

    Returns:
        SchemaResult with columns, schema_hash, data_hash, counts.
    """
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        raise ValueError(
            f"Unsupported file type for schema extraction: {path}"
        )

    df_slice = df.head(sample_rows) if sample_rows is not None else df

    # Build column specs preserving order
    columns = [ColumnSpec(name=c, dtype=str(df[c].dtype)) for c in df.columns]

    # Canonical JSON for schema hash (sorted by name for stability)
    canonical_schema = {
        "columns": sorted(
            [{"name": c.name, "dtype": c.dtype} for c in columns],
            key=lambda x: x["name"],
        )
    }
    schema_hash = _hash_bytes(
        json.dumps(canonical_schema, separators=(",", ":")).encode()
    )

    # Data hash: hash each column's bytes then aggregate
    col_hashes = []
    for c in df_slice.columns:
        try:
            # Convert to numpy bytes representation for hashing
            values = df_slice[c].to_numpy()
            col_hash = _hash_bytes(values.tobytes())
        except Exception:
            # Fallback: JSON serialize values (slower)
            serialized = json.dumps(
                df_slice[c].tolist(), separators=(",", ":")
            ).encode()
            col_hash = _hash_bytes(serialized)
        col_hashes.append(f"{c}:{col_hash}")

    data_hash = _hash_bytes("|".join(sorted(col_hashes)).encode())

    return SchemaResult(
        columns=columns,
        schema_hash=schema_hash,
        data_hash=data_hash,
        n_rows=len(df),
        n_cols=len(df.columns),
    )


def save_schema(result: SchemaResult, out_path: str) -> None:
    """Persist schema result as JSON."""
    with open(out_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, sort_keys=True)


__all__ = ["extract_schema", "save_schema", "SchemaResult", "ColumnSpec"]
