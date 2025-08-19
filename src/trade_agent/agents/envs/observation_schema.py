"""Utilities for validating RL observation schema.

compute_observation_schema() inspects a feature dataframe and derives
expected flattened observation dimensionality given a rolling window
and optional inclusion of target columns (mu_hat, sigma_hat).

Keys produced:
    n_features: int
    has_mu_hat: bool
    has_sigma_hat: bool
    missing_targets: list[str]
    include_targets: bool
    window_size: int
    expected_dim: int
    feature_columns: list[str]
    schema_hash: str
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


TARGET_COLS = ["mu_hat", "sigma_hat"]


@dataclass
class ObservationSchema:
    n_features: int
    has_mu_hat: bool
    has_sigma_hat: bool
    missing_targets: list[str]
    include_targets: bool
    window_size: int
    expected_dim: int
    feature_columns: list[str]
    schema_hash: str

    def to_dict(self) -> dict[str, Any]:  # pragma: no cover - trivial
        return asdict(self)


def compute_observation_schema(
    df: pd.DataFrame, window_size: int, include_targets: bool = True
) -> ObservationSchema:
    has_mu = "mu_hat" in df.columns
    has_sigma = "sigma_hat" in df.columns
    missing = [c for c in TARGET_COLS if c not in df.columns]

    feature_cols = [c for c in df.columns if c not in TARGET_COLS]
    n_features = len(feature_cols)

    target_count = 2 if include_targets else 0
    # +2 for position & cash/equity
    expected_dim = window_size * n_features + target_count + 2

    canonical = {
        "features": feature_cols,
        "include_targets": include_targets,
        "window_size": window_size,
        "target_present": {"mu_hat": has_mu, "sigma_hat": has_sigma},
    }
    schema_hash = hashlib.sha256(
        json.dumps(canonical, separators=(",", ":")).encode()
    ).hexdigest()

    return ObservationSchema(
        n_features=n_features,
        has_mu_hat=has_mu,
        has_sigma_hat=has_sigma,
        missing_targets=missing,
        include_targets=include_targets,
        window_size=window_size,
        expected_dim=expected_dim,
        feature_columns=feature_cols,
        schema_hash=schema_hash,
    )


def save_observation_schema(schema: ObservationSchema, path: str) -> None:
    with open(path, "w") as f:
        json.dump(schema.to_dict(), f, indent=2, sort_keys=True)


__all__ = [
    "compute_observation_schema",
    "save_observation_schema",
    "ObservationSchema",
]
