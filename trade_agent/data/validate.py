"""Simple validation utilities for timeseries data."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ValidationIssue:
	code: str
	details: str


@dataclass
class ValidationReport:
	passed: bool
	issues: list[ValidationIssue]


def validate_timeseries(
	df: pd.DataFrame, na_ratio_threshold: float = 0.05
) -> ValidationReport:
	issues: list[ValidationIssue] = []
	if df.empty:
		issues.append(
			ValidationIssue(code="empty", details="DataFrame is empty")
		)
	else:
		col_na = df.isna().mean()
		for col, ratio in col_na.items():
			if ratio > na_ratio_threshold:
				issues.append(
					ValidationIssue(
						code="nan_ratio",
						details=f"Column {col} NA ratio {ratio:.2%} > threshold",
					)
				)
	return ValidationReport(passed=len(issues) == 0, issues=issues)

__all__ = ["validate_timeseries", "ValidationReport", "ValidationIssue"]
