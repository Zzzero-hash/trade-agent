"""Tests for MLflow integration utilities and Makefile target.

Validates that:
  * Each mlflow_run context creates a distinct run with params, metrics,
    and artifacts recorded in a temporary tracking store.
  * The Makefile contains the `mlflow-ui` target invoking `mlflow ui`.
"""
from __future__ import annotations

import glob
from pathlib import Path

import pytest


mlflow = pytest.importorskip("mlflow")  # noqa: F401  (skip if mlflow missing)

from trade_agent.logging.mlflow_utils import (  # type: ignore  # noqa: E402
    log_artifact,
    log_metrics,
    mlflow_run,
)


def _collect_run_dirs(base: Path) -> list[Path]:
    """Return list of run directories containing meta.yaml under base.

    Layout: <base>/<experiment_id>/<run_id>/meta.yaml
    """
    return [
        Path(p).parent
        for p in glob.glob(str(base / "*" / "*" / "meta.yaml"))
    ]


def test_mlflow_run_logging(  # type: ignore[no-untyped-def]
    tmp_path: Path,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    monkeypatch.setenv(  # type: ignore[attr-defined]
        "MLFLOW_TRACKING_URI", str(tmp_path)
    )

    # First run
    artifact1 = tmp_path / "artifact1.txt"
    artifact1.write_text("hello1")
    with mlflow_run(run_name="test_run_one", params={"param_a": 1}):
        log_metrics({"metric_x": 0.123})
        log_artifact(artifact1)

    # Second run
    artifact2 = tmp_path / "artifact2.txt"
    artifact2.write_text("hello2")
    with mlflow_run(run_name="test_run_two", params={"param_b": 2}):
        log_metrics({"metric_y": 9.87})
        log_artifact(artifact2)

    run_dirs = _collect_run_dirs(tmp_path)
    assert len(run_dirs) >= 2, "Expected at least two MLflow run directories"

    for rd in run_dirs:
        meta = rd / "meta.yaml"
        params_dir = rd / "params"
        metrics_dir = rd / "metrics"
        artifacts_dir = rd / "artifacts"
        assert meta.is_file(), f"Missing meta.yaml in {rd}"
        assert any(params_dir.glob("*")), f"No params logged in {rd}"
        assert any(metrics_dir.glob("*")), f"No metrics logged in {rd}"
        assert artifacts_dir.is_dir(), f"No artifacts dir in {rd}"


def test_makefile_has_mlflow_ui_target() -> None:
    makefile = Path.cwd() / "Makefile"
    assert makefile.is_file(), "Makefile not found at project root"
    content = makefile.read_text().splitlines()
    inside_target = False
    lines: list[str] = []
    for line in content:
        if line.startswith("mlflow-ui:"):
            inside_target = True
            continue
        if inside_target:
            if line.strip().startswith("#"):
                continue
            if line and not line.startswith("\t"):
                break  # next target encountered
            lines.append(line.strip())
    combined = "\n".join(lines)
    assert "mlflow ui" in combined, (
        "mlflow-ui target does not invoke 'mlflow ui'"
    )
