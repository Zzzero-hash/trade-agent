"""MLflow integration helpers.

Provides a thin wrapper to:
 - start a run
 - log flattened hyperparameters (pruning large values)
 - log metrics and artifacts

Safe to import even if MLflow is missing (no-op behavior).
"""
from __future__ import annotations

from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any


try:  # optional dependency
    import mlflow  # type: ignore
    _mlflow_available = True
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore
    _mlflow_available = False


def _flatten(
    prefix: str,
    obj: Any,
    out: dict[str, Any],
    depth: int = 0,
    max_depth: int = 4,
) -> None:
    if depth > max_depth:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(
                f"{prefix}.{k}" if prefix else k,
                v,
                out,
                depth + 1,
                max_depth,
            )
    elif isinstance(obj, list | tuple):
        out[prefix] = str(obj)[:120]
    else:
        out[prefix] = obj


@contextmanager
def mlflow_run(
    run_name: str | None = None,
    params: dict[str, Any] | None = None,
) -> Any:
    if not _mlflow_available:
        yield None
        return
    # Ensure an experiment exists; set_experiment creates if missing and
    # sets the active experiment id (avoids reliance on implicit id '0').
    with suppress(Exception):  # pragma: no cover
        mlflow.set_experiment("Default")  # type: ignore[attr-defined]
    with mlflow.start_run(run_name=run_name):  # type: ignore[attr-defined]
        if params:
            flat: dict[str, Any] = {}
            _flatten("", params, flat)
            # Truncate very long strings
            cleaned = {
                k: (v if isinstance(v, int | float) else str(v)[:200])
                for k, v in flat.items()
            }
            mlflow.log_params(cleaned)  # type: ignore[attr-defined]
        yield mlflow


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    if not _mlflow_available:
        return
    for k, v in metrics.items():
        try:
            if mlflow is not None:  # type: ignore
                mlflow.log_metric(
                    k, float(v), step=step  # type: ignore[attr-defined]
                )
        except Exception:  # pragma: no cover
            continue


def log_artifact(path: str | Path) -> None:
    if not _mlflow_available:
        return
    try:
        if mlflow is not None:  # type: ignore
            mlflow.log_artifact(str(path))  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        pass


class SB3MLflowCallback:
    """Light-weight callback interface for SB3 training loops.

    Minimal dependency (not inheriting BaseCallback to avoid import if unused).
    Attach via custom callback list; expects model.logger to provide
    rollout/episode info.
    """
    def __init__(self, every_n_steps: int = 1000) -> None:
        self.every_n_steps = every_n_steps

    def __call__(
        self, locals_: dict[str, Any], globals_: dict[str, Any]
    ) -> bool:  # noqa: D401
        # Try to extract progress info
        step = int(locals_.get("num_timesteps", 0))
        if step % self.every_n_steps != 0:
            return True
        info_dict: dict[str, float] = {}
        # Common SB3 logger keys
        logger = locals_.get("logger")
        if logger and hasattr(logger, "name_to_value"):
            for k, v in logger.name_to_value.items():
                try:
                    info_dict[k] = float(v)
                except Exception:
                    continue
        if info_dict:
            log_metrics(info_dict, step=step)
            return True
        return None


__all__ = ["mlflow_run", "log_metrics", "log_artifact", "SB3MLflowCallback"]
