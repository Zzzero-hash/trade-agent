from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


MetricFn = Callable[[Any, Any], float]


@dataclass(frozen=True)
class CVFoldResult:
    fold: int
    train_size: int
    val_size: int
    scores: Mapping[str, float]


class PurgedTimeSeriesSplit:
    def __init__(
        self, n_splits: int, gap: int = 0, embargo: int = 0
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.gap = max(0, gap)
        self.embargo = max(0, embargo)

    def split(
        self, X: Sequence[Any]
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:  # type: ignore[type-arg]
        n = len(X)
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1
        idx = np.arange(n)
        cursor = 0
        for fold_size in fold_sizes:
            start = cursor
            stop = start + fold_size
            cursor = stop
            val_idx = idx[start:stop]
            train_mask = idx < start
            if self.embargo:
                train_mask &= idx < (start - self.embargo)
            train_idx = idx[train_mask]
            if self.gap and train_idx.size:
                train_idx = train_idx[train_idx < (start - self.gap)]
            yield train_idx, val_idx


def temporal_cv_scores(
    model: Any,
    X: Any,
    y: Any,
    splitter: PurgedTimeSeriesSplit,
    metrics: dict[str, MetricFn],
    maximize: bool | dict[str, bool] = False,
    prune_callback: Callable[[int, float], None] | None = None,
) -> dict[str, Any]:
    if X.shape[0] != y.shape[0]:  # type: ignore[attr-defined]
        raise ValueError("X and y must align")

    if isinstance(maximize, bool):
        max_flags = {m: maximize for m in metrics}
    else:
        max_flags = {m: maximize.get(m, True) for m in metrics}

    folds: list[CVFoldResult] = []
    agg: dict[str, list[float]] = {m: [] for m in metrics}

    for fold_idx, (tr, va) in enumerate(
        splitter.split(range(X.shape[0]))  # type: ignore[arg-type]
    ):
        if tr.size == 0 or va.size == 0:
            continue
        model.fit(X[tr], y[tr])  # type: ignore[index]
        preds = model.predict(X[va])  # type: ignore[index]
        scores: dict[str, float] = {}
        for name, fn in metrics.items():
            raw = float(fn(y[va], preds))  # type: ignore[index]
            val = raw if max_flags.get(name, True) else -raw
            scores[name] = val
            agg[name].append(val)
        folds.append(
            CVFoldResult(
                fold=fold_idx,
                train_size=int(tr.size),
                val_size=int(va.size),
                scores=scores,
            )
        )
        if prune_callback is not None:
            primary = list(metrics.keys())[0]
            running = float(np.mean(agg[primary]))
            prune_callback(fold_idx, running)

    summary = {
        f"mean_{m}": float(np.mean(v)) if v else 0.0
        for m, v in agg.items()
    }
    for name, flag in max_flags.items():
        if not flag:
            summary[f"mean_{name}"] = -summary[f"mean_{name}"]

    return {"folds": [f.__dict__ for f in folds], "summary": summary}


def optuna_objective_factory(
    model_factory: Callable[[dict[str, float]], Any],
    X: Any,
    y: Any,
    splitter: PurgedTimeSeriesSplit,
    metric: str = "mse",
    maximize: bool = False,
    metrics: dict[str, MetricFn] | None = None,
):  # pragma: no cover
    if metrics is None:
        def _mse(a: Any, b: Any) -> float:
            return float(np.mean((a - b) ** 2))
        metrics = {metric: _mse}

    def _objective(trial):  # type: ignore[no-untyped-def]
        params = getattr(trial, "params", {})
        model = model_factory(params)

        def _prune_cb(f_idx: int, score: float) -> None:
            trial.report(score if maximize else -score, f_idx)
            from optuna.exceptions import TrialPruned  # local import
            if trial.should_prune():  # pragma: no cover
                raise TrialPruned()

        result = temporal_cv_scores(
            model,
            X,
            y,
            splitter=splitter,
            metrics=metrics,
            maximize={metric: maximize},
            prune_callback=_prune_cb,
        )
        mean_metric = result["summary"][f"mean_{metric}"]
        return -mean_metric if maximize else mean_metric

    return _objective

# Public exports


__all__ = [
    "PurgedTimeSeriesSplit",
    "temporal_cv_scores",
    "optuna_objective_factory",
    "CVFoldResult",
]
