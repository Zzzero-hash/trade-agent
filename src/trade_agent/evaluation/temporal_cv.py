"""Temporal (purged / embargoed) cross‑validation utilities.

This module provides a *minimal, dependency-light* implementation of
time-ordered CV splitting designed to prevent look‑ahead leakage when
features consume trailing windows. It integrates with Optuna by optionally
invoking a pruning callback after each fold.

Exports:
    PurgedTimeSeriesSplit – generator yielding (train_idx, val_idx)
    temporal_cv_scores – run CV, return per-fold + aggregate metrics
    optuna_objective_factory – helper returning an Optuna objective

The splitter implements both a *gap* (purge) between training and
validation folds and an *embargo* that removes trailing samples from the
training set which would overlap with future validation information.

All metrics are expressed in a maximization-friendly way internally; if
`maximize=False` for a given metric its sign is inverted during fold
aggregation and restored in the returned summary.
"""
# mypy: ignore-errors
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Sequence, Mapping, Any

import numpy as np
import numpy.typing as npt

ArrayLike = npt.NDArray[np.float_]
MetricFn = Callable[[ArrayLike, ArrayLike], float]


@dataclass(frozen=True)
class CVFoldResult:
    """Container for an individual CV fold result."""

    fold: int
    train_size: int
    val_size: int
    scores: Mapping[str, float]


class PurgedTimeSeriesSplit:
    """Time series split with *gap purge* + *embargo*.

    Parameters
    ----------
    n_splits : int
        Number of validation folds. Must be >= 2.
    gap : int, default 0
        Number of samples to exclude immediately before each validation
        fold (purging potential label leakage via rolling features).
    embargo : int, default 0
        Number of *final* samples in the training window removed so that
        information that will soon overlap the validation period is not
        used for fitting (López de Prado style embargo).
    """

    def __init__(self, n_splits: int, gap: int = 0, embargo: int = 0):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.gap = max(gap, 0)
        self.embargo = max(embargo, 0)

    def split(
        self, X: Sequence[Any]
    ) -> Iterator[tuple[ArrayLike, ArrayLike]]:
        n_samples = len(X)
        fold_sizes = np.full(
            self.n_splits, n_samples // self.n_splits, dtype=int
        )
        fold_sizes[: n_samples % self.n_splits] += 1
        indices = np.arange(n_samples)
        current = 0
        for fold, fold_size in enumerate(fold_sizes):
            start = current
            stop = start + fold_size
            current = stop
            val_idx = indices[start:stop]
            train_mask = indices < start
            if self.embargo > 0:
                train_mask &= indices < (start - self.embargo)
            train_idx = indices[train_mask]
            if self.gap > 0 and train_idx.size:
                train_idx = train_idx[train_idx < (start - self.gap)]
            yield train_idx, val_idx

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"PurgedTimeSeriesSplit(n_splits={self.n_splits}, gap={self.gap}, "
            f"embargo={self.embargo})"
        )


def temporal_cv_scores(
    model: Any,
    X: ArrayLike,
    y: ArrayLike,
    splitter: PurgedTimeSeriesSplit,
    metrics: dict[str, MetricFn],
    maximize: bool | dict[str, bool] = False,
    prune_callback: Callable[[int, float], None] | None = None,
) -> dict[str, object]:
    """Run temporal CV returning per-fold + aggregate scores.

    Parameters
    ----------
    model : estimator implementing fit(X, y) and predict(X)
    X, y : np.ndarray
        Ordered features / target.
    splitter : PurgedTimeSeriesSplit
        Defines fold boundaries.
    metrics : dict[str, MetricFn]
        Mapping of metric name -> function.
    maximize : bool | dict[str, bool]
        Global or per-metric maximize flags (True = higher is better).
    prune_callback : callable, optional
        Invoked with (fold_index, running_primary_metric) after each fold
        (used for Optuna pruning via trial.report + should_prune).
    """
    if getattr(X, "shape", None) is None or getattr(y, "shape", None) is None:
        raise ValueError("X and y must be array-like with shape attribute")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of rows")

    if isinstance(maximize, bool):
        max_flags = {m: maximize for m in metrics}
    else:
        max_flags = {m: maximize.get(m, True) for m in metrics}

    fold_results: list[CVFoldResult] = []
    aggregates: dict[str, list[float]] = {m: [] for m in metrics}

    for fold_idx, (tr, va) in enumerate(splitter.split(X)):
        if tr.size == 0 or va.size == 0:
            continue
        model.fit(X[tr], y[tr])
        preds = model.predict(X[va])
        scores: dict[str, float] = {}
        for name, fn in metrics.items():
            score_val = float(fn(y[va], preds))
            # Invert for internal maximization uniformity when needed
            if not max_flags.get(name, True):
                score_val = -score_val
            scores[name] = score_val
            aggregates[name].append(score_val)
        fold_results.append(
            CVFoldResult(
                """Temporal cross‑validation (purged + embargo) helpers.

                Focused, dependency‑light implementation for SL model selection &
                Optuna integration. Designed to minimise complexity while providing
                leakage‑aware validation folds.
                """
                from __future__ import annotations

                from dataclasses import dataclass
                from typing import Any, Callable, Iterator, Mapping, Sequence

                import numpy as np

                MetricFn = Callable[[np.ndarray, np.ndarray], float]


                @dataclass(frozen=True)
                class CVFoldResult:
                    fold: int
                    train_size: int
                    val_size: int
                    scores: Mapping[str, float]


                class PurgedTimeSeriesSplit:
                    def __init__(self, n_splits: int, gap: int = 0, embargo: int = 0) -> None:
                        if n_splits < 2:
                            raise ValueError("n_splits must be >= 2")
                        self.n_splits = n_splits
                        self.gap = max(0, gap)
                        self.embargo = max(0, embargo)

                    def split(self, X: Sequence[Any]) -> Iterator[tuple[np.ndarray, np.ndarray]]:
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

                    fold_results: list[CVFoldResult] = []
                    aggregates: dict[str, list[float]] = {m: [] for m in metrics}

                    for fold_idx, (tr, va) in enumerate(splitter.split(range(X.shape[0]))):  # type: ignore[arg-type]
                        if tr.size == 0 or va.size == 0:
                            continue
                        model.fit(X[tr], y[tr])  # type: ignore[index]
                        preds = model.predict(X[va])  # type: ignore[index]
                        scores: dict[str, float] = {}
                        for name, fn in metrics.items():
                            raw = float(fn(y[va], preds))  # type: ignore[index]
                            val = raw if max_flags.get(name, True) else -raw
                            scores[name] = val
                            aggregates[name].append(val)
                        fold_results.append(
                            CVFoldResult(
                                fold=fold_idx,
                                train_size=int(tr.size),
                                val_size=int(va.size),
                                scores=scores,
                            )
                        )
                        if prune_callback is not None:
                            primary = list(metrics.keys())[0]
                            running_mean = float(np.mean(aggregates[primary]))
                            prune_callback(fold_idx, running_mean)

                    summary = {
                        f"mean_{m}": float(np.mean(vals)) if vals else 0.0
                        for m, vals in aggregates.items()
                    }
                    for name, flag in max_flags.items():
                        if not flag:
                            summary[f"mean_{name}"] = -summary[f"mean_{name}"]

                    return {"folds": [fr.__dict__ for fr in fold_results], "summary": summary}


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
                        def _mse(a: np.ndarray, b: np.ndarray) -> float:
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


                __all__ = [
                    "PurgedTimeSeriesSplit",
                    "temporal_cv_scores",
                    "optuna_objective_factory",
                    "CVFoldResult",
                ]
