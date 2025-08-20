"""Unified random seed utility.

Provides a single ``set_seed`` helper covering:
 - Python's ``random`` module
 - NumPy
 - PyTorch (CPU + CUDA, with optional deterministic algorithms)
 - Gymnasium / Gym style environments (single or vector)

Design goals:
 1. Determinism-by-default while allowing opt-out.
 2. Idempotent and safe to call multiple times.
 3. Minimal external dependencies (only uses already present packages).

Notes on determinism:
Full determinism in PyTorch may reduce performance and still cannot be
guaranteed for all operations (some kernels have no deterministic
implementation). We enable the most common deterministic guards but do
not raise if unavailable.
"""
from __future__ import annotations

import os
import random
from collections.abc import Iterable
from typing import Any

import numpy as np


try:  # Optional torch dependency (fallback if running w/o torch context)
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

# Ensure cuBLAS workspace config is present as early as possible so any
# downstream call to torch.use_deterministic_algorithms(True) won't error.
if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:  # pragma: no cover - env setup
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

try:  # Gymnasium preferred
    import gymnasium as gym  # type: ignore
except Exception:  # pragma: no cover
    try:
        import gym  # type: ignore  # noqa: F401
        gym = gym  # type: ignore
    except Exception:  # pragma: no cover
        gym = None  # type: ignore

__all__ = ["set_seed"]


def _seed_env(env: Any, seed: int) -> None:
    """Attempt to seed a (vector) environment in a version-agnostic way.

    Supports:
     - gymnasium / gym single env: ``env.reset(seed=seed)``
     - Stable Baselines3 VecEnv: ``env.seed(seed)`` (older gym) or reset
     - Action/observation spaces, if they expose ``seed``.
    """
    if env is None:
        return

    # VecEnv (SB3) exposes .envs list
    if (
        hasattr(env, "envs")
        and isinstance(env.envs, Iterable)
    ):  # pragma: no cover - runtime shape
        for i, e in enumerate(env.envs):
            try:
                if hasattr(e, "reset"):
                    e.reset(seed=seed + i)
                elif hasattr(e, "seed"):
                    e.seed(seed + i)
            except Exception:
                pass
    else:  # Single env
        try:
            if hasattr(env, "reset"):
                env.reset(seed=seed)
            elif hasattr(env, "seed"):
                env.seed(seed)
        except Exception:
            pass

    # Seed spaces if possible
    for space_name in ("action_space", "observation_space"):
        space = getattr(env, space_name, None)
        if space is not None and hasattr(space, "seed"):
            try:
                space.seed(seed)
            except Exception:  # pragma: no cover - non critical
                pass


def set_seed(
    seed: int | None = 42,
    *,
    env: Any | None = None,
    deterministic: bool = True,
    torch_deterministic_algorithms: bool = True,
    disable_cudnn_benchmark: bool = True,
    verbose: bool = False,
) -> int:
    """Set random seeds across supported libraries & optionally an env.

    Args:
        seed: Desired base seed. ``None`` -> no action, returns 0.
        env: Optional gym/gymnasium environment (single or vector) to seed.
        deterministic: If True, set flags for deterministic execution.
        torch_deterministic_algorithms: Call
            ``torch.use_deterministic_algorithms(True)`` when available.
        disable_cudnn_benchmark: Force ``cudnn.benchmark = False`` when
            deterministic to avoid nondeterministic autotuning.
        verbose: Emit a short diagnostic to stdout.

    Returns:
        The integer seed actually used (0 if no seeding performed).
    """
    if seed is None:
        if verbose:
            pass
        return 0

    # Normalise seed into Python int range
    seed_int = int(seed) & 0xFFFFFFFF

    # Python / NumPy
    os.environ["PYTHONHASHSEED"] = str(seed_int)
    random.seed(seed_int)
    np.random.seed(seed_int)

    # CUDA library workspace determinism (must be set BEFORE enabling
    # deterministic algorithms to avoid runtime errors in some ops).
    if deterministic and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        # Try smaller workspace first; will fall back to larger if needed.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    # PyTorch
    if torch is not None:  # pragma: no branch - simple guard
        try:  # Seed RNGs first
            torch.manual_seed(seed_int)
            if torch.cuda.is_available():  # type: ignore[attr-defined]
                torch.cuda.manual_seed(seed_int)  # type: ignore[attr-defined]
                torch.cuda.manual_seed_all(  # type: ignore[attr-defined]
                    seed_int
                )
        except Exception:  # pragma: no cover - defensive
            pass
        if deterministic:
            try:  # cuDNN flags
                torch.backends.cudnn.deterministic = True  # type: ignore
                if disable_cudnn_benchmark:
                    torch.backends.cudnn.benchmark = False  # type: ignore
            except Exception:  # pragma: no cover
                pass
            if torch_deterministic_algorithms:
                try:
                    torch.use_deterministic_algorithms(True)  # type: ignore
                except RuntimeError as e:  # workspace config issues
                    # If failure mentions CUBLAS workspace, try alternate size.
                    if "CUBLAS_WORKSPACE_CONFIG" in str(e):
                        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                        try:
                            torch.use_deterministic_algorithms(True)  # retry
                        except Exception:  # pragma: no cover
                            pass
                except Exception:  # pragma: no cover
                    pass

    # Seed environment if provided
    _seed_env(env, seed_int)

    if verbose:  # pragma: no cover - logging side effect
        libs = ["python", "numpy", "torch" if torch else "(no torch)"]
        if env is not None:
            libs.append("env")

    return seed_int
