"""Backward‑compatibility shim.

The observation schema utilities were moved to ``trade_agent.envs``. This
module re-exports the public API to avoid breaking older import paths
(``trade_agent.agents.envs.observation_schema``).
"""
from trade_agent.envs.observation_schema import (  # noqa: F401
    ObservationSchema,
    compute_observation_schema,
    save_observation_schema,
)


__all__ = [
    "compute_observation_schema",
    "save_observation_schema",
    "ObservationSchema",
]
