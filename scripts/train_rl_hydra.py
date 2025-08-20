#!/usr/bin/env python3
"""Hydra-enabled RL training entrypoint (PPO/SAC) using new rl/ config group.

Examples:
  # Single PPO run
  python scripts/train_rl_hydra.py \
      experiment.name=test_run algo=ppo training.total_timesteps=10000

  # Switch to SAC
  python scripts/train_rl_hydra.py algo=sac

  # Optuna sweep (requires hydra-optuna-sweeper)
  python scripts/train_rl_hydra.py -m hydra/sweeper=optuna rl_optuna \
      ppo.learning_rate=loguniform(1e-5,1e-3) ppo.batch_size=choice(32,64)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


# Ensure src on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trade_agent.agents.rl.train_ppo import PPOTrainer  # type: ignore  # noqa: E402
from trade_agent.agents.rl.train_sac import SACTrainer  # type: ignore  # noqa: E402
from trade_agent.logging.mlflow_utils import (  # noqa: E402
    log_metrics,
    mlflow_run,
)


def _build_config_dict(cfg: DictConfig) -> dict[str, Any]:
    """Flatten Hydra cfg into dict expected by trainers.

    Tolerant to missing algo section; falls back to cfg.algo name if present.
    """
    as_dict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(as_dict, dict)
    # Determine algo section name
    algo_name = as_dict.get('algo')  # type: ignore[assignment]
    for candidate in ['ppo', 'sac']:
        if candidate in as_dict:
            algo_section_key = candidate
            break
    else:
        # No section; fabricate minimal from algo name
        algo_section_key = (
            str(algo_name) if algo_name in ['ppo', 'sac'] else 'ppo'
        )
        as_dict.setdefault(algo_section_key, {})
    return {
        algo_section_key: as_dict.get(algo_section_key, {}),
        'training': as_dict.get('training', {}),
        'mlp_features': as_dict.get('mlp_features', {}),
    }


@hydra.main(version_base=None, config_path="../conf/rl", config_name="config")
def main(cfg: DictConfig) -> None:  # noqa: D401
    # Prepare flat config dict for params logging
    cfg_dict = _build_config_dict(cfg)
    run_name = (
        getattr(cfg, 'experiment', {}).get('name', None)
        if hasattr(cfg, 'experiment')
        else None
    )  # type: ignore[union-attr]
    # Param logging handled inside trainers; omit here to keep types simple
    params_obj = None
    with mlflow_run(
        run_name=run_name,
        params=params_obj,
    ):  # type: ignore[arg-type]
        algo = 'ppo' if 'ppo' in cfg_dict else 'sac'
        if algo == 'ppo':
            trainer = PPOTrainer(config=cfg_dict)
        else:
            trainer = SACTrainer(config=cfg_dict)
        data_path = to_absolute_path(cfg.env.data_path)
        # Determine total timesteps (training override or default)
        total_timesteps = 1000
        if 'training' in cfg_dict:
            total_timesteps = int(
                cfg_dict['training'].get('total_timesteps', total_timesteps)
            )
        else:
            try:  # pragma: no cover
                total_timesteps = int(
                    cfg.training.total_timesteps  # type: ignore[attr-defined]
                )
            except Exception:
                pass
        model = trainer.train(
            data_file=data_path,
            total_timesteps=total_timesteps,
            n_envs=4,
            window_size=cfg.env.window_size,
            initial_capital=cfg.env.initial_capital,
            transaction_cost=cfg.env.transaction_cost,
        )
        # Simple metric logging: final total timesteps and algo
        log_metrics({
            'total_timesteps': float(total_timesteps),
            'algo_flag_ppo': 1.0 if algo == 'ppo' else 0.0,
        })
        # Optionally record parameter count
        try:  # pragma: no cover - best effort
            # Iterate model parameters (policy attr best-effort)
            n_params = sum(
                p.numel()
                # policy attribute exists for SB3 models
                for p in model.policy.parameters()  # type: ignore
            )
            log_metrics({'policy_num_params': float(n_params)})
        except Exception:  # pragma: no cover
            pass


if __name__ == "__main__":  # pragma: no cover
    main()
