import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]):
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def test_hydra_rl_ppo_smoke() -> None:
    script = Path('scripts/train_rl_hydra.py')
    assert script.exists(), 'Hydra RL training script missing'
    cmd = [
        sys.executable,
        str(script),
        'algo=ppo',
        'env.data_path=data/features.parquet',
    ]
    result = _run(cmd)
    assert 'ppo' in result.stdout.lower() or result.returncode == 0


def test_hydra_rl_sac_smoke() -> None:
    script = Path('scripts/train_rl_hydra.py')
    assert script.exists(), 'Hydra RL training script missing'
    cmd = [
        sys.executable,
        str(script),
        'algo=sac',
        'env.data_path=data/features.parquet',
    ]
    result = _run(cmd)
    assert 'sac' in result.stdout.lower() or result.returncode == 0
