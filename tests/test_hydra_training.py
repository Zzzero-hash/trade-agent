import os
import subprocess
import sys
from pathlib import Path


def test_hydra_ridge_single_run() -> None:
    script = Path('scripts/train_sl_hydra.py')
    assert script.exists(), 'Hydra training script missing'
    # Run a quick ridge training using sample CSV (small dataset)
    cmd = [
        sys.executable,
        str(script),
        'model=ridge',
        'train.data_path=data/sample_data.csv',
        'train.target=close',
    ]
    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    result = subprocess.run(" ".join(cmd), capture_output=True, text=True, env=env, shell=True)
    # Basic sanity checks
    assert 'Resolved Config' in result.stdout
