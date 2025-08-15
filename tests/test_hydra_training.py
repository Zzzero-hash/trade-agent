import subprocess
import sys
from pathlib import Path


def test_hydra_ridge_single_run():
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
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # Basic sanity checks
    assert 'Resolved Config' in result.stdout
    assert 'Results' in result.stdout
    assert 'train_mse' in result.stdout
