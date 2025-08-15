#!/usr/bin/env python3
"""THIS SCRIPT HAS BEEN PERMANENTLY REMOVED - Use Hydra multirun instead.

Migration Example:
python scripts/train_sl_hydra.py -m model=ridge,linear,garch,mlp,\\
    cnn_lstm,transformer train.data_path=... train.target=...

This file will be removed in the next release.
"""
import sys

sys.exit(
    "ERROR: Legacy script removed. Use Hydra multirun for model sweeps."
)
