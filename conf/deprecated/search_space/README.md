# Search Space Examples

This directory contains modular Hydra include files defining Optuna search spaces. Use them by adding `+search_space=<name>` when launching a multirun with the Optuna sweeper.

Example (ridge):

```
python scripts/train_sl_hydra.py -m hydra/sweeper=optuna \
  model=ridge +search_space=ridge_alpha optuna.n_trials=25 \
  optimization.metric=cv_mse
```

Supervised models now support temporal cross‑validation based objectives with pruning. Set `optimization.metric=cv_mse` (or `mse`) to optimize negative validation MSE aggregated over folds. Pruning is triggered using fold‑level intermediate reports (`trial.report`).

Available search space includes:

- ridge_alpha.yaml – Ridge `alpha` and `random_state`.
- mlp_basic.yaml – Core MLP depth/regularization/learning rate.
- cnn_lstm_seq.yaml – CNN channel/kernel, LSTM size/layers, dropout, seq length.
- ppo_core.yaml – PPO learning rate, steps, batch size, gamma, clip, entropy.
- sac_core.yaml – SAC learning rate, batch size, gamma, tau, entropy coef.

## Key Notes

1. Temporal CV vs Train MSE: The objective has been switched from single train MSE to fold‑based temporal CV metrics to enable effective pruning.
2. Pruning: Median pruner evaluates intermediate mean score after each fold.
3. Metric Choices: `sharpe`, `mse` / `cv_mse`, `mae`.
4. Maximization: All objectives are cast as maximization (loss metrics negated).

## Customizing

Create a new file (e.g. `conf/search_space/custom.yaml`) with:

```
hydra:
  sweeper:
    search_space:
      model.param_name:
        type: float
        low: 0.001
        high: 0.1
        log: true
```

Then launch with `+search_space=custom`.
