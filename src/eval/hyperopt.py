from typing import Any, Optional

import numpy as np
import optuna
import pandas as pd
from optuna import Study, Trial
from scipy.stats import spearmanr

from src.eval.walkforward import WalkForwardConfig, WalkForwardValidator


class RobustHyperparameterOptimizer(Study):
    """Optuna study with integrated walk-forward validation and robustness checks."""

    def __init__(
        self,
        data: pd.DataFrame,
        model: Any,
        base_config: WalkForwardConfig = WalkForwardConfig(),
        study_name: str = "robust_optimization",
        n_trials: int = 100,
        timeout: Optional[float] = None,
        early_stopping_patience: int = 10
    ):
        self.data = data
        self.model = model
        self.base_config = base_config
        self.study: optuna.Study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(n_startup_trials=20)
        )
        self.n_trials = n_trials
        self.timeout = timeout
        self.early_stopping_patience = early_stopping_patience
        self.best_params: dict[str, Any] = {}
        self.volatility_history: list[float] = self._calculate_historical_volatility()

    def _calculate_historical_volatility(self, window: int = 21) -> list[float]:
        """Calculate rolling volatility for dynamic parameter bounds."""
        returns = np.log(self.data['close']).diff().dropna()
        return returns.rolling(window).std().dropna().tolist()

    def _dynamic_bounds(self, trial: Trial, param_name: str) -> tuple:
        """Generate dynamic parameter bounds based on market volatility."""
        current_vol = self.volatility_history[-1] if self.volatility_history else 0.1
        base_ranges = {
            'alpha': (0.5, 3.0),
            'beta': (0.1, 2.0)
        }

        # Scale ranges inversely with volatility
        vol_factor = 1 / (1 + current_vol)
        scaled_min = base_ranges[param_name][0] * vol_factor
        scaled_max = base_ranges[param_name][1] * vol_factor

        return (scaled_min, scaled_max)

    def _calculate_reward(
        self,
        val_metrics: dict[str, float],
        test_metrics: dict[str, float],
        analysis: dict[str, Any]
    ) -> float:
        """Simplified reward function with robustness penalties."""
        base_reward = val_metrics['sharpe_ratio'] * 0.6 + test_metrics['sharpe_ratio'] * 0.4

        # Penalty terms
        param_stability_penalty = sum(
            v['cv'] for v in analysis['parameter_stability'].values()
        ) * 0.2

        overfit_penalty = len(analysis['overfitting_flags']) * 0.1
        vol_penalty = np.std(self.volatility_history[-10:]) * 0.05

        return base_reward - param_stability_penalty - overfit_penalty - vol_penalty

    def _objective(self, trial: Trial) -> float:
        """Optuna objective function with integrated validation."""
        params = {
            'alpha': trial.suggest_float(
                'alpha', *self._dynamic_bounds(trial, 'alpha')
            ),
            'beta': trial.suggest_float(
                'beta', *self._dynamic_bounds(trial, 'beta')
            )
        }

        validator = WalkForwardValidator(self.base_config)
        results = validator.validate(self.data, self.model, params)

        # Store trial metadata
        trial.set_user_attr("param_stability", results['parameter_stability'])
        trial.set_user_attr("overfitting_flags", results['overfitting_flags'])
        trial.set_user_attr("stationarity", results['stationarity'])

        # Calculate composite reward
        val_metric = np.mean([r['val_metrics']['sharpe_ratio'] for r in results['results']])
        test_metric = np.mean([r['test_metrics']['sharpe_ratio'] for r in results['results']])
        reward = self._calculate_reward(
            {'sharpe_ratio': val_metric},
            {'sharpe_ratio': test_metric},
            results['analysis']
        )

        # Early stopping check
        if trial.number > self.early_stopping_patience:
            recent_rewards = [
                t.value for t in trial.study.trials[-self.early_stopping_patience:]
                if t.value is not None
            ]
            if np.std(recent_rewards) < 0.01:
                trial.study.stop()

        return reward

    def optimize(self) -> dict[str, Any]:
        """Execute optimization with robustness checks."""
        self.study.optimize(
            self._objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            callbacks=[self._stop_check]
        )

        # Store best parameters with metadata
        self.best_params = {
            **self.study.best_params,
            'metadata': {
                'validation_report': self.generate_report(),
                'robustness_metrics': self.study.best_trial.user_attrs
            }
        }
        return self.best_params

    def _stop_check(self, study: Study, trial: Trial) -> None:
        """Custom early stopping callback."""
        if len(study.trials) >= self.early_stopping_patience:
            recent_trials = study.trials[-self.early_stopping_patience:]
            if not any(t.value > study.best_value for t in recent_trials):
                study.stop()

    def generate_report(self) -> str:
        """Generate comprehensive optimization report."""
        report = [
            "# Robust Hyperparameter Optimization Report",
            f"## Best Parameters\n{self.study.best_params}",
            "## Robustness Metrics",
            "### Parameter Stability"
        ]

        stability = self.study.best_trial.user_attrs.get('param_stability', {})
        for param, metrics in stability.items():
            report.append(
                f"- {param}: CV={metrics['cv']:.3f}, "
                f"Levene p={metrics['levene'].pvalue:.4f}"
            )

        report.append("\n## Validation-Test Correlation")
        val_scores = [
            t.user_attrs.get('val_sharpe', 0) for t in self.study.trials
        ]
        test_scores = [
            t.user_attrs.get('test_sharpe', 0) for t in self.study.trials
        ]
        corr = spearmanr(val_scores, test_scores)[0]
        report.append(f"Spearman Correlation: {corr:.3f}")

        return '\n'.join(report)

    def save_study(self, filepath: str) -> None:
        """Save complete study state."""
        optuna.save_study(self.study, filepath)

    @classmethod
    def load_study(
        cls,
        filepath: str,
        data: pd.DataFrame,
        model: Any
    ) -> 'RobustHyperparameterOptimizer':
        """Load existing study with data and model."""
        study = optuna.load_study(
            study_name=study_name,
            storage=optuna.storages.RDBStorage(
                f"sqlite:///{storage_path}",  # Now matches parameter name
                engine_kwargs={"pool_pre_ping": True}
            )
        )
        optimizer = cls(data, model)
        optimizer.study = study
        return optimizer
