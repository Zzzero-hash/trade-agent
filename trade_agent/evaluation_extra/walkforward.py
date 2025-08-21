"""Shim module re-exporting new location trade_agent.eval.walkforward."""

from trade_agent.eval.walkforward import *  # type: ignore  # noqa: F401,F403


class ModelProtocol(Protocol):  # type: ignore
    """Protocol defining the interface for trading models.

    Requires implementations of fit, predict, and get_params methods.
    """
    def fit(self, data: pd.DataFrame, **kwargs: Any) -> None: ...
    def predict(self, data: pd.DataFrame) -> pd.Series: ...
    def get_params(self) -> dict[str, Any]: ...


@dataclass
class WalkForwardConfig:
    train_years: int = 1
    val_months: int = 3
    test_months: int = 3
    min_train_samples: int = 252  # 1 year of daily data
    stability_threshold: float = 0.2  # 20% maximum parameter variation
    p_value_threshold: float = 0.05
    metrics: tuple[str, ...] = ('sharpe_ratio', 'cagr', 'max_drawdown')
    stationarity_test: bool = True
    max_workers: int = 4


class WalkForwardValidator:
    """Implements walk-forward validation with overfitting detection."""

    def __init__(self, config: WalkForwardConfig = WalkForwardConfig()) -> None:
        self.config = config
        self.windows = []
        self.metrics_history = []
        self.parameter_stability = {}
        self.stationarity_results = {}

    def create_windows(
        self,
        data: pd.DataFrame
    ) -> list[dict[str, pd.DatetimeIndex]]:
        """Create temporal splits ensuring no data leakage.

        Args:
            data: DataFrame with datetime index
        Returns:
            List of dictionaries with train/val/test date ranges
        """
        dates = cast(pd.DatetimeIndex, data.index.sort_values())
        """Create temporal splits ensuring no data leakage."""
        dates = data.index.sort_values()
        windows = []
        start_idx = 0

        while True:
            current_date = dates[start_idx]
            train_end = current_date + pd.DateOffset(
                years=self.config.train_years)
            val_end = train_end + pd.DateOffset(months=self.config.val_months)
            test_end = val_end + pd.DateOffset(months=self.config.test_months)

            if test_end > dates[-1]:
                break

            # Get indices for each period
            train_mask = (dates >= dates[start_idx]) & (dates < train_end)
            val_mask = (dates >= train_end) & (dates < val_end)
            test_mask = (dates >= val_end) & (dates < test_end)

            if train_mask.sum() < self.config.min_train_samples:  # type: ignore # noqa
                start_idx += 1
                continue

            windows.append({
                'train': dates[train_mask],
                'val': dates[val_mask],
                'test': dates[test_mask]
            })

            # Move window by validation period
            # noinspection PyTypeChecker
            start_idx = int(np.where(dates == val_end)[0][0])  # type: ignore

        return windows

    def validate(  # type: ignore
        self,
        data: pd.DataFrame,
        model: ModelProtocol,
        params: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute walk-forward validation across all windows."""
        self.windows = self.create_windows(data)
        results = []

        for i, window in enumerate(self.windows):
            # Split data
            train_data = data.loc[window['train']]
            val_data = data.loc[window['val']]
            test_data = data.loc[window['test']]

            # Train model
            model.fit(train_data, **params)

            # Validate
            val_preds = model.predict(val_data)
            val_metrics = self._calculate_performance(val_data, val_preds)

            # Test
            test_preds = model.predict(test_data)
            test_metrics = self._calculate_performance(test_data, test_preds)

            # Track parameters
            self._track_parameters(model.get_params(), i)

            results.append({
                'window': i,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'params': model.get_params()
            })

        return self._analyze_results(results)

    def _calculate_performance(
        self,
        data: pd.DataFrame,
        signals: pd.Series
    ) -> dict[str, float]:
        """Run backtest and return metrics."""
        engine = BacktestEngine()
        prices = data['close']  # Assuming OHLCV data
        results = engine.run_backtest(signals, prices)
        return {
            k: v for k, v in results['metrics'].items()
            if k in self.config.metrics
        }

    def _track_parameters(  # type: ignore
        self,
        params: dict[str, Any],
        window_idx: int
    ) -> None:
        """Track parameter values across validation windows."""
        for param, value in params.items():
            if param not in self.parameter_stability:
                self.parameter_stability[param] = []
            self.parameter_stability[param].append(value)
        """Track parameter stability across windows."""
        for param, value in params.items():
            self.parameter_stability.setdefault(param, []).append(value)
        """Track parameter stability across windows."""
        for param, value in params.items():
            if param not in self.parameter_stability:
                self.parameter_stability[param] = []
            self.parameter_stability[param].append(value)

    def _analyze_results(
        self,
        results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Perform statistical analysis of validation results."""
        analysis = {
            'metrics_summary': {},
            'stationarity': {},
            'parameter_stability': {},
            'overfitting_flags': []
        }

        # Metrics analysis
        for metric in self.config.metrics:
            val_values = [r['val_metrics'][metric] for r in results]
            test_values = [r['test_metrics'][metric] for r in results]

            # Performance consistency
            analysis['metrics_summary'][metric] = {
                'val_mean': np.mean(val_values),
                'test_mean': np.mean(test_values),
                'delta': np.mean(val_values) - np.mean(test_values),
                't_test': ttest_ind(val_values, test_values),
                'wilcoxon': wilcoxon(val_values, test_values)
            }

            # Stationarity testing
            if self.config.stationarity_test:
                stat_test = adfuller(val_values)
                analysis['stationarity'][metric] = {
                    'adf': stat_test[0],
                    'p_value': stat_test[1],
                    'stationary': stat_test[1] < self.config.p_value_threshold
                }

        # Parameter stability analysis
        for param, values in self.parameter_stability.items():
            if len(values) < 2:
                continue

            analysis['parameter_stability'][param] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'cv': np.std(values) / np.mean(values),
                'levene': levene(*np.array_split(values, 2))  # type: ignore
            }

        # Overfitting detection
        for r in results:
            flags = []
            for metric in self.config.metrics:
                val_metric = r['val_metrics'][metric]
                test_metric = r['test_metrics'][metric]

                # Performance degradation check
                perf_ratio = (val_metric - test_metric) / val_metric
                if perf_ratio > self.config.stability_threshold:
                    flags.append(f'{metric}_degradation')

                # Statistical significance check
                wilcoxon_result = analysis['metrics_summary'][metric][
                    'wilcoxon']  # type: ignore
                pval = wilcoxon_result.pvalue  # type: ignore
                if pval < self.config.p_value_threshold:
                    flags.append(f'{metric}_wilcoxon')  # type: ignore

            analysis['overfitting_flags'].append(flags)  # type: ignore

        return analysis

    def generate_report(self, analysis: dict[str, Any]) -> str:
        """Generate validation report."""
        report = [
            "# Walk-Forward Validation Report",
            "## Metrics Summary",
            "| Metric | Val Mean | Test Mean | Delta |",
            "| T-Test p-value | Wilcoxon p-value |",
            "|--------|----------|-----------|-------|"
            "----------------|------------------|"
        ]

        for metric, summary in analysis['metrics_summary'].items():
            report.append(
                f"| {metric} | {summary['val_mean']:.4f} | "
                f"{summary['test_mean']:.4f} | "
                f"{summary['delta']:.4f} | {summary['t_test'].pvalue:.4f} | "
                f"{summary['wilcoxon'].pvalue:.4f} |"
            )

        report.extend([
            "\n## Parameter Stability",
            "| Parameter | Mean | CV | Levene p-value |",
            "|-----------|------|----|----------------|"
        ])

        for param, stats in analysis['parameter_stability'].items():
            report.append(
                f"| {param} | {stats['mean']:.4f} | {stats['cv']:.4f} | "
                f"{stats['levene'].pvalue:.4f} |"
            )

        report.append("\n## Overfitting Flags")
        for i, flags in enumerate(analysis['overfitting_flags']):
            flag_text = ', '.join(flags) if flags else 'None'
            report.append(f"Window {i+1}: {flag_text}")

        return '\n'.join(report)
