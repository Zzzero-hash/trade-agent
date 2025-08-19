import numpy as np
import pandas as pd

from trade_agent.evaluation.backtest import run_backtest
from trade_agent.evaluation.report import write_report


def test_write_report_creates_files(tmp_path) -> None:
    idx = pd.date_range('2024-02-01', periods=10, freq='D')
    prices = pd.Series(np.linspace(100, 105, len(idx)), index=idx)
    signals = pd.Series(1.0, index=idx)
    result = run_backtest(prices, signals, fee_rate=0.0002)
    summary = write_report(result, run_id='unit_test_run', base_dir=tmp_path)
    run_dir = tmp_path / 'unit_test_run'
    assert (run_dir / 'summary.json').exists()
    assert (run_dir / 'trades.csv').exists()
    if 'html_report_path' in summary:
        assert (run_dir / 'report.html').exists()
    assert 'metrics' in summary
