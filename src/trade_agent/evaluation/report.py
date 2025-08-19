"""Reporting utilities for vectorised backtest results."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd


def generate_report(
    result: Mapping[str, Any],
    html_path: str | None = None,
) -> dict[str, Any]:
    """Generate a JSON style summary (and optional HTML stub).

    Args:
        result: Output dictionary from ``run_backtest``.
        html_path: If provided, write a self‑contained minimal HTML report to
            this path (parent directories are created as needed).

    Returns:
        JSON serialisable dictionary with metrics & scalar info.
    """
    metrics = result.get("metrics", {})
    equity_curve = result.get("equity_curve")
    if isinstance(equity_curve, pd.Series):
        final_equity = (
            float(equity_curve.iloc[-1]) if not equity_curve.empty else 1.0
        )
        num_periods = int(len(equity_curve))
    else:
        final_equity = 1.0
        num_periods = 0
    summary: dict[str, Any] = {
        "final_equity": final_equity,
        "num_periods": num_periods,
        "metrics": metrics,
    }
    if html_path:
        path = Path(html_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = _build_html(summary)
        path.write_text(html, encoding="utf-8")
        summary["html_report_path"] = str(path)
    return summary


def _build_html(summary: Mapping[str, Any]) -> str:
    metrics = summary.get("metrics", {})
    rows = "".join(
        f"<tr><td>{k}</td><td>{v:.6f}</td></tr>"
        for k, v in sorted(metrics.items())
        if isinstance(v, int | float)
    )
    body = f"""
<!DOCTYPE html>
<html lang='en'>
<head><meta charset='utf-8'><title>Backtest Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 1.5rem; }}
table {{ border-collapse: collapse; }}
td, th {{ border: 1px solid #ccc; padding: 4px 8px; text-align: right; }}
th {{ background: #f5f5f5; }}
</style></head>
<body>
<h1>Backtest Report</h1>
<p>Final Equity: {summary.get('final_equity', 1.0):.4f}</p>
<p>Periods: {summary.get('num_periods', 0)}</p>
<table>
<thead><tr><th>Metric</th><th>Value</th></tr></thead>
<tbody>{rows}</tbody>
</table>
<pre style='margin-top:1rem;'>Raw JSON:\n{{json}}</pre>
</body>
</html>
"""
    return body.replace("{json}", json.dumps(summary, indent=2))


__all__ = ["generate_report"]
