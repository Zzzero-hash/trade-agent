"""Reporting utilities for vectorised backtest results.

Adds convenience writer ``write_report`` which materialises a structured
artifact directory:
``reports/<run_id>/{summary.json,trades.csv,report.html?}``.
The run identifier can be user supplied or auto‑generated (UUID4) for
traceability in experiment workflows.
"""
from __future__ import annotations

import json
import uuid
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
            float(equity_curve.iloc[-1]) if not equity_curve.empty else 1.0  # type: ignore[index]
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


def write_report(
    result: Mapping[str, Any],
    run_id: str | None = None,
    base_dir: str | Path = "reports",
    include_html: bool = True,
) -> dict[str, Any]:
    """Persist backtest outputs (summary + trades) to disk.

    Parameters
    ----------
    result : Mapping[str, Any]
        Output from :func:`run_backtest`.
    run_id : str | None, default None
        Identifier for directory name. If omitted a UUID4 is generated.
    base_dir : str | Path, default "reports"
        Root directory for report runs.
    include_html : bool, default True
        Whether to also write an HTML summary file.

    Returns
    -------
    dict[str, Any]
        The JSON summary dictionary enriched with file path references.
    """
    rid = run_id or str(uuid.uuid4())
    run_path = Path(base_dir) / rid
    run_path.mkdir(parents=True, exist_ok=True)

    html_path = (run_path / "report.html") if include_html else None
    summary = generate_report(
        result, html_path=str(html_path) if html_path else None
    )

    # Build trades/activity frame. Columns: price, position, gross_return,
    # trading_cost, net_return, turnover.
    import pandas as pd  # local import to keep module thin when unused

    prices = result.get("prices")
    positions = result.get("positions")
    gross = result.get("gross_returns")
    costs = result.get("trading_cost")
    net = result.get("returns")
    turnover = result.get("turnover")
    if isinstance(prices, pd.Series):
        trades = pd.DataFrame(
            {
                "price": prices,
                "position": (
                    positions if isinstance(positions, pd.Series) else None
                ),
                "gross_return": (
                    gross if isinstance(gross, pd.Series) else None
                ),
                "trading_cost": (
                    costs if isinstance(costs, pd.Series) else None
                ),
                "net_return": net if isinstance(net, pd.Series) else None,
                "turnover": (
                    turnover if isinstance(turnover, pd.Series) else None
                ),
            }
        )
        trades_path = run_path / "trades.csv"
        trades.to_csv(trades_path, index=True)
        summary["trades_path"] = str(trades_path)

    summary_path = run_path / "summary.json"
    summary["run_id"] = rid
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


__all__ = ["generate_report", "write_report"]
