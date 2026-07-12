from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from summarize_monthly_panel_results import summarize_run


def _write_synthetic_results(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "backend": "osqp",
            "model_name": "PCA-monthly-panel",
            "rebalance_date": "2020-01-31",
            "window_id": "w1",
            "status": "optimal",
            "realized_next_return": "0.02",
            "gross_long": "1.2",
            "gross_short": "0.2",
            "max_constraint_violation": "1e-8",
            "box_violation": "0",
            "short_budget_violation": "0",
            "sum_weights": "1",
            "turnover": "0.1",
        },
        {
            "backend": "osqp",
            "model_name": "PCA-monthly-panel",
            "rebalance_date": "2020-02-29",
            "window_id": "w2",
            "status": "optimal",
            "realized_next_return": "-0.01",
            "gross_long": "1.2",
            "gross_short": "0.2",
            "max_constraint_violation": "2e-8",
            "box_violation": "0",
            "short_budget_violation": "0",
            "sum_weights": "1",
            "turnover": "0.2",
        },
        {
            "backend": "cuopt",
            "model_name": "PCA-monthly-panel",
            "rebalance_date": "2020-01-31",
            "window_id": "w1",
            "status": "skipped",
            "skip_reason": "cuOpt unavailable",
        },
    ]
    fields = sorted({field for row in rows for field in row})
    with (path / "per_window_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_empirical_summarizer_writes_table2_outputs_and_plots(tmp_path):
    _write_synthetic_results(tmp_path)
    metrics = summarize_run(tmp_path, plots=True)

    required = {
        "start_period",
        "end_period",
        "time_in_market",
        "cagr",
        "annualized_volatility",
        "Sharpe",
        "max_drawdown",
        "Calmar",
        "best_month",
        "worst_month",
        "best_year",
        "worst_year",
        "average_gross_long",
        "average_gross_short",
        "maximum_constraint_violation",
        "optimal_count",
        "skipped_count",
        "failed_count",
    }
    assert required.issubset(metrics[0])
    for filename in (
        "metrics_table.csv",
        "metrics_table.md",
        "cumulative_returns.csv",
        "monthly_returns.csv",
        "constraint_diagnostics.csv",
        "weights_summary.csv",
    ):
        assert (tmp_path / filename).exists()
    for filename in (
        "cumulative_returns.png",
        "underwater.png",
        "monthly_returns_heatmap.png",
        "constraint_violation.png",
        "gross_exposure.png",
    ):
        assert (tmp_path / filename).exists()
