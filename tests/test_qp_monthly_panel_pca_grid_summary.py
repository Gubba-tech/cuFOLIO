from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_monthly_panel_pca_grid import summarize_grid  # noqa: E402


def test_grid_summary_ranks_and_writes_heatmaps(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    pd.DataFrame(
        {
            "date": ["2020-01-31", "2020-02-29"],
            "osqp:PCA": [0.01, -0.02],
        }
    ).to_csv(artifact / "monthly_returns.csv", index=False)
    rows = [
        {
            "config_id": "k2_l1_1e-6_l2_1e-6",
            "backend": "osqp",
            "k": 2,
            "lambda_l1": 1e-6,
            "lambda_l2": 1e-6,
            "annualized_sharpe": 1.2,
            "Sharpe": 1.2,
            "annualized_return": 0.2,
            "cagr": 0.2,
            "max_drawdown": -0.1,
            "Calmar": 2.0,
            "maximum_constraint_violation": 1e-8,
            "artifact_path": str(artifact),
        },
        {
            "config_id": "k2_l1_1e-3_l2_1e-3",
            "backend": "osqp",
            "k": 2,
            "lambda_l1": 1e-3,
            "lambda_l2": 1e-3,
            "annualized_sharpe": 0.8,
            "Sharpe": 0.8,
            "annualized_return": 0.1,
            "cagr": 0.1,
            "max_drawdown": -0.2,
            "Calmar": 0.5,
            "maximum_constraint_violation": 1e-8,
            "artifact_path": str(artifact),
        },
    ]
    grid_path = tmp_path / "grid_results.csv"
    with grid_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize_grid(grid_path, tmp_path / "summary")

    assert summary["rows"] == 2
    ranked = pd.read_csv(tmp_path / "summary/grid_results_ranked.csv")
    assert ranked.iloc[0]["config_id"] == rows[0]["config_id"]
    assert (tmp_path / "summary/heatmap_data/heatmap_K2.csv").exists()
    assert (tmp_path / "summary/plots/sharpe_heatmap_K2.png").exists()
    assert (tmp_path / "summary/plots/best_config_cumulative_returns.png").exists()
