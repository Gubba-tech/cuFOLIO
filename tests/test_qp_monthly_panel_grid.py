from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from qp_monthly_panel_test_utils import expanded_monthly_panel
from run_monthly_panel_pca_grid import run_grid

from cufolio.qp_monthly_panel import (
    build_paper_managed_portfolios,
    convert_monthly_panel,
)


def test_monthly_panel_grid_runs_on_synthetic_data(tmp_path):
    raw_path = expanded_monthly_panel(tmp_path, months=16, assets=12)
    cleaned_path = tmp_path / "processed/monthly_panel.parquet"
    convert_monthly_panel(raw_path, cleaned_path, format="tsv", sep="tab")
    managed_dir = tmp_path / "managed"
    build_paper_managed_portfolios(
        cleaned_path,
        managed_dir,
        characteristics="beta,a2me,beme,lme,r12_2",
        n_bins=3,
        weighting="value",
        min_assets_per_bin=2,
    )
    output_dir = tmp_path / "grid"
    rows = run_grid(
        managed_dir / "managed_portfolio_returns.parquet",
        output_dir,
        start_date="2000-12-31",
        end_date="2001-04-30",
        lookback_months=12,
        k_values=[2],
        lambda_l1_values=[0.0, 1e-6],
        lambda_l2_values=[0.0],
        backend="osqp",
        max_windows=2,
    )

    assert rows
    assert all(row["backend"] == "osqp" for row in rows)
    assert all(row["optimal_count"] == 2 for row in rows)
    for filename in (
        "grid_results.csv",
        "grid_summary.md",
        "heatmap_data.csv",
        "k_sensitivity_sharpe.png",
        "lambda_grid_sharpe.png",
        "lambda_grid_sharpe_osqp.png",
    ):
        assert (output_dir / filename).exists()
    table = pd.read_csv(output_dir / "grid_results.csv")
    assert {"k", "lambda_l1", "lambda_l2", "Sharpe"}.issubset(table.columns)
