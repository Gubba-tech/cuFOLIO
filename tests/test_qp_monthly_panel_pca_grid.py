from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from qp_monthly_panel_test_utils import expanded_monthly_panel  # noqa: E402
from run_monthly_panel_pca_paper_grid import build_parser, run_paper_grid  # noqa: E402

from cufolio.qp_monthly_panel import (  # noqa: E402
    build_paper_managed_portfolios,
    convert_monthly_panel,
)


def test_paper_grid_runner_writes_results_on_synthetic_data(tmp_path):
    raw = expanded_monthly_panel(tmp_path, months=16, assets=12)
    processed = tmp_path / "processed/monthly_panel.parquet"
    convert_monthly_panel(raw, processed, format="tsv", sep="tab")
    managed_dir = tmp_path / "managed"
    build_paper_managed_portfolios(
        processed,
        managed_dir,
        characteristics="beta,a2me,beme,lme,r12_2",
        n_bins=3,
        weighting="value",
        min_assets_per_bin=2,
    )
    args = build_parser().parse_args(
        [
            "--managed-portfolio-returns",
            str(managed_dir / "managed_portfolio_returns.parquet"),
            "--output-dir",
            str(tmp_path / "grid"),
            "--start-date",
            "2000-12-31",
            "--end-date",
            "2001-04-30",
            "--lookback-months",
            "12",
            "--k-values",
            "2",
            "--lambda-l1-grid",
            "1e-6",
            "--lambda-l2-grid",
            "1e-3",
            "--backend",
            "osqp",
            "--allow-short-lookback",
            "--max-combinations",
            "1",
        ]
    )
    rows = run_paper_grid(args)

    assert len(rows) == 1
    assert rows[0]["backend"] == "osqp"
    assert rows[0]["optimal_count"] == 4
    for filename in ("grid_results.csv", "grid_summary.md", "last_completed.json"):
        assert (tmp_path / "grid" / filename).exists()


def test_paper_grid_resume_does_not_duplicate_completed_config(tmp_path):
    raw = expanded_monthly_panel(tmp_path, months=16, assets=12)
    processed = tmp_path / "processed/monthly_panel.parquet"
    convert_monthly_panel(raw, processed, format="tsv", sep="tab")
    managed_dir = tmp_path / "managed"
    build_paper_managed_portfolios(
        processed,
        managed_dir,
        characteristics="beta,a2me,beme,lme,r12_2",
        n_bins=3,
        weighting="value",
        min_assets_per_bin=2,
    )
    base = [
        "--managed-portfolio-returns",
        str(managed_dir / "managed_portfolio_returns.parquet"),
        "--output-dir",
        str(tmp_path / "grid"),
        "--start-date",
        "2000-12-31",
        "--end-date",
        "2001-04-30",
        "--lookback-months",
        "12",
        "--k-values",
        "2",
        "--lambda-l1-grid",
        "1e-6",
        "--lambda-l2-grid",
        "1e-3",
        "--backend",
        "osqp",
        "--allow-short-lookback",
    ]
    run_paper_grid(build_parser().parse_args(base))
    resumed = build_parser().parse_args(base + ["--resume"])
    rows = run_paper_grid(resumed)
    assert len(rows) == 1
    assert len(pd.read_csv(tmp_path / "grid/grid_results.csv")) == 1
