from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from diagnose_managed_portfolios import diagnose_managed_portfolios  # noqa: E402
from diagnose_monthly_panel_qp_window import diagnose_window  # noqa: E402
from export_pca_replay_windows import export_pca_windows  # noqa: E402
from qp_monthly_panel_test_utils import expanded_monthly_panel  # noqa: E402

from cufolio.qp_monthly_panel import (  # noqa: E402
    build_paper_managed_portfolios,
    convert_monthly_panel,
)


def test_diagnostics_run_on_synthetic_managed_portfolios(tmp_path):
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
    diagnostic_dir = tmp_path / "diagnostics"
    summary = diagnose_managed_portfolios(
        managed_dir / "managed_portfolio_returns.parquet",
        diagnostic_dir,
        covariance_lookback=12,
    )
    assert summary["managed_portfolios"] == 15
    assert (diagnostic_dir / "pca_factor_mean_stats.csv").exists()
    assert (diagnostic_dir / "covariance_condition_by_window.csv").exists()

    windows = tmp_path / "windows"
    export_pca_windows(
        managed_dir / "managed_portfolio_returns.parquet",
        windows,
        k_values=[2],
        start_date="2000-12-31",
        end_date="2001-04-30",
        lookback_months=12,
        allow_short_lookback=True,
    )
    window = sorted(windows.glob("*.npz"))[0]
    window_summary = diagnose_window(window, backend="osqp")
    assert window_summary["k"] == 2
    assert window_summary["recovered_managed_weights_sum"] == pytest.approx(1.0, abs=1e-5)
