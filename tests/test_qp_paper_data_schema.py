from pathlib import Path

import pandas as pd

from cufolio.qp_paper_data import (
    build_amp_universe,
    build_managed_portfolios,
    validate_cleaned_data,
    write_validation_report,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/fixtures/paper_data"


def test_cleaned_data_validator_accepts_synthetic_parquet(tmp_path):
    report = validate_cleaned_data(
        monthly_returns=DATA / "monthly_returns.parquet",
        characteristics=DATA / "characteristics_monthly.parquet",
        start_date="2004-01-31",
        end_date="2005-12-31",
        lookback_months=12,
    )
    path = write_validation_report(report, tmp_path / "validation.md")

    assert report["status"] == "pass"
    assert report["characteristic_count"] == 3
    assert report["candidate_first_oos_date"] == "2004-01-31"
    assert path.exists()
    assert (Path("artifacts/paper_replay/data_validation") / "validation.json").exists()


def test_cleaned_data_validator_reports_missing_inputs(tmp_path):
    report = validate_cleaned_data(
        monthly_returns=tmp_path / "missing.parquet",
        characteristics=tmp_path / "missing_chars.parquet",
    )

    assert report["status"] == "blocked"
    assert len(report["issues"]) == 2


def test_amp_universe_and_managed_portfolio_builder_use_synthetic_data(tmp_path):
    universe, summary = build_amp_universe(
        monthly_returns=DATA / "monthly_returns.parquet",
        characteristics=DATA / "characteristics_monthly.parquet",
        start_date="2004-01-31",
        end_date="2005-12-31",
        market_cap_coverage=0.90,
        lookback_months=12,
    )
    assert not universe.empty
    assert summary.shape[0] == 24
    assert universe["selected"].all()

    output = tmp_path / "universe.parquet"
    universe.to_parquet(output, index=False)
    managed, metadata = build_managed_portfolios(
        monthly_returns=DATA / "monthly_returns.parquet",
        characteristics=DATA / "characteristics_monthly.parquet",
        universe=output,
        selected_characteristics="all",
        n_bins=5,
        lag_months=1,
        start_date="2004-01-31",
        end_date="2005-12-31",
    )

    assert not managed.empty
    assert managed["portfolio_id"].nunique() >= 15
    assert metadata["mapping_semantics"].startswith("managed_portfolio_returns")
    assert pd.api.types.is_datetime64_any_dtype(managed["date"])
