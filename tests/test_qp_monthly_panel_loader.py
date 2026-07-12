from __future__ import annotations

import pandas as pd
from qp_monthly_panel_test_utils import expanded_monthly_panel

from cufolio.qp_monthly_panel import (
    PAPER_CHARACTERISTICS,
    convert_monthly_panel,
    validate_monthly_panel,
    write_monthly_panel_validation,
)


def test_monthly_panel_validator_reports_schema_and_lookback(tmp_path):
    input_path = expanded_monthly_panel(tmp_path, months=25, assets=12)
    report = validate_monthly_panel(input_path, format="tsv", sep="tab")

    assert report["status"] == "pass"
    assert report["min_date"] == "2000-01-31"
    assert report["number_of_months"] == 25
    assert report["average_assets_per_month"] == 12.0
    assert report["available_characteristics"]
    assert report["missing_characteristics"] == []
    assert report["earliest_feasible_20_year_lookback_oos_date"] == "2020-01-31"
    assert report["2005_01_31_oos_possible_with_240_months"] is False
    assert report["2020_01_31_oos_possible_with_240_months"] is True
    assert report["warnings"]

    summary = write_monthly_panel_validation(report, tmp_path / "validation")
    assert summary.exists()
    assert (tmp_path / "validation/monthly_coverage.csv").exists()
    assert (tmp_path / "validation/characteristic_coverage.csv").exists()


def test_monthly_panel_converter_writes_clean_parquet_and_metadata(tmp_path):
    input_path = expanded_monthly_panel(tmp_path, months=4, assets=6)
    output_path = tmp_path / "processed/monthly_characteristic_panel.parquet"
    cleaned, metadata = convert_monthly_panel(
        input_path,
        output_path,
        format="tsv",
        sep="tab",
        characteristics="all",
        drop_missing_ret=True,
        winsorize_characteristics=(0.01, 0.99),
        standardize_characteristics="cross_sectional",
    )

    loaded = pd.read_parquet(output_path)
    assert len(cleaned) == len(loaded)
    assert {"date", "asset_id", "cusip", "price", "ret", "market_cap"}.issubset(
        loaded.columns
    )
    assert "permco" in loaded.columns
    assert set(PAPER_CHARACTERISTICS).issubset(loaded.columns)
    assert metadata["source_data_committed"] is False
    assert output_path.with_suffix(".metadata.json").exists()
