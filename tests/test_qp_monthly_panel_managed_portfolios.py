from __future__ import annotations

import pandas as pd
from qp_monthly_panel_test_utils import expanded_monthly_panel

from cufolio.qp_monthly_panel import (
    build_paper_managed_portfolios,
    convert_monthly_panel,
)


def test_managed_portfolio_builder_sorts_and_lags_returns(tmp_path):
    raw_path = expanded_monthly_panel(tmp_path, months=16, assets=12)
    cleaned_path = tmp_path / "processed/monthly_panel.parquet"
    convert_monthly_panel(raw_path, cleaned_path, format="tsv", sep="tab")
    output_dir = tmp_path / "managed"

    returns, membership, metadata = build_paper_managed_portfolios(
        cleaned_path,
        output_dir,
        characteristics="beta,a2me,beme,lme,r12_2",
        n_bins=3,
        weighting="value",
        min_assets_per_bin=2,
        start_date="2000-03-31",
        end_date="2001-04-30",
        assume_characteristics_lagged=True,
    )

    assert not returns.empty
    assert not membership.empty
    assert metadata["managed_portfolios_created"] == 15
    assert set(returns["portfolio_id"]) == {
        f"{name}_bin{bin_number:02d}"
        for name in ("beta", "a2me", "beme", "lme", "r12_2")
        for bin_number in range(1, 4)
    }
    assert (returns["date"] > returns["sort_date"]).all()
    weights = membership.groupby(["date", "characteristic_name", "bin"])["weight"].sum()
    assert (weights > 0.999999).all()
    assert (weights < 1.000001).all()
    assert (output_dir / "managed_portfolio_returns.parquet").exists()
    assert (output_dir / "managed_portfolio_membership.parquet").exists()
    assert pd.api.types.is_datetime64_any_dtype(returns["date"])
