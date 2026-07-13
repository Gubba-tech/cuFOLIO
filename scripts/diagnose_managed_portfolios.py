#!/usr/bin/env python3
"""Audit managed-portfolio returns, missingness, covariance, and PCA inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cufolio.qp_factor_workflows import build_pca_factor_qp_data


def _load_returns(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.to_period("M").dt.to_timestamp("M")
    if {"portfolio_id", "ret"}.issubset(frame.columns):
        frame = frame.pivot(index="date", columns="portfolio_id", values="ret")
    else:
        frame = frame.set_index("date").select_dtypes(include=[np.number])
    return frame.sort_index().sort_index(axis=1)


def _condition_rows(returns: pd.DataFrame, lookback: int) -> pd.DataFrame:
    rows = []
    for end_index in range(lookback - 1, len(returns)):
        history = returns.iloc[end_index - lookback + 1 : end_index + 1]
        complete = history.dropna(axis=1, how="any")
        if complete.shape[1] < 2:
            rows.append(
                {
                    "window_end": returns.index[end_index].date(),
                    "lookback_months": lookback,
                    "n_complete_portfolios": complete.shape[1],
                    "condition_number": np.nan,
                }
            )
            continue
        covariance = np.cov(complete.to_numpy(dtype=float), rowvar=False)
        eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
        positive = eigenvalues[eigenvalues > 1e-14]
        condition = float(positive.max() / positive.min()) if positive.size else np.inf
        rows.append(
            {
                "window_end": returns.index[end_index].date(),
                "lookback_months": lookback,
                "n_complete_portfolios": complete.shape[1],
                "condition_number": condition,
                "minimum_eigenvalue": float(eigenvalues.min()),
                "maximum_eigenvalue": float(eigenvalues.max()),
            }
        )
    return pd.DataFrame(rows)


def _pca_explained_variance(returns: pd.DataFrame) -> pd.DataFrame:
    complete = returns.dropna(axis=1, how="any").dropna(axis=0, how="any")
    matrix = complete.to_numpy(dtype=float)
    centered = matrix - matrix.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False, full_matrices=False)
    variance = singular_values**2
    ratios = variance / variance.sum() if variance.sum() > 0 else variance
    rows = []
    for k in range(2, min(10, len(ratios)) + 1):
        rows.append(
            {
                "k": k,
                "n_observations": matrix.shape[0],
                "n_portfolios": matrix.shape[1],
                "explained_variance_k": float(ratios[k - 1]),
                "explained_variance_cumulative": float(ratios[:k].sum()),
            }
        )
    return pd.DataFrame(rows)


def _pca_factor_mean_stats(returns: pd.DataFrame) -> pd.DataFrame:
    complete = returns.dropna(axis=1, how="any").dropna(axis=0, how="any")
    matrix = complete.to_numpy(dtype=float)
    rows = []
    for k in range(2, min(6, min(matrix.shape)) + 1):
        data = build_pca_factor_qp_data(matrix, n_components=k, center=True)
        for index, value in enumerate(data.factor_mean):
            rows.append(
                {
                    "k": k,
                    "factor": index,
                    "factor_mean": float(value),
                    "factor_volatility": float(np.std(data.factor_returns[:, index], ddof=1)),
                    "factor_sharpe_monthly": float(
                        value / np.std(data.factor_returns[:, index], ddof=1)
                    ),
                }
            )
    return pd.DataFrame(rows)


def diagnose_managed_portfolios(
    managed_portfolio_returns: str | Path,
    output_dir: str | Path,
    membership_file: str | Path | None = None,
    covariance_lookback: int = 240,
    min_assets_per_bin: int = 5,
) -> dict[str, object]:
    returns = _load_returns(Path(managed_portfolio_returns))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    missing = returns.isna().sum()
    stats = pd.DataFrame(
        {
            "portfolio_id": returns.columns.astype(str),
            "observations": returns.notna().sum().to_numpy(),
            "missing_count": missing.to_numpy(),
            "missing_rate": returns.isna().mean().to_numpy(),
            "mean_monthly_return": returns.mean().to_numpy(),
            "volatility_monthly": returns.std(ddof=1).to_numpy(),
            "annualized_volatility": (returns.std(ddof=1) * np.sqrt(12)).to_numpy(),
            "monthly_sharpe": (returns.mean() / returns.std(ddof=1)).to_numpy(),
            "annualized_sharpe": (returns.mean() / returns.std(ddof=1) * np.sqrt(12)).to_numpy(),
            "minimum_return": returns.min().to_numpy(),
            "maximum_return": returns.max().to_numpy(),
        }
    )
    stats.to_csv(output / "managed_portfolio_return_stats.csv", index=False)
    stats[["portfolio_id", "missing_count", "missing_rate"]].to_csv(
        output / "missingness_by_portfolio.csv", index=False
    )

    stacked = returns.stack(future_stack=True).rename("ret").reset_index()
    stacked = stacked.dropna(subset=["ret"])
    stacked.columns = ["date", "portfolio_id", "ret"]
    extremes = pd.concat(
        [
            stacked.nsmallest(20, "ret").assign(side="lowest"),
            stacked.nlargest(20, "ret").assign(side="highest"),
        ],
        ignore_index=True,
    )
    extremes.to_csv(output / "portfolio_extreme_returns.csv", index=False)
    _condition_rows(returns, covariance_lookback).to_csv(
        output / "covariance_condition_by_window.csv", index=False
    )
    _pca_explained_variance(returns).to_csv(output / "pca_explained_variance.csv", index=False)
    _pca_factor_mean_stats(returns).to_csv(output / "pca_factor_mean_stats.csv", index=False)

    membership_path = Path(membership_file) if membership_file else Path(managed_portfolio_returns).parent / "managed_portfolio_membership.parquet"
    coverage = pd.DataFrame()
    concentration = pd.DataFrame()
    if membership_path.exists():
        membership = pd.read_parquet(membership_path)
        grouped = membership.groupby(["date", "portfolio_id"], as_index=False).agg(
            n_assets=("asset_id", "nunique"),
            max_weight=("weight", "max"),
            weight_sum=("weight", "sum"),
            sum_weight_squared=("weight", lambda value: float(np.square(value).sum())),
        )
        grouped["effective_n"] = 1.0 / grouped["sum_weight_squared"].replace(0, np.nan)
        coverage = grouped
        coverage.to_csv(output / "portfolio_coverage.csv", index=False)
        concentration = grouped.groupby("portfolio_id", as_index=False).agg(
            mean_max_weight=("max_weight", "mean"),
            maximum_max_weight=("max_weight", "max"),
            mean_effective_n=("effective_n", "mean"),
            minimum_assets=("n_assets", "min"),
        )
        concentration.to_csv(output / "weight_concentration_by_portfolio.csv", index=False)

    summary = {
        "managed_portfolio_returns": str(managed_portfolio_returns),
        "months_available": int(len(returns)),
        "first_month": str(returns.index.min().date()),
        "last_month": str(returns.index.max().date()),
        "managed_portfolios": int(returns.shape[1]),
        "missing_rate_overall": float(returns.isna().mean().mean()),
        "minimum_assets_per_bin_threshold": min_assets_per_bin,
        "membership_file": str(membership_path) if membership_path.exists() else None,
        "portfolio_groups_below_min_assets": int(
            (coverage["n_assets"] < min_assets_per_bin).sum()
        )
        if not coverage.empty
        else None,
        "covariance_lookback_months": covariance_lookback,
        "pca_mean_construction": "raw managed returns projected onto PCA V; V estimated from demeaned returns",
    }
    (output / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Managed-Portfolio Diagnostics",
        "",
        f"- Managed portfolios: **{summary['managed_portfolios']}**",
        f"- Months: **{summary['months_available']}** ({summary['first_month']} to {summary['last_month']})",
        f"- Overall missing return rate: **{summary['missing_rate_overall']:.6g}**",
        f"- Covariance lookback: **{covariance_lookback} months**",
        f"- Membership file: `{summary['membership_file']}`",
        f"- Portfolio groups below {min_assets_per_bin} assets: **{summary['portfolio_groups_below_min_assets']}**",
        "",
        "PCA directions are estimated from demeaned managed returns, while factor means and covariance are computed from raw returns projected by V.",
        "",
        "The companion CSV files contain per-portfolio return statistics, missingness, extreme returns, rolling covariance condition numbers, PCA explained variance, factor means, and membership/weight concentration diagnostics.",
    ]
    (output / "managed_portfolio_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--managed-portfolio-returns", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--membership-file", type=Path)
    parser.add_argument("--covariance-lookback", type=int, default=240)
    parser.add_argument("--min-assets-per-bin", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = diagnose_managed_portfolios(
        args.managed_portfolio_returns,
        args.output_dir,
        membership_file=args.membership_file,
        covariance_lookback=args.covariance_lookback,
        min_assets_per_bin=args.min_assets_per_bin,
    )
    print(
        f"managed_portfolios={summary['managed_portfolios']} "
        f"months={summary['months_available']} output_dir={args.output_dir}"
    )


if __name__ == "__main__":
    main()
