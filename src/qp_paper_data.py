# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cleaned paper-style data validation and managed-portfolio helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

BASE_COLUMNS = {"date", "asset_id", "ticker"}
MONTHLY_COLUMNS = BASE_COLUMNS | {"ret", "price", "market_cap"}


def load_clean_table(path: str | Path) -> pd.DataFrame:
    """Load a cleaned Parquet or CSV table with normalized month-end dates."""
    path = Path(path)
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported cleaned-data format: {path.suffix}")
    if "date" in frame:
        parsed = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.copy()
        frame["date"] = parsed.dt.to_period("M").dt.to_timestamp("M")
    return frame


def characteristic_wide(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert long characteristic rows to the accepted wide representation."""
    if "characteristic_name" not in frame.columns:
        return frame.copy()
    value_columns = [
        column
        for column in frame.columns
        if column not in BASE_COLUMNS | {"characteristic_name"}
    ]
    if len(value_columns) != 1:
        raise ValueError("Long characteristics data requires one value column.")
    value_column = value_columns[0]
    return (
        frame.pivot_table(
            index=["date", "asset_id", "ticker"],
            columns="characteristic_name",
            values=value_column,
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )


def characteristic_columns(frame: pd.DataFrame) -> list[str]:
    """Return wide characteristic columns after excluding identifiers."""
    return [column for column in frame.columns if column not in BASE_COLUMNS]


def _date_bounds(frame: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    result = frame
    if start_date is not None:
        result = result[result["date"] >= pd.Timestamp(start_date).to_period("M").to_timestamp("M")]
    if end_date is not None:
        result = result[result["date"] <= pd.Timestamp(end_date).to_period("M").to_timestamp("M")]
    return result


def _coverage(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {
            "date": str(date.date()),
            "rows": int(group.shape[0]),
            "assets": int(group["asset_id"].nunique()),
        }
        for date, group in frame.groupby("date", sort=True)
    ]


def validate_cleaned_data(
    monthly_returns: str | Path | None = None,
    daily_returns: str | Path | None = None,
    characteristics: str | Path | None = None,
    benchmark_returns: str | Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_months: int = 240,
) -> dict[str, object]:
    """Validate cleaned inputs and return a JSON-serializable report."""
    report: dict[str, object] = {"status": "pass", "issues": [], "files": {}}
    monthly = None
    if monthly_returns is None or not Path(monthly_returns).exists():
        report["issues"].append(f"monthly returns missing: {monthly_returns}")
    else:
        monthly = load_clean_table(monthly_returns)
        report["files"]["monthly_returns"] = str(monthly_returns)
        missing = sorted(MONTHLY_COLUMNS - set(monthly.columns))
        if missing:
            report["issues"].append(f"monthly returns missing columns: {missing}")
        else:
            duplicate_count = int(monthly.duplicated(["date", "asset_id"]).sum())
            if duplicate_count:
                report["issues"].append(f"monthly duplicate date/asset rows: {duplicate_count}")
            if monthly["date"].isna().any():
                report["issues"].append("monthly returns contain unparseable dates")
            if not np.isfinite(monthly["ret"].dropna()).all():
                report["issues"].append("monthly returns contain non-finite returns")
            if (monthly["market_cap"].dropna() < 0).any():
                report["issues"].append("monthly returns contain negative market caps")
            report["files"]["monthly_coverage"] = _coverage(monthly)
            dates = sorted(monthly["date"].dropna().unique())
            candidate = dates[lookback_months] if len(dates) > lookback_months else None
            report["candidate_first_oos_date"] = (
                None if candidate is None else str(pd.Timestamp(candidate).date())
            )
            report["month_count"] = len(dates)
            report["asset_count"] = int(monthly["asset_id"].nunique())

    for label, path, required in (
        ("daily_returns", daily_returns, MONTHLY_COLUMNS),
        ("benchmark_returns", benchmark_returns, {"date", "benchmark_name", "ret"}),
    ):
        if path is None:
            continue
        if not Path(path).exists():
            report["issues"].append(f"{label} missing: {path}")
            continue
        frame = load_clean_table(path)
        report["files"][label] = str(path)
        missing = sorted(required - set(frame.columns))
        if missing:
            report["issues"].append(f"{label} missing columns: {missing}")
        if "ret" in frame and not np.isfinite(frame["ret"].dropna()).all():
            report["issues"].append(f"{label} contains non-finite returns")

    if characteristics is not None:
        if not Path(characteristics).exists():
            report["issues"].append(f"characteristics missing: {characteristics}")
        else:
            chars = characteristic_wide(load_clean_table(characteristics))
            report["files"]["characteristics"] = str(characteristics)
            missing = sorted(BASE_COLUMNS - set(chars.columns))
            if missing:
                report["issues"].append(f"characteristics missing columns: {missing}")
            else:
                report["characteristic_count"] = len(characteristic_columns(chars))
                report["characteristic_coverage"] = _coverage(chars)
                duplicate_count = int(chars.duplicated(["date", "asset_id"]).sum())
                if duplicate_count:
                    report["issues"].append(
                        f"characteristics duplicate date/asset rows: {duplicate_count}"
                    )
                report["characteristic_missing_fraction"] = {
                    column: float(chars[column].isna().mean())
                    for column in characteristic_columns(chars)
                }

    if start_date is not None:
        report["requested_start_date"] = start_date
    if end_date is not None:
        report["requested_end_date"] = end_date
    report["lookback_months"] = lookback_months
    report["status"] = "blocked" if report["issues"] else "pass"
    return report


def write_validation_report(report: dict[str, object], report_path: str | Path) -> Path:
    """Write Markdown and the companion ignored JSON summary."""
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Cleaned Paper Data Validation",
        "",
        f"Status: **{report['status']}**",
        "",
        "## Issues",
        "",
    ]
    issues = report.get("issues", [])
    lines.extend([f"- {issue}" for issue in issues] or ["- None"])
    lines.extend(["", "## Summary", "", "```json", json.dumps(report, indent=2, default=str), "```", ""])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    json_path = Path("artifacts/paper_replay/data_validation") / f"{report_path.stem}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    return report_path


def build_amp_universe(
    monthly_returns: str | Path,
    characteristics: str | Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    market_cap_coverage: float = 0.90,
    min_price: float = 0.0,
    max_missing_fraction: float = 0.20,
    lookback_months: int = 240,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build configurable AMP-style selected-universe rows by rebalance date."""
    if not 0 < market_cap_coverage <= 1:
        raise ValueError("market_cap_coverage must be in (0, 1].")
    monthly = load_clean_table(monthly_returns)
    required = MONTHLY_COLUMNS
    missing = required - set(monthly.columns)
    if missing:
        raise ValueError(f"monthly returns missing columns: {sorted(missing)}")
    if characteristics is not None:
        characteristic_wide(load_clean_table(characteristics))
    selected_period = _date_bounds(monthly, start_date, end_date)
    dates = sorted(selected_period["date"].dropna().unique())
    selected_rows = []
    summary_rows = []
    for date in dates:
        prior = monthly[monthly["date"] <= date]
        window_dates = sorted(prior["date"].unique())[-lookback_months:]
        window = monthly[monthly["date"].isin(window_dates)]
        missing_fraction = window.groupby("asset_id")["ret"].apply(lambda values: values.isna().mean())
        current = monthly[monthly["date"] == date].copy()
        current["missing_fraction"] = current["asset_id"].map(missing_fraction).fillna(1.0)
        current = current[
            (current["price"] >= min_price)
            & (current["missing_fraction"] <= max_missing_fraction)
            & current["market_cap"].notna()
            & (current["market_cap"] >= 0)
        ].sort_values("market_cap", ascending=False)
        total_market_cap = float(current["market_cap"].sum())
        if total_market_cap > 0 and not current.empty:
            current["cumulative_coverage"] = current["market_cap"].cumsum() / total_market_cap
            crossing = np.flatnonzero(current["cumulative_coverage"].to_numpy() >= market_cap_coverage)
            end = int(crossing[0]) + 1 if crossing.size else len(current)
            current = current.iloc[:end]
        current["date"] = date
        current["selected"] = True
        selected_rows.append(
            current[["date", "asset_id", "ticker", "price", "market_cap", "missing_fraction", "selected"]]
        )
        summary_rows.append(
            {
                "date": str(pd.Timestamp(date).date()),
                "selected_assets": int(current["asset_id"].nunique()),
                "market_cap": total_market_cap,
                "coverage_target": market_cap_coverage,
            }
        )
    selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    return selected, summary


def build_managed_portfolios(
    monthly_returns: str | Path,
    characteristics: str | Path,
    universe: pd.DataFrame | str | Path,
    selected_characteristics: Iterable[str] | str = "all",
    n_bins: int = 10,
    weighting: str = "equal",
    lag_months: int = 6,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build lagged characteristic-sorted managed portfolio returns."""
    if n_bins < 2 or weighting not in {"equal", "value"}:
        raise ValueError("n_bins must be >= 2 and weighting must be equal or value.")
    monthly = load_clean_table(monthly_returns)
    chars = characteristic_wide(load_clean_table(characteristics))
    universe_frame = load_clean_table(universe) if isinstance(universe, (str, Path)) else universe.copy()
    target_period = _date_bounds(monthly, start_date, end_date)
    all_dates = sorted(monthly["date"].dropna().unique())
    dates = sorted(target_period["date"].dropna().unique())
    char_names = characteristic_columns(chars) if selected_characteristics == "all" else list(selected_characteristics)
    missing = sorted(set(char_names) - set(chars.columns))
    if missing:
        raise ValueError(f"characteristics missing requested columns: {missing}")
    outputs = []
    for date in dates:
        index = all_dates.index(date)
        if index < lag_months:
            continue
        lag_date = all_dates[index - lag_months]
        eligible = universe_frame[universe_frame["date"] == date]
        current = monthly[monthly["date"] == date].merge(
            eligible[["asset_id"]], on="asset_id", how="inner"
        )
        lag = chars[chars["date"] == lag_date][["asset_id", *char_names]]
        current = current.merge(lag, on="asset_id", how="left")
        for name in char_names:
            valid = current.dropna(subset=[name, "ret"]).copy()
            if valid.empty:
                continue
            valid["bin"] = pd.qcut(
                valid[name].rank(method="first"), q=n_bins, labels=False, duplicates="drop"
            )
            for bin_number, group in valid.groupby("bin", sort=True):
                weights = (
                    group["market_cap"].clip(lower=0)
                    if weighting == "value"
                    else pd.Series(1.0, index=group.index)
                )
                if float(weights.sum()) <= 0:
                    continue
                outputs.append(
                    {
                        "date": date,
                        "portfolio_id": f"{name}__decile_{int(bin_number) + 1}",
                        "ret": float(np.average(group["ret"], weights=weights)),
                        "n_assets": int(group.shape[0]),
                        "characteristic_name": name,
                        "decile": int(bin_number) + 1,
                    }
                )
    result = pd.DataFrame(outputs)
    metadata = {
        "characteristics": char_names,
        "n_bins": n_bins,
        "weighting": weighting,
        "lag_months": lag_months,
        "mapping_semantics": "managed_portfolio_returns; not individual stock weights",
    }
    return result, metadata
