# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load, validate, clean, and sort the Sprint 14 monthly stock panel."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("date", "cusip", "permco", "prc", "ret", "mktcap")
SUPPORTED_CHARACTERISTICS = (
    "beta",
    "a2me",
    "at",
    "ac",
    "lme",
    "lt_rev",
    "ato",
    "beme",
    "beme_adj",
    "c2d",
    "c",
    "cto",
    "e2p",
    "idio_vol",
    "lev",
    "mktcap",
    "lturnover",
    "noa",
    "oa",
    "ol",
    "pcm",
    "pm",
    "dto",
    "q",
    "high_52w",
    "rna",
    "roa",
    "roe",
    "r12_2",
    "r12_7",
    "r2_1",
    "r36_13",
    "s2p",
    "sga2s",
    "suv",
)
EXTRA_CHARACTERISTICS = {"beme_adj", "c2d"}
PAPER_CHARACTERISTICS = tuple(
    name for name in SUPPORTED_CHARACTERISTICS if name not in EXTRA_CHARACTERISTICS
)
PAPER_TABLE_A1_MAPPING = {
    "a2me": "A2ME",
    "at": "AT",
    "ac": "AC",
    "lme": "LME / Size",
    "lt_rev": "Lt_Rev",
    "ato": "ATO",
    "beme": "BEME",
    "c": "C",
    "cto": "CTO",
    "e2p": "E2P",
    "idio_vol": "Idio vol",
    "lev": "Lev",
    "mktcap": "Mktcap",
    "lturnover": "LTurnover",
    "noa": "NOA",
    "oa": "OA",
    "ol": "OL",
    "pcm": "PCM",
    "pm": "PM",
    "dto": "DTO",
    "q": "Q",
    "high_52w": "Rel to High",
    "rna": "RNA",
    "roa": "ROA",
    "roe": "ROE",
    "r12_2": "r12-2",
    "r12_7": "r12-7",
    "r2_1": "r2-1",
    "r36_13": "r36-13",
    "s2p": "S2P",
    "sga2s": "SGA2S",
    "suv": "SUV",
}


def _month_end(value: object) -> pd.Timestamp:
    return pd.Timestamp(value).to_period("M").to_timestamp("M")


def _parse_separator(path: Path, sep: str) -> str:
    if sep == "comma":
        return ","
    if sep == "tab":
        return "\t"
    if sep != "auto":
        raise ValueError("sep must be auto, comma, or tab.")
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        header = handle.readline()
    return "\t" if "\t" in header else ","


def read_monthly_panel(
    path: str | Path,
    format: str = "auto",
    sep: str = "auto",
) -> pd.DataFrame:
    """Read a raw CSV/TSV/Parquet panel without imposing column names."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    selected_format = path.suffix.lstrip(".").lower() if format == "auto" else format
    if selected_format in {"csv", "tsv"}:
        delimiter = (
            ("\t" if selected_format == "tsv" else ",")
            if sep == "auto"
            else _parse_separator(path, sep)
        )
        frame = pd.read_csv(path, sep=delimiter, low_memory=False)
    elif selected_format == "parquet":
        frame = pd.read_parquet(path)
    else:
        raise ValueError("format must be auto, csv, tsv, or parquet.")
    unnamed = [column for column in frame.columns if str(column).startswith("Unnamed")]
    if unnamed:
        frame = frame.drop(columns=unnamed)
    return frame


def _resolve_column(frame: pd.DataFrame, requested: str, aliases: Iterable[str]) -> str:
    if requested in frame.columns:
        return requested
    for alias in aliases:
        if alias in frame.columns:
            return alias
    raise KeyError(requested)


def normalize_monthly_panel(
    frame: pd.DataFrame,
    date_col: str = "date",
    asset_id_col: str = "permco",
    cusip_col: str = "cusip",
    return_col: str = "ret",
    price_col: str = "prc",
    market_cap_col: str = "mktcap",
) -> pd.DataFrame:
    """Normalize raw or already-cleaned input while preserving source IDs."""
    frame = frame.copy()
    columns = {
        "date": _resolve_column(frame, date_col, ("date",)),
        "asset_id": _resolve_column(frame, asset_id_col, ("asset_id", "permco")),
        "cusip": _resolve_column(frame, cusip_col, ("cusip", "ticker")),
        "ret": _resolve_column(frame, return_col, ("ret",)),
        "price": _resolve_column(frame, price_col, ("price", "prc")),
        "market_cap": _resolve_column(frame, market_cap_col, ("market_cap", "mktcap")),
    }
    frame["date"] = pd.to_datetime(frame[columns["date"]], errors="coerce")
    valid_dates = frame["date"].notna()
    frame.loc[valid_dates, "date"] = frame.loc[valid_dates, "date"].map(_month_end)
    frame["asset_id"] = frame[columns["asset_id"]].astype("string")
    frame["cusip"] = frame[columns["cusip"]].astype("string")
    for target in ("ret", "price", "market_cap", *SUPPORTED_CHARACTERISTICS):
        source = columns.get(target, target)
        if source in frame.columns:
            frame[target] = pd.to_numeric(frame[source], errors="coerce")
    if columns["asset_id"] != "asset_id":
        frame["permco"] = frame[columns["asset_id"]]
    if columns["price"] != "price":
        frame["prc"] = frame[columns["price"]]
    if columns["market_cap"] != "market_cap":
        frame["mktcap"] = frame[columns["market_cap"]]
    if columns["ret"] != "ret":
        frame["ret"] = frame[columns["ret"]]
    return frame


def load_normalized_monthly_panel(
    path: str | Path,
    format: str = "auto",
    sep: str = "auto",
    **kwargs: str,
) -> pd.DataFrame:
    return normalize_monthly_panel(read_monthly_panel(path, format=format, sep=sep), **kwargs)


def _json_value(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value).date())
    if pd.isna(value):
        return None
    return value


def _date_filter(frame: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    result = frame
    if start_date:
        result = result[result["date"] >= _month_end(start_date)]
    if end_date:
        result = result[result["date"] <= _month_end(end_date)]
    return result


def _coverage_rows(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for date, group in frame.groupby("date", sort=True, dropna=True):
        rows.append(
            {
                "date": str(pd.Timestamp(date).date()),
                "rows": int(len(group)),
                "unique_assets": int(group["asset_id"].nunique()),
            }
        )
    return rows


def validate_monthly_panel(
    path: str | Path,
    format: str = "auto",
    sep: str = "auto",
    date_col: str = "date",
    asset_id_col: str = "permco",
    cusip_col: str = "cusip",
    return_col: str = "ret",
    price_col: str = "prc",
    market_cap_col: str = "mktcap",
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_months: int = 240,
) -> dict[str, object]:
    """Return a JSON-serializable validation report for the raw panel."""
    report: dict[str, object] = {
        "status": "pass",
        "input": str(path),
        "issues": [],
        "warnings": [],
        "required_columns": list(REQUIRED_COLUMNS),
        "supported_characteristics": list(SUPPORTED_CHARACTERISTICS),
        "paper_characteristics": list(PAPER_CHARACTERISTICS),
        "lookback_months": int(lookback_months),
    }
    if not Path(path).exists():
        report["issues"] = [f"input missing: {path}"]
        report["status"] = "blocked"
        return report
    raw = read_monthly_panel(path, format=format, sep=sep)
    missing_source = [
        column
        for column in (date_col, asset_id_col, cusip_col, return_col, price_col, market_cap_col)
        if column not in raw.columns
        and not (column == asset_id_col and "asset_id" in raw.columns)
        and not (column == price_col and "price" in raw.columns)
        and not (column == market_cap_col and "market_cap" in raw.columns)
    ]
    if missing_source:
        report["issues"].append(f"missing required source columns: {missing_source}")
        report["status"] = "blocked"
        return report
    frame = normalize_monthly_panel(
        raw,
        date_col=date_col,
        asset_id_col=asset_id_col,
        cusip_col=cusip_col,
        return_col=return_col,
        price_col=price_col,
        market_cap_col=market_cap_col,
    )
    dates = frame["date"].dropna().sort_values().unique()
    invalid_date_count = int(frame["date"].isna().sum())
    if invalid_date_count:
        report["issues"].append(f"unparseable dates: {invalid_date_count}")
    if len(dates):
        min_date = pd.Timestamp(dates[0])
        max_date = pd.Timestamp(dates[-1])
        expected = pd.period_range(min_date, max_date, freq="M")
        actual = {pd.Timestamp(value).to_period("M") for value in dates}
        missing_months = [str(period) for period in expected if period not in actual]
        report.update(
            {
                "min_date": str(min_date.date()),
                "max_date": str(max_date.date()),
                "number_of_months": int(len(dates)),
                "missing_months": missing_months,
                "date_range_complete": not missing_months,
            }
        )
        if missing_months:
            report["issues"].append(f"missing monthly periods: {len(missing_months)}")
        earliest = _month_end(min_date + pd.DateOffset(months=lookback_months))
        report["earliest_feasible_20_year_lookback_oos_date"] = str(earliest.date())
        report["2005_01_31_oos_possible_with_240_months"] = earliest <= pd.Timestamp("2005-01-31")
        report["2020_01_31_oos_possible_with_240_months"] = earliest <= pd.Timestamp("2020-01-31")
        if earliest > pd.Timestamp("2005-01-31"):
            report["warnings"].append(
                "Full 20-year lookback replication from 2005 is not possible. "
                "2005-start experiments require shorter lookback and are pilot-only."
            )
    selected = _date_filter(frame, start_date, end_date)
    duplicate_count = int(frame.duplicated(["date", "asset_id"]).sum())
    if duplicate_count:
        report["issues"].append(f"duplicate date/asset rows: {duplicate_count}")
    missing_returns = int(selected["ret"].isna().sum())
    nonfinite_returns = int((~np.isfinite(selected["ret"].fillna(0))).sum())
    missing_price = int(selected["price"].isna().sum())
    missing_market_cap = int(selected["market_cap"].isna().sum())
    available = [name for name in SUPPORTED_CHARACTERISTICS if name in frame.columns]
    missing_chars = [name for name in SUPPORTED_CHARACTERISTICS if name not in frame.columns]
    report.update(
        {
            "selected_start_date": start_date,
            "selected_end_date": end_date,
            "selected_rows": int(len(selected)),
            "unique_assets": int(selected["asset_id"].nunique()),
            "average_assets_per_month": float(selected.groupby("date")["asset_id"].nunique().mean())
            if not selected.empty
            else 0.0,
            "min_assets_per_month": int(selected.groupby("date")["asset_id"].nunique().min())
            if not selected.empty
            else 0,
            "median_assets_per_month": float(selected.groupby("date")["asset_id"].nunique().median())
            if not selected.empty
            else 0.0,
            "max_assets_per_month": int(selected.groupby("date")["asset_id"].nunique().max())
            if not selected.empty
            else 0,
            "duplicate_date_asset_rows": duplicate_count,
            "missing_returns": missing_returns,
            "nonfinite_returns": nonfinite_returns,
            "price_coverage": float(selected["price"].notna().mean()) if len(selected) else 0.0,
            "market_cap_coverage": float(selected["market_cap"].notna().mean()) if len(selected) else 0.0,
            "missing_price_rows": missing_price,
            "missing_market_cap_rows": missing_market_cap,
            "available_characteristics": available,
            "missing_characteristics": missing_chars,
            "characteristic_missing_rate": {
                name: float(frame[name].isna().mean()) for name in available
            },
            "monthly_coverage": _coverage_rows(selected),
        }
    )
    if report["status"] != "blocked" and report["issues"]:
        report["status"] = "blocked"
    return report


def write_monthly_panel_validation(report: dict[str, object], output_dir: str | Path) -> Path:
    """Write the four requested validation artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation_summary.json").write_text(
        json.dumps(report, indent=2, default=_json_value) + "\n", encoding="utf-8"
    )
    pd.DataFrame(report.get("monthly_coverage", [])).to_csv(
        output_dir / "monthly_coverage.csv", index=False
    )
    characteristic_rows = [
        {
            "characteristic": name,
            "available": name in report.get("available_characteristics", []),
            "missing_rate": report.get("characteristic_missing_rate", {}).get(name),
        }
        for name in report.get("supported_characteristics", [])
    ]
    pd.DataFrame(characteristic_rows).to_csv(
        output_dir / "characteristic_coverage.csv", index=False
    )
    issues = report.get("issues", [])
    lines = [
        "# Monthly Characteristic Panel Validation",
        "",
        f"Status: **{report.get('status')}**",
        "",
        "## Scope",
        "",
        "This is the uploaded CRSP/Compustat-style monthly panel validation. "
        "The raw panel is not copied into the repository.",
        "",
        "## Issues",
        "",
    ]
    lines.extend([f"- {issue}" for issue in issues] or ["- None"])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {warning}" for warning in report.get("warnings", [])] or ["- None"])
    lines.extend(["", "## Summary", "", "```json"])
    lines.append(json.dumps(report, indent=2, default=_json_value))
    lines.extend(["```", ""])
    summary_path = output_dir / "validation_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def _selected_characteristics(frame: pd.DataFrame, value: str | Iterable[str]) -> list[str]:
    if value == "all":
        return [name for name in PAPER_CHARACTERISTICS if name in frame.columns]
    if isinstance(value, str):
        selected = [name.strip() for name in value.split(",") if name.strip()]
    else:
        selected = list(value)
    missing = [name for name in selected if name not in frame.columns]
    if missing:
        raise ValueError(f"characteristics missing from panel: {missing}")
    return selected


def _cross_sectional_winsorize(frame: pd.DataFrame, names: Iterable[str], lower: float, upper: float) -> None:
    for name in names:
        frame[name] = frame.groupby("date")[name].transform(
            lambda values: values.clip(values.quantile(lower), values.quantile(upper))
        )


def _cross_sectional_standardize(frame: pd.DataFrame, names: Iterable[str]) -> None:
    for name in names:
        grouped = frame.groupby("date")[name]
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, np.nan)
        frame[name] = (frame[name] - mean) / std


def convert_monthly_panel(
    input_path: str | Path,
    output_path: str | Path,
    format: str = "auto",
    sep: str = "auto",
    date_col: str = "date",
    asset_id_col: str = "permco",
    cusip_col: str = "cusip",
    return_col: str = "ret",
    price_col: str = "prc",
    market_cap_col: str = "mktcap",
    characteristics: str | Iterable[str] = "all",
    drop_missing_ret: bool = False,
    min_price: float | None = None,
    winsorize_characteristics: tuple[float, float] | None = None,
    standardize_characteristics: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Convert a raw panel to the cleaned Parquet contract."""
    frame = load_normalized_monthly_panel(
        input_path,
        format=format,
        sep=sep,
        date_col=date_col,
        asset_id_col=asset_id_col,
        cusip_col=cusip_col,
        return_col=return_col,
        price_col=price_col,
        market_cap_col=market_cap_col,
    )
    required = {"date", "asset_id", "cusip", "price", "ret", "market_cap"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"normalized panel missing columns: {missing}")
    if frame["date"].isna().any():
        raise ValueError("cannot convert a panel with unparseable dates")
    if frame.duplicated(["date", "asset_id"]).any():
        raise ValueError("cannot convert duplicate date/asset rows")
    selected = _selected_characteristics(frame, characteristics)
    if drop_missing_ret:
        frame = frame.loc[frame["ret"].notna()].copy()
    if min_price is not None:
        frame = frame.loc[frame["price"].notna() & (frame["price"] >= min_price)].copy()
    if winsorize_characteristics is not None:
        lower, upper = winsorize_characteristics
        if not 0 <= lower < upper <= 1:
            raise ValueError("winsorization bounds must satisfy 0 <= lower < upper <= 1")
        _cross_sectional_winsorize(frame, selected, lower, upper)
    if standardize_characteristics:
        if standardize_characteristics != "cross_sectional":
            raise ValueError("standardize_characteristics must be cross_sectional")
        _cross_sectional_standardize(frame, selected)
    keep = ["date", "asset_id", "cusip", "price", "ret", "market_cap"]
    for source in ("permco", "prc", "mktcap"):
        if source in frame.columns and source not in keep:
            keep.append(source)
    keep.extend(name for name in selected if name not in keep)
    cleaned = frame.loc[:, [column for column in keep if column in frame.columns]].copy()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(output_path, index=False)
    metadata = {
        "source_path": str(input_path),
        "source_data_committed": False,
        "output_path": str(output_path),
        "rows": int(len(cleaned)),
        "min_date": str(cleaned["date"].min().date()) if not cleaned.empty else None,
        "max_date": str(cleaned["date"].max().date()) if not cleaned.empty else None,
        "selected_characteristics": selected,
        "drop_missing_ret": drop_missing_ret,
        "min_price": min_price,
        "winsorize_characteristics": winsorize_characteristics,
        "standardize_characteristics": standardize_characteristics,
        "internal_columns": ["date", "asset_id", "cusip", "price", "ret", "market_cap"],
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, default=_json_value) + "\n", encoding="utf-8")
    return cleaned, metadata


def _eligible_universe(
    frame: pd.DataFrame,
    sort_date: pd.Timestamp,
    universe_method: str,
    market_cap_coverage: float,
    min_price: float | None,
    max_missing_fraction: float,
) -> pd.DataFrame:
    current = frame.loc[frame["date"] == sort_date].copy()
    current = current.loc[current["market_cap"].notna() & (current["market_cap"] >= 0)]
    if max_missing_fraction < 1.0:
        missing_fraction = frame.groupby("asset_id")["ret"].apply(
            lambda values: float(values.isna().mean())
        )
        current = current.loc[
            current["asset_id"].map(missing_fraction).fillna(1.0) <= max_missing_fraction
        ]
    if min_price is not None:
        current = current.loc[current["price"].notna() & (current["price"] >= min_price)]
    if universe_method == "amp":
        current = current.sort_values("market_cap", ascending=False)
        total = float(current["market_cap"].sum())
        if total > 0 and not current.empty:
            cumulative = current["market_cap"].cumsum() / total
            crossing = np.flatnonzero(cumulative.to_numpy() >= market_cap_coverage)
            if crossing.size:
                current = current.iloc[: int(crossing[0]) + 1]
    return current


def build_paper_managed_portfolios(
    monthly_panel: str | Path,
    output_dir: str | Path,
    characteristics: str | Iterable[str] = "all",
    n_bins: int = 10,
    weighting: str = "value",
    market_cap_col: str = "market_cap",
    start_date: str | None = None,
    end_date: str | None = None,
    min_assets_per_bin: int = 5,
    universe_method: str = "all",
    market_cap_coverage: float = 0.90,
    min_price: float | None = None,
    max_missing_fraction: float = 1.0,
    assume_characteristics_lagged: bool = False,
    sort_at_t_return_at_t_plus_1: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build next-month characteristic-sorted managed portfolios."""
    if n_bins < 2 or weighting not in {"equal", "value"}:
        raise ValueError("n_bins must be >= 2 and weighting must be equal or value")
    if universe_method not in {"all", "amp"}:
        raise ValueError("universe_method must be all or amp")
    if not 0 <= max_missing_fraction <= 1:
        raise ValueError("max_missing_fraction must be between 0 and 1")
    if not sort_at_t_return_at_t_plus_1:
        raise ValueError("only sort-at-t/return-at-t-plus-1 is supported")
    frame = load_normalized_monthly_panel(monthly_panel, date_col="date", asset_id_col="asset_id")
    frame = frame.rename(columns={market_cap_col: "market_cap"}) if market_cap_col != "market_cap" else frame
    selected = _selected_characteristics(frame, characteristics)
    dates = sorted(frame["date"].dropna().unique())
    target_start = _month_end(start_date) if start_date else None
    target_end = _month_end(end_date) if end_date else None
    returns = []
    membership_parts = []
    coverage = []
    for index, sort_date_value in enumerate(dates[:-1]):
        sort_date = pd.Timestamp(sort_date_value)
        return_date = pd.Timestamp(dates[index + 1])
        if target_start is not None and return_date < target_start:
            continue
        if target_end is not None and return_date > target_end:
            continue
        current = _eligible_universe(
            frame,
            sort_date,
            universe_method,
            market_cap_coverage,
            min_price,
            max_missing_fraction,
        )
        if current.empty:
            continue
        next_returns = frame.loc[frame["date"] == return_date, ["asset_id", "ret"]].rename(
            columns={"ret": "next_ret"}
        )
        current = current.merge(next_returns, on="asset_id", how="inner")
        current = current.loc[current["next_ret"].notna() & np.isfinite(current["next_ret"])]
        if current.empty:
            continue
        for name in selected:
            valid = current.loc[current[name].notna() & np.isfinite(current[name])].copy()
            if max_missing_fraction < 1.0:
                valid = valid.loc[valid[name].notna()]
            if valid.empty:
                continue
            valid["bin"] = pd.qcut(
                valid[name].rank(method="first"),
                q=n_bins,
                labels=False,
                duplicates="drop",
            )
            valid = valid.loc[
                valid["bin"].map(valid["bin"].value_counts()) >= min_assets_per_bin
            ].copy()
            if valid.empty:
                continue
            if weighting == "value":
                valid["weight"] = valid.groupby("bin")["market_cap"].transform(
                    lambda values: values.clip(lower=0) / values.clip(lower=0).sum()
                )
            else:
                valid["weight"] = valid.groupby("bin")["asset_id"].transform(
                    lambda values: 1.0 / len(values)
                )
            valid["date"] = return_date
            valid["sort_date"] = sort_date
            valid["characteristic_name"] = name
            valid["bin"] = valid["bin"].astype(int) + 1
            valid["portfolio_id"] = valid["bin"].map(
                lambda number: f"{name}_bin{int(number):02d}"
            )
            membership_parts.append(
                valid[
                    [
                        "date",
                        "sort_date",
                        "asset_id",
                        "cusip",
                        "characteristic_name",
                        "bin",
                        "portfolio_id",
                        "weight",
                    ]
                ]
            )
            for bin_number, group in valid.groupby("bin", sort=True):
                if weighting == "value":
                    weights = group["market_cap"].clip(lower=0).astype(float)
                else:
                    weights = pd.Series(1.0, index=group.index)
                if float(weights.sum()) <= 0:
                    continue
                weights = weights / float(weights.sum())
                portfolio_id = f"{name}_bin{int(bin_number):02d}"
                returns.append(
                    {
                        "date": return_date,
                        "sort_date": sort_date,
                        "portfolio_id": portfolio_id,
                        "ret": float(np.dot(group["next_ret"], weights)),
                        "n_assets": int(len(group)),
                        "characteristic_name": name,
                        "bin": int(bin_number) + 1,
                    }
                )
            coverage.append(
                {
                    "sort_date": sort_date,
                    "return_date": return_date,
                    "characteristic": name,
                    "eligible_assets": int(len(valid)),
                    "bins_created": int(valid["bin"].nunique()),
                }
            )
    returns_frame = pd.DataFrame(returns)
    memberships_frame = (
        pd.concat(membership_parts, ignore_index=True)
        if membership_parts
        else pd.DataFrame()
    )
    coverage_frame = pd.DataFrame(coverage)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    returns_frame.to_parquet(output_dir / "managed_portfolio_returns.parquet", index=False)
    memberships_frame.to_parquet(output_dir / "managed_portfolio_membership.parquet", index=False)
    coverage_frame.to_csv(output_dir / "characteristic_bin_coverage.csv", index=False)
    metadata = {
        "characteristics": selected,
        "n_bins": n_bins,
        "weighting": weighting,
        "universe_method": universe_method,
        "market_cap_coverage": market_cap_coverage,
        "min_assets_per_bin": min_assets_per_bin,
        "managed_portfolios_created": int(returns_frame["portfolio_id"].nunique())
        if not returns_frame.empty
        else 0,
        "mapping_semantics": "managed_portfolio_returns; not individual stock weights",
        "timing": "sort using characteristics at month t; return at month t+1",
        "assume_characteristics_lagged": assume_characteristics_lagged,
        "source_data_committed": False,
    }
    (output_dir / "managed_portfolio_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=_json_value) + "\n", encoding="utf-8"
    )
    lines = [
        "# Monthly Panel Managed Portfolios",
        "",
        "The portfolios sort on month-t characteristics and realize month-t+1 returns.",
        "",
        f"- Characteristics: {len(selected)}",
        f"- Bins: {n_bins}",
        f"- Weighting: {weighting}",
        f"- Portfolios created: {metadata['managed_portfolios_created']}",
        f"- Mapping: {metadata['mapping_semantics']}",
        "",
        "These are paper-style managed portfolios, not individual-stock weights.",
        "",
    ]
    (output_dir / "managed_portfolio_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return returns_frame, memberships_frame, metadata
