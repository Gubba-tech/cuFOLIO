#!/usr/bin/env python3
"""Export factor-space replay windows from managed portfolio returns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cufolio.qp_factor_workflows import build_pca_factor_qp_data
from cufolio.qp_paper_replay import PaperReplayWindow, save_replay_window


def _load_managed_returns(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "date" not in frame.columns:
        raise ValueError("managed returns require a date column")
    frame["date"] = pd.to_datetime(frame["date"]).dt.to_period("M").dt.to_timestamp("M")
    if {"portfolio_id", "ret"}.issubset(frame.columns):
        return frame.pivot(index="date", columns="portfolio_id", values="ret").sort_index()
    return frame.set_index("date").select_dtypes(include=[np.number]).sort_index()


def export_pca_windows(
    managed_portfolio_returns: str | Path,
    output_dir: str | Path,
    k_values: list[int],
    start_date: str,
    end_date: str,
    lookback_months: int = 240,
    lambda_l1: float = 0.0,
    lambda_l2: float = 0.0,
    short_budget: float = 0.2,
    w_min: float = -0.08,
    w_max: float = 0.08,
    risk_free_rate: float = 0.0,
    max_windows: int | None = None,
    allow_short_lookback: bool = False,
    model_name: str = "PCA",
) -> dict[str, object]:
    """Export PCA windows and return provenance metadata."""
    if lookback_months < 2:
        raise ValueError("lookback_months must be at least 2")
    returns = _load_managed_returns(Path(managed_portfolio_returns))
    if returns.empty:
        raise ValueError("managed portfolio returns are empty")
    dates = list(returns.index)
    start = pd.Timestamp(start_date).to_period("M").to_timestamp("M")
    end = pd.Timestamp(end_date).to_period("M").to_timestamp("M")
    earliest = (
        dates[0] + pd.DateOffset(months=lookback_months - 1)
    ).to_period("M").to_timestamp("M")
    pilot_only = bool(lookback_months != 240 or (start < earliest and allow_short_lookback))
    if start < earliest and not allow_short_lookback:
        raise ValueError(
            f"{lookback_months}-month lookback first becomes feasible around "
            f"{pd.Timestamp(earliest).date()}, before requested start {start.date()}. "
            "Use a shorter lookback or pass --allow-short-lookback for a pilot-only run."
        )
    output_dir = Path(output_dir)
    written: list[str] = []
    effective_lookbacks: list[int] = []
    for index, date in enumerate(dates[:-1]):
        if not start <= date <= end or index < 1:
            continue
        effective = min(lookback_months, index + 1)
        if effective < 2:
            continue
        history = returns.iloc[index - effective + 1 : index + 1]
        next_returns = returns.iloc[index + 1]
        complete = history.notna().all() & next_returns.notna()
        history = history.loc[:, complete]
        next_returns = next_returns.loc[history.columns]
        if history.shape[1] == 0:
            continue
        for k in k_values:
            if k > min(history.shape):
                continue
            factor_data = build_pca_factor_qp_data(
                history.to_numpy(dtype=float),
                n_components=k,
                # Estimate PCA directions from demeaned returns, then project
                # raw returns so the optimizer receives a non-zero factor mean.
                center=True,
                tickers=[str(column) for column in history.columns],
            )
            note = (
                "V maps PCA factor weights to managed-portfolio weights; these are "
                "not individual stock weights. "
                f"requested_lookback_months={lookback_months}; "
                f"effective_lookback_months={effective}; "
                f"pilot_only={pilot_only}"
            )
            window = PaperReplayWindow(
                schema_version="1.0",
                window_id=f"pca_monthly_panel_k{k}_{date.date()}",
                rebalance_date=str(date.date()),
                model_name=model_name,
                objective="max_sharpe",
                mapping_mode="factor_space",
                risk_free_rate=risk_free_rate,
                lambda_l1=lambda_l1,
                lambda_l2=lambda_l2,
                short_budget=short_budget,
                w_min=w_min,
                w_max=w_max,
                tickers=[str(column) for column in history.columns],
                factor_names=factor_data.factor_names,
                factor_mean=factor_data.factor_mean,
                factor_covariance=factor_data.factor_covariance,
                stock_mapping=factor_data.stock_mapping,
                realized_next_returns=next_returns.to_numpy(dtype=float),
                notes=note,
            )
            target = output_dir / window.window_id
            written.append(str(save_replay_window(window, target)))
            effective_lookbacks.append(effective)
            if max_windows is not None and len(written) >= max_windows:
                break
        if max_windows is not None and len(written) >= max_windows:
            break
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_name": model_name,
        "k_values": k_values,
        "lookback_months": lookback_months,
        "effective_lookback_months": sorted(set(effective_lookbacks)),
        "requested_start_date": str(start.date()),
        "requested_end_date": str(end.date()),
        "earliest_feasible_date": str(pd.Timestamp(earliest).date()),
        "pilot_only": pilot_only,
        "written_windows": written,
        "mapping_semantics": "managed_portfolio_weights",
        "source_data_committed": False,
    }
    (output_dir / "export_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--managed-portfolio-returns", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k-values", nargs="+", type=int, default=[6])
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--lookback-months", type=int, default=240)
    parser.add_argument("--lambda-l1", type=float, default=1.7e-4)
    parser.add_argument("--lambda-l2", type=float, default=1e-3)
    parser.add_argument("--short-budget", type=float, default=0.2)
    parser.add_argument("--w-min", type=float, default=-0.08)
    parser.add_argument("--w-max", type=float, default=0.08)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--allow-short-lookback", action="store_true")
    parser.add_argument("--model-name", default="PCA")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = export_pca_windows(
        managed_portfolio_returns=args.managed_portfolio_returns,
        output_dir=args.output_dir,
        k_values=args.k_values,
        start_date=args.start_date,
        end_date=args.end_date,
        lookback_months=args.lookback_months,
        lambda_l1=args.lambda_l1,
        lambda_l2=args.lambda_l2,
        short_budget=args.short_budget,
        w_min=args.w_min,
        w_max=args.w_max,
        risk_free_rate=args.risk_free_rate,
        max_windows=args.max_windows,
        allow_short_lookback=args.allow_short_lookback,
        model_name=args.model_name,
    )
    print(f"windows_written={len(metadata['written_windows'])} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
