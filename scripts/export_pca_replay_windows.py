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
    frame["date"] = pd.to_datetime(frame["date"]).dt.to_period("M").dt.to_timestamp("M")
    if {"portfolio_id", "ret"}.issubset(frame.columns):
        return frame.pivot(index="date", columns="portfolio_id", values="ret").sort_index()
    if "date" not in frame.columns:
        raise ValueError("managed returns require a date column.")
    return frame.set_index("date").select_dtypes(include=[np.number]).sort_index()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--managed-portfolio-returns", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k-values", nargs="+", type=int, default=[6])
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--lookback-months", type=int, default=240)
    parser.add_argument("--lambda-l1", type=float, default=0.0)
    parser.add_argument("--lambda-l2", type=float, default=0.0)
    parser.add_argument("--short-budget", type=float, default=0.2)
    parser.add_argument("--w-min", type=float, default=-0.08)
    parser.add_argument("--w-max", type=float, default=0.08)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--max-windows", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    returns = _load_managed_returns(args.managed_portfolio_returns)
    dates = list(returns.index)
    start = pd.Timestamp(args.start_date).to_period("M").to_timestamp("M")
    end = pd.Timestamp(args.end_date).to_period("M").to_timestamp("M")
    rebalance_dates = [date for date in dates if start <= date <= end]
    written = []
    for date in rebalance_dates:
        index = dates.index(date)
        if index < args.lookback_months or index + 1 >= len(dates):
            continue
        history = returns.iloc[index - args.lookback_months : index]
        next_returns = returns.iloc[index + 1]
        complete = history.notna().all() & next_returns.notna()
        history = history.loc[:, complete]
        next_returns = next_returns.loc[history.columns]
        for k in args.k_values:
            if k > min(history.shape):
                continue
            factor_data = build_pca_factor_qp_data(
                history.to_numpy(dtype=float),
                n_components=k,
                center=False,
                tickers=[str(column) for column in history.columns],
            )
            window = PaperReplayWindow(
                schema_version="1.0",
                window_id=f"pca_k{k}_{date.date()}",
                rebalance_date=str(date.date()),
                model_name="PCA",
                objective="max_sharpe",
                mapping_mode="factor_space",
                risk_free_rate=args.risk_free_rate,
                lambda_l1=args.lambda_l1,
                lambda_l2=args.lambda_l2,
                short_budget=args.short_budget,
                w_min=args.w_min,
                w_max=args.w_max,
                tickers=[str(column) for column in history.columns],
                factor_names=factor_data.factor_names,
                factor_mean=factor_data.factor_mean,
                factor_covariance=factor_data.factor_covariance,
                stock_mapping=factor_data.stock_mapping,
                realized_next_returns=next_returns.to_numpy(dtype=float),
                notes=(
                    "V maps factor weights to managed-portfolio weights; "
                    "these are not individual stock weights. "
                    f"lookback_months={args.lookback_months}"
                ),
            )
            target = args.output_dir / window.window_id
            written.append(str(save_replay_window(window, target)))
            if args.max_windows is not None and len(written) >= args.max_windows:
                break
        if args.max_windows is not None and len(written) >= args.max_windows:
            break
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "export_metadata.json").write_text(
        json.dumps(
            {
                "model_name": "PCA",
                "k_values": args.k_values,
                "lookback_months": args.lookback_months,
                "written_windows": written,
                "mapping_semantics": "managed_portfolio_weights",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"windows_written={len(written)} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()

