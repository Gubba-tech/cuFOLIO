#!/usr/bin/env python3
"""Run a stock-level QP solver demonstration on the monthly panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cufolio.qp_monthly_panel import load_normalized_monthly_panel
from cufolio.qp_paper_replay import (
    PaperReplayWindow,
    run_replay_directory,
    save_replay_window,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monthly-panel", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/paper_replay/results/monthly_panel_stock_qp_pilot"),
    )
    parser.add_argument("--objective", choices=("min_variance", "mean_variance", "max_sharpe"), default="min_variance")
    parser.add_argument("--lookback-months", type=int, choices=(60, 120, 240), default=60)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--backend", choices=("osqp", "cuopt", "both"), default="osqp")
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--w-min", type=float, default=-0.08)
    parser.add_argument("--w-max", type=float, default=0.08)
    parser.add_argument("--short-budget", type=float, default=0.2)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--format", choices=("auto", "csv", "tsv", "parquet"), default="auto")
    parser.add_argument("--sep", choices=("auto", "comma", "tab"), default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frame = load_normalized_monthly_panel(args.monthly_panel, format=args.format, sep=args.sep)
    frame = frame.sort_values(["date", "asset_id"])
    dates = sorted(frame["date"].dropna().unique())
    if not dates:
        raise SystemExit("monthly panel has no valid dates")
    start = pd.Timestamp(args.start_date) if args.start_date else pd.Timestamp(dates[args.lookback_months])
    start = start.to_period("M").to_timestamp("M")
    end = pd.Timestamp(args.end_date).to_period("M").to_timestamp("M") if args.end_date else pd.Timestamp(dates[-1])
    output_dir = args.output_dir
    window_dir = output_dir / "windows"
    written = []
    for index, date_value in enumerate(dates):
        date = pd.Timestamp(date_value)
        if index < args.lookback_months or not start <= date <= end:
            continue
        history = frame.loc[frame["date"].isin(dates[index - args.lookback_months : index])]
        wide = history.pivot(index="date", columns="asset_id", values="ret").sort_index()
        next_rows = frame.loc[frame["date"] == date, ["asset_id", "ret"]].set_index("asset_id")
        columns = [column for column in wide.columns if wide[column].notna().all() and column in next_rows.index]
        if len(columns) < 2:
            continue
        wide = wide.loc[:, columns]
        next_returns = next_rows.loc[columns, "ret"]
        covariance = np.asarray(np.cov(wide.to_numpy(dtype=float), rowvar=False), dtype=float)
        if covariance.ndim == 0:
            covariance = covariance.reshape(1, 1)
        covariance = covariance + np.eye(len(columns)) * 1e-8
        window = PaperReplayWindow(
            schema_version="1.0",
            window_id=f"stock_monthly_panel_{args.objective}_{date.date()}",
            rebalance_date=str(date.date()),
            model_name="monthly-panel-stock-QP",
            objective=args.objective,
            mapping_mode="stock_space",
            risk_free_rate=args.risk_free_rate,
            lambda_l1=0.0,
            lambda_l2=1e-3,
            short_budget=args.short_budget if args.objective == "max_sharpe" else None,
            w_min=args.w_min,
            w_max=args.w_max,
            mean=wide.mean(axis=0).to_numpy(dtype=float),
            covariance=covariance,
            tickers=[str(column) for column in columns],
            realized_next_returns=next_returns.to_numpy(dtype=float),
            notes=(
                "Stock-level solver demonstration only. Sample-mean max-Sharpe is "
                "noisy and is not expected to match factor-model performance."
            ),
        )
        written.append(str(save_replay_window(window, window_dir / window.window_id)))
        if args.max_windows is not None and len(written) >= args.max_windows:
            break
    if not written:
        raise SystemExit("stock-level pilot produced no replay windows")
    window_dir.mkdir(parents=True, exist_ok=True)
    (window_dir / "export_metadata.json").write_text(
        json.dumps(
            {
                "objective": args.objective,
                "lookback_months": args.lookback_months,
                "written_windows": written,
                "stock_level_solver_demonstration_only": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    rows = run_replay_directory(window_dir, output_dir / "replay", backend=args.backend, write_summary=True)
    (output_dir / "pilot_manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "pilot_manifest.json").write_text(
        json.dumps(
            {
                "objective": args.objective,
                "lookback_months": args.lookback_months,
                "written_windows": len(written),
                "rows": len(rows),
                "stock_level_solver_demonstration_only": True,
                "full_paper_replication": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"windows={len(written)} rows={len(rows)} output_dir={output_dir}")


if __name__ == "__main__":
    main()
