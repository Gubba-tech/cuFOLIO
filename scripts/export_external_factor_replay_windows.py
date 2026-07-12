#!/usr/bin/env python3
"""Export replay windows from externally supplied factor outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cufolio.qp_paper_replay import PaperReplayWindow, save_replay_window


def _load_dated_matrix(path: Path, value_name: str = "ret") -> pd.DataFrame:
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
        frame["date"] = pd.to_datetime(frame["date"]).dt.to_period("M").dt.to_timestamp("M")
        if {"portfolio_id", value_name}.issubset(frame.columns):
            return frame.pivot(index="date", columns="portfolio_id", values=value_name).sort_index()
        if {"asset_id", value_name}.issubset(frame.columns):
            return frame.pivot(index="date", columns="asset_id", values=value_name).sort_index()
        return frame.set_index("date").select_dtypes(include=[np.number]).sort_index()
    if path.suffix == ".csv":
        return _load_dated_matrix_csv(path)
    raise ValueError("dated matrices must be Parquet or CSV.")


def _load_dated_matrix_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.to_period("M").dt.to_timestamp("M")
    if {"portfolio_id", "ret"}.issubset(frame.columns):
        return frame.pivot(index="date", columns="portfolio_id", values="ret").sort_index()
    return frame.set_index("date").select_dtypes(include=[np.number]).sort_index()


def _load_mapping(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False), dtype=float)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if len(data.files) != 1:
                raise ValueError("mapping NPZ must contain one array.")
            return np.asarray(data[data.files[0]], dtype=float)
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    return frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-returns", type=Path, required=True)
    parser.add_argument("--stock-mapping", type=Path, required=True)
    parser.add_argument("--asset-returns", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", choices=("IPCA", "RP-PCA", "AP-Trees", "External"), default="External")
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
    factors = _load_dated_matrix(args.factor_returns)
    assets = _load_dated_matrix(args.asset_returns)
    mapping = _load_mapping(args.stock_mapping)
    if mapping.shape[0] != assets.shape[1] or mapping.shape[1] not in args.k_values:
        raise ValueError("mapping dimensions must match asset returns and requested K.")
    dates = list(factors.index)
    start = pd.Timestamp(args.start_date).to_period("M").to_timestamp("M")
    end = pd.Timestamp(args.end_date).to_period("M").to_timestamp("M")
    written = []
    for date in [date for date in dates if start <= date <= end]:
        index = dates.index(date)
        if index < args.lookback_months or index + 1 >= len(dates):
            continue
        history = factors.iloc[index - args.lookback_months : index].dropna(axis=1)
        next_date = dates[index + 1]
        if next_date not in assets.index or history.empty:
            continue
        next_returns = assets.loc[next_date].reindex(assets.columns)
        if next_returns.isna().any():
            continue
        for k in args.k_values:
            if k != mapping.shape[1]:
                continue
            mean = history.to_numpy(dtype=float).mean(axis=0)
            covariance = np.cov(history.to_numpy(dtype=float), rowvar=False)
            covariance = np.atleast_2d(covariance)
            window = PaperReplayWindow(
                schema_version="1.0",
                window_id=f"{args.model_name.lower()}_k{k}_{date.date()}",
                rebalance_date=str(date.date()),
                model_name=args.model_name,
                objective="max_sharpe",
                mapping_mode="factor_space",
                risk_free_rate=args.risk_free_rate,
                lambda_l1=args.lambda_l1,
                lambda_l2=args.lambda_l2,
                short_budget=args.short_budget,
                w_min=args.w_min,
                w_max=args.w_max,
                tickers=[str(column) for column in assets.columns],
                factor_names=[f"factor_{idx}" for idx in range(k)],
                factor_mean=mean,
                factor_covariance=covariance,
                stock_mapping=mapping,
                realized_next_returns=next_returns.to_numpy(dtype=float),
                notes="Externally supplied factor returns and mapping; verify mapping semantics.",
            )
            written.append(str(save_replay_window(window, args.output_dir / window.window_id)))
            if args.max_windows is not None and len(written) >= args.max_windows:
                break
        if args.max_windows is not None and len(written) >= args.max_windows:
            break
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "export_metadata.json").write_text(
        json.dumps({"model_name": args.model_name, "written_windows": written}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"windows_written={len(written)} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
