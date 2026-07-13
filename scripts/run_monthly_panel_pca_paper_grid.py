#!/usr/bin/env python3
"""Run a resumable paper-style PCA K/lambda grid."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
from export_pca_replay_windows import export_pca_windows
from summarize_monthly_panel_results import summarize_run

from cufolio.qp_monthly_panel import (
    build_paper_managed_portfolios,
    convert_monthly_panel,
)
from cufolio.qp_paper_replay import run_replay_directory

PAPER_LAMBDA_GRID = [1e-6, 5.6e-6, 3.1e-5, 1.7e-4, 9.5e-4, 5.3e-3, 2.9e-2, 1.6e-1, 9.0e-1, 5.0]


def _parse_grid(values: list[str]) -> list[float]:
    if len(values) == 1 and values[0].lower() == "paper":
        return PAPER_LAMBDA_GRID.copy()
    parsed: list[float] = []
    for value in values:
        parsed.extend(float(part) for part in value.split(",") if part)
    if not parsed:
        raise ValueError("lambda grid cannot be empty")
    return parsed


def _label(value: float) -> str:
    return f"{value:.8g}".replace("-", "m").replace(".", "p")


def _config_id(k: int, lambda_l1: float, lambda_l2: float) -> str:
    return f"k{k}_l1_{_label(lambda_l1)}_l2_{_label(lambda_l2)}"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _prepare_managed(args: argparse.Namespace) -> Path:
    if args.managed_portfolio_returns:
        return args.managed_portfolio_returns
    if not args.monthly_panel:
        raise ValueError("provide --monthly-panel or --managed-portfolio-returns")
    prep_dir = args.output_dir / "data_prep"
    processed = prep_dir / "monthly_characteristic_panel.parquet"
    convert_monthly_panel(
        args.monthly_panel,
        processed,
        format=args.format,
        sep=args.sep,
        characteristics=args.characteristics,
        drop_missing_ret=True,
    )
    managed_dir = prep_dir / "managed_portfolios"
    build_paper_managed_portfolios(
        processed,
        managed_dir,
        characteristics=args.characteristics,
        n_bins=args.n_bins,
        weighting=args.weighting,
        min_assets_per_bin=args.min_assets_per_bin,
        universe_method="all",
        market_cap_coverage=0.90,
        assume_characteristics_lagged=args.assume_characteristics_lagged,
    )
    return managed_dir / "managed_portfolio_returns.parquet"


def _load_existing(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def _summary_row(metric: dict[str, object], config_id: str, args: argparse.Namespace, result_dir: Path) -> dict[str, object]:
    row = dict(metric)
    row.update(
        {
            "config_id": config_id,
            "k": int(config_id.split("_", 1)[0][1:]),
            "lambda_l1": args._current_lambda_l1,
            "lambda_l2": args._current_lambda_l2,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "lookback_months": args.lookback_months,
            "artifact_path": str(result_dir),
            "annualized_sharpe": metric.get("annualized_sharpe", metric.get("Sharpe")),
            "failed_windows": metric.get("failed_count"),
            "skipped_windows": metric.get("skipped_count"),
        }
    )
    return row


def run_paper_grid(args: argparse.Namespace) -> list[dict[str, object]]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    managed_returns = _prepare_managed(args)
    lambda_l1_values = _parse_grid(args.lambda_l1_grid)
    lambda_l2_values = _parse_grid(args.lambda_l2_grid)
    existing_path = args.output_dir / "grid_results.csv"
    aggregate = _load_existing(existing_path) if (args.resume or args.skip_existing) else []
    existing_keys = {
        (str(row.get("config_id")), str(row.get("backend"))) for row in aggregate
    }
    backends = ["osqp", "cuopt"] if args.backend == "both" else [args.backend]
    all_combinations = [
        (k, lambda_l1, lambda_l2)
        for k in args.k_values
        for lambda_l1 in lambda_l1_values
        for lambda_l2 in lambda_l2_values
    ]
    total_combinations = len(all_combinations)
    combinations = all_combinations
    if args.combination_offset:
        combinations = combinations[args.combination_offset :]
    if args.max_combinations is not None:
        combinations = combinations[: args.max_combinations]

    for k, lambda_l1, lambda_l2 in combinations:
        config_id = _config_id(k, lambda_l1, lambda_l2)
        result_dir = args.output_dir / "configs" / config_id / "replay"
        window_dir = args.output_dir / "configs" / config_id / "windows_pca"
        if (args.resume or args.skip_existing) and all(
            (config_id, backend) in existing_keys for backend in backends
        ):
            continue
        args._current_lambda_l1 = lambda_l1
        args._current_lambda_l2 = lambda_l2
        export_pca_windows(
            managed_returns,
            window_dir,
            k_values=[k],
            start_date=args.start_date,
            end_date=args.end_date,
            lookback_months=args.lookback_months,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
            short_budget=args.short_budget,
            w_min=args.w_min,
            w_max=args.w_max,
            risk_free_rate=args.risk_free_rate,
            allow_short_lookback=args.allow_short_lookback,
            model_name=f"PCA-monthly-panel-K{k}",
        )
        rows = run_replay_directory(
            window_dir,
            result_dir,
            backend=args.backend,
            write_summary=True,
        )
        metrics = summarize_run(result_dir, plots=False)
        for metric in metrics:
            aggregate.append(_summary_row(metric, config_id, args, result_dir))
        _write_csv(existing_path, aggregate)
        (args.output_dir / "last_completed.json").write_text(
            json.dumps({"config_id": config_id, "replay_rows": len(rows)}, indent=2) + "\n",
            encoding="utf-8",
        )

    aggregate = sorted(aggregate, key=lambda row: (str(row.get("backend")), str(row.get("config_id"))))
    _write_csv(existing_path, aggregate)
    lines = [
        "# Monthly Panel PCA Paper-Style Grid Summary",
        "",
        "Results are empirical uploaded-panel evidence and are not full paper replication.",
        "",
        f"- Configurations requested: {total_combinations}",
        f"- Grid sizes: K={args.k_values}; L1={len(lambda_l1_values)}; L2={len(lambda_l2_values)}",
        f"- Rows written: {len(aggregate)}",
        "",
        "| config_id | backend | K | lambda_l1 | lambda_l2 | annualized Sharpe | CAGR | max drawdown | optimal | failed | skipped | max violation |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate:
        lines.append(
            f"| {row.get('config_id')} | {row.get('backend')} | {row.get('k')} | "
            f"{row.get('lambda_l1')} | {row.get('lambda_l2')} | {row.get('annualized_sharpe')} | "
            f"{row.get('cagr')} | {row.get('max_drawdown')} | {row.get('optimal_count')} | "
            f"{row.get('failed_windows')} | {row.get('skipped_windows')} | {row.get('maximum_constraint_violation')} |"
        )
    (args.output_dir / "grid_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monthly-panel", type=Path)
    parser.add_argument("--managed-portfolio-returns", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--format", choices=("auto", "csv", "tsv", "parquet"), default="auto")
    parser.add_argument("--sep", choices=("auto", "comma", "tab"), default="auto")
    parser.add_argument("--characteristics", default="all")
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--weighting", choices=("value", "equal"), default="value")
    parser.add_argument("--min-assets-per-bin", type=int, default=5)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--lookback-months", type=int, required=True)
    parser.add_argument("--k-values", nargs="+", type=int, default=[2, 3, 4, 5, 6])
    parser.add_argument("--lambda-l1-grid", nargs="+", default=["paper"])
    parser.add_argument("--lambda-l2-grid", nargs="+", default=["paper"])
    parser.add_argument("--backend", choices=("osqp", "cuopt", "both"), default="osqp")
    parser.add_argument("--short-budget", type=float, default=0.2)
    parser.add_argument("--w-min", type=float, default=-0.08)
    parser.add_argument("--w-max", type=float, default=0.08)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--allow-short-lookback", action="store_true")
    parser.add_argument("--assume-characteristics-lagged", action="store_true")
    parser.add_argument("--max-combinations", type=int)
    parser.add_argument("--combination-offset", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--write-summary", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = run_paper_grid(args)
    print(f"grid_rows={len(rows)} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
