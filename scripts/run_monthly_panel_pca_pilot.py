#!/usr/bin/env python3
"""Run the monthly-panel validation, PCA managed portfolios, and QP replay."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from export_pca_replay_windows import export_pca_windows

from cufolio.qp_monthly_panel import (
    build_paper_managed_portfolios,
    convert_monthly_panel,
    validate_monthly_panel,
    write_monthly_panel_validation,
)
from cufolio.qp_paper_replay import run_replay_directory


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _metrics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("backend")), str(row.get("model_name")))].append(row)
    output = []
    for (backend, model), group in sorted(grouped.items()):
        successful = [row for row in group if row.get("status") == "optimal"]
        returns = np.asarray(
            [value for row in successful if (value := _number(row.get("realized_next_return"))) is not None],
            dtype=float,
        )
        annualized_return = (
            float(np.prod(1.0 + returns) ** (12.0 / len(returns)) - 1.0)
            if len(returns) and np.all(returns > -1)
            else None
        )
        volatility = float(np.std(returns, ddof=1) * np.sqrt(12.0)) if len(returns) > 1 else None
        sharpe = (
            float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(12.0))
            if len(returns) > 1 and np.std(returns, ddof=1) > 0
            else None
        )
        cumulative = np.cumprod(1.0 + returns) if len(returns) else np.asarray([])
        drawdown = cumulative / np.maximum.accumulate(cumulative) - 1.0 if len(cumulative) else np.asarray([])

        def values(name: str) -> list[float]:
            return [
                value
                for row in successful
                if (value := _number(row.get(name))) is not None
            ]

        violations = values("max_constraint_violation")
        output.append(
            {
                "backend": backend,
                "model_name": model,
                "rows": len(group),
                "successful_rows": len(successful),
                "status_counts": dict(Counter(str(row.get("status")) for row in group)),
                "annualized_return": annualized_return,
                "annualized_volatility": volatility,
                "Sharpe": sharpe,
                "max_drawdown": float(np.min(drawdown)) if len(drawdown) else None,
                "average_gross_long": float(np.mean(values("gross_long"))) if values("gross_long") else None,
                "average_gross_short": float(np.mean(values("gross_short"))) if values("gross_short") else None,
                "max_constraint_violation": max(violations) if violations else None,
                "average_turnover": float(np.mean(values("turnover"))) if values("turnover") else None,
            }
        )
    return output


def _write_summary(output_dir: Path, rows: list[dict[str, object]], metrics: list[dict[str, object]], manifest: dict[str, object]) -> None:
    (output_dir / "pilot_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8"
    )
    lines = [
        "# Monthly Panel PCA Pilot Summary",
        "",
        "This is a PCA managed-portfolio pilot, not full paper replication.",
        "",
        f"- Replay windows: {len({row.get('window_id') for row in rows})}",
        f"- Pilot-only flag: {manifest['pilot_only']}",
        "- Full IPCA/AP-Trees replication: not claimed",
        "- Old-solution parity: not claimed without old weights/objectives",
        "- Global QP speedup: not claimed",
        "- Mean-CVaR workflow: untouched",
        "",
        "| backend | model | rows | optimal | statuses | annualized return | annualized volatility | Sharpe | max drawdown | average gross long | average gross short | max constraint violation | average turnover |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics:
        lines.append(
            "| {backend} | {model_name} | {rows} | {successful_rows} | {status_counts} | "
            "{annualized_return} | {annualized_volatility} | {Sharpe} | {max_drawdown} | "
            "{average_gross_long} | {average_gross_short} | {max_constraint_violation} | "
            "{average_turnover} |".format(**row)
        )
    (output_dir / "pilot_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monthly-panel", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/paper_replay/results/monthly_panel_pca_pilot"),
    )
    parser.add_argument("--format", choices=("auto", "csv", "tsv", "parquet"), default="auto")
    parser.add_argument("--sep", choices=("auto", "comma", "tab"), default="auto")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--lookback-months", type=int, default=240)
    parser.add_argument("--k-values", nargs="+", type=int, default=[6])
    parser.add_argument("--characteristics", default="all")
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--weighting", choices=("value", "equal"), default="value")
    parser.add_argument("--lambda-l1", type=float, default=1.7e-4)
    parser.add_argument("--lambda-l2", type=float, default=1e-3)
    parser.add_argument("--short-budget", type=float, default=0.2)
    parser.add_argument("--w-min", type=float, default=-0.08)
    parser.add_argument("--w-max", type=float, default=0.08)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--backend", choices=("osqp", "cuopt", "both"), default="osqp")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--allow-short-lookback", action="store_true")
    parser.add_argument("--assume-characteristics-lagged", action="store_true")
    parser.add_argument("--min-assets-per-bin", type=int, default=5)
    parser.add_argument("--universe-method", choices=("all", "amp"), default="all")
    parser.add_argument("--market-cap-coverage", type=float, default=0.90)
    parser.add_argument("--min-price", type=float)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_dir = output_dir / "validation"
    report = validate_monthly_panel(
        args.monthly_panel,
        format=args.format,
        sep=args.sep,
        start_date=args.start_date,
        end_date=args.end_date,
        lookback_months=args.lookback_months,
    )
    write_monthly_panel_validation(report, validation_dir)
    if report["status"] == "blocked":
        raise SystemExit("monthly panel validation blocked the pilot")
    processed_path = output_dir / "processed" / "monthly_characteristic_panel.parquet"
    _, conversion_metadata = convert_monthly_panel(
        args.monthly_panel,
        processed_path,
        format=args.format,
        sep=args.sep,
        characteristics=args.characteristics,
        drop_missing_ret=True,
        min_price=args.min_price,
    )
    managed_dir = output_dir / "managed_portfolios"
    managed, _, managed_metadata = build_paper_managed_portfolios(
        processed_path,
        managed_dir,
        characteristics=args.characteristics,
        n_bins=args.n_bins,
        weighting=args.weighting,
        # Keep the full history available; start/end only define the OOS replay range.
        start_date=None,
        end_date=None,
        min_assets_per_bin=args.min_assets_per_bin,
        universe_method=args.universe_method,
        market_cap_coverage=args.market_cap_coverage,
        min_price=args.min_price,
        assume_characteristics_lagged=args.assume_characteristics_lagged,
    )
    if managed.empty:
        raise SystemExit("managed portfolio builder produced no returns")
    start_date = args.start_date or str(managed["date"].min().date())
    end_date = args.end_date or str(managed["date"].max().date())
    window_dir = output_dir / "windows_pca"
    export_metadata = export_pca_windows(
        managed_dir / "managed_portfolio_returns.parquet",
        window_dir,
        k_values=args.k_values,
        start_date=start_date,
        end_date=end_date,
        lookback_months=args.lookback_months,
        lambda_l1=args.lambda_l1,
        lambda_l2=args.lambda_l2,
        short_budget=args.short_budget,
        w_min=args.w_min,
        w_max=args.w_max,
        risk_free_rate=args.risk_free_rate,
        max_windows=args.max_windows,
        allow_short_lookback=args.allow_short_lookback,
        model_name="PCA-monthly-panel",
    )
    replay_dir = output_dir / "replay"
    rows = run_replay_directory(
        window_dir,
        replay_dir,
        backend=args.backend,
        write_summary=True,
        workers=args.workers,
    )
    metrics = _metrics(rows)
    manifest = {
        "input_path": str(args.monthly_panel),
        "source_data_committed": False,
        "date_range": {"start": report.get("min_date"), "end": report.get("max_date")},
        "selected_oos_range": {"start": start_date, "end": end_date},
        "number_of_months": report.get("number_of_months"),
        "average_stocks_per_month": report.get("average_assets_per_month"),
        "characteristics_used": managed_metadata.get("characteristics"),
        "managed_portfolios_created": managed_metadata.get("managed_portfolios_created"),
        "number_of_replay_windows": len(export_metadata["written_windows"]),
        "lookback_months": args.lookback_months,
        "effective_lookback_months": export_metadata.get("effective_lookback_months"),
        "pilot_only": export_metadata.get("pilot_only", False),
        "backend": args.backend,
        "metrics": metrics,
        "conversion_metadata": conversion_metadata,
        "claims": {
            "full_paper_replication": False,
            "old_solution_parity": False,
            "full_ipca": False,
            "full_ap_trees": False,
            "global_qp_speedup": False,
            "mean_cvar_changed": False,
        },
    }
    (output_dir / "pilot_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    _write_summary(output_dir, rows, metrics, manifest)
    print(
        f"managed_portfolios={manifest['managed_portfolios_created']} "
        f"windows={manifest['number_of_replay_windows']} output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
