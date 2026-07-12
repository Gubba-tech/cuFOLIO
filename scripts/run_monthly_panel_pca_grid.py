#!/usr/bin/env python3
"""Run PCA K/lambda sensitivity grids on managed-portfolio returns."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
from export_pca_replay_windows import export_pca_windows
from summarize_monthly_panel_results import summarize_run

from cufolio.qp_monthly_panel import (
    build_paper_managed_portfolios,
    convert_monthly_panel,
)
from cufolio.qp_paper_replay import run_replay_directory


def _parse_values(values: list[str]) -> list[float]:
    parsed = []
    for value in values:
        parsed.extend(float(part) for part in value.split(",") if part)
    return parsed


def _label(value: float) -> str:
    return f"{value:.8g}".replace("-", "m").replace(".", "p")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_grid_plots(output_dir: Path, rows: list[dict[str, object]]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    generated = []
    frame = pd.DataFrame(rows)
    if frame.empty or "Sharpe" not in frame:
        return generated
    frame["Sharpe"] = pd.to_numeric(frame["Sharpe"], errors="coerce")
    frame["k"] = pd.to_numeric(frame["k"], errors="coerce")
    frame["lambda_l1"] = pd.to_numeric(frame["lambda_l1"], errors="coerce")
    frame["lambda_l2"] = pd.to_numeric(frame["lambda_l2"], errors="coerce")
    first_l1 = frame["lambda_l1"].dropna().min()
    first_l2 = frame["lambda_l2"].dropna().min()
    k_frame = frame[(frame["lambda_l1"] == first_l1) & (frame["lambda_l2"] == first_l2)]
    figure, axis = plt.subplots(figsize=(8, 5))
    for backend, group in k_frame.groupby("backend"):
        group = group.sort_values("k")
        axis.plot(group["k"], group["Sharpe"], marker="o", label=str(backend))
    axis.set_xlabel("PCA K")
    axis.set_ylabel("Sharpe")
    axis.set_title("PCA K sensitivity")
    axis.legend()
    figure.savefig(output_dir / "k_sensitivity_sharpe.png", bbox_inches="tight")
    plt.close(figure)
    generated.append("k_sensitivity_sharpe.png")

    selected_k = int(frame["k"].dropna().max())
    lambda_frame = frame[frame["k"] == selected_k]
    for backend, group in lambda_frame.groupby("backend"):
        pivot = group.pivot_table(index="lambda_l1", columns="lambda_l2", values="Sharpe", aggfunc="mean")
        if pivot.empty:
            continue
        figure, axis = plt.subplots(figsize=(8, 6))
        image = axis.imshow(pivot.to_numpy(dtype=float), aspect="auto", interpolation="nearest")
        axis.set_xticks(range(len(pivot.columns)), [f"{value:g}" for value in pivot.columns])
        axis.set_yticks(range(len(pivot.index)), [f"{value:g}" for value in pivot.index])
        axis.set_xlabel("lambda_l2")
        axis.set_ylabel("lambda_l1")
        axis.set_title(f"Lambda-grid Sharpe ({backend}, K={selected_k})")
        figure.colorbar(image, ax=axis)
        filename = f"lambda_grid_sharpe_{backend}.png"
        figure.savefig(output_dir / filename, bbox_inches="tight")
        generated.append(filename)
        if frame["backend"].dropna().nunique() == 1:
            generic_filename = "lambda_grid_sharpe.png"
            figure.savefig(output_dir / generic_filename, bbox_inches="tight")
            generated.append(generic_filename)
        plt.close(figure)
    return generated


def run_grid(
    managed_portfolio_returns: str | Path,
    output_dir: str | Path,
    start_date: str,
    end_date: str,
    lookback_months: int,
    k_values: list[int],
    lambda_l1_values: list[float],
    lambda_l2_values: list[float],
    backend: str = "osqp",
    max_windows: int | None = None,
    workers: int = 1,
) -> list[dict[str, object]]:
    """Run a grid and write per-config and aggregate artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate: list[dict[str, object]] = []
    for k in k_values:
        for lambda_l1 in lambda_l1_values:
            for lambda_l2 in lambda_l2_values:
                config = f"k{k}_l1_{_label(lambda_l1)}_l2_{_label(lambda_l2)}"
                window_dir = output_dir / "windows" / config
                result_dir = output_dir / "runs" / config
                export_pca_windows(
                    managed_portfolio_returns,
                    window_dir,
                    k_values=[k],
                    start_date=start_date,
                    end_date=end_date,
                    lookback_months=lookback_months,
                    lambda_l1=lambda_l1,
                    lambda_l2=lambda_l2,
                    short_budget=0.2,
                    w_min=-0.08,
                    w_max=0.08,
                    risk_free_rate=0.0,
                    max_windows=max_windows,
                    allow_short_lookback=lookback_months != 240,
                    model_name=f"PCA-monthly-panel-K{k}",
                )
                replay_rows = run_replay_directory(
                    window_dir,
                    result_dir,
                    backend=backend,
                    write_summary=True,
                    workers=workers,
                )
                metrics = summarize_run(result_dir, plots=False)
                for metric in metrics:
                    aggregate.append(
                        {
                            **metric,
                            "config": config,
                            "k": k,
                            "lambda_l1": lambda_l1,
                            "lambda_l2": lambda_l2,
                            "lookback_months": lookback_months,
                            "replay_rows": len(replay_rows),
                            "result_dir": str(result_dir),
                        }
                    )
    _write_csv(output_dir / "grid_results.csv", aggregate)
    _write_csv(output_dir / "heatmap_data.csv", aggregate)
    generated = _write_grid_plots(output_dir, aggregate)
    lines = [
        "# Monthly Panel PCA Grid Summary",
        "",
        "Grid results are pilot evidence and are not full paper replication.",
        "",
        f"- Configurations: {len({row['config'] for row in aggregate})}",
        f"- Backends represented: {sorted({row['backend'] for row in aggregate})}",
        f"- Plots: {', '.join(generated) if generated else 'none'}",
        "",
        "| config | backend | K | lambda_l1 | lambda_l2 | Sharpe | CAGR | optimal | skipped | failed | max violation |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['config']} | {row['backend']} | {row['k']} | {row['lambda_l1']} | "
            f"{row['lambda_l2']} | {row.get('Sharpe')} | {row.get('cagr')} | "
            f"{row.get('optimal_count')} | {row.get('skipped_count')} | {row.get('failed_count')} | "
            f"{row.get('maximum_constraint_violation')} |"
        )
    (output_dir / "grid_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "plot_manifest.txt").write_text(
        "\n".join(generated) + ("\n" if generated else ""), encoding="utf-8"
    )
    return aggregate


def _prepare_managed(args: argparse.Namespace) -> Path:
    if args.managed_portfolio_returns:
        return args.managed_portfolio_returns
    if not args.monthly_panel:
        raise ValueError("provide --managed-portfolio-returns or --monthly-panel")
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
    )
    return managed_dir / "managed_portfolio_returns.parquet"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--managed-portfolio-returns", type=Path)
    parser.add_argument("--monthly-panel", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--format", choices=("auto", "csv", "tsv", "parquet"), default="auto")
    parser.add_argument("--sep", choices=("auto", "comma", "tab"), default="auto")
    parser.add_argument("--characteristics", default="all")
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--weighting", choices=("equal", "value"), default="value")
    parser.add_argument("--min-assets-per-bin", type=int, default=5)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--lookback-months", type=int, default=240)
    parser.add_argument("--k-values", nargs="+", type=int, default=[6])
    parser.add_argument("--lambda-l1-values", nargs="+", default=["1.7e-4"])
    parser.add_argument("--lambda-l2-values", nargs="+", default=["1e-3"])
    parser.add_argument("--backend", choices=("osqp", "cuopt", "both"), default="osqp")
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    managed = _prepare_managed(args)
    rows = run_grid(
        managed,
        args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        lookback_months=args.lookback_months,
        k_values=args.k_values,
        lambda_l1_values=_parse_values(args.lambda_l1_values),
        lambda_l2_values=_parse_values(args.lambda_l2_values),
        backend=args.backend,
        max_windows=args.max_windows,
        workers=args.workers,
    )
    print(f"grid_rows={len(rows)} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
