#!/usr/bin/env python3
"""Rank monthly-panel PCA grid results and generate matplotlib heatmaps."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _read_grid(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"config_id", "k", "lambda_l1", "lambda_l2", "backend"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"grid_results.csv is missing columns: {sorted(missing)}")
    for column in ("k", "lambda_l1", "lambda_l2", "annualized_sharpe", "Sharpe", "cagr", "max_drawdown", "Calmar"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "annualized_sharpe" not in frame:
        frame["annualized_sharpe"] = frame["Sharpe"]
    return frame


def _plot_heatmap(frame: pd.DataFrame, metric: str, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = frame.pivot_table(index="lambda_l1", columns="lambda_l2", values=metric, aggfunc="mean")
    if pivot.empty:
        return
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(pivot.to_numpy(dtype=float), aspect="auto", interpolation="nearest")
    axis.set_xticks(range(len(pivot.columns)), [f"{value:g}" for value in pivot.columns])
    axis.set_yticks(range(len(pivot.index)), [f"{value:g}" for value in pivot.index])
    axis.set_xlabel("lambda_l2")
    axis.set_ylabel("lambda_l1")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label=metric)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _plot_best_returns(best: pd.Series, output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    artifact = best.get("artifact_path")
    if not artifact:
        return []
    returns_path = Path(str(artifact)) / "monthly_returns.csv"
    if not returns_path.exists():
        return []
    returns = pd.read_csv(returns_path, parse_dates=["date"])
    columns = [column for column in returns.columns if column != "date"]
    if not columns:
        return []
    values = pd.to_numeric(returns[columns[0]], errors="coerce").fillna(0.0).to_numpy()
    dates = returns["date"]
    cumulative = np.cumprod(1.0 + values)
    generated = []
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(dates, cumulative)
    axis.set_title(f"Best grid configuration cumulative returns: {best['config_id']}")
    figure.autofmt_xdate()
    figure.savefig(output_dir / "best_config_cumulative_returns.png", bbox_inches="tight")
    plt.close(figure)
    generated.append("best_config_cumulative_returns.png")

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(dates, cumulative / np.maximum.accumulate(cumulative) - 1.0)
    axis.set_title("Best grid configuration underwater")
    figure.autofmt_xdate()
    figure.savefig(output_dir / "best_config_underwater.png", bbox_inches="tight")
    plt.close(figure)
    generated.append("best_config_underwater.png")

    figure, axis = plt.subplots(figsize=(10, 4))
    image = axis.imshow(values.reshape(1, -1), aspect="auto", interpolation="nearest")
    axis.set_yticks([0], ["monthly return"])
    axis.set_title("Best grid configuration monthly returns")
    figure.colorbar(image, ax=axis)
    figure.savefig(output_dir / "best_config_monthly_returns_heatmap.png", bbox_inches="tight")
    plt.close(figure)
    generated.append("best_config_monthly_returns_heatmap.png")
    return generated


def summarize_grid(
    grid_results: str | Path,
    output_dir: str | Path,
    max_drawdown_threshold: float = -0.25,
    constraint_tolerance: float = 1e-5,
) -> dict[str, object]:
    frame = _read_grid(grid_results)
    output = Path(output_dir)
    heatmap_dir = output / "heatmap_data"
    plot_dir = output / "plots"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    frame["rank_by_annualized_sharpe"] = frame["annualized_sharpe"].rank(ascending=False, method="min")
    frame["rank_by_calmar"] = frame["Calmar"].rank(ascending=False, method="min") if "Calmar" in frame else np.nan
    ranked = frame.sort_values(["annualized_sharpe", "Calmar"], ascending=[False, False])
    ranked.to_csv(output / "grid_results_ranked.csv", index=False)

    generated = []
    for k in sorted(frame["k"].dropna().astype(int).unique()):
        k_frame = frame[frame["k"] == k]
        k_frame[["lambda_l1", "lambda_l2", "annualized_sharpe"]].to_csv(
            heatmap_dir / f"heatmap_K{k}.csv", index=False
        )
        for backend, backend_frame in k_frame.groupby("backend"):
            suffix = f"_{backend}" if frame["backend"].nunique() > 1 else ""
            sharpe_name = f"sharpe_heatmap_K{k}{suffix}.png"
            _plot_heatmap(
                backend_frame,
                "annualized_sharpe",
                plot_dir / sharpe_name,
                f"Annualized Sharpe heatmap, K={k}, {backend}",
            )
            generated.append(f"plots/{sharpe_name}")
    k6 = frame[frame["k"] == frame["k"].max()]
    if not k6.empty:
        for metric, filename in (
            ("max_drawdown", "max_drawdown_heatmap_K6.png"),
            ("annualized_return", "annual_return_heatmap_K6.png"),
        ):
            if metric in k6:
                backend = k6["backend"].iloc[0]
                _plot_heatmap(k6[k6["backend"] == backend], metric, plot_dir / filename, f"{metric} heatmap, K=6")
                generated.append(f"plots/{filename}")

    best_sharpe = ranked.iloc[0] if not ranked.empty else pd.Series(dtype=object)
    best_calmar = frame.sort_values("Calmar", ascending=False).iloc[0] if "Calmar" in frame and frame["Calmar"].notna().any() else pd.Series(dtype=object)
    constrained = frame[
        (frame["max_drawdown"] >= max_drawdown_threshold)
        & (frame["maximum_constraint_violation"] <= constraint_tolerance)
    ] if {"max_drawdown", "maximum_constraint_violation"}.issubset(frame.columns) else frame.iloc[0:0]
    best_constrained = constrained.sort_values("annualized_sharpe", ascending=False).iloc[0] if not constrained.empty else pd.Series(dtype=object)
    chosen = frame[
        (frame["k"] == 6)
        & np.isclose(frame["lambda_l1"], 1.7e-4)
        & np.isclose(frame["lambda_l2"], 1e-3)
    ]
    chosen_rank = int(chosen["rank_by_annualized_sharpe"].iloc[0]) if not chosen.empty else None
    generated.extend(_plot_best_returns(best_sharpe, plot_dir))

    def record(row: pd.Series) -> dict[str, object] | None:
        return None if row.empty else {key: row.get(key) for key in ("config_id", "backend", "k", "lambda_l1", "lambda_l2", "annualized_sharpe", "cagr", "max_drawdown", "Calmar", "artifact_path")}

    summary = {
        "grid_results": str(grid_results),
        "rows": int(len(frame)),
        "best_by_annualized_sharpe": record(best_sharpe),
        "best_by_calmar": record(best_calmar),
        "best_with_drawdown_and_constraint_filters": record(best_constrained),
        "sprint15_chosen_config_rank_by_annualized_sharpe": chosen_rank,
        "max_drawdown_threshold": max_drawdown_threshold,
        "constraint_tolerance": constraint_tolerance,
        "plots": generated,
    }
    lines = ["# Monthly Panel PCA Grid Summary", "", f"Rows: {len(frame)}", ""]
    for label, row in (
        ("Best by annualized Sharpe", best_sharpe),
        ("Best by Calmar", best_calmar),
        ("Best within drawdown/constraint filters", best_constrained),
    ):
        lines.append(f"## {label}")
        lines.append("")
        lines.append(f"`{record(row)}`")
        lines.append("")
    lines.append(f"Sprint 15 chosen K=6, lambda_l1=1.7e-4, lambda_l2=1e-3 rank: `{chosen_rank}`.")
    lines.append("")
    lines.append("This grid is empirical uploaded-panel evidence, not full paper replication or old-solution parity.")
    (output / "grid_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "plot_manifest.txt").write_text("\n".join(generated) + ("\n" if generated else ""), encoding="utf-8")
    (output / "grid_summary.json").write_text(pd.Series(summary).to_json(indent=2) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-drawdown-threshold", type=float, default=-0.25)
    parser.add_argument("--constraint-tolerance", type=float, default=1e-5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = summarize_grid(
        args.grid_results,
        args.output_dir,
        max_drawdown_threshold=args.max_drawdown_threshold,
        constraint_tolerance=args.constraint_tolerance,
    )
    print(f"grid_rows={summary['rows']} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
