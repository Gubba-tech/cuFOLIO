#!/usr/bin/env python3
"""Run short-budget and box-bound sensitivity for top PCA grid configurations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from export_pca_replay_windows import export_pca_windows
from summarize_monthly_panel_results import summarize_run

from cufolio.qp_paper_replay import (
    load_replay_window,
    run_replay_directory,
    solve_replay_window,
)


def _label(value: float) -> str:
    return f"{value:.8g}".replace("-", "m").replace(".", "p")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _activity(window_dir: Path, backend: str, w_min: float, w_max: float, short_budget: float) -> dict[str, float]:
    rows = []
    for path in sorted(window_dir.glob("*.npz")):
        result = solve_replay_window(load_replay_window(path), backend=backend)
        weights = result.stock_weights
        gross_short = float(np.maximum(-weights, 0.0).sum())
        active = int(
            np.isclose(weights, w_min, atol=1e-5).sum()
            + np.isclose(weights, w_max, atol=1e-5).sum()
        )
        rows.append(
            {
                "lower_bound_count": int(np.isclose(weights, w_min, atol=1e-5).sum()),
                "upper_bound_count": int(np.isclose(weights, w_max, atol=1e-5).sum()),
                "near_zero_count": int(np.isclose(weights, 0.0, atol=1e-5).sum()),
                "l1_norm": float(np.abs(weights).sum()),
                "l2_norm": float(np.linalg.norm(weights)),
                "gross_long": float(np.maximum(weights, 0.0).sum()),
                "gross_short": gross_short,
                "short_budget_binding": gross_short >= short_budget - 1e-5,
                "many_bounds_active": active >= max(5, int(0.1 * weights.size)),
            }
        )
    frame = pd.DataFrame(rows)
    return {
        "windows": len(frame),
        "average_lower_bound_count": frame["lower_bound_count"].mean(),
        "average_upper_bound_count": frame["upper_bound_count"].mean(),
        "average_near_zero_count": frame["near_zero_count"].mean(),
        "average_l1_norm": frame["l1_norm"].mean(),
        "average_l2_norm": frame["l2_norm"].mean(),
        "average_gross_long": frame["gross_long"].mean(),
        "average_gross_short": frame["gross_short"].mean(),
        "fraction_short_budget_binding": frame["short_budget_binding"].mean(),
        "fraction_many_bounds_active": frame["many_bounds_active"].mean(),
    }


def run_constraint_sensitivity(
    grid_results: str | Path,
    managed_portfolio_returns: str | Path,
    output_dir: str | Path,
    start_date: str,
    end_date: str,
    lookback_months: int = 240,
    backend: str = "osqp",
    top_n: int = 3,
    short_budgets: list[float] | None = None,
    bound_pairs: list[tuple[float, float]] | None = None,
) -> list[dict[str, object]]:
    frame = pd.read_csv(grid_results)
    sharpe_column = "annualized_sharpe" if "annualized_sharpe" in frame else "Sharpe"
    top = frame.sort_values(sharpe_column, ascending=False).head(top_n)
    short_budgets = short_budgets or [0.0, 0.1, 0.2, 0.5]
    bound_pairs = bound_pairs or [(-0.05, 0.05), (-0.08, 0.08), (-0.10, 0.10), (-0.20, 0.20)]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for _, selected in top.iterrows():
        base = f"k{int(selected['k'])}_l1_{_label(float(selected['lambda_l1']))}_l2_{_label(float(selected['lambda_l2']))}"
        for short_budget in short_budgets:
            for w_min, w_max in bound_pairs:
                config = f"{base}_sb_{_label(short_budget)}_b_{_label(w_min)}_{_label(w_max)}"
                window_dir = output / "windows" / config
                result_dir = output / "runs" / config
                export_pca_windows(
                    managed_portfolio_returns,
                    window_dir,
                    k_values=[int(selected["k"])],
                    start_date=start_date,
                    end_date=end_date,
                    lookback_months=lookback_months,
                    lambda_l1=float(selected["lambda_l1"]),
                    lambda_l2=float(selected["lambda_l2"]),
                    short_budget=short_budget,
                    w_min=w_min,
                    w_max=w_max,
                    allow_short_lookback=False,
                    model_name=f"PCA-constraint-sensitivity-{config}",
                )
                replay_rows = run_replay_directory(window_dir, result_dir, backend=backend, write_summary=True)
                metrics = summarize_run(result_dir, plots=False)
                activity = _activity(window_dir, backend, w_min, w_max, short_budget)
                for metric in metrics:
                    results.append(
                        {
                            **metric,
                            **activity,
                            "config_id": config,
                            "k": int(selected["k"]),
                            "lambda_l1": float(selected["lambda_l1"]),
                            "lambda_l2": float(selected["lambda_l2"]),
                            "short_budget": short_budget,
                            "w_min": w_min,
                            "w_max": w_max,
                            "replay_rows": len(replay_rows),
                            "artifact_path": str(result_dir),
                        }
                    )
    _write_csv(output / "constraint_sensitivity.csv", results)
    _write_csv(output / "bound_activity.csv", results)
    _write_csv(output / "regularization_activity.csv", results)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-results", type=Path, required=True)
    parser.add_argument("--managed-portfolio-returns", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-date", default="2020-01-31")
    parser.add_argument("--end-date", default="2022-12-31")
    parser.add_argument("--lookback-months", type=int, default=240)
    parser.add_argument("--backend", choices=("osqp", "cuopt"), default="osqp")
    parser.add_argument("--top-n", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = run_constraint_sensitivity(
        args.grid_results,
        args.managed_portfolio_returns,
        args.output_dir,
        args.start_date,
        args.end_date,
        args.lookback_months,
        args.backend,
        args.top_n,
    )
    print(f"constraint_rows={len(rows)} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
