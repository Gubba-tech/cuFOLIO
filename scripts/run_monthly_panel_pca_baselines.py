#!/usr/bin/env python3
"""Run managed-portfolio and PCA baseline comparisons on two OOS designs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from cufolio.qp_backend import solve_compiled_qp_cuopt, solve_compiled_qp_osqp
from cufolio.qp_factor_workflows import build_pca_factor_qp_data
from cufolio.qp_formulations import compile_portfolio_qp
from cufolio.qp_paper_replay import (
    PaperReplayWindow,
    ReplaySolveResult,
    compute_replay_diagnostics,
)
from cufolio.qp_parameters import QPParameters


def _load_returns(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.to_period("M").dt.to_timestamp("M")
    if {"portfolio_id", "ret"}.issubset(frame.columns):
        frame = frame.pivot(index="date", columns="portfolio_id", values="ret")
    else:
        frame = frame.set_index("date").select_dtypes(include=[np.number])
    return frame.sort_index().sort_index(axis=1)


def _configs() -> list[dict[str, object]]:
    return [
        {"name": "equal_weight_managed", "kind": "equal"},
        {"name": "min_variance_managed", "kind": "min_variance"},
        {"name": "mean_variance_ra_0p1", "kind": "mean_variance", "risk_aversion": 0.1},
        {"name": "mean_variance_ra_1", "kind": "mean_variance", "risk_aversion": 1.0},
        {"name": "mean_variance_ra_10", "kind": "mean_variance", "risk_aversion": 10.0},
        {"name": "pca_k6_unregularized", "kind": "pca", "lambda_l1": 0.0, "lambda_l2": 0.0},
        {"name": "pca_k6_l1_only", "kind": "pca", "lambda_l1": 1.7e-4, "lambda_l2": 0.0},
        {"name": "pca_k6_l2_only", "kind": "pca", "lambda_l1": 0.0, "lambda_l2": 1e-3},
        {"name": "pca_k6_l1_l2", "kind": "pca", "lambda_l1": 1.7e-4, "lambda_l2": 1e-3},
        {
            "name": "pca_k6_long_only",
            "kind": "pca",
            "lambda_l1": 1.7e-4,
            "lambda_l2": 1e-3,
            "long_only": True,
        },
        {"name": "pca_k6_long_short", "kind": "pca", "lambda_l1": 0.0, "lambda_l2": 0.0},
    ]


def _manual_row(config: dict[str, object], date: str, next_returns: np.ndarray) -> dict[str, object]:
    weights = np.full(next_returns.size, 1.0 / next_returns.size)
    return {
        "backend": "analytic",
        "model_name": str(config["name"]),
        "rebalance_date": date,
        "window_id": f"{config['name']}_{date}",
        "status": "optimal",
        "realized_next_return": float(weights @ next_returns),
        "sum_weights": float(weights.sum()),
        "gross_long": float(weights.sum()),
        "gross_short": 0.0,
        "short_budget_violation": 0.0,
        "box_violation": 0.0,
        "max_constraint_violation": 0.0,
        "turnover": None,
    }


def _solve_job(job: tuple[dict[str, object], str, np.ndarray, np.ndarray, str]) -> dict[str, object]:
    config, date, history, next_returns, backend = job
    name = str(config["name"])
    if config["kind"] == "equal":
        return _manual_row(config, date, next_returns)

    if config["kind"] == "pca":
        factor_data = build_pca_factor_qp_data(history, n_components=6, center=True)
        returns_dict = factor_data.to_returns_dict()
        mapping_mode = "factor_space"
        mean = None
        covariance = None
        mapping = factor_data.stock_mapping
        lambda_l1 = float(config["lambda_l1"])
        lambda_l2 = float(config["lambda_l2"])
        long_only = bool(config.get("long_only", False))
    else:
        mean = history.mean(axis=0)
        covariance = np.cov(history, rowvar=False)
        returns_dict = {
            "mean": mean,
            "covariance": covariance,
        }
        mapping_mode = "stock_space"
        mapping = np.eye(history.shape[1])
        lambda_l1 = 0.0
        lambda_l2 = 0.0
        long_only = False

    objective = str(config["kind"])
    if objective not in {"min_variance", "mean_variance"}:
        objective = "max_sharpe"
    params = QPParameters(
        objective=objective,
        risk_aversion=float(config.get("risk_aversion", 1.0)),
        risk_free_rate=0.0,
        w_min=0.0 if long_only else -0.08,
        w_max=0.08,
        short_budget=None if long_only else 0.2,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        mapping_mode=mapping_mode,
        V=mapping if mapping_mode == "factor_space" else None,
        backend=backend,
    )
    if mapping_mode == "factor_space":
        window_kwargs = {
            "factor_mean": returns_dict["factor_mean"],
            "factor_covariance": returns_dict["factor_covariance"],
            "stock_mapping": mapping,
            "factor_names": factor_data.factor_names,
        }
    else:
        window_kwargs = {"mean": mean, "covariance": covariance}
    window = PaperReplayWindow(
        schema_version="1.0",
        window_id=f"{name}_{date}",
        rebalance_date=date,
        model_name=name,
        objective=objective,
        mapping_mode=mapping_mode,
        risk_free_rate=0.0,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        short_budget=None if long_only else 0.2,
        w_min=0.0 if long_only else -0.08,
        w_max=0.08,
        realized_next_returns=next_returns,
        **window_kwargs,
    )
    compiled = compile_portfolio_qp(returns_dict, params)
    solver = solve_compiled_qp_cuopt if backend == "cuopt" else solve_compiled_qp_osqp
    solution = solver(compiled)
    weights = compiled.recover_stock_weights(solution.x)
    factor_weights = compiled.recover_factor_weights(solution.x)
    result = ReplaySolveResult(window, compiled, solution, weights, factor_weights)
    row = compute_replay_diagnostics(None, result, window)
    row.update(
        {
            "backend": backend,
            "model_name": name,
            "rebalance_date": date,
            "window_id": window.window_id,
        }
    )
    return row


def _safe_solve_job(job: tuple[dict[str, object], str, np.ndarray, np.ndarray, str]) -> dict[str, object]:
    """Convert an infeasible or backend error into a per-window failure row."""
    config, date, _history, _next_returns, backend = job
    try:
        return _solve_job(job)
    except Exception as exc:  # noqa: BLE001 - diagnostics must continue past bad windows
        name = str(config["name"])
        return {
            "backend": backend,
            "model_name": name,
            "rebalance_date": date,
            "window_id": f"{name}_{date}",
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "realized_next_return": None,
        }


def _metric_rows(rows: list[dict[str, object]], design: str) -> list[dict[str, object]]:
    try:
        from summarize_monthly_panel_results import metrics_for_rows
    except ModuleNotFoundError:
        from scripts.summarize_monthly_panel_results import metrics_for_rows

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["backend"]), str(row["model_name"]))].append(row)
    output = []
    for _, group in sorted(grouped.items()):
        metric = metrics_for_rows(group)
        metric["design"] = design
        metric["config_id"] = group[0]["model_name"]
        output.append(metric)
    return output


def run_design(
    returns: pd.DataFrame,
    start_date: str,
    end_date: str,
    lookback_months: int,
    backend: str,
    workers: int,
) -> tuple[str, list[dict[str, object]], list[dict[str, object]]]:
    start = pd.Timestamp(start_date).to_period("M").to_timestamp("M")
    end = pd.Timestamp(end_date).to_period("M").to_timestamp("M")
    jobs: list[tuple[dict[str, object], str, np.ndarray, np.ndarray, str]] = []
    for index, date in enumerate(returns.index[:-1]):
        if not start <= date <= end:
            continue
        effective = min(lookback_months, index + 1)
        if lookback_months == 240 and effective < lookback_months:
            continue
        if effective < 2:
            continue
        history = returns.iloc[index - effective + 1 : index + 1]
        next_returns = returns.iloc[index + 1]
        complete = history.notna().all() & next_returns.notna()
        history = history.loc[:, complete]
        next_returns = next_returns.loc[history.columns]
        if history.empty:
            continue
        for config in _configs():
            jobs.append((config, str(date.date()), history.to_numpy(float), next_returns.to_numpy(float), backend))
    if backend == "cuopt" and workers > 1:
        raise ValueError("use workers=1 for a single-GPU cuOpt baseline run")
    if workers > 1 or backend == "cuopt":
        context = __import__("multiprocessing").get_context("spawn")
        executor_kwargs = {"max_workers": workers, "mp_context": context}
        if backend == "cuopt":
            executor_kwargs["max_tasks_per_child"] = 20
        with ProcessPoolExecutor(**executor_kwargs) as executor:
            rows = list(executor.map(_safe_solve_job, jobs))
    else:
        rows = [_safe_solve_job(job) for job in jobs]
    design = f"{start_date}_{end_date}_{lookback_months}m"
    return design, rows, _metric_rows(rows, design)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_baselines(
    managed_portfolio_returns: str | Path,
    output_dir: str | Path,
    backend: str = "osqp",
    workers: int = 1,
) -> list[dict[str, object]]:
    returns = _load_returns(Path(managed_portfolio_returns))
    all_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for design_args in (("2020-01-31", "2022-12-31", 240), ("2005-01-31", "2022-12-31", 60)):
        design, rows, metrics = run_design(returns, *design_args, backend=backend, workers=workers)
        for row in rows:
            row["design"] = design
        all_rows.extend(rows)
        metric_rows.extend(metrics)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "baseline_per_window_results.csv", all_rows)
    _write_csv(output / "baseline_metrics.csv", metric_rows)
    frame = pd.DataFrame(all_rows)
    if not frame.empty:
        frame["date"] = pd.to_datetime(frame["rebalance_date"])
        pivot = frame.pivot_table(index="date", columns="model_name", values="realized_next_return", aggfunc="first").sort_index()
        pivot.to_csv(output / "baseline_cumulative_returns.csv", index=True)
    lines = [
        "# Monthly Panel PCA Baselines",
        "",
        "These are diagnostic uploaded-panel baselines, not full paper replication.",
        "",
        "| design | config | backend | CAGR | annualized vol | annualized Sharpe | max drawdown | optimal | failed |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metric_rows:
        lines.append(
            f"| {row.get('design')} | {row.get('config_id')} | {row.get('backend')} | "
            f"{row.get('cagr')} | {row.get('annualized_volatility')} | {row.get('annualized_sharpe', row.get('Sharpe'))} | "
            f"{row.get('max_drawdown')} | {row.get('optimal_count')} | {row.get('failed_count')} |"
        )
    (output / "baseline_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metric_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--managed-portfolio-returns", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("osqp", "cuopt"), default="osqp")
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = run_baselines(args.managed_portfolio_returns, args.output_dir, args.backend, args.workers)
    print(f"baseline_rows={len(rows)} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
