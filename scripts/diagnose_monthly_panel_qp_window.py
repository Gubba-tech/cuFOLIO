#!/usr/bin/env python3
"""Diagnose one saved monthly-panel PCA max-Sharpe QP window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cufolio.qp_paper_replay import load_replay_window, solve_replay_window


def _constraint_row(compiled, name: str) -> list[float] | None:
    if name not in compiled.constraint_names:
        return None
    index = compiled.constraint_names.index(name)
    if index < compiled.A_eq.shape[0]:
        row = compiled.A_eq.getrow(index).toarray().reshape(-1)
    else:
        row = compiled.A_ineq.getrow(index - compiled.A_eq.shape[0]).toarray().reshape(-1)
    return row.tolist()


def diagnose_window(window_file: str | Path, backend: str = "osqp") -> dict[str, object]:
    window = load_replay_window(window_file)
    result = solve_replay_window(window, backend=backend)
    compiled = result.compiled
    weights = result.stock_weights
    lower = np.broadcast_to(window.w_min, window.n_assets)
    upper = np.broadcast_to(window.w_max, window.n_assets)
    covariance = np.asarray(window.factor_covariance, dtype=float)
    eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
    factor_weights = result.factor_weights
    if factor_weights is None:
        expected_return = float(weights @ window.mean)
        variance = float(weights @ window.covariance @ weights)
    else:
        expected_return = float(factor_weights @ window.factor_mean)
        variance = float(factor_weights @ window.factor_covariance @ factor_weights)
    scale = compiled.recover_scale(result.solution.x)
    realized = (
        float(weights @ window.realized_next_returns)
        if window.realized_next_returns is not None
        else None
    )
    names = np.asarray(window.tickers, dtype=str)
    diagnostics = {
        "window_file": str(window_file),
        "window_id": window.window_id,
        "rebalance_date": window.rebalance_date,
        "backend": backend,
        "solver": result.solution.solver_name,
        "status": result.solution.status,
        "raw_status": result.solution.raw_status,
        "objective": window.objective,
        "mapping_mode": window.mapping_mode,
        "k": window.n_factors,
        "lambda_l1": window.lambda_l1,
        "lambda_l2": window.lambda_l2,
        "risk_free_rate": window.risk_free_rate,
        "short_budget": window.short_budget,
        "w_min": lower.tolist(),
        "w_max": upper.tolist(),
        "factor_mean": window.factor_mean.tolist() if window.factor_mean is not None else None,
        "factor_covariance_eigenvalues": eigenvalues.tolist(),
        "factor_covariance_condition_number": float(np.linalg.cond(covariance)),
        "max_sharpe_excess_return_row": _constraint_row(compiled, "max_sharpe_excess_return"),
        "fully_invested_row": _constraint_row(compiled, "fully_invested"),
        "scaled_budget_row": _constraint_row(compiled, "scaled_budget"),
        "c_scale": float(scale) if scale is not None else None,
        "factor_weights": factor_weights.tolist() if factor_weights is not None else None,
        "recovered_managed_weights_sum": float(weights.sum()),
        "expected_monthly_return_training": expected_return,
        "variance_training": variance,
        "implied_training_sharpe": (
            (expected_return - window.risk_free_rate) / np.sqrt(variance)
            if variance > 0
            else None
        ),
        "realized_next_month_return": realized,
        "gross_long": float(np.maximum(weights, 0.0).sum()),
        "gross_short": float(np.maximum(-weights, 0.0).sum()),
        "number_weights_at_lower_bound": int(np.isclose(weights, lower, atol=1e-6).sum()),
        "number_weights_at_upper_bound": int(np.isclose(weights, upper, atol=1e-6).sum()),
        "number_near_zero_weights": int(np.isclose(weights, 0.0, atol=1e-6).sum()),
        "l1_norm": float(np.abs(weights).sum()),
        "l2_norm": float(np.linalg.norm(weights)),
        "max_constraint_violation": float(result.solution.max_constraint_violation),
        "solve_time": result.solution.solve_time,
        "total_time": result.solution.total_time,
        "managed_portfolio_names": names.tolist(),
    }
    return diagnostics


def _write_markdown(path: Path, diagnostics: dict[str, object]) -> None:
    scalar_items = [
        (key, value)
        for key, value in diagnostics.items()
        if not isinstance(value, (list, dict))
    ]
    lines = ["# Monthly-Panel QP Window Diagnostics", ""]
    lines.extend(f"- **{key}**: `{value}`" for key, value in scalar_items)
    lines.extend(
        [
            "",
            "## Max-Sharpe Rows",
            "",
            "```json",
            json.dumps(
                {
                    key: diagnostics[key]
                    for key in (
                        "max_sharpe_excess_return_row",
                        "fully_invested_row",
                        "scaled_budget_row",
                    )
                },
                indent=2,
            ),
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("osqp", "cuopt"), default="osqp")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    diagnostics = diagnose_window(args.window_file, backend=args.backend)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "window_qp_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )
    _write_markdown(args.output_dir / "window_qp_diagnostics.md", diagnostics)
    print(f"diagnosed_window={diagnostics['window_id']} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
