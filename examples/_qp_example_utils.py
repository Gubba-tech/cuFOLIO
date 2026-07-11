"""Shared synthetic data and reporting helpers for the QP examples."""

from __future__ import annotations

import numpy as np

from cufolio.qp_backend import solve_compiled_qp_cuopt, solve_compiled_qp_osqp
from cufolio.qp_formulations import compile_portfolio_qp


def synthetic_returns(n_assets: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    design = rng.normal(size=(n_assets, n_assets))
    covariance = design.T @ design / n_assets + 0.02 * np.eye(n_assets)
    mean = rng.normal(0.0, 0.004, size=n_assets) + 0.03
    return {"mean": mean, "covariance": covariance}


def box_feasible_asset_count(n_assets: int, upper_bound: float = 0.08) -> int:
    return max(n_assets, int(np.ceil(1.0 / upper_bound)))


def solve_and_report(compiled, params, previous_weights=None, label=None) -> None:
    if params.backend == "osqp":
        solution = solve_compiled_qp_osqp(compiled)
    else:
        solution = solve_compiled_qp_cuopt(compiled)
    weights = compiled.recover_stock_weights(solution.x)
    if compiled.mapping_mode == "factor_space":
        factor_weights = compiled.recover_factor_weights(solution.x)
        expected_return = float(compiled.mean @ factor_weights)
        variance = float(factor_weights @ compiled.covariance @ factor_weights)
    else:
        factor_weights = compiled.recover_factor_weights(solution.x)
        expected_return = float(compiled.mean @ weights)
        variance = float(weights @ compiled.covariance @ weights)
    volatility = float(np.sqrt(max(variance, 0.0)))
    excess_return = expected_return - compiled.risk_free_rate
    sharpe = excess_return / volatility if volatility > 0 else np.nan

    if label:
        print(f"workflow: {label}")
    print(f"objective: {compiled.objective_value(solution.x):.10g}")
    print(f"backend: {solution.solver_name}")
    print(f"status: {solution.status}")
    print(f"expected_return: {expected_return:.10g}")
    print(f"variance: {variance:.10g}")
    print(f"volatility: {volatility:.10g}")
    print(f"sharpe: {sharpe:.10g}")
    print(f"gross_long: {np.maximum(weights, 0.0).sum():.10g}")
    print(f"gross_short: {np.maximum(-weights, 0.0).sum():.10g}")
    if previous_weights is not None:
        print(
            "turnover: "
            f"{np.abs(weights - np.asarray(previous_weights, dtype=float)).sum():.10g}"
        )
    print(f"max_constraint_violation: {solution.max_constraint_violation:.10g}")
    if compiled.objective == "max_sharpe":
        print(f"c_scale: {compiled.recover_scale(solution.x):.10g}")
        print(f"excess_return: {excess_return:.10g}")
        print(f"recovered_weights_sum: {weights.sum():.10g}")
    if compiled.mapping_mode == "factor_space":
        print(f"factor_weights_shape: {factor_weights.shape}")
        print(f"stock_weights_shape: {weights.shape}")
        print(f"mapping_V_shape: {compiled.stock_mapping.shape}")


def solve(returns_dict, params, previous_weights=None, label=None) -> None:
    compiled = compile_portfolio_qp(returns_dict, params)
    solve_and_report(compiled, params, previous_weights=previous_weights, label=label)
