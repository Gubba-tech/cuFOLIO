import importlib.util

import numpy as np
import pandas as pd
import pytest
from qp_test_utils import (
    assert_feasible_solution,
    assert_objective_gap_within,
    require_cuopt,
    small_returns_dict,
)

from cufolio.exceptions import GPUBackendUnavailable
from cufolio.qp_backend import solve_compiled_qp_cuopt, solve_compiled_qp_osqp
from cufolio.qp_formulations import compile_portfolio_qp
from cufolio.qp_optimizer import QuadraticPortfolioOptimizer
from cufolio.qp_parameters import QPParameters


def _interior_returns_dict():
    excess = np.array([0.04, 0.06, 0.08])
    covariance = np.diag([0.04, 0.09, 0.16])
    return {"mean": excess, "covariance": covariance}


def _max_sharpe_params(backend="osqp", **overrides):
    options = {
        "objective": "max_sharpe",
        "w_min": 0.0,
        "w_max": 1.0,
        "risk_free_rate": 0.0,
        "backend": backend,
    }
    options.update(overrides)
    return QPParameters(**options)


def test_max_sharpe_base_compiler_rows():
    compiled = compile_portfolio_qp(
        _interior_returns_dict(),
        _max_sharpe_params(),
    )
    decision = compiled.variable_slices["decision"]
    scale = compiled.variable_slices["scale"]

    assert decision.size == 3
    assert scale.size == 1
    assert compiled.n_variables == 4
    assert compiled.constraint_names[:2] == [
        "max_sharpe_excess_return",
        "scaled_budget",
    ]
    np.testing.assert_allclose(
        compiled.A_eq.getrow(0).toarray().reshape(-1),
        [0.04, 0.06, 0.08, 0.0],
    )
    np.testing.assert_allclose(
        compiled.A_eq.getrow(1).toarray().reshape(-1),
        [1.0, 1.0, 1.0, -1.0],
    )
    np.testing.assert_allclose(compiled.b_eq, [1.0, 0.0])
    assert compiled.lower[scale.start] > 0.0


def test_max_sharpe_recovered_weights_sum_to_one():
    compiled = compile_portfolio_qp(
        small_returns_dict(),
        _max_sharpe_params(),
    )
    solution = solve_compiled_qp_osqp(compiled)
    weights = compiled.recover_max_sharpe_weights(solution.x)

    assert_feasible_solution(compiled, solution)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)
    assert (compiled.mean - compiled.risk_free_rate) @ weights > 0.0
    assert compiled.recover_scale(solution.x) > 0.0


def test_max_sharpe_matches_closed_form_unconstrained_or_long_only_case():
    returns_dict = _interior_returns_dict()
    compiled = compile_portfolio_qp(
        returns_dict,
        _max_sharpe_params(),
    )
    solution = solve_compiled_qp_osqp(compiled)
    weights = compiled.recover_stock_weights(solution.x)
    excess = returns_dict["mean"]
    tangent = np.linalg.solve(returns_dict["covariance"], excess)
    tangent /= tangent.sum()

    np.testing.assert_allclose(weights, tangent, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
def test_max_sharpe_cuopt_matches_osqp_when_available():
    require_cuopt()
    returns_dict = _interior_returns_dict()
    osqp_compiled = compile_portfolio_qp(returns_dict, _max_sharpe_params())
    cuopt_compiled = compile_portfolio_qp(
        returns_dict,
        _max_sharpe_params(backend="cuopt"),
    )
    osqp_solution = solve_compiled_qp_osqp(osqp_compiled)
    cuopt_solution = solve_compiled_qp_cuopt(cuopt_compiled)

    assert_feasible_solution(osqp_compiled, osqp_solution)
    assert_feasible_solution(cuopt_compiled, cuopt_solution, tol=1e-5)
    assert_objective_gap_within(
        cuopt_compiled,
        cuopt_solution,
        osqp_solution,
        tol=5e-4,
    )
    np.testing.assert_allclose(
        cuopt_compiled.recover_stock_weights(cuopt_solution.x),
        osqp_compiled.recover_stock_weights(osqp_solution.x),
        atol=1e-4,
        rtol=1e-4,
    )


def test_max_sharpe_cuopt_backend_never_calls_cpu_solver(monkeypatch):
    original_find_spec = importlib.util.find_spec

    def find_spec_without_cuopt(name, *args, **kwargs):
        if name == "cuopt":
            return None
        return original_find_spec(name, *args, **kwargs)

    compiled = compile_portfolio_qp(
        small_returns_dict(),
        _max_sharpe_params(backend="cuopt"),
    )
    monkeypatch.setattr(importlib.util, "find_spec", find_spec_without_cuopt)

    with pytest.raises(GPUBackendUnavailable):
        solve_compiled_qp_cuopt(compiled)


def test_max_sharpe_optimizer_reports_scaled_solution_fields():
    returns_dict = _interior_returns_dict()
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    returns_dict.update(
        {
            "return_type": "LOG",
            "returns": pd.DataFrame(
                np.tile(returns_dict["mean"], (3, 1)),
                index=dates,
                columns=["A", "B", "C"],
            ),
            "regime": {"name": "toy", "range": (dates[0], dates[-1])},
            "dates": dates,
            "tickers": ["A", "B", "C"],
        }
    )
    optimizer = QuadraticPortfolioOptimizer(
        returns_dict,
        _max_sharpe_params(),
    )
    result, portfolio = optimizer.solve_optimization_problem(print_results=False)

    assert result["c_scale"] > 0.0
    assert result["excess_return"] > 0.0
    assert result["raw_status"]
    np.testing.assert_allclose(result["recovered_weights"], portfolio.weights)
    np.testing.assert_allclose(result["stock_weights"], portfolio.weights)
