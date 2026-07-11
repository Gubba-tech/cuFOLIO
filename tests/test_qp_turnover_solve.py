import numpy as np
import pytest
from qp_test_utils import (
    assert_feasible_solution,
    assert_objective_gap_within,
    assert_turnover_budget,
    require_cuopt,
    small_returns_dict,
)

from cufolio.qp_backend import solve_compiled_qp_cuopt, solve_compiled_qp_osqp
from cufolio.qp_formulations import compile_portfolio_qp
from cufolio.qp_parameters import QPParameters


def _previous():
    return np.ones(4) / 4


def _params(backend="osqp", **overrides):
    options = {
        "objective": "mean_variance",
        "risk_aversion": 2.0,
        "previous_weights": _previous(),
        "turnover_budget": 0.25,
        "backend": backend,
    }
    options.update(overrides)
    return QPParameters(**options)


def test_turnover_osqp_solution_is_feasible():
    compiled = compile_portfolio_qp(small_returns_dict(), _params())
    solution = solve_compiled_qp_osqp(compiled)
    weights = compiled.recover_stock_weights(solution.x)

    assert_feasible_solution(compiled, solution)
    assert_turnover_budget(weights, _previous(), 0.25)


@pytest.mark.gpu
def test_turnover_cuopt_matches_osqp_when_available():
    require_cuopt()
    returns_dict = small_returns_dict()
    osqp_compiled = compile_portfolio_qp(returns_dict, _params())
    cuopt_compiled = compile_portfolio_qp(
        returns_dict,
        _params(backend="cuopt"),
    )
    osqp_solution = solve_compiled_qp_osqp(osqp_compiled)
    cuopt_solution = solve_compiled_qp_cuopt(cuopt_compiled)
    cuopt_weights = cuopt_compiled.recover_stock_weights(cuopt_solution.x)

    assert_feasible_solution(osqp_compiled, osqp_solution)
    assert_feasible_solution(cuopt_compiled, cuopt_solution, tol=1e-5)
    assert_objective_gap_within(
        cuopt_compiled,
        cuopt_solution,
        osqp_solution,
        tol=5e-4,
    )
    assert_turnover_budget(cuopt_weights, _previous(), 0.25, tol=1e-5)


def test_max_sharpe_turnover_osqp_solution_is_feasible():
    compiled = compile_portfolio_qp(
        small_returns_dict(),
        _params(objective="max_sharpe", w_min=0.0, w_max=1.0),
    )
    solution = solve_compiled_qp_osqp(compiled)
    weights = compiled.recover_stock_weights(solution.x)

    assert_feasible_solution(compiled, solution)
    assert_turnover_budget(weights, _previous(), 0.25)


@pytest.mark.gpu
def test_max_sharpe_turnover_cuopt_matches_osqp_when_available():
    require_cuopt()
    returns_dict = small_returns_dict()
    options = {"objective": "max_sharpe", "w_min": 0.0, "w_max": 1.0}
    osqp_compiled = compile_portfolio_qp(returns_dict, _params(**options))
    cuopt_compiled = compile_portfolio_qp(
        returns_dict,
        _params(backend="cuopt", **options),
    )
    osqp_solution = solve_compiled_qp_osqp(osqp_compiled)
    cuopt_solution = solve_compiled_qp_cuopt(cuopt_compiled)
    cuopt_weights = cuopt_compiled.recover_stock_weights(cuopt_solution.x)

    assert_feasible_solution(osqp_compiled, osqp_solution)
    assert_feasible_solution(cuopt_compiled, cuopt_solution, tol=1e-5)
    assert_objective_gap_within(
        cuopt_compiled,
        cuopt_solution,
        osqp_solution,
        tol=5e-4,
    )
    assert_turnover_budget(cuopt_weights, _previous(), 0.25, tol=1e-5)
