import numpy as np
import pytest
from qp_test_utils import (
    assert_feasible_solution,
    assert_long_short_budget,
    assert_objective_gap_within,
    require_cuopt,
)

from cufolio.qp_backend import solve_compiled_qp_cuopt, solve_compiled_qp_osqp
from cufolio.qp_formulations import compile_portfolio_qp
from cufolio.qp_parameters import QPParameters


def _factor_inputs():
    mapping = np.array(
        [
            [0.7, 0.0, 0.0],
            [0.3, 0.2, 0.0],
            [0.0, 0.5, 0.1],
            [0.0, 0.3, 0.4],
            [0.0, 0.0, 0.4],
            [0.0, 0.0, 0.1],
        ]
    )
    returns_dict = {
        "mean": np.array([0.04, 0.03, 0.02]),
        "covariance": np.diag([0.04, 0.06, 0.08]),
    }
    return returns_dict, mapping


def _params(mapping, backend="osqp", **overrides):
    options = {
        "mapping_mode": "factor_space",
        "V": mapping,
        "objective": "max_sharpe",
        "risk_free_rate": 0.01,
        "w_min": 0.0,
        "w_max": 1.0,
        "backend": backend,
    }
    options.update(overrides)
    return QPParameters(**options)


def test_factor_space_max_sharpe_rows():
    returns_dict, mapping = _factor_inputs()
    compiled = compile_portfolio_qp(returns_dict, _params(mapping))
    scale = compiled.variable_slices["scale"]
    expected_excess = np.zeros(compiled.n_variables)
    expected_excess[:3] = returns_dict["mean"]
    expected_excess[scale.start] = -0.01
    expected_budget = np.zeros(compiled.n_variables)
    expected_budget[:3] = np.ones(6) @ mapping
    expected_budget[scale.start] = -1.0

    np.testing.assert_allclose(
        compiled.A_eq.getrow(0).toarray().reshape(-1),
        expected_excess,
    )
    np.testing.assert_allclose(
        compiled.A_eq.getrow(1).toarray().reshape(-1),
        expected_budget,
    )
    np.testing.assert_allclose(compiled.b_eq, [1.0, 0.0])


def test_factor_space_max_sharpe_recovery():
    returns_dict, mapping = _factor_inputs()
    compiled = compile_portfolio_qp(returns_dict, _params(mapping))
    solution = solve_compiled_qp_osqp(compiled)
    factor_weights = compiled.recover_factor_weights(solution.x)
    stock_weights = compiled.recover_stock_weights(solution.x)

    assert_feasible_solution(compiled, solution)
    np.testing.assert_allclose(stock_weights, mapping @ factor_weights)
    np.testing.assert_allclose(stock_weights.sum(), 1.0, atol=1e-6)
    assert returns_dict["mean"] @ factor_weights - 0.01 > 0.0


def test_factor_space_max_sharpe_box_long_short_l1_l2():
    returns_dict, mapping = _factor_inputs()
    compiled = compile_portfolio_qp(
        returns_dict,
        _params(
            mapping,
            risk_free_rate=0.0,
            w_min=-1.0,
            w_max=1.0,
            short_budget=0.2,
            lambda_l1=0.05,
            lambda_l2=0.05,
        ),
    )
    solution = solve_compiled_qp_osqp(compiled)
    stock_weights = compiled.recover_stock_weights(solution.x)

    assert_feasible_solution(compiled, solution, tol=1e-5)
    assert_long_short_budget(stock_weights, 0.2, tol=1e-5)
    np.testing.assert_allclose(stock_weights.sum(), 1.0, atol=1e-5)


@pytest.mark.gpu
def test_factor_space_max_sharpe_cuopt_matches_osqp_when_available():
    require_cuopt()
    returns_dict, mapping = _factor_inputs()
    osqp_compiled = compile_portfolio_qp(returns_dict, _params(mapping))
    cuopt_compiled = compile_portfolio_qp(
        returns_dict,
        _params(mapping, backend="cuopt"),
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
