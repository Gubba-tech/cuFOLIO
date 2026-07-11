import numpy as np
import pytest
from qp_test_utils import (
    assert_feasible_solution,
    assert_objective_gap_within,
    require_cuopt,
    small_returns_dict,
)

from cufolio.qp_backend import solve_compiled_qp_cuopt, solve_compiled_qp_osqp
from cufolio.qp_formulations import compile_portfolio_qp
from cufolio.qp_parameters import QPParameters


def _params(backend="osqp", **overrides):
    options = {
        "objective": "max_sharpe",
        "w_min": 0.0,
        "w_max": 1.0,
        "backend": backend,
    }
    options.update(overrides)
    return QPParameters(**options)


def test_max_sharpe_l2_q_increment_identity():
    returns_dict = small_returns_dict()
    base = compile_portfolio_qp(returns_dict, _params())
    lambda_l2 = 0.35
    regularized = compile_portfolio_qp(
        returns_dict,
        _params(lambda_l2=lambda_l2),
    )
    decision = regularized.variable_slices["decision"].slice

    np.testing.assert_allclose(
        regularized.Q[decision, decision].toarray()
        - base.Q[decision, decision].toarray(),
        2.0 * lambda_l2 * np.eye(decision.stop - decision.start),
        atol=1e-12,
    )


def test_max_sharpe_l1_auxiliary_identity():
    lambda_l1 = 0.25
    compiled = compile_portfolio_qp(
        small_returns_dict(),
        _params(lambda_l1=lambda_l1),
    )
    decision = compiled.variable_slices["decision"]
    l1_pos = compiled.variable_slices["l1_pos"]
    l1_neg = compiled.variable_slices["l1_neg"]

    assert l1_pos.size == l1_neg.size == 4
    np.testing.assert_allclose(compiled.q[l1_pos.slice], lambda_l1)
    np.testing.assert_allclose(compiled.q[l1_neg.slice], lambda_l1)
    for asset_idx in range(4):
        row_idx = 2 + asset_idx
        row = compiled.A_eq.getrow(row_idx).toarray().reshape(-1)
        expected = np.zeros(compiled.n_variables)
        expected[decision.start + asset_idx] = 1.0
        expected[l1_pos.start + asset_idx] = -1.0
        expected[l1_neg.start + asset_idx] = 1.0
        np.testing.assert_allclose(row, expected)


def test_max_sharpe_l1_l2_osqp_solution_feasible():
    compiled = compile_portfolio_qp(
        small_returns_dict(),
        _params(lambda_l1=0.10, lambda_l2=0.15),
    )
    solution = solve_compiled_qp_osqp(compiled)
    weights = compiled.recover_stock_weights(solution.x)

    assert_feasible_solution(compiled, solution)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)
    assert compiled.mean @ weights > 0.0


@pytest.mark.gpu
def test_max_sharpe_l1_l2_cuopt_matches_osqp_when_available():
    require_cuopt()
    returns_dict = small_returns_dict()
    options = {"lambda_l1": 0.10, "lambda_l2": 0.15}
    osqp_compiled = compile_portfolio_qp(returns_dict, _params(**options))
    cuopt_compiled = compile_portfolio_qp(
        returns_dict,
        _params(backend="cuopt", **options),
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
