import numpy as np
import pytest
from qp_test_utils import (
    assert_feasible_solution,
    assert_long_short_budget,
    assert_objective_gap_within,
    require_cuopt,
    small_returns_dict,
)

from cufolio.qp_backend import solve_compiled_qp_cuopt, solve_compiled_qp_osqp
from cufolio.qp_formulations import compile_portfolio_qp
from cufolio.qp_parameters import QPParameters


def _mapping():
    return np.array(
        [
            [0.8, 0.0],
            [0.2, 0.1],
            [0.0, 0.4],
            [0.0, 0.5],
        ]
    )


def _params(backend="osqp", **overrides):
    options = {
        "objective": "max_sharpe",
        "V": _mapping(),
        "w_min": 0.0,
        "w_max": 1.0,
        "backend": backend,
    }
    options.update(overrides)
    return QPParameters(**options)


def test_max_sharpe_generic_v_base_rows():
    returns_dict = small_returns_dict()
    mapping = _mapping()
    compiled = compile_portfolio_qp(returns_dict, _params())
    decision = compiled.variable_slices["decision"]
    scale = compiled.variable_slices["scale"]
    excess = returns_dict["mean"]

    expected_excess = mapping.T @ excess
    np.testing.assert_allclose(
        compiled.A_eq.getrow(0).toarray().reshape(-1),
        [*expected_excess, 0.0],
    )
    expected_budget = np.zeros(compiled.n_variables)
    expected_budget[decision.slice] = np.ones(4) @ mapping
    expected_budget[scale.start] = -1.0
    np.testing.assert_allclose(
        compiled.A_eq.getrow(1).toarray().reshape(-1),
        expected_budget,
    )


def test_max_sharpe_generic_v_scaled_bounds():
    mapping = _mapping()
    compiled = compile_portfolio_qp(
        small_returns_dict(),
        _params(w_min=-0.2, w_max=0.6),
    )
    decision = compiled.variable_slices["decision"]
    scale = compiled.variable_slices["scale"]
    ineq_names = compiled.constraint_names[compiled.A_eq.shape[0] :]
    upper_idx = ineq_names.index("upper_bound_2")
    lower_idx = ineq_names.index("lower_bound_2")
    expected_upper = np.zeros(compiled.n_variables)
    expected_upper[decision.slice] = mapping[2]
    expected_upper[scale.start] = -0.6
    expected_lower = np.zeros(compiled.n_variables)
    expected_lower[decision.slice] = -mapping[2]
    expected_lower[scale.start] = -0.2

    np.testing.assert_allclose(
        compiled.A_ineq.getrow(upper_idx).toarray().reshape(-1),
        expected_upper,
    )
    np.testing.assert_allclose(
        compiled.A_ineq.getrow(lower_idx).toarray().reshape(-1),
        expected_lower,
    )


def test_max_sharpe_generic_v_scaled_long_short():
    mapping = _mapping()
    compiled = compile_portfolio_qp(
        small_returns_dict(),
        _params(w_min=-1.0, w_max=1.0, short_budget=0.2),
    )
    decision = compiled.variable_slices["decision"]
    scale = compiled.variable_slices["scale"]
    pos = compiled.variable_slices["pos"]
    neg = compiled.variable_slices["neg"]
    for asset_idx in range(4):
        row = compiled.A_eq.getrow(2 + asset_idx).toarray().reshape(-1)
        expected = np.zeros(compiled.n_variables)
        expected[decision.slice] = mapping[asset_idx]
        expected[pos.start + asset_idx] = -1.0
        expected[neg.start + asset_idx] = 1.0
        np.testing.assert_allclose(row, expected)

    ineq_names = compiled.constraint_names[compiled.A_eq.shape[0] :]
    long_idx = ineq_names.index("long_budget")
    short_idx = ineq_names.index("short_budget")
    expected_long = np.zeros(compiled.n_variables)
    expected_long[pos.slice] = 1.0
    expected_long[scale.start] = -1.2
    expected_short = np.zeros(compiled.n_variables)
    expected_short[neg.slice] = 1.0
    expected_short[scale.start] = -0.2
    np.testing.assert_allclose(
        compiled.A_ineq.getrow(long_idx).toarray().reshape(-1),
        expected_long,
    )
    np.testing.assert_allclose(
        compiled.A_ineq.getrow(short_idx).toarray().reshape(-1),
        expected_short,
    )


def test_max_sharpe_generic_v_l1_l2_compiler_terms():
    mapping = _mapping()
    lambda_l1 = 0.15
    lambda_l2 = 0.25
    base = compile_portfolio_qp(
        small_returns_dict(),
        _params(),
    )
    compiled = compile_portfolio_qp(
        small_returns_dict(),
        _params(lambda_l1=lambda_l1, lambda_l2=lambda_l2),
    )
    decision = compiled.variable_slices["decision"].slice
    l1_pos = compiled.variable_slices["l1_pos"]
    l1_neg = compiled.variable_slices["l1_neg"]
    np.testing.assert_allclose(
        compiled.Q[decision, decision].toarray()
        - base.Q[decision, decision].toarray(),
        2.0 * lambda_l2 * mapping.T @ mapping,
    )
    np.testing.assert_allclose(compiled.q[l1_pos.slice], lambda_l1)
    np.testing.assert_allclose(compiled.q[l1_neg.slice], lambda_l1)
    for asset_idx in range(4):
        row = compiled.A_eq.getrow(2 + asset_idx).toarray().reshape(-1)
        expected = np.zeros(compiled.n_variables)
        expected[compiled.variable_slices["decision"].slice] = mapping[asset_idx]
        expected[l1_pos.start + asset_idx] = -1.0
        expected[l1_neg.start + asset_idx] = 1.0
        np.testing.assert_allclose(row, expected)


def test_max_sharpe_generic_v_osqp_solution_feasible():
    compiled = compile_portfolio_qp(
        small_returns_dict(),
        _params(objective="max_sharpe"),
    )
    solution = solve_compiled_qp_osqp(compiled)
    weights = compiled.recover_stock_weights(solution.x)

    assert_feasible_solution(compiled, solution)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)
    assert np.all(weights >= -1e-7)


@pytest.mark.gpu
def test_max_sharpe_generic_v_cuopt_matches_osqp_when_available():
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
    np.testing.assert_allclose(cuopt_weights.sum(), 1.0, atol=1e-5)


def test_max_sharpe_generic_v_long_short_solution_feasible():
    compiled = compile_portfolio_qp(
        small_returns_dict(),
        _params(w_min=-1.0, w_max=1.0, short_budget=0.2),
    )
    solution = solve_compiled_qp_osqp(compiled)
    weights = compiled.recover_stock_weights(solution.x)

    assert_feasible_solution(compiled, solution, tol=1e-5)
    assert_long_short_budget(weights, short_budget=0.2, tol=1e-5)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-5)
