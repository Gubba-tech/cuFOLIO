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


def _generic_mapping():
    return np.array(
        [
            [0.8, -0.2],
            [0.2, 0.4],
            [0.0, 0.4],
            [0.0, 0.4],
        ]
    )


def _generic_params(backend):
    return QPParameters(
        V=_generic_mapping(),
        objective="mean_variance",
        risk_aversion=2.0,
        w_min=-1.0,
        w_max=1.0,
        short_budget=0.2,
        backend=backend,
    )


def test_short_budget_generic_v_auxiliary_dimension():
    mapping = _generic_mapping()
    compiled = compile_portfolio_qp(
        small_returns_dict(),
        _generic_params("osqp"),
    )
    n_assets, n_factors = mapping.shape
    decision = compiled.variable_slices["decision"]
    pos = compiled.variable_slices["pos"]
    neg = compiled.variable_slices["neg"]

    assert decision.size == n_factors
    assert pos.size == n_assets
    assert neg.size == n_assets
    assert compiled.n_variables == n_factors + 2 * n_assets
    for asset_idx in range(n_assets):
        row_idx = 1 + asset_idx
        row = compiled.A_eq.getrow(row_idx).toarray().reshape(-1)
        expected = np.zeros(compiled.n_variables)
        expected[decision.slice] = mapping[asset_idx]
        expected[pos.start + asset_idx] = -1.0
        expected[neg.start + asset_idx] = 1.0
        np.testing.assert_allclose(row, expected, atol=1e-12)
        assert compiled.constraint_names[row_idx] == f"long_short_split_{asset_idx}"

    ineq_names = compiled.constraint_names[compiled.A_eq.shape[0] :]
    long_idx = ineq_names.index("long_budget")
    short_idx = ineq_names.index("short_budget")
    expected_long = np.zeros(compiled.n_variables)
    expected_long[pos.slice] = 1.0
    expected_short = np.zeros(compiled.n_variables)
    expected_short[neg.slice] = 1.0
    np.testing.assert_allclose(
        compiled.A_ineq.getrow(long_idx).toarray().reshape(-1),
        expected_long,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        compiled.A_ineq.getrow(short_idx).toarray().reshape(-1),
        expected_short,
        atol=1e-12,
    )
    assert compiled.b_ineq[long_idx] == pytest.approx(1.2)
    assert compiled.b_ineq[short_idx] == pytest.approx(0.2)


def test_short_budget_generic_v_osqp_solution_feasible():
    compiled = compile_portfolio_qp(
        small_returns_dict(),
        _generic_params("osqp"),
    )
    solution = solve_compiled_qp_osqp(compiled)
    weights = compiled.recover_stock_weights(solution.x)

    assert_feasible_solution(compiled, solution)
    assert_long_short_budget(weights, short_budget=0.2)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)


@pytest.mark.gpu
def test_short_budget_generic_v_cuopt_matches_osqp_when_available():
    require_cuopt()
    returns_dict = small_returns_dict()
    osqp_compiled = compile_portfolio_qp(returns_dict, _generic_params("osqp"))
    cuopt_compiled = compile_portfolio_qp(returns_dict, _generic_params("cuopt"))
    osqp_solution = solve_compiled_qp_osqp(osqp_compiled)
    cuopt_solution = solve_compiled_qp_cuopt(cuopt_compiled)
    cuopt_weights = cuopt_compiled.recover_stock_weights(cuopt_solution.x)

    assert_feasible_solution(osqp_compiled, osqp_solution, tol=1e-6)
    assert_feasible_solution(cuopt_compiled, cuopt_solution, tol=1e-5)
    assert_objective_gap_within(
        cuopt_compiled,
        cuopt_solution,
        osqp_solution,
        tol=5e-4,
    )
    assert_long_short_budget(cuopt_weights, short_budget=0.2, tol=1e-5)
    np.testing.assert_allclose(cuopt_weights.sum(), 1.0, atol=1e-5)
