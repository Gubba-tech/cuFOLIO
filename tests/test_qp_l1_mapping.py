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


def _generic_mapping():
    return np.array(
        [
            [0.8, 0.0],
            [0.2, 0.1],
            [0.0, 0.4],
            [0.0, 0.5],
        ]
    )


def test_l1_generic_v_auxiliary_dimension():
    returns_dict = small_returns_dict()
    mapping = _generic_mapping()
    lambda_l1 = 0.30
    compiled = compile_portfolio_qp(
        returns_dict,
        QPParameters(V=mapping, lambda_l1=lambda_l1, backend="osqp"),
    )
    n_assets, n_factors = mapping.shape
    decision = compiled.variable_slices["decision"]
    l1_pos = compiled.variable_slices["l1_pos"]
    l1_neg = compiled.variable_slices["l1_neg"]

    assert decision.size == n_factors
    assert l1_pos.size == n_assets
    assert l1_neg.size == n_assets
    assert compiled.n_variables == n_factors + 2 * n_assets
    assert all(name.startswith("l1_pos_") for name in compiled.variable_names[l1_pos.slice])
    assert all(name.startswith("l1_neg_") for name in compiled.variable_names[l1_neg.slice])
    np.testing.assert_allclose(compiled.q[l1_pos.slice], lambda_l1)
    np.testing.assert_allclose(compiled.q[l1_neg.slice], lambda_l1)

    for asset_idx in range(n_assets):
        row_idx = 1 + asset_idx
        row = compiled.A_eq.getrow(row_idx).toarray().reshape(-1)
        expected = np.zeros(compiled.n_variables)
        expected[decision.slice] = mapping[asset_idx]
        expected[l1_pos.start + asset_idx] = -1.0
        expected[l1_neg.start + asset_idx] = 1.0
        np.testing.assert_allclose(row, expected, atol=1e-12)
        assert compiled.b_eq[row_idx] == pytest.approx(0.0)


def test_l1_l2_generic_v_compiler_matrix_terms():
    returns_dict = small_returns_dict()
    mapping = _generic_mapping()
    lambda_l1 = 0.30
    lambda_l2 = 0.25
    l1_only = compile_portfolio_qp(
        returns_dict,
        QPParameters(V=mapping, lambda_l1=lambda_l1, backend="osqp"),
    )
    l1_l2 = compile_portfolio_qp(
        returns_dict,
        QPParameters(
            V=mapping,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
            backend="osqp",
        ),
    )
    decision = l1_l2.variable_slices["decision"].slice
    l1_pos = l1_l2.variable_slices["l1_pos"].slice
    l1_neg = l1_l2.variable_slices["l1_neg"].slice

    np.testing.assert_allclose(
        l1_l2.Q[decision, decision].toarray()
        - l1_only.Q[decision, decision].toarray(),
        2.0 * lambda_l2 * mapping.T @ mapping,
        atol=1e-12,
    )
    np.testing.assert_allclose(l1_l2.q[l1_pos], lambda_l1)
    np.testing.assert_allclose(l1_l2.q[l1_neg], lambda_l1)


def test_l1_generic_v_osqp_solution_feasible():
    compiled = compile_portfolio_qp(
        small_returns_dict(),
        QPParameters(
            V=_generic_mapping(),
            objective="mean_variance",
            risk_aversion=2.0,
            lambda_l1=0.15,
            backend="osqp",
        ),
    )
    solution = solve_compiled_qp_osqp(compiled)

    assert_feasible_solution(compiled, solution)


@pytest.mark.gpu
def test_l1_generic_v_cuopt_matches_osqp_when_available():
    require_cuopt()
    compiled = compile_portfolio_qp(
        small_returns_dict(),
        QPParameters(
            V=_generic_mapping(),
            objective="mean_variance",
            risk_aversion=2.0,
            lambda_l1=0.15,
            backend="cuopt",
        ),
    )
    osqp_solution = solve_compiled_qp_osqp(compiled)
    cuopt_solution = solve_compiled_qp_cuopt(compiled)

    assert_feasible_solution(compiled, osqp_solution, tol=1e-6)
    assert_feasible_solution(compiled, cuopt_solution, tol=1e-5)
    assert_objective_gap_within(compiled, cuopt_solution, osqp_solution, tol=5e-4)
