import numpy as np
import pytest
from qp_test_utils import (
    assert_feasible_solution,
    gross_exposure,
    require_cuopt,
    small_returns_dict,
)

from cufolio.qp_backend import solve_compiled_qp_cuopt, solve_compiled_qp_osqp
from cufolio.qp_formulations import compile_portfolio_qp
from cufolio.qp_parameters import QPParameters


def _short_incentive_returns_dict():
    return {
        "mean": np.array([0.12, -0.08, 0.05, -0.03]),
        "covariance": np.array(
            [
                [0.020, 0.001, 0.000, 0.000],
                [0.001, 0.020, 0.000, 0.000],
                [0.000, 0.000, 0.025, 0.001],
                [0.000, 0.000, 0.001, 0.025],
            ]
        ),
    }


def _l1_split_row(compiled, row_idx):
    return compiled.A_eq.getrow(row_idx).toarray().reshape(-1)


def test_l1_regularization_adds_auxiliary_variables_identity():
    returns_dict = small_returns_dict()
    lambda_l1 = 0.35
    base = compile_portfolio_qp(
        returns_dict,
        QPParameters(objective="mean_variance", risk_aversion=2.0, backend="osqp"),
    )
    compiled = compile_portfolio_qp(
        returns_dict,
        QPParameters(
            objective="mean_variance",
            risk_aversion=2.0,
            lambda_l1=lambda_l1,
            backend="osqp",
        ),
    )
    n_assets = len(returns_dict["mean"])
    decision = compiled.variable_slices["decision"].slice
    l1_pos = compiled.variable_slices["l1_pos"]
    l1_neg = compiled.variable_slices["l1_neg"]

    assert compiled.n_variables == base.n_variables + 2 * n_assets
    assert l1_pos.size == n_assets
    assert l1_neg.size == n_assets
    assert all(name.startswith("l1_pos_") for name in compiled.variable_names[l1_pos.slice])
    assert all(name.startswith("l1_neg_") for name in compiled.variable_names[l1_neg.slice])
    np.testing.assert_allclose(compiled.lower[l1_pos.slice], 0.0)
    np.testing.assert_allclose(compiled.lower[l1_neg.slice], 0.0)
    assert np.all(np.isinf(compiled.upper[l1_pos.slice]))
    assert np.all(np.isinf(compiled.upper[l1_neg.slice]))
    np.testing.assert_allclose(compiled.q[l1_pos.slice], lambda_l1)
    np.testing.assert_allclose(compiled.q[l1_neg.slice], lambda_l1)
    np.testing.assert_allclose(
        compiled.Q[decision, decision].toarray(),
        base.Q.toarray(),
        atol=1e-12,
    )

    for asset_idx in range(n_assets):
        row_idx = 1 + asset_idx
        row = _l1_split_row(compiled, row_idx)
        expected = np.zeros(compiled.n_variables)
        expected[decision.start + asset_idx] = 1.0
        expected[l1_pos.start + asset_idx] = -1.0
        expected[l1_neg.start + asset_idx] = 1.0
        np.testing.assert_allclose(row, expected, atol=1e-12)
        assert compiled.b_eq[row_idx] == pytest.approx(0.0)


def test_l1_long_only_identity_is_constant_and_does_not_change_weights():
    returns_dict = small_returns_dict()
    base = compile_portfolio_qp(
        returns_dict,
        QPParameters(
            objective="mean_variance",
            risk_aversion=3.0,
            w_min=0.0,
            w_max=1.0,
            backend="osqp",
        ),
    )
    l1 = compile_portfolio_qp(
        returns_dict,
        QPParameters(
            objective="mean_variance",
            risk_aversion=3.0,
            lambda_l1=0.50,
            w_min=0.0,
            w_max=1.0,
            backend="osqp",
        ),
    )
    base_solution = solve_compiled_qp_osqp(base)
    l1_solution = solve_compiled_qp_osqp(l1)

    assert_feasible_solution(base, base_solution)
    assert_feasible_solution(l1, l1_solution)
    np.testing.assert_allclose(
        l1.recover_stock_weights(l1_solution.x),
        base.recover_stock_weights(base_solution.x),
        atol=1e-5,
    )


@pytest.mark.gpu
def test_l1_long_only_identity_cuopt_weights_match_when_available():
    require_cuopt()
    returns_dict = small_returns_dict()
    base = compile_portfolio_qp(
        returns_dict,
        QPParameters(
            objective="mean_variance",
            risk_aversion=3.0,
            w_min=0.0,
            w_max=1.0,
            backend="cuopt",
        ),
    )
    l1 = compile_portfolio_qp(
        returns_dict,
        QPParameters(
            objective="mean_variance",
            risk_aversion=3.0,
            lambda_l1=0.50,
            w_min=0.0,
            w_max=1.0,
            backend="cuopt",
        ),
    )
    base_solution = solve_compiled_qp_cuopt(base)
    l1_solution = solve_compiled_qp_cuopt(l1)

    assert_feasible_solution(base, base_solution, tol=1e-5)
    assert_feasible_solution(l1, l1_solution, tol=1e-5)
    np.testing.assert_allclose(
        l1.recover_stock_weights(l1_solution.x),
        base.recover_stock_weights(base_solution.x),
        atol=1e-4,
        rtol=1e-4,
    )


def test_l1_reduces_gross_exposure_when_shorts_allowed_by_bounds():
    returns_dict = _short_incentive_returns_dict()
    base = compile_portfolio_qp(
        returns_dict,
        QPParameters(
            objective="mean_variance",
            risk_aversion=8.0,
            w_min=-1.0,
            w_max=1.0,
            backend="osqp",
        ),
    )
    l1 = compile_portfolio_qp(
        returns_dict,
        QPParameters(
            objective="mean_variance",
            risk_aversion=8.0,
            lambda_l1=0.75,
            w_min=-1.0,
            w_max=1.0,
            backend="osqp",
        ),
    )
    base_solution = solve_compiled_qp_osqp(base)
    l1_solution = solve_compiled_qp_osqp(l1)

    assert_feasible_solution(base, base_solution)
    assert_feasible_solution(l1, l1_solution)
    assert gross_exposure(l1, l1_solution.x) <= gross_exposure(base, base_solution.x) + 1e-6


@pytest.mark.gpu
def test_l1_cuopt_reduces_gross_exposure_when_shorts_allowed_by_bounds():
    require_cuopt()
    returns_dict = _short_incentive_returns_dict()
    base = compile_portfolio_qp(
        returns_dict,
        QPParameters(
            objective="mean_variance",
            risk_aversion=8.0,
            w_min=-1.0,
            w_max=1.0,
            backend="cuopt",
        ),
    )
    l1 = compile_portfolio_qp(
        returns_dict,
        QPParameters(
            objective="mean_variance",
            risk_aversion=8.0,
            lambda_l1=0.75,
            w_min=-1.0,
            w_max=1.0,
            backend="cuopt",
        ),
    )
    base_solution = solve_compiled_qp_cuopt(base)
    l1_solution = solve_compiled_qp_cuopt(l1)

    assert_feasible_solution(base, base_solution, tol=1e-5)
    assert_feasible_solution(l1, l1_solution, tol=1e-5)
    assert gross_exposure(l1, l1_solution.x) <= gross_exposure(base, base_solution.x) + 1e-5
