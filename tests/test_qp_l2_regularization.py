import numpy as np
import pytest
from qp_test_utils import (
    assert_cuopt_matches_osqp,
    assert_feasible_solution,
    small_returns_dict,
)

from cufolio.qp_backend import solve_compiled_qp_osqp
from cufolio.qp_formulations import compile_portfolio_qp
from cufolio.qp_parameters import QPParameters


def test_l2_regularization_adds_identity_when_mapping_is_absent():
    returns_dict = small_returns_dict()
    lambda_l2 = 0.35
    base = compile_portfolio_qp(
        returns_dict,
        QPParameters(objective="mean_variance", risk_aversion=2.0, backend="osqp"),
    )
    regularized = compile_portfolio_qp(
        returns_dict,
        QPParameters(
            objective="mean_variance",
            risk_aversion=2.0,
            lambda_l2=lambda_l2,
            backend="osqp",
        ),
    )

    expected = 2.0 * lambda_l2 * np.eye(len(returns_dict["mean"]))
    np.testing.assert_allclose(
        regularized.Q.toarray() - base.Q.toarray(),
        expected,
        atol=1e-12,
    )


def test_l2_regularization_adds_v_transpose_v_for_generic_mapping():
    returns_dict = small_returns_dict()
    lambda_l2 = 0.20
    mapping = np.array(
        [
            [0.8, 0.0],
            [0.2, 0.1],
            [0.0, 0.4],
            [0.0, 0.5],
        ]
    )
    base = compile_portfolio_qp(
        returns_dict,
        QPParameters(V=mapping, backend="osqp"),
    )
    regularized = compile_portfolio_qp(
        returns_dict,
        QPParameters(V=mapping, lambda_l2=lambda_l2, backend="osqp"),
    )

    expected = 2.0 * lambda_l2 * mapping.T @ mapping
    np.testing.assert_allclose(
        regularized.Q.toarray() - base.Q.toarray(),
        expected,
        atol=1e-12,
    )


def test_l2_regularized_qp_osqp_solution_is_feasible():
    compiled = compile_portfolio_qp(
        small_returns_dict(),
        QPParameters(
            objective="mean_variance",
            risk_aversion=2.0,
            lambda_l2=0.25,
            backend="osqp",
        ),
    )
    solution = solve_compiled_qp_osqp(compiled)

    assert_feasible_solution(compiled, solution)


@pytest.mark.gpu
def test_l2_regularized_qp_cuopt_matches_osqp_when_available():
    compiled = compile_portfolio_qp(
        small_returns_dict(),
        QPParameters(
            objective="mean_variance",
            risk_aversion=2.0,
            lambda_l2=0.25,
            backend="cuopt",
        ),
    )

    assert_cuopt_matches_osqp(compiled)
