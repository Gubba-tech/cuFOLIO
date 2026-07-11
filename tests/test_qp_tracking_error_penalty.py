import numpy as np
import pytest
from qp_test_utils import (
    assert_feasible_solution,
    assert_objective_gap_within,
    require_cuopt,
    small_returns_dict,
)

from cufolio.exceptions import QPCompilationError
from cufolio.qp_backend import solve_compiled_qp_cuopt, solve_compiled_qp_osqp
from cufolio.qp_formulations import compile_portfolio_qp
from cufolio.qp_parameters import QPParameters


def _benchmark():
    return np.ones(4) / 4


def _params(backend="osqp", **overrides):
    options = {
        "objective": "mean_variance",
        "risk_aversion": 2.0,
        "benchmark_weights": _benchmark(),
        "lambda_tracking_error": 0.4,
        "backend": backend,
    }
    options.update(overrides)
    return QPParameters(**options)


def test_tracking_error_penalty_q_q_terms_identity():
    returns_dict = small_returns_dict()
    base = compile_portfolio_qp(
        returns_dict,
        _params(lambda_tracking_error=0.0),
    )
    penalty = compile_portfolio_qp(returns_dict, _params())
    lambda_tracking_error = 0.4

    np.testing.assert_allclose(
        penalty.Q.toarray() - base.Q.toarray(),
        2.0 * lambda_tracking_error * returns_dict["covariance"],
    )
    np.testing.assert_allclose(
        penalty.q - base.q,
        -2.0
        * lambda_tracking_error
        * returns_dict["covariance"]
        @ _benchmark(),
    )


def test_tracking_error_penalty_q_q_terms_generic_v():
    mapping = np.array(
        [
            [0.8, 0.0],
            [0.2, 0.1],
            [0.0, 0.4],
            [0.0, 0.5],
        ]
    )
    returns_dict = small_returns_dict()
    base = compile_portfolio_qp(
        returns_dict,
        _params(V=mapping, lambda_tracking_error=0.0),
    )
    penalty = compile_portfolio_qp(
        returns_dict,
        _params(V=mapping),
    )
    lambda_tracking_error = 0.4
    decision = penalty.variable_slices["decision"].slice
    sigma = returns_dict["covariance"]
    benchmark = _benchmark()

    np.testing.assert_allclose(
        penalty.Q[decision, decision].toarray()
        - base.Q[decision, decision].toarray(),
        2.0 * lambda_tracking_error * mapping.T @ sigma @ mapping,
    )
    np.testing.assert_allclose(
        penalty.q[decision] - base.q[decision],
        -2.0 * lambda_tracking_error * mapping.T @ sigma @ benchmark,
    )


def test_tracking_error_requires_benchmark_weights():
    with pytest.raises(QPCompilationError, match="benchmark_weights"):
        compile_portfolio_qp(
            small_returns_dict(),
            QPParameters(lambda_tracking_error=0.4, backend="osqp"),
        )


def test_tracking_error_penalty_ordinary_factor_space_requires_stock_covariance():
    returns_dict = {
        "factor_mean": np.array([0.04, 0.03]),
        "factor_covariance": np.diag([0.04, 0.06]),
        "stock_mapping": np.ones((4, 2)) / 4.0,
    }
    with pytest.raises(QPCompilationError, match="stock_covariance"):
        compile_portfolio_qp(
            returns_dict,
            QPParameters(
                mapping_mode="factor_space",
                V=returns_dict["stock_mapping"],
                benchmark_weights=_benchmark(),
                lambda_tracking_error=0.4,
                backend="osqp",
            ),
        )


def test_tracking_error_penalty_osqp_solution_feasible():
    compiled = compile_portfolio_qp(small_returns_dict(), _params())
    solution = solve_compiled_qp_osqp(compiled)

    assert_feasible_solution(compiled, solution)


@pytest.mark.gpu
def test_tracking_error_penalty_cuopt_matches_osqp_when_available():
    require_cuopt()
    returns_dict = small_returns_dict()
    osqp_compiled = compile_portfolio_qp(returns_dict, _params())
    cuopt_compiled = compile_portfolio_qp(
        returns_dict,
        _params(backend="cuopt"),
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


def test_tracking_error_is_documented_as_penalty_not_hard_constraint():
    with open("docs/nvidia_qp_backend.md", encoding="utf-8") as handle:
        text = handle.read()
    assert "Tracking-error hard constraints" in text
    assert "quadratic objective penalty" in text
