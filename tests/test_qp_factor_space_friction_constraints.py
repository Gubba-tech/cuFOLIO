import numpy as np
import pytest
from qp_test_utils import (
    assert_feasible_solution,
    assert_objective_gap_within,
    require_cuopt,
)

from cufolio.exceptions import QPCompilationError
from cufolio.qp_backend import solve_compiled_qp_cuopt, solve_compiled_qp_osqp
from cufolio.qp_formulations import compile_portfolio_qp
from cufolio.qp_parameters import QPParameters


def _mapping():
    return np.array(
        [
            [0.7, 0.0, 0.0],
            [0.3, 0.2, 0.0],
            [0.0, 0.5, 0.1],
            [0.0, 0.3, 0.4],
            [0.0, 0.0, 0.4],
            [0.0, 0.0, 0.1],
        ]
    )


def _returns():
    return {
        "mean": np.array([0.04, 0.03, 0.02]),
        "covariance": np.diag([0.04, 0.06, 0.08]),
    }


def _anchor():
    return _mapping() @ np.array([0.4, 0.3, 0.3])


def _params(backend="osqp", **overrides):
    options = {
        "mapping_mode": "factor_space",
        "V": _mapping(),
        "objective": "mean_variance",
        "previous_weights": _anchor(),
        "turnover_budget": 0.25,
        "benchmark_weights": _anchor(),
        "benchmark_l1_budget": 0.30,
        "factor_exposure_matrix": np.array(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, -0.5], [0.0, 0.5], [0.5, 0.0]]
        ),
        "factor_exposure_lower": np.array([0.0, 0.0]),
        "factor_exposure_upper": np.array([1.0, 1.0]),
        "backend": backend,
    }
    options.update(overrides)
    return QPParameters(**options)


def test_factor_space_friction_rows_use_stock_exposure():
    compiled = compile_portfolio_qp(_returns(), _params())
    decision = compiled.variable_slices["decision"]
    mapping = _mapping()
    for name, anchor, row_start in (
        ("turnover", _anchor(), 1),
        ("benchmark", _anchor(), 7),
    ):
        pos = compiled.variable_slices[f"{name}_pos"]
        neg = compiled.variable_slices[f"{name}_neg"]
        for asset_idx, value in enumerate(anchor):
            expected = np.zeros(compiled.n_variables)
            expected[decision.slice] = mapping[asset_idx]
            expected[pos.start + asset_idx] = -1.0
            expected[neg.start + asset_idx] = 1.0
            np.testing.assert_allclose(
                compiled.A_eq.getrow(row_start + asset_idx).toarray().reshape(-1),
                expected,
            )
            assert compiled.b_eq[row_start + asset_idx] == pytest.approx(value)


def test_factor_space_max_sharpe_scaled_friction_rows():
    compiled = compile_portfolio_qp(
        _returns(),
        _params(
            objective="max_sharpe",
            previous_weights=None,
            turnover_budget=None,
        ),
    )
    decision = compiled.variable_slices["decision"]
    scale = compiled.variable_slices["scale"]
    pos = compiled.variable_slices["benchmark_pos"]
    neg = compiled.variable_slices["benchmark_neg"]
    for asset_idx, value in enumerate(_anchor()):
        expected = np.zeros(compiled.n_variables)
        expected[decision.slice] = _mapping()[asset_idx]
        expected[scale.start] = -value
        expected[pos.start + asset_idx] = -1.0
        expected[neg.start + asset_idx] = 1.0
        np.testing.assert_allclose(
            compiled.A_eq.getrow(2 + asset_idx).toarray().reshape(-1),
            expected,
        )


def test_factor_space_factor_exposure_rows_use_v():
    compiled = compile_portfolio_qp(_returns(), _params())
    decision = compiled.variable_slices["decision"]
    exposure_matrix = _params().factor_exposure_matrix.T @ _mapping()
    ineq_names = compiled.constraint_names[compiled.A_eq.shape[0] :]
    upper_idx = ineq_names.index("factor_exposure_upper_0")
    np.testing.assert_allclose(
        compiled.A_ineq.getrow(upper_idx).toarray().reshape(-1)[decision.slice],
        exposure_matrix[0],
    )


def test_factor_space_tracking_error_requires_stock_covariance():
    with pytest.raises(QPCompilationError, match="stock_covariance"):
        compile_portfolio_qp(
            _returns(),
            _params(lambda_tracking_error=0.2),
        )


def test_factor_space_tracking_error_matrix_terms_use_stock_covariance():
    stock_covariance = np.diag(np.linspace(0.02, 0.07, 6))
    returns_dict = {
        **_returns(),
        "stock_covariance": stock_covariance,
    }
    base = compile_portfolio_qp(returns_dict, _params(lambda_tracking_error=0.0))
    penalty = compile_portfolio_qp(returns_dict, _params(lambda_tracking_error=0.2))
    decision = penalty.variable_slices["decision"].slice
    expected = 2.0 * 0.2 * _mapping().T @ stock_covariance @ _mapping()
    np.testing.assert_allclose(
        penalty.Q[decision, decision].toarray()
        - base.Q[decision, decision].toarray(),
        expected,
    )


@pytest.mark.gpu
def test_factor_space_friction_cuopt_matches_osqp_when_available():
    require_cuopt()
    returns_dict = {
        **_returns(),
        "stock_covariance": np.diag(np.linspace(0.02, 0.07, 6)),
    }
    osqp_compiled = compile_portfolio_qp(
        returns_dict,
        _params(lambda_tracking_error=0.2),
    )
    cuopt_compiled = compile_portfolio_qp(
        returns_dict,
        _params(backend="cuopt", lambda_tracking_error=0.2),
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
