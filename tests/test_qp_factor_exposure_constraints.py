import numpy as np
import pytest
from qp_test_utils import (
    assert_factor_exposure_bounds,
    assert_feasible_solution,
    assert_objective_gap_within,
    require_cuopt,
    small_returns_dict,
)

from cufolio.exceptions import QPCompilationError
from cufolio.qp_backend import solve_compiled_qp_cuopt, solve_compiled_qp_osqp
from cufolio.qp_formulations import compile_portfolio_qp
from cufolio.qp_parameters import QPParameters


def _factor_matrix():
    return np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, -0.5],
        ]
    )


def _bounds():
    return np.array([0.0, 0.0]), np.array([1.0, 1.0])


def _params(backend="osqp", **overrides):
    lower, upper = _bounds()
    options = {
        "objective": "mean_variance",
        "risk_aversion": 2.0,
        "factor_exposure_matrix": _factor_matrix(),
        "factor_exposure_lower": lower,
        "factor_exposure_upper": upper,
        "backend": backend,
    }
    options.update(overrides)
    return QPParameters(**options)


def test_factor_exposure_upper_lower_rows():
    factor_matrix = _factor_matrix()
    lower, upper = _bounds()
    compiled = compile_portfolio_qp(small_returns_dict(), _params())
    ineq_names = compiled.constraint_names[compiled.A_eq.shape[0] :]
    for factor_idx in range(2):
        upper_idx = ineq_names.index(f"factor_exposure_upper_{factor_idx}")
        lower_idx = ineq_names.index(f"factor_exposure_lower_{factor_idx}")
        np.testing.assert_allclose(
            compiled.A_ineq.getrow(upper_idx).toarray().reshape(-1),
            factor_matrix[:, factor_idx],
        )
        np.testing.assert_allclose(
            compiled.A_ineq.getrow(lower_idx).toarray().reshape(-1),
            -factor_matrix[:, factor_idx],
        )
        assert compiled.b_ineq[upper_idx] == pytest.approx(upper[factor_idx])
        assert compiled.b_ineq[lower_idx] == pytest.approx(-lower[factor_idx])


def test_factor_exposure_shape_mismatch_rejected():
    with pytest.raises(QPCompilationError, match="factor_exposure_matrix"):
        compile_portfolio_qp(
            small_returns_dict(),
            _params(factor_exposure_matrix=np.ones((3, 2))),
        )


def test_factor_exposure_requires_bound():
    with pytest.raises(QPCompilationError, match="require lower"):
        compile_portfolio_qp(
            small_returns_dict(),
            _params(
                factor_exposure_lower=None,
                factor_exposure_upper=None,
            ),
        )


def test_factor_exposure_osqp_solution_feasible():
    lower, upper = _bounds()
    compiled = compile_portfolio_qp(small_returns_dict(), _params())
    solution = solve_compiled_qp_osqp(compiled)
    weights = compiled.recover_stock_weights(solution.x)

    assert_feasible_solution(compiled, solution)
    assert_factor_exposure_bounds(weights, _factor_matrix(), lower, upper)

@pytest.mark.gpu
def test_factor_exposure_cuopt_matches_osqp_when_available():
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


def test_max_sharpe_scaled_factor_exposure_rows():
    factor_matrix = _factor_matrix()
    lower, upper = _bounds()
    compiled = compile_portfolio_qp(
        small_returns_dict(),
        _params(objective="max_sharpe", w_min=0.0, w_max=1.0),
    )
    scale = compiled.variable_slices["scale"]
    for factor_idx in range(2):
        ineq_names = compiled.constraint_names[compiled.A_eq.shape[0] :]
        upper_idx = ineq_names.index(f"factor_exposure_upper_{factor_idx}")
        lower_idx = ineq_names.index(f"factor_exposure_lower_{factor_idx}")
        expected_upper = np.zeros(compiled.n_variables)
        expected_upper[:4] = factor_matrix[:, factor_idx]
        expected_upper[scale.start] = -upper[factor_idx]
        expected_lower = np.zeros(compiled.n_variables)
        expected_lower[:4] = -factor_matrix[:, factor_idx]
        expected_lower[scale.start] = lower[factor_idx]
        np.testing.assert_allclose(
            compiled.A_ineq.getrow(upper_idx).toarray().reshape(-1),
            expected_upper,
        )
        np.testing.assert_allclose(
            compiled.A_ineq.getrow(lower_idx).toarray().reshape(-1),
            expected_lower,
        )


def test_max_sharpe_factor_exposure_solution_feasible():
    lower, upper = _bounds()
    compiled = compile_portfolio_qp(
        small_returns_dict(),
        _params(objective="max_sharpe", w_min=0.0, w_max=1.0),
    )
    solution = solve_compiled_qp_osqp(compiled)
    weights = compiled.recover_stock_weights(solution.x)

    assert_feasible_solution(compiled, solution)
    assert_factor_exposure_bounds(weights, _factor_matrix(), lower, upper)
