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


def _twenty_asset_returns_dict():
    return {
        "mean": np.linspace(0.03, 0.06, 20),
        "covariance": np.diag(np.linspace(0.02, 0.08, 20)),
    }


def _params(backend="osqp", **overrides):
    options = {
        "objective": "max_sharpe",
        "risk_free_rate": 0.0,
        "backend": backend,
    }
    options.update(overrides)
    return QPParameters(**options)


def test_max_sharpe_scaled_box_constraint_rows():
    compiled = compile_portfolio_qp(
        _twenty_asset_returns_dict(),
        _params(w_min=-0.08, w_max=0.08),
    )
    decision = compiled.variable_slices["decision"]
    scale = compiled.variable_slices["scale"]
    ineq_names = compiled.constraint_names[compiled.A_eq.shape[0] :]

    upper_idx = ineq_names.index("upper_bound_3")
    lower_idx = ineq_names.index("lower_bound_3")
    upper_row = compiled.A_ineq.getrow(upper_idx).toarray().reshape(-1)
    lower_row = compiled.A_ineq.getrow(lower_idx).toarray().reshape(-1)
    expected_upper = np.zeros(compiled.n_variables)
    expected_upper[decision.start + 3] = 1.0
    expected_upper[scale.start] = -0.08
    expected_lower = np.zeros(compiled.n_variables)
    expected_lower[decision.start + 3] = -1.0
    expected_lower[scale.start] = -0.08

    np.testing.assert_allclose(upper_row, expected_upper)
    np.testing.assert_allclose(lower_row, expected_lower)
    assert compiled.b_ineq[upper_idx] == pytest.approx(0.0)
    assert compiled.b_ineq[lower_idx] == pytest.approx(0.0)


def test_max_sharpe_box_recovered_weights_respect_bounds():
    compiled = compile_portfolio_qp(
        _twenty_asset_returns_dict(),
        _params(w_min=-0.08, w_max=0.08),
    )
    solution = solve_compiled_qp_osqp(compiled)
    weights = compiled.recover_stock_weights(solution.x)

    assert_feasible_solution(compiled, solution)
    assert np.all(weights >= -0.08 - 1e-6)
    assert np.all(weights <= 0.08 + 1e-6)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)


def test_max_sharpe_scaled_long_short_constraint_rows():
    compiled = compile_portfolio_qp(
        small_returns_dict(),
        _params(w_min=-1.0, w_max=1.0, short_budget=0.2),
    )
    scale = compiled.variable_slices["scale"]
    pos = compiled.variable_slices["pos"]
    neg = compiled.variable_slices["neg"]
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
    np.testing.assert_allclose(compiled.b_ineq[[long_idx, short_idx]], [0.0, 0.0])


def test_max_sharpe_long_short_recovered_weights_respect_budget():
    compiled = compile_portfolio_qp(
        small_returns_dict(),
        _params(w_min=-1.0, w_max=1.0, short_budget=0.2),
    )
    solution = solve_compiled_qp_osqp(compiled)
    weights = compiled.recover_stock_weights(solution.x)

    assert_feasible_solution(compiled, solution, tol=1e-5)
    assert_long_short_budget(weights, short_budget=0.2, tol=1e-5)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-5)


@pytest.mark.gpu
def test_max_sharpe_cuopt_long_short_matches_osqp_when_available():
    require_cuopt()
    returns_dict = small_returns_dict()
    options = {"w_min": -1.0, "w_max": 1.0, "short_budget": 0.2}
    osqp_compiled = compile_portfolio_qp(returns_dict, _params(**options))
    cuopt_compiled = compile_portfolio_qp(
        returns_dict,
        _params(backend="cuopt", **options),
    )
    osqp_solution = solve_compiled_qp_osqp(osqp_compiled)
    cuopt_solution = solve_compiled_qp_cuopt(cuopt_compiled)
    cuopt_weights = cuopt_compiled.recover_stock_weights(cuopt_solution.x)

    assert_feasible_solution(osqp_compiled, osqp_solution, tol=1e-5)
    assert_feasible_solution(cuopt_compiled, cuopt_solution, tol=1e-5)
    assert_objective_gap_within(
        cuopt_compiled,
        cuopt_solution,
        osqp_solution,
        tol=5e-4,
    )
    assert_long_short_budget(cuopt_weights, short_budget=0.2, tol=1e-5)
