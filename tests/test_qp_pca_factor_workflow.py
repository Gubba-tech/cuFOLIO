import numpy as np
import pytest
from qp_test_utils import (
    assert_feasible_solution,
    assert_objective_gap_within,
    require_cuopt,
)

from cufolio.qp_backend import solve_compiled_qp_cuopt, solve_compiled_qp_osqp
from cufolio.qp_factor_workflows import build_pca_factor_qp_data
from cufolio.qp_formulations import compile_portfolio_qp
from cufolio.qp_parameters import QPParameters


def _stock_returns(n_observations=60, n_assets=10):
    rng = np.random.default_rng(321)
    market = rng.normal(0.001, 0.01, size=(n_observations, 1))
    loadings = np.linspace(0.6, 1.4, n_assets).reshape(1, -1)
    noise = rng.normal(0.0, 0.004, size=(n_observations, n_assets))
    return 0.002 + market @ loadings + noise


def _params(data, backend="osqp", **overrides):
    options = {
        "mapping_mode": "factor_space",
        "V": data.stock_mapping,
        "backend": backend,
    }
    options.update(overrides)
    return QPParameters(**options)


def test_build_pca_factor_qp_data_shapes():
    data = build_pca_factor_qp_data(_stock_returns(), n_components=3)

    assert data.stock_mapping.shape == (10, 3)
    assert data.factor_returns.shape == (60, 3)
    assert data.factor_mean.shape == (3,)
    assert data.factor_covariance.shape == (3, 3)


def test_pca_factor_qp_mean_variance_osqp():
    data = build_pca_factor_qp_data(_stock_returns(), n_components=3, center=False)
    compiled = compile_portfolio_qp(
        data.to_returns_dict(),
        _params(data, objective="mean_variance", risk_aversion=2.0, w_min=-1.0, w_max=1.0),
    )
    solution = solve_compiled_qp_osqp(compiled)
    weights = compiled.recover_stock_weights(solution.x)

    assert_feasible_solution(compiled, solution)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)


def test_pca_factor_qp_max_sharpe_l1_l2_long_short_osqp():
    data = build_pca_factor_qp_data(_stock_returns(60, 20), n_components=3, center=False)
    compiled = compile_portfolio_qp(
        data.to_returns_dict(),
        _params(
            data,
            objective="max_sharpe",
            w_min=-0.08,
            w_max=0.08,
            short_budget=0.2,
            lambda_l1=0.01,
            lambda_l2=0.01,
        ),
    )
    solution = solve_compiled_qp_osqp(compiled)
    weights = compiled.recover_stock_weights(solution.x)

    assert_feasible_solution(compiled, solution, tol=1e-5)
    assert np.all(weights >= -0.08 - 1e-5)
    assert np.all(weights <= 0.08 + 1e-5)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-5)


@pytest.mark.gpu
def test_pca_factor_qp_cuopt_matches_osqp_when_available():
    require_cuopt()
    data = build_pca_factor_qp_data(_stock_returns(60, 20), n_components=3, center=False)
    options = {
        "objective": "mean_variance",
        "risk_aversion": 2.0,
        "w_min": -0.08,
        "w_max": 0.08,
    }
    osqp_compiled = compile_portfolio_qp(data.to_returns_dict(), _params(data, **options))
    cuopt_compiled = compile_portfolio_qp(
        data.to_returns_dict(),
        _params(data, backend="cuopt", **options),
    )
    osqp_solution = solve_compiled_qp_osqp(osqp_compiled)
    cuopt_solution = solve_compiled_qp_cuopt(cuopt_compiled)

    assert_feasible_solution(osqp_compiled, osqp_solution, tol=1e-5)
    assert_feasible_solution(cuopt_compiled, cuopt_solution, tol=1e-5)
    assert_objective_gap_within(
        cuopt_compiled,
        cuopt_solution,
        osqp_solution,
        tol=5e-4,
    )
