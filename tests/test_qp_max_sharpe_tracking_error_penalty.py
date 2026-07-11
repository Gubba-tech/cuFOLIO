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

LAMBDA_TE = 0.4


def _benchmark(n_assets=4):
    return np.ones(n_assets) / n_assets


def _max_params(backend="osqp", **overrides):
    options = {
        "objective": "max_sharpe",
        "w_min": -1.0,
        "w_max": 1.0,
        "benchmark_weights": _benchmark(),
        "lambda_tracking_error": LAMBDA_TE,
        "backend": backend,
    }
    options.update(overrides)
    return QPParameters(**options)


def _compile_pair(params_with_tracking, params_without_tracking=None, returns=None):
    returns = small_returns_dict() if returns is None else returns
    if params_without_tracking is None:
        params_without_tracking = params_with_tracking.model_copy(
            update={"lambda_tracking_error": 0.0}
        )
    return (
        compile_portfolio_qp(returns, params_without_tracking),
        compile_portfolio_qp(returns, params_with_tracking),
    )


def test_max_sharpe_tracking_error_scaled_q_terms_identity():
    base, penalty = _compile_pair(_max_params())
    decision = penalty.variable_slices["decision"]
    scale = penalty.variable_slices["scale"].start
    sigma = small_returns_dict()["covariance"]
    benchmark = _benchmark()
    q_delta = penalty.Q - base.Q

    np.testing.assert_allclose(
        q_delta[decision.slice, decision.slice].toarray(),
        2.0 * LAMBDA_TE * sigma,
    )
    np.testing.assert_allclose(
        q_delta[decision.slice, scale].toarray().reshape(-1),
        -2.0 * LAMBDA_TE * sigma @ benchmark,
    )
    np.testing.assert_allclose(
        q_delta[scale, decision.slice].toarray().reshape(-1),
        -2.0 * LAMBDA_TE * benchmark @ sigma,
    )
    assert q_delta[scale, scale] == pytest.approx(
        2.0 * LAMBDA_TE * benchmark @ sigma @ benchmark
    )
    np.testing.assert_allclose(penalty.q - base.q, 0.0)
    np.testing.assert_allclose(penalty.Q.toarray(), penalty.Q.toarray().T)


def test_max_sharpe_tracking_error_scaled_q_terms_generic_v():
    mapping = np.array(
        [
            [0.6, 0.1],
            [0.2, 0.4],
            [0.1, 0.3],
            [0.1, 0.2],
        ]
    )
    params = _max_params(V=mapping)
    base, penalty = _compile_pair(params)
    decision = penalty.variable_slices["decision"]
    scale = penalty.variable_slices["scale"].start
    sigma = small_returns_dict()["covariance"]
    benchmark = _benchmark()
    q_delta = penalty.Q - base.Q

    np.testing.assert_allclose(
        q_delta[decision.slice, decision.slice].toarray(),
        2.0 * LAMBDA_TE * mapping.T @ sigma @ mapping,
    )
    np.testing.assert_allclose(
        q_delta[decision.slice, scale].toarray().reshape(-1),
        -2.0 * LAMBDA_TE * mapping.T @ sigma @ benchmark,
    )
    assert q_delta[scale, scale] == pytest.approx(
        2.0 * LAMBDA_TE * benchmark @ sigma @ benchmark
    )
    np.testing.assert_allclose(penalty.q - base.q, 0.0)


def test_max_sharpe_tracking_error_scaled_q_terms_factor_space():
    mapping = np.array(
        [
            [0.6, 0.1],
            [0.2, 0.4],
            [0.1, 0.3],
            [0.1, 0.2],
        ]
    )
    stock_covariance = np.diag([0.02, 0.03, 0.04, 0.05])
    returns = {
        "factor_mean": np.array([0.04, 0.03]),
        "factor_covariance": np.diag([0.20, 0.30]),
        "stock_mapping": mapping,
        "stock_covariance": stock_covariance,
    }
    params = _max_params(
        mapping_mode="factor_space",
        V=mapping,
        benchmark_weights=_benchmark(),
    )
    base, penalty = _compile_pair(params, returns=returns)
    decision = penalty.variable_slices["decision"]
    scale = penalty.variable_slices["scale"].start
    benchmark = _benchmark()
    q_delta = penalty.Q - base.Q

    np.testing.assert_allclose(
        q_delta[decision.slice, decision.slice].toarray(),
        2.0 * LAMBDA_TE * mapping.T @ stock_covariance @ mapping,
    )
    np.testing.assert_allclose(
        q_delta[decision.slice, scale].toarray().reshape(-1),
        -2.0 * LAMBDA_TE * mapping.T @ stock_covariance @ benchmark,
    )
    assert q_delta[scale, scale] == pytest.approx(
        2.0 * LAMBDA_TE * benchmark @ stock_covariance @ benchmark
    )
    np.testing.assert_allclose(penalty.q - base.q, 0.0)


def test_max_sharpe_tracking_error_scaled_q_terms_block_is_psd():
    mapping = np.array(
        [
            [0.6, 0.1],
            [0.2, 0.4],
            [0.1, 0.3],
            [0.1, 0.2],
        ]
    )
    sigma = small_returns_dict()["covariance"]
    benchmark = _benchmark()
    exposure = np.column_stack([mapping, -benchmark])
    block = exposure.T @ sigma @ exposure
    np.testing.assert_allclose(block, block.T)
    assert np.linalg.eigvalsh(block).min() >= -1e-10


def test_max_sharpe_tracking_error_penalty_zero_at_benchmark_exposure():
    compiled = compile_portfolio_qp(small_returns_dict(), _max_params())
    decision = compiled.variable_slices["decision"]
    scale = compiled.variable_slices["scale"].start
    c_value = 2.5
    x = np.zeros(compiled.n_variables)
    x[decision.slice] = c_value * _benchmark()
    x[scale] = c_value
    base, penalty = _compile_pair(_max_params())
    delta_q = penalty.Q - base.Q
    delta_q_value = 0.5 * x @ (delta_q @ x)
    assert delta_q_value == pytest.approx(0.0, abs=1e-12)


def test_max_sharpe_tracking_error_osqp_solution_feasible():
    compiled = compile_portfolio_qp(small_returns_dict(), _max_params())
    solution = solve_compiled_qp_osqp(compiled)
    weights = compiled.recover_stock_weights(solution.x)

    assert_feasible_solution(compiled, solution, tol=1e-5)
    assert weights.sum() == pytest.approx(1.0, abs=1e-5)
    assert compiled.mean @ weights > 0.0


@pytest.mark.gpu
def test_max_sharpe_tracking_error_cuopt_matches_osqp_when_available():
    require_cuopt()
    returns = small_returns_dict()
    osqp_compiled = compile_portfolio_qp(returns, _max_params())
    cuopt_compiled = compile_portfolio_qp(
        returns,
        _max_params(backend="cuopt"),
    )
    osqp_solution = solve_compiled_qp_osqp(osqp_compiled)
    cuopt_solution = solve_compiled_qp_cuopt(cuopt_compiled)

    assert cuopt_solution.solver_name == "cuopt_qp"
    assert_feasible_solution(osqp_compiled, osqp_solution, tol=1e-5)
    assert_feasible_solution(cuopt_compiled, cuopt_solution, tol=1e-5)
    assert_objective_gap_within(
        cuopt_compiled,
        cuopt_solution,
        osqp_solution,
        tol=5e-4,
    )
