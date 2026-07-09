import numpy as np
import pandas as pd
import pytest

from cufolio.exceptions import GPUBackendUnavailable
from cufolio.qp_backend import solve_compiled_qp_cuopt
from cufolio.qp_formulations import compile_portfolio_qp
from cufolio.qp_optimizer import QuadraticPortfolioOptimizer
from cufolio.qp_parameters import QPParameters


@pytest.fixture()
def returns_dict():
    covariance = np.array(
        [
            [0.050, 0.010, 0.004, 0.002, 0.001],
            [0.010, 0.060, 0.006, 0.003, 0.002],
            [0.004, 0.006, 0.040, 0.005, 0.001],
            [0.002, 0.003, 0.005, 0.030, 0.004],
            [0.001, 0.002, 0.001, 0.004, 0.020],
        ]
    )
    mean = np.array([0.030, 0.040, 0.025, 0.020, 0.015])
    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    returns = pd.DataFrame(
        np.tile(mean, (20, 1)),
        index=dates,
        columns=["A", "B", "C", "D", "E"],
    )
    return {
        "return_type": "LOG",
        "returns": returns,
        "regime": {"name": "toy", "range": (dates[0], dates[-1])},
        "dates": dates,
        "mean": mean,
        "covariance": covariance,
        "tickers": list(returns.columns),
    }


def test_qp_min_variance_compiler_dimensions(returns_dict):
    params = QPParameters(objective="min_variance", backend="osqp")
    compiled = compile_portfolio_qp(returns_dict, params)

    assert compiled.Q.shape == (5, 5)
    assert compiled.q.shape == (5,)
    assert compiled.A_eq.shape == (1, 5)
    assert compiled.A_ineq.shape == (10, 5)
    assert compiled.n_constraints == 11


def test_qp_min_variance_osqp_solution(returns_dict):
    params = QPParameters(objective="min_variance", backend="osqp")
    optimizer = QuadraticPortfolioOptimizer(returns_dict, params)
    result, portfolio = optimizer.solve_optimization_problem(print_results=False)

    assert result["solver"] == "OSQP"
    assert result["status"] in {"optimal", "optimal_inaccurate"}
    np.testing.assert_allclose(portfolio.weights.sum(), 1.0, atol=1e-6)
    assert np.all(portfolio.weights >= -1e-7)
    assert result["max_constraint_violation"] <= 1e-5


def test_qp_mean_variance_differs_from_min_variance(returns_dict):
    min_params = QPParameters(objective="min_variance", backend="osqp")
    mv_params = QPParameters(
        objective="mean_variance",
        risk_aversion=10.0,
        backend="osqp",
    )
    _, min_portfolio = QuadraticPortfolioOptimizer(
        returns_dict, min_params
    ).solve_optimization_problem(print_results=False)
    _, mv_portfolio = QuadraticPortfolioOptimizer(
        returns_dict, mv_params
    ).solve_optimization_problem(print_results=False)

    assert not np.allclose(min_portfolio.weights, mv_portfolio.weights, atol=1e-4)


def test_qp_target_return_constraint(returns_dict):
    target = 0.032
    params = QPParameters(
        objective="target_return",
        target_return=target,
        backend="osqp",
    )
    result, portfolio = QuadraticPortfolioOptimizer(
        returns_dict, params
    ).solve_optimization_problem(print_results=False)

    assert result["expected_return"] >= target - 1e-6
    np.testing.assert_allclose(portfolio.weights.sum(), 1.0, atol=1e-6)


def test_qp_max_sharpe_recovered_weights_sum_to_one(returns_dict):
    params = QPParameters(
        objective="max_sharpe",
        w_min=0.0,
        w_max=1.0,
        risk_free_rate=0.0,
        backend="osqp",
    )
    result, portfolio = QuadraticPortfolioOptimizer(
        returns_dict, params
    ).solve_optimization_problem(print_results=False)

    np.testing.assert_allclose(portfolio.weights.sum(), 1.0, atol=1e-6)
    assert result["variance"] >= 0


def test_qp_l1_regularization_split(returns_dict):
    params = QPParameters(lambda_l1=0.1, backend="osqp")
    compiled = compile_portfolio_qp(returns_dict, params)
    assert "l1_pos" in compiled.variable_slices
    assert "l1_neg" in compiled.variable_slices
    assert compiled.n_variables == 15

    _, portfolio = QuadraticPortfolioOptimizer(
        returns_dict, params
    ).solve_optimization_problem(print_results=False)
    np.testing.assert_allclose(portfolio.weights.sum(), 1.0, atol=1e-6)


def test_qp_l2_regularization_changes_q(returns_dict):
    base = compile_portfolio_qp(returns_dict, QPParameters(backend="osqp"))
    reg = compile_portfolio_qp(
        returns_dict, QPParameters(lambda_l2=0.5, backend="osqp")
    )
    diff = reg.Q.toarray() - base.Q.toarray()
    np.testing.assert_allclose(np.diag(diff), np.ones(5), atol=1e-12)


def test_qp_long_short_budget(returns_dict):
    params = QPParameters(
        objective="mean_variance",
        risk_aversion=10.0,
        w_min=-0.4,
        w_max=0.8,
        short_budget=0.2,
        backend="osqp",
    )
    _, portfolio = QuadraticPortfolioOptimizer(
        returns_dict, params
    ).solve_optimization_problem(print_results=False)
    weights = portfolio.weights
    assert np.maximum(weights, 0).sum() <= 1.2 + 1e-6
    assert np.maximum(-weights, 0).sum() <= 0.2 + 1e-6


def test_qp_turnover_constraint(returns_dict):
    previous = np.ones(5) / 5
    params = QPParameters(
        objective="mean_variance",
        risk_aversion=10.0,
        previous_weights=previous,
        turnover_budget=0.1,
        backend="osqp",
    )
    _, portfolio = QuadraticPortfolioOptimizer(
        returns_dict, params
    ).solve_optimization_problem(print_results=False)
    assert np.abs(portfolio.weights - previous).sum() <= 0.1 + 1e-6


def test_qp_benchmark_l1_constraint(returns_dict):
    benchmark = np.ones(5) / 5
    params = QPParameters(
        objective="mean_variance",
        risk_aversion=10.0,
        benchmark_weights=benchmark,
        benchmark_l1_budget=0.2,
        backend="osqp",
    )
    _, portfolio = QuadraticPortfolioOptimizer(
        returns_dict, params
    ).solve_optimization_problem(print_results=False)
    assert np.abs(portfolio.weights - benchmark).sum() <= 0.2 + 1e-6


def test_qp_factor_mapping_returns_stock_weights(returns_dict):
    V = np.array(
        [
            [0.8, 0.0],
            [0.2, 0.1],
            [0.0, 0.4],
            [0.0, 0.3],
            [0.0, 0.2],
        ]
    )
    params = QPParameters(V=V, w_min=0.0, w_max=1.0, backend="osqp")
    result, portfolio = QuadraticPortfolioOptimizer(
        returns_dict, params
    ).solve_optimization_problem(print_results=False)

    assert result["mapping_matrix_shape"] == (5, 2)
    assert result["factor_weights"] is not None
    assert portfolio.weights.shape == (5,)
    np.testing.assert_allclose(portfolio.weights.sum(), 1.0, atol=1e-6)


def test_qp_cuopt_guard_no_cpu_fallback(returns_dict):
    params = QPParameters(backend="cuopt")
    compiled = compile_portfolio_qp(returns_dict, params)
    with pytest.raises((GPUBackendUnavailable, NotImplementedError)):
        solve_compiled_qp_cuopt(compiled)
