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


def _benchmark_weights():
    return np.ones(4) / 4


def _params(backend="osqp", **overrides):
    options = {
        "objective": "mean_variance",
        "risk_aversion": 2.0,
        "benchmark_weights": _benchmark_weights(),
        "benchmark_l1_budget": 0.30,
        "backend": backend,
    }
    options.update(overrides)
    return QPParameters(**options)


def test_benchmark_l1_adds_auxiliary_variables():
    compiled = compile_portfolio_qp(small_returns_dict(), _params())
    decision = compiled.variable_slices["decision"]
    pos = compiled.variable_slices["benchmark_pos"]
    neg = compiled.variable_slices["benchmark_neg"]

    assert compiled.n_variables == 12
    assert pos.size == neg.size == 4
    np.testing.assert_allclose(compiled.lower[pos.slice], 0.0)
    np.testing.assert_allclose(compiled.lower[neg.slice], 0.0)
    for asset_idx, benchmark in enumerate(_benchmark_weights()):
        row_idx = 1 + asset_idx
        expected = np.zeros(compiled.n_variables)
        expected[decision.start + asset_idx] = 1.0
        expected[pos.start + asset_idx] = -1.0
        expected[neg.start + asset_idx] = 1.0
        np.testing.assert_allclose(
            compiled.A_eq.getrow(row_idx).toarray().reshape(-1),
            expected,
        )
        assert compiled.b_eq[row_idx] == pytest.approx(benchmark)
        assert compiled.constraint_names[row_idx] == f"benchmark_split_{asset_idx}"

    ineq_names = compiled.constraint_names[compiled.A_eq.shape[0] :]
    budget_idx = ineq_names.index("benchmark_budget")
    expected_budget = np.zeros(compiled.n_variables)
    expected_budget[pos.slice] = 1.0
    expected_budget[neg.slice] = 1.0
    np.testing.assert_allclose(
        compiled.A_ineq.getrow(budget_idx).toarray().reshape(-1),
        expected_budget,
    )
    assert compiled.b_ineq[budget_idx] == pytest.approx(0.30)


def test_benchmark_requires_benchmark_weights():
    with pytest.raises(QPCompilationError, match="benchmark_weights"):
        compile_portfolio_qp(
            small_returns_dict(),
            QPParameters(benchmark_l1_budget=0.30, backend="osqp"),
        )


def test_benchmark_l1_osqp_solution_feasible():
    compiled = compile_portfolio_qp(small_returns_dict(), _params())
    solution = solve_compiled_qp_osqp(compiled)

    assert_feasible_solution(compiled, solution)


@pytest.mark.gpu
def test_benchmark_l1_cuopt_matches_osqp_when_available():
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


def test_max_sharpe_scaled_benchmark_rows():
    compiled = compile_portfolio_qp(
        small_returns_dict(),
        _params(objective="max_sharpe", w_min=0.0, w_max=1.0),
    )
    decision = compiled.variable_slices["decision"]
    scale = compiled.variable_slices["scale"]
    pos = compiled.variable_slices["benchmark_pos"]
    neg = compiled.variable_slices["benchmark_neg"]
    for asset_idx, benchmark in enumerate(_benchmark_weights()):
        row_idx = 2 + asset_idx
        expected = np.zeros(compiled.n_variables)
        expected[decision.start + asset_idx] = 1.0
        expected[scale.start] = -benchmark
        expected[pos.start + asset_idx] = -1.0
        expected[neg.start + asset_idx] = 1.0
        np.testing.assert_allclose(
            compiled.A_eq.getrow(row_idx).toarray().reshape(-1),
            expected,
        )

    ineq_names = compiled.constraint_names[compiled.A_eq.shape[0] :]
    budget_idx = ineq_names.index("benchmark_budget")
    expected_budget = np.zeros(compiled.n_variables)
    expected_budget[pos.slice] = 1.0
    expected_budget[neg.slice] = 1.0
    expected_budget[scale.start] = -0.30
    np.testing.assert_allclose(
        compiled.A_ineq.getrow(budget_idx).toarray().reshape(-1),
        expected_budget,
    )


def test_max_sharpe_benchmark_recovered_weights_feasible():
    compiled = compile_portfolio_qp(
        small_returns_dict(),
        _params(objective="max_sharpe", w_min=0.0, w_max=1.0),
    )
    solution = solve_compiled_qp_osqp(compiled)
    weights = compiled.recover_stock_weights(solution.x)

    assert_feasible_solution(compiled, solution)
    assert np.sum(np.abs(weights - _benchmark_weights())) <= 0.30 + 1e-6


@pytest.mark.gpu
def test_max_sharpe_benchmark_cuopt_matches_osqp_when_available():
    require_cuopt()
    returns_dict = small_returns_dict()
    options = {"objective": "max_sharpe", "w_min": 0.0, "w_max": 1.0}
    osqp_compiled = compile_portfolio_qp(returns_dict, _params(**options))
    cuopt_compiled = compile_portfolio_qp(
        returns_dict,
        _params(backend="cuopt", **options),
    )
    osqp_solution = solve_compiled_qp_osqp(osqp_compiled)
    cuopt_solution = solve_compiled_qp_cuopt(cuopt_compiled)
    cuopt_weights = cuopt_compiled.recover_stock_weights(cuopt_solution.x)

    assert_feasible_solution(osqp_compiled, osqp_solution)
    assert_feasible_solution(cuopt_compiled, cuopt_solution, tol=1e-5)
    assert_objective_gap_within(
        cuopt_compiled,
        cuopt_solution,
        osqp_solution,
        tol=5e-4,
    )
    assert np.sum(np.abs(cuopt_weights - _benchmark_weights())) <= 0.30 + 1e-5
