import numpy as np
import pytest
from qp_test_utils import (
    assert_benchmark_l1_budget,
    assert_feasible_solution,
    assert_objective_gap_within,
    assert_turnover_budget,
    require_cuopt,
    small_returns_dict,
)

from cufolio.qp_backend import solve_compiled_qp_cuopt, solve_compiled_qp_osqp
from cufolio.qp_formulations import compile_portfolio_qp
from cufolio.qp_parameters import QPParameters


def _mapping():
    return np.array(
        [
            [0.8, 0.0],
            [0.2, 0.1],
            [0.0, 0.4],
            [0.0, 0.5],
        ]
    )


def _anchor():
    return _mapping() @ np.array([0.5, 0.5])


def _params(backend="osqp", **overrides):
    options = {
        "V": _mapping(),
        "objective": "mean_variance",
        "risk_aversion": 2.0,
        "w_min": 0.0,
        "w_max": 1.0,
        "previous_weights": _anchor(),
        "turnover_budget": 0.25,
        "benchmark_weights": _anchor(),
        "benchmark_l1_budget": 0.30,
        "backend": backend,
    }
    options.update(overrides)
    return QPParameters(**options)


def _assert_anchor_rows(compiled, pos_name, neg_name, anchor, row_start):
    decision = compiled.variable_slices["decision"]
    pos = compiled.variable_slices[pos_name]
    neg = compiled.variable_slices[neg_name]
    for asset_idx, value in enumerate(anchor):
        expected = np.zeros(compiled.n_variables)
        expected[decision.slice] = _mapping()[asset_idx]
        expected[pos.start + asset_idx] = -1.0
        expected[neg.start + asset_idx] = 1.0
        np.testing.assert_allclose(
            compiled.A_eq.getrow(row_start + asset_idx).toarray().reshape(-1),
            expected,
        )
        assert compiled.b_eq[row_start + asset_idx] == pytest.approx(value)


def test_turnover_generic_v_rows():
    compiled = compile_portfolio_qp(
        small_returns_dict(),
        _params(benchmark_weights=None, benchmark_l1_budget=None),
    )
    _assert_anchor_rows(
        compiled,
        "turnover_pos",
        "turnover_neg",
        _anchor(),
        row_start=1,
    )
    ineq_names = compiled.constraint_names[compiled.A_eq.shape[0] :]
    budget_idx = ineq_names.index("turnover_budget")
    pos = compiled.variable_slices["turnover_pos"]
    neg = compiled.variable_slices["turnover_neg"]
    expected = np.zeros(compiled.n_variables)
    expected[pos.slice] = 1.0
    expected[neg.slice] = 1.0
    np.testing.assert_allclose(
        compiled.A_ineq.getrow(budget_idx).toarray().reshape(-1),
        expected,
    )
    assert compiled.b_ineq[budget_idx] == pytest.approx(0.25)


def test_benchmark_generic_v_rows():
    compiled = compile_portfolio_qp(
        small_returns_dict(),
        _params(previous_weights=None, turnover_budget=None),
    )
    _assert_anchor_rows(
        compiled,
        "benchmark_pos",
        "benchmark_neg",
        _anchor(),
        row_start=1,
    )
    ineq_names = compiled.constraint_names[compiled.A_eq.shape[0] :]
    budget_idx = ineq_names.index("benchmark_budget")
    pos = compiled.variable_slices["benchmark_pos"]
    neg = compiled.variable_slices["benchmark_neg"]
    expected = np.zeros(compiled.n_variables)
    expected[pos.slice] = 1.0
    expected[neg.slice] = 1.0
    np.testing.assert_allclose(
        compiled.A_ineq.getrow(budget_idx).toarray().reshape(-1),
        expected,
    )
    assert compiled.b_ineq[budget_idx] == pytest.approx(0.30)


def test_max_sharpe_turnover_generic_v_scaled_rows():
    compiled = compile_portfolio_qp(
        small_returns_dict(),
        _params(
            objective="max_sharpe",
            w_min=-1.0,
            w_max=1.0,
            benchmark_weights=None,
            benchmark_l1_budget=None,
        ),
    )
    decision = compiled.variable_slices["decision"]
    scale = compiled.variable_slices["scale"]
    pos = compiled.variable_slices["turnover_pos"]
    neg = compiled.variable_slices["turnover_neg"]
    for asset_idx, previous in enumerate(_anchor()):
        row_idx = 2 + asset_idx
        expected = np.zeros(compiled.n_variables)
        expected[decision.slice] = _mapping()[asset_idx]
        expected[scale.start] = -previous
        expected[pos.start + asset_idx] = -1.0
        expected[neg.start + asset_idx] = 1.0
        np.testing.assert_allclose(
            compiled.A_eq.getrow(row_idx).toarray().reshape(-1),
            expected,
        )
    ineq_names = compiled.constraint_names[compiled.A_eq.shape[0] :]
    budget_idx = ineq_names.index("turnover_budget")
    expected_budget = np.zeros(compiled.n_variables)
    expected_budget[pos.slice] = 1.0
    expected_budget[neg.slice] = 1.0
    expected_budget[scale.start] = -0.25
    np.testing.assert_allclose(
        compiled.A_ineq.getrow(budget_idx).toarray().reshape(-1),
        expected_budget,
    )


def test_max_sharpe_benchmark_generic_v_scaled_rows():
    compiled = compile_portfolio_qp(
        small_returns_dict(),
        _params(
            objective="max_sharpe",
            w_min=-1.0,
            w_max=1.0,
            previous_weights=None,
            turnover_budget=None,
        ),
    )
    decision = compiled.variable_slices["decision"]
    scale = compiled.variable_slices["scale"]
    pos = compiled.variable_slices["benchmark_pos"]
    neg = compiled.variable_slices["benchmark_neg"]
    for asset_idx, benchmark in enumerate(_anchor()):
        row_idx = 2 + asset_idx
        expected = np.zeros(compiled.n_variables)
        expected[decision.slice] = _mapping()[asset_idx]
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


def test_generic_v_turnover_benchmark_osqp_solution_feasible():
    compiled = compile_portfolio_qp(small_returns_dict(), _params())
    solution = solve_compiled_qp_osqp(compiled)
    weights = compiled.recover_stock_weights(solution.x)

    assert_feasible_solution(compiled, solution)
    assert_turnover_budget(weights, _anchor(), 0.25)
    assert_benchmark_l1_budget(weights, _anchor(), 0.30)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)


@pytest.mark.gpu
def test_generic_v_turnover_benchmark_cuopt_matches_osqp_when_available():
    require_cuopt()
    returns_dict = small_returns_dict()
    osqp_compiled = compile_portfolio_qp(returns_dict, _params())
    cuopt_compiled = compile_portfolio_qp(
        returns_dict,
        _params(backend="cuopt"),
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
    assert_turnover_budget(cuopt_weights, _anchor(), 0.25, tol=1e-5)
    assert_benchmark_l1_budget(cuopt_weights, _anchor(), 0.30, tol=1e-5)
