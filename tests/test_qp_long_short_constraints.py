import numpy as np
import pytest
from qp_test_utils import assert_feasible_solution, small_returns_dict

from cufolio.qp_backend import solve_compiled_qp_osqp
from cufolio.qp_formulations import compile_portfolio_qp
from cufolio.qp_parameters import QPParameters


def _long_short_params(**overrides):
    options = {
        "objective": "mean_variance",
        "risk_aversion": 2.0,
        "w_min": -1.0,
        "w_max": 1.0,
        "short_budget": 0.2,
        "backend": "osqp",
    }
    options.update(overrides)
    return QPParameters(**options)


def test_short_budget_adds_position_split_variables():
    returns_dict = small_returns_dict()
    n_assets = len(returns_dict["mean"])
    compiled = compile_portfolio_qp(
        returns_dict,
        _long_short_params(),
    )

    decision = compiled.variable_slices["decision"]
    pos = compiled.variable_slices["pos"]
    neg = compiled.variable_slices["neg"]
    assert compiled.n_variables == n_assets * 3
    assert decision.size == n_assets
    assert pos.size == n_assets
    assert neg.size == n_assets
    np.testing.assert_allclose(compiled.lower[pos.slice], 0.0)
    np.testing.assert_allclose(compiled.lower[neg.slice], 0.0)
    assert np.all(np.isinf(compiled.upper[pos.slice]))
    assert np.all(np.isinf(compiled.upper[neg.slice]))

    assert compiled.constraint_names[0] == "fully_invested"
    for asset_idx in range(n_assets):
        row_idx = 1 + asset_idx
        row = compiled.A_eq.getrow(row_idx).toarray().reshape(-1)
        expected = np.zeros(compiled.n_variables)
        expected[decision.start + asset_idx] = 1.0
        expected[pos.start + asset_idx] = -1.0
        expected[neg.start + asset_idx] = 1.0
        np.testing.assert_allclose(row, expected, atol=1e-12)
        assert compiled.constraint_names[row_idx] == f"long_short_split_{asset_idx}"
        assert compiled.b_eq[row_idx] == pytest.approx(0.0)

    ineq_names = compiled.constraint_names[compiled.A_eq.shape[0] :]
    long_idx = ineq_names.index("long_budget")
    short_idx = ineq_names.index("short_budget")
    long_row = compiled.A_ineq.getrow(long_idx).toarray().reshape(-1)
    short_row = compiled.A_ineq.getrow(short_idx).toarray().reshape(-1)
    expected_long = np.zeros(compiled.n_variables)
    expected_long[pos.slice] = 1.0
    expected_short = np.zeros(compiled.n_variables)
    expected_short[neg.slice] = 1.0
    np.testing.assert_allclose(long_row, expected_long, atol=1e-12)
    np.testing.assert_allclose(short_row, expected_short, atol=1e-12)
    assert compiled.b_ineq[long_idx] == pytest.approx(1.2)
    assert compiled.b_ineq[short_idx] == pytest.approx(0.2)


def test_short_budget_zero_equivalent_to_long_only():
    returns_dict = small_returns_dict()
    split_compiled = compile_portfolio_qp(
        returns_dict,
        _long_short_params(short_budget=0.0),
    )
    long_only_compiled = compile_portfolio_qp(
        returns_dict,
        QPParameters(
            objective="mean_variance",
            risk_aversion=2.0,
            w_min=0.0,
            w_max=1.0,
            backend="osqp",
        ),
    )

    split_solution = solve_compiled_qp_osqp(split_compiled)
    long_only_solution = solve_compiled_qp_osqp(long_only_compiled)
    assert_feasible_solution(split_compiled, split_solution)
    assert_feasible_solution(long_only_compiled, long_only_solution)
    split_weights = split_compiled.recover_stock_weights(split_solution.x)
    long_only_weights = long_only_compiled.recover_stock_weights(
        long_only_solution.x
    )
    assert np.all(split_weights >= -1e-7)
    np.testing.assert_allclose(split_weights, long_only_weights, atol=1e-5)
    np.testing.assert_allclose(
        split_compiled.objective_value(split_solution.x),
        long_only_compiled.objective_value(long_only_solution.x),
        atol=1e-8,
    )


def test_short_budget_caps_gross_short_exposure():
    returns_dict = small_returns_dict()
    compiled = compile_portfolio_qp(
        returns_dict,
        _long_short_params(),
    )
    solution = solve_compiled_qp_osqp(compiled)
    weights = compiled.recover_stock_weights(solution.x)

    assert_feasible_solution(compiled, solution)
    assert np.maximum(weights, 0.0).sum() <= 1.2 + 1e-6
    assert np.maximum(-weights, 0.0).sum() <= 0.2 + 1e-6
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)


def test_short_budget_invalid_negative_rejected():
    with pytest.raises(ValueError, match="short_budget must be non-negative"):
        QPParameters(short_budget=-0.1)


def test_long_short_split_does_not_use_l1_names():
    compiled = compile_portfolio_qp(
        small_returns_dict(),
        _long_short_params(lambda_l1=0.1),
    )
    assert "pos" in compiled.variable_slices
    assert "neg" in compiled.variable_slices
    assert "l1_pos" in compiled.variable_slices
    assert "l1_neg" in compiled.variable_slices
    assert compiled.variable_slices["pos"] != compiled.variable_slices["l1_pos"]
