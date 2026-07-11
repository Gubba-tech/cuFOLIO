import numpy as np
import pytest
from qp_test_utils import (
    assert_benchmark_l1_budget,
    assert_factor_exposure_bounds,
    assert_feasible_solution,
    assert_long_short_budget,
    assert_objective_gap_within,
    assert_turnover_budget,
    max_sharpe_medium_returns_dict,
    require_cuopt,
)

from cufolio.qp_backend import solve_compiled_qp_cuopt, solve_compiled_qp_osqp
from cufolio.qp_formulations import compile_portfolio_qp
from cufolio.qp_parameters import QPParameters


def _factor_matrix(n_assets=20):
    return np.column_stack(
        [np.ones(n_assets), np.linspace(-1.0, 1.0, n_assets)]
    )


def _anchors(n_assets=20):
    anchor = np.ones(n_assets) / n_assets
    return anchor, anchor.copy()


def _factor_options():
    return {
        "factor_exposure_matrix": _factor_matrix(),
        "factor_exposure_lower": np.array([0.5, -0.5]),
        "factor_exposure_upper": np.array([1.5, 0.5]),
    }


ORDINARY_CASES = [
    (
        "turnover",
        {
            "objective": "mean_variance",
            "risk_aversion": 2.0,
            "previous_weights": _anchors()[0],
            "turnover_budget": 0.5,
        },
    ),
    (
        "benchmark_l1",
        {
            "objective": "mean_variance",
            "risk_aversion": 2.0,
            "benchmark_weights": _anchors()[1],
            "benchmark_l1_budget": 0.5,
        },
    ),
    (
        "factor_exposure",
        {"objective": "mean_variance", "risk_aversion": 2.0, **_factor_options()},
    ),
    (
        "tracking_error",
        {
            "objective": "mean_variance",
            "risk_aversion": 2.0,
            "benchmark_weights": _anchors()[1],
            "lambda_tracking_error": 0.10,
        },
    ),
    (
        "combined",
        {
            "objective": "mean_variance",
            "risk_aversion": 2.0,
            "w_min": -1.0,
            "w_max": 1.0,
            "short_budget": 0.2,
            "lambda_l1": 0.02,
            "lambda_l2": 0.02,
            "previous_weights": _anchors()[0],
            "turnover_budget": 0.5,
            "benchmark_weights": _anchors()[1],
            "benchmark_l1_budget": 0.5,
        },
    ),
]


MAX_SHARPE_CASES = [
    (
        "max_sharpe_turnover",
        {
            "objective": "max_sharpe",
            "previous_weights": _anchors()[0],
            "turnover_budget": 0.5,
        },
    ),
    (
        "max_sharpe_benchmark_l1",
        {
            "objective": "max_sharpe",
            "benchmark_weights": _anchors()[1],
            "benchmark_l1_budget": 0.5,
        },
    ),
    (
        "max_sharpe_factor_exposure",
        {"objective": "max_sharpe", **_factor_options()},
    ),
    (
        "max_sharpe_combined",
        {
            "objective": "max_sharpe",
            "w_min": -1.0,
            "w_max": 1.0,
            "short_budget": 0.2,
            "lambda_l1": 0.02,
            "lambda_l2": 0.02,
            "previous_weights": _anchors()[0],
            "turnover_budget": 0.5,
            "benchmark_weights": _anchors()[1],
            "benchmark_l1_budget": 0.5,
        },
    ),
]


def _params(options, backend):
    return QPParameters(**options, backend=backend)


def _assert_recovered_constraints(compiled, weights, options):
    assert np.sum(weights) == pytest.approx(1.0, abs=1e-5)
    if "short_budget" in options:
        assert_long_short_budget(weights, options["short_budget"], tol=1e-5)
    if "previous_weights" in options:
        assert_turnover_budget(
            weights,
            options["previous_weights"],
            options["turnover_budget"],
            tol=1e-5,
        )
    if "benchmark_l1_budget" in options:
        assert_benchmark_l1_budget(
            weights,
            options["benchmark_weights"],
            options["benchmark_l1_budget"],
            tol=1e-5,
        )
    if "factor_exposure_matrix" in options:
        assert_factor_exposure_bounds(
            weights,
            options["factor_exposure_matrix"],
            options["factor_exposure_lower"],
            options["factor_exposure_upper"],
            tol=1e-5,
        )


@pytest.mark.parametrize(
    "case_name, options",
    ORDINARY_CASES,
    ids=[case[0] for case in ORDINARY_CASES],
)
def test_medium_friction_constraints_osqp_solution_is_feasible(case_name, options):
    del case_name
    returns_dict = max_sharpe_medium_returns_dict(n_assets=20)
    compiled = compile_portfolio_qp(returns_dict, _params(options, "osqp"))
    solution = solve_compiled_qp_osqp(compiled)
    weights = compiled.recover_stock_weights(solution.x)

    assert_feasible_solution(compiled, solution, tol=1e-5)
    _assert_recovered_constraints(compiled, weights, options)


@pytest.mark.parametrize(
    "case_name, options",
    MAX_SHARPE_CASES,
    ids=[case[0] for case in MAX_SHARPE_CASES],
)
def test_medium_friction_constraints_max_sharpe_osqp_solution_is_feasible(
    case_name,
    options,
):
    del case_name
    returns_dict = max_sharpe_medium_returns_dict(n_assets=20)
    compiled = compile_portfolio_qp(returns_dict, _params(options, "osqp"))
    solution = solve_compiled_qp_osqp(compiled)
    weights = compiled.recover_stock_weights(solution.x)

    assert_feasible_solution(compiled, solution, tol=1e-5)
    _assert_recovered_constraints(compiled, weights, options)
    assert (compiled.mean - compiled.risk_free_rate) @ weights > 0.0


@pytest.mark.gpu
@pytest.mark.parametrize(
    "case_name, options",
    ORDINARY_CASES + MAX_SHARPE_CASES,
    ids=[case[0] for case in ORDINARY_CASES + MAX_SHARPE_CASES],
)
def test_medium_friction_constraints_cuopt_matches_osqp_when_available(
    case_name,
    options,
):
    del case_name
    require_cuopt()
    returns_dict = max_sharpe_medium_returns_dict(n_assets=20)
    osqp_compiled = compile_portfolio_qp(returns_dict, _params(options, "osqp"))
    cuopt_compiled = compile_portfolio_qp(
        returns_dict,
        _params(options, "cuopt"),
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
    _assert_recovered_constraints(cuopt_compiled, cuopt_weights, options)
