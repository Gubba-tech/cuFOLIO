import numpy as np
import pytest
from qp_test_utils import (
    assert_feasible_solution,
    assert_long_short_budget,
    assert_objective_gap_within,
    max_sharpe_medium_returns_dict,
    require_cuopt,
)

from cufolio.qp_backend import solve_compiled_qp_cuopt, solve_compiled_qp_osqp
from cufolio.qp_formulations import compile_portfolio_qp
from cufolio.qp_parameters import QPParameters

MEDIUM_MAX_SHARPE_CASES = [
    ("long_only", {"w_min": 0.0, "w_max": 1.0}),
    ("box", {"w_min": -0.08, "w_max": 0.08}),
    (
        "long_short",
        {"w_min": -1.0, "w_max": 1.0, "short_budget": 0.2},
    ),
    (
        "long_short_l1",
        {
            "w_min": -1.0,
            "w_max": 1.0,
            "short_budget": 0.2,
            "lambda_l1": 0.05,
        },
    ),
    (
        "long_short_l2",
        {
            "w_min": -1.0,
            "w_max": 1.0,
            "short_budget": 0.2,
            "lambda_l2": 0.10,
        },
    ),
    (
        "long_short_l1_l2",
        {
            "w_min": -1.0,
            "w_max": 1.0,
            "short_budget": 0.2,
            "lambda_l1": 0.05,
            "lambda_l2": 0.10,
        },
    ),
]


def _params(options, backend):
    return QPParameters(
        objective="max_sharpe",
        backend=backend,
        **options,
    )


@pytest.mark.parametrize(
    "case_name, options",
    MEDIUM_MAX_SHARPE_CASES,
    ids=[case[0] for case in MEDIUM_MAX_SHARPE_CASES],
)
def test_medium_max_sharpe_osqp_solution_is_feasible(case_name, options):
    del case_name
    compiled = compile_portfolio_qp(
        max_sharpe_medium_returns_dict(n_assets=20),
        _params(options, "osqp"),
    )
    solution = solve_compiled_qp_osqp(compiled)
    weights = compiled.recover_stock_weights(solution.x)

    assert_feasible_solution(compiled, solution, tol=1e-5)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-5)
    assert (compiled.mean - compiled.risk_free_rate) @ weights > 0.0
    if "short_budget" in options:
        assert_long_short_budget(
            weights,
            short_budget=options["short_budget"],
            tol=1e-5,
        )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "case_name, options",
    MEDIUM_MAX_SHARPE_CASES,
    ids=[case[0] for case in MEDIUM_MAX_SHARPE_CASES],
)
def test_medium_max_sharpe_cuopt_matches_osqp_when_available(case_name, options):
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
    np.testing.assert_allclose(cuopt_weights.sum(), 1.0, atol=1e-5)
    if "short_budget" in options:
        assert_long_short_budget(
            cuopt_weights,
            short_budget=options["short_budget"],
            tol=1e-5,
        )
