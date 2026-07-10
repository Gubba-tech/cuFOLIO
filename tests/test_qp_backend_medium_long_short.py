import pytest
from qp_test_utils import (
    assert_feasible_solution,
    assert_long_short_budget,
    assert_objective_gap_within,
    medium_returns_dict,
    require_cuopt,
)

from cufolio.qp_backend import solve_compiled_qp_cuopt, solve_compiled_qp_osqp
from cufolio.qp_formulations import compile_portfolio_qp
from cufolio.qp_parameters import QPParameters

MEDIUM_LONG_SHORT_CASES = [
    (
        "mean_variance_short",
        {"objective": "mean_variance", "risk_aversion": 2.0},
    ),
    (
        "mean_variance_short_l1",
        {
            "objective": "mean_variance",
            "risk_aversion": 2.0,
            "lambda_l1": 0.05,
        },
    ),
    (
        "mean_variance_short_l2",
        {
            "objective": "mean_variance",
            "risk_aversion": 2.0,
            "lambda_l2": 0.10,
        },
    ),
    (
        "mean_variance_short_l1_l2",
        {
            "objective": "mean_variance",
            "risk_aversion": 2.0,
            "lambda_l1": 0.05,
            "lambda_l2": 0.10,
        },
    ),
]


def _params(options, backend):
    return QPParameters(
        **options,
        w_min=-1.0,
        w_max=1.0,
        short_budget=0.2,
        backend=backend,
    )


@pytest.mark.parametrize(
    "case_name, options",
    MEDIUM_LONG_SHORT_CASES,
    ids=[case[0] for case in MEDIUM_LONG_SHORT_CASES],
)
def test_medium_long_short_qp_osqp_solution_is_feasible(case_name, options):
    del case_name
    compiled = compile_portfolio_qp(
        medium_returns_dict(n_assets=20),
        _params(options, "osqp"),
    )
    solution = solve_compiled_qp_osqp(compiled)
    weights = compiled.recover_stock_weights(solution.x)

    assert_feasible_solution(compiled, solution, tol=1e-6)
    assert_long_short_budget(weights, short_budget=0.2)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "case_name, options",
    MEDIUM_LONG_SHORT_CASES,
    ids=[case[0] for case in MEDIUM_LONG_SHORT_CASES],
)
def test_medium_long_short_qp_cuopt_matches_osqp_when_available(case_name, options):
    del case_name
    require_cuopt()
    returns_dict = medium_returns_dict(n_assets=20)
    osqp_compiled = compile_portfolio_qp(returns_dict, _params(options, "osqp"))
    cuopt_compiled = compile_portfolio_qp(returns_dict, _params(options, "cuopt"))
    osqp_solution = solve_compiled_qp_osqp(osqp_compiled)
    cuopt_solution = solve_compiled_qp_cuopt(cuopt_compiled)
    cuopt_weights = cuopt_compiled.recover_stock_weights(cuopt_solution.x)

    assert_feasible_solution(osqp_compiled, osqp_solution, tol=1e-6)
    assert_feasible_solution(cuopt_compiled, cuopt_solution, tol=1e-5)
    assert_objective_gap_within(
        cuopt_compiled,
        cuopt_solution,
        osqp_solution,
        tol=5e-4,
    )
    assert_long_short_budget(cuopt_weights, short_budget=0.2, tol=1e-5)
