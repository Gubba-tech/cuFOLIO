import pytest
from qp_test_utils import (
    assert_feasible_solution,
    assert_objective_gap_within,
    medium_returns_dict,
    require_cuopt,
)

from cufolio.qp_backend import solve_compiled_qp_cuopt, solve_compiled_qp_osqp
from cufolio.qp_formulations import compile_portfolio_qp
from cufolio.qp_parameters import QPParameters

MEDIUM_L1_CASES = [
    (
        "mean_variance_l1",
        {"objective": "mean_variance", "risk_aversion": 2.0, "lambda_l1": 0.05},
    ),
    (
        "mean_variance_l1_l2",
        {
            "objective": "mean_variance",
            "risk_aversion": 2.0,
            "lambda_l1": 0.05,
            "lambda_l2": 0.10,
        },
    ),
]


@pytest.mark.parametrize(
    "case_name, options",
    MEDIUM_L1_CASES,
    ids=[case[0] for case in MEDIUM_L1_CASES],
)
def test_medium_l1_qp_osqp_solution_is_feasible(case_name, options):
    del case_name
    compiled = compile_portfolio_qp(
        medium_returns_dict(n_assets=20),
        QPParameters(**options, backend="osqp"),
    )
    solution = solve_compiled_qp_osqp(compiled)

    assert_feasible_solution(compiled, solution, tol=1e-6)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "case_name, options",
    MEDIUM_L1_CASES,
    ids=[case[0] for case in MEDIUM_L1_CASES],
)
def test_medium_l1_qp_cuopt_solution_matches_osqp_when_available(case_name, options):
    del case_name
    require_cuopt()
    compiled = compile_portfolio_qp(
        medium_returns_dict(n_assets=20),
        QPParameters(**options, backend="cuopt"),
    )
    osqp_solution = solve_compiled_qp_osqp(compiled)
    cuopt_solution = solve_compiled_qp_cuopt(compiled)

    assert_feasible_solution(compiled, osqp_solution, tol=1e-6)
    assert_feasible_solution(compiled, cuopt_solution, tol=1e-5)
    assert_objective_gap_within(compiled, cuopt_solution, osqp_solution, tol=5e-4)
