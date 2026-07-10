import importlib.util

import numpy as np
import pytest
from qp_test_utils import (
    assert_feasible_solution,
    assert_long_short_budget,
    assert_objective_gap_within,
    require_cuopt,
    small_returns_dict,
)

from cufolio.exceptions import GPUBackendUnavailable
from cufolio.qp_backend import solve_compiled_qp_cuopt, solve_compiled_qp_osqp
from cufolio.qp_formulations import compile_portfolio_qp
from cufolio.qp_parameters import QPParameters

LONG_SHORT_CASES = [
    ("min_variance", {"objective": "min_variance"}),
    (
        "mean_variance",
        {"objective": "mean_variance", "risk_aversion": 2.0},
    ),
    (
        "target_return",
        {"objective": "target_return", "target_return": 0.040},
    ),
    (
        "mean_variance_l1",
        {
            "objective": "mean_variance",
            "risk_aversion": 2.0,
            "lambda_l1": 0.15,
        },
    ),
    (
        "mean_variance_l2",
        {
            "objective": "mean_variance",
            "risk_aversion": 2.0,
            "lambda_l2": 0.10,
        },
    ),
    (
        "mean_variance_l1_l2",
        {
            "objective": "mean_variance",
            "risk_aversion": 2.0,
            "lambda_l1": 0.15,
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
    LONG_SHORT_CASES,
    ids=[case[0] for case in LONG_SHORT_CASES],
)
def test_long_short_stock_qp_osqp_solution_is_feasible(case_name, options):
    del case_name
    compiled = compile_portfolio_qp(
        small_returns_dict(),
        _params(options, "osqp"),
    )
    solution = solve_compiled_qp_osqp(compiled)
    weights = compiled.recover_stock_weights(solution.x)

    assert_feasible_solution(compiled, solution)
    assert_long_short_budget(weights, short_budget=0.2)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "case_name, options",
    LONG_SHORT_CASES,
    ids=[case[0] for case in LONG_SHORT_CASES],
)
def test_long_short_stock_qp_cuopt_matches_osqp_when_available(case_name, options):
    del case_name
    require_cuopt()
    returns_dict = small_returns_dict()
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
    np.testing.assert_allclose(cuopt_weights.sum(), 1.0, atol=1e-5)


def test_long_short_cuopt_backend_never_falls_back_to_cpu(monkeypatch):
    original_find_spec = importlib.util.find_spec

    def find_spec_without_cuopt(name, *args, **kwargs):
        if name == "cuopt":
            return None
        return original_find_spec(name, *args, **kwargs)

    compiled = compile_portfolio_qp(
        small_returns_dict(),
        _params({"objective": "mean_variance", "risk_aversion": 2.0}, "cuopt"),
    )
    monkeypatch.setattr(importlib.util, "find_spec", find_spec_without_cuopt)

    with pytest.raises(GPUBackendUnavailable):
        solve_compiled_qp_cuopt(compiled)
