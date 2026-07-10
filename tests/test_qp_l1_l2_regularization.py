import importlib.util

import pytest
from qp_test_utils import (
    assert_feasible_solution,
    assert_objective_gap_within,
    require_cuopt,
    small_returns_dict,
)

from cufolio.exceptions import GPUBackendUnavailable
from cufolio.qp_backend import solve_compiled_qp_cuopt, solve_compiled_qp_osqp
from cufolio.qp_formulations import compile_portfolio_qp
from cufolio.qp_parameters import QPParameters


def _l1_l2_params(backend: str):
    return QPParameters(
        objective="mean_variance",
        risk_aversion=2.5,
        lambda_l1=0.20,
        lambda_l2=0.15,
        backend=backend,
    )


def test_l1_l2_regularized_qp_osqp_solution_feasible():
    compiled = compile_portfolio_qp(
        small_returns_dict(),
        _l1_l2_params("osqp"),
    )
    solution = solve_compiled_qp_osqp(compiled)

    assert_feasible_solution(compiled, solution)


@pytest.mark.gpu
def test_l1_l2_regularized_qp_cuopt_matches_osqp_when_available():
    require_cuopt()
    compiled = compile_portfolio_qp(
        small_returns_dict(),
        _l1_l2_params("cuopt"),
    )
    osqp_solution = solve_compiled_qp_osqp(compiled)
    cuopt_solution = solve_compiled_qp_cuopt(compiled)

    assert_feasible_solution(compiled, osqp_solution, tol=1e-6)
    assert_feasible_solution(compiled, cuopt_solution, tol=1e-5)
    assert_objective_gap_within(compiled, cuopt_solution, osqp_solution, tol=5e-4)


def test_l1_l2_cuopt_backend_never_falls_back_to_cpu(monkeypatch):
    original_find_spec = importlib.util.find_spec

    def find_spec_without_cuopt(name, *args, **kwargs):
        if name == "cuopt":
            return None
        return original_find_spec(name, *args, **kwargs)

    compiled = compile_portfolio_qp(
        small_returns_dict(),
        _l1_l2_params("cuopt"),
    )
    monkeypatch.setattr(importlib.util, "find_spec", find_spec_without_cuopt)

    with pytest.raises(GPUBackendUnavailable):
        solve_compiled_qp_cuopt(compiled)
