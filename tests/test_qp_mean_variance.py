import importlib.util

import numpy as np
import pytest
from qp_test_utils import (
    assert_cuopt_matches_osqp,
    assert_feasible_solution,
    small_returns_dict,
)

from cufolio.exceptions import GPUBackendUnavailable
from cufolio.qp_backend import solve_compiled_qp_cuopt, solve_compiled_qp_osqp
from cufolio.qp_formulations import compile_portfolio_qp
from cufolio.qp_parameters import QPParameters


def test_mean_variance_compiler_linear_term_matches_gamma_mu():
    returns_dict = small_returns_dict()
    gamma = 4.25
    compiled = compile_portfolio_qp(
        returns_dict,
        QPParameters(
            objective="mean_variance",
            risk_aversion=gamma,
            backend="osqp",
        ),
    )

    np.testing.assert_allclose(compiled.q, -gamma * returns_dict["mean"])


def test_mean_variance_osqp_solution_is_feasible():
    compiled = compile_portfolio_qp(
        small_returns_dict(),
        QPParameters(objective="mean_variance", risk_aversion=3.0, backend="osqp"),
    )
    solution = solve_compiled_qp_osqp(compiled)

    assert_feasible_solution(compiled, solution)
    np.testing.assert_allclose(solution.x.sum(), 1.0, atol=1e-7)
    assert np.all(solution.x >= -1e-7)


def test_mean_variance_cuopt_backend_never_falls_back_to_cpu(monkeypatch):
    original_find_spec = importlib.util.find_spec

    def find_spec_without_cuopt(name, *args, **kwargs):
        if name == "cuopt":
            return None
        return original_find_spec(name, *args, **kwargs)

    compiled = compile_portfolio_qp(
        small_returns_dict(),
        QPParameters(objective="mean_variance", risk_aversion=3.0, backend="cuopt"),
    )
    monkeypatch.setattr(importlib.util, "find_spec", find_spec_without_cuopt)

    with pytest.raises(GPUBackendUnavailable):
        solve_compiled_qp_cuopt(compiled)


@pytest.mark.gpu
def test_mean_variance_cuopt_matches_osqp_when_available():
    compiled = compile_portfolio_qp(
        small_returns_dict(),
        QPParameters(objective="mean_variance", risk_aversion=3.0, backend="cuopt"),
    )

    assert_cuopt_matches_osqp(compiled)
