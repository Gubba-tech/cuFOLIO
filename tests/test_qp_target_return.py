import numpy as np
import pytest
from qp_test_utils import (
    assert_cuopt_matches_osqp,
    assert_feasible_solution,
    small_returns_dict,
)

from cufolio.exceptions import QPCompilationError
from cufolio.qp_backend import solve_compiled_qp_osqp
from cufolio.qp_formulations import compile_portfolio_qp
from cufolio.qp_parameters import QPParameters


def test_target_return_requires_target_value():
    with pytest.raises(QPCompilationError):
        compile_portfolio_qp(
            small_returns_dict(),
            QPParameters(objective="target_return", backend="osqp"),
        )


def test_target_return_compiled_row_has_expected_sign():
    returns_dict = small_returns_dict()
    target = 0.040
    compiled = compile_portfolio_qp(
        returns_dict,
        QPParameters(
            objective="target_return",
            target_return=target,
            backend="osqp",
        ),
    )

    ineq_names = compiled.constraint_names[compiled.A_eq.shape[0] :]
    target_row_idx = ineq_names.index("target_return")
    target_row = compiled.A_ineq.getrow(target_row_idx).toarray().reshape(-1)

    np.testing.assert_allclose(target_row, -returns_dict["mean"])
    assert compiled.b_ineq[target_row_idx] == pytest.approx(-target)


def test_target_return_osqp_solution_satisfies_target():
    returns_dict = small_returns_dict()
    target = 0.040
    compiled = compile_portfolio_qp(
        returns_dict,
        QPParameters(
            objective="target_return",
            target_return=target,
            backend="osqp",
        ),
    )
    solution = solve_compiled_qp_osqp(compiled)
    realized_return = float(returns_dict["mean"] @ solution.x)

    assert_feasible_solution(compiled, solution)
    assert realized_return >= target - 1e-6


@pytest.mark.gpu
def test_target_return_cuopt_solution_satisfies_target_when_available():
    returns_dict = small_returns_dict()
    target = 0.040
    compiled = compile_portfolio_qp(
        returns_dict,
        QPParameters(
            objective="target_return",
            target_return=target,
            backend="cuopt",
        ),
    )
    _, cuopt_solution = assert_cuopt_matches_osqp(compiled)
    realized_return = float(returns_dict["mean"] @ cuopt_solution.x)

    assert realized_return >= target - 1e-6
