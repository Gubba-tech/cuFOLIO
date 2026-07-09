import numpy as np

from cufolio.qp_backend import max_constraint_violation
from cufolio.qp_formulations import OBJECTIVE_CONVENTION, compile_portfolio_qp
from cufolio.qp_parameters import QPParameters


def _tiny_returns_dict():
    return {
        "mean": np.array([0.0, 0.0, 0.0]),
        "covariance": np.diag([1.0, 2.0, 4.0]),
    }


def test_compiled_qp_shapes_and_convention():
    compiled = compile_portfolio_qp(
        _tiny_returns_dict(),
        QPParameters(objective="min_variance", backend="osqp"),
    )

    assert compiled.objective_convention == OBJECTIVE_CONVENTION
    assert compiled.Q.shape == (3, 3)
    assert compiled.q.shape == (3,)
    assert compiled.A.shape == (7, 3)
    assert compiled.A_eq.shape == (1, 3)
    assert compiled.A_ineq.shape == (6, 3)
    assert compiled.lower.shape == (3,)
    assert compiled.upper.shape == (3,)
    assert compiled.row_lower.shape == (7,)
    assert compiled.row_upper.shape == (7,)
    assert len(compiled.variable_names) == compiled.n_variables == 3


def test_compiled_objective_uses_half_quadratic_convention():
    compiled = compile_portfolio_qp(
        _tiny_returns_dict(),
        QPParameters(objective="min_variance", backend="osqp"),
    )
    x = np.array([0.5, 0.25, 0.25])
    expected = 0.5 * (1.0 * 0.5**2 + 2.0 * 0.25**2 + 4.0 * 0.25**2)
    assert compiled.objective_value(x) == expected


def test_max_constraint_violation_calculation():
    compiled = compile_portfolio_qp(
        _tiny_returns_dict(),
        QPParameters(objective="min_variance", backend="osqp"),
    )
    feasible = np.array([0.5, 0.25, 0.25])
    infeasible = np.array([0.5, 0.5, 0.5])

    assert max_constraint_violation(compiled, feasible) <= 1e-12
    assert max_constraint_violation(compiled, infeasible) >= 0.5 - 1e-12
