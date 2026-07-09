import numpy as np

from cufolio.qp_backend import solve_compiled_qp_osqp
from cufolio.qp_formulations import compile_portfolio_qp
from cufolio.qp_parameters import QPParameters


def test_osqp_validation_backend_solves_tiny_min_variance_qp():
    returns_dict = {
        "mean": np.array([0.0, 0.0, 0.0]),
        "covariance": np.diag([1.0, 2.0, 4.0]),
    }
    compiled = compile_portfolio_qp(
        returns_dict,
        QPParameters(objective="min_variance", backend="osqp"),
    )
    solution = solve_compiled_qp_osqp(compiled)

    expected = np.array([1.0, 0.5, 0.25])
    expected = expected / expected.sum()

    assert solution.status in {"optimal", "optimal_inaccurate"}
    assert solution.solver_name == "OSQP"
    assert abs(solution.x.sum() - 1.0) < 1e-6
    assert np.all(solution.x >= -1e-7)
    np.testing.assert_allclose(solution.x, expected, atol=1e-5)
    assert solution.max_constraint_violation <= 1e-6
