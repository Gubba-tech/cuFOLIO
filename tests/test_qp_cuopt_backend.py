import numpy as np
import pytest
from qp_test_utils import require_cuopt

from cufolio.qp_backend import solve_compiled_qp_cuopt
from cufolio.qp_formulations import compile_portfolio_qp
from cufolio.qp_parameters import QPParameters


@pytest.mark.gpu
def test_cuopt_backend_solves_tiny_min_variance_qp_directly():
    require_cuopt()

    returns_dict = {
        "mean": np.array([0.0, 0.0, 0.0]),
        "covariance": np.diag([1.0, 2.0, 4.0]),
    }
    compiled = compile_portfolio_qp(
        returns_dict,
        QPParameters(objective="min_variance", backend="cuopt"),
    )
    solution = solve_compiled_qp_cuopt(compiled)

    expected = np.array([1.0, 0.5, 0.25])
    expected = expected / expected.sum()

    assert solution.status == "optimal"
    assert solution.raw_status is not None
    assert solution.solver_name == "cuopt_qp"
    assert abs(solution.x.sum() - 1.0) < 1e-6
    assert np.all(solution.x >= -1e-7)
    np.testing.assert_allclose(solution.x, expected, atol=1e-4)
    assert solution.max_constraint_violation <= 1e-5
    assert abs(solution.objective_value - compiled.objective_value(solution.x)) < 1e-5
