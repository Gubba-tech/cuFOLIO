import numpy as np
import pytest
from qp_test_utils import small_returns_dict

from cufolio.exceptions import QPCompilationError
from cufolio.qp_formulations import compile_portfolio_qp
from cufolio.qp_parameters import QPParameters


def _params(**overrides):
    options = {"objective": "max_sharpe", "backend": "osqp"}
    options.update(overrides)
    return QPParameters(**options)


def test_max_sharpe_missing_mu_raises_clear_error():
    with pytest.raises(QPCompilationError, match="mean"):
        compile_portfolio_qp(
            {"covariance": np.eye(3)},
            _params(),
        )


def test_max_sharpe_all_nonpositive_excess_long_only_rejected():
    returns_dict = {
        "mean": np.array([-0.03, 0.0, -0.01]),
        "covariance": np.eye(3),
    }
    with pytest.raises(QPCompilationError, match="strictly positive.*excess"):
        compile_portfolio_qp(returns_dict, _params(w_min=0.0, w_max=1.0))


def test_max_sharpe_risk_free_rate_is_used_in_excess_row():
    returns_dict = {
        "mean": np.array([0.03, 0.05]),
        "covariance": np.diag([0.04, 0.05]),
    }
    compiled = compile_portfolio_qp(
        returns_dict,
        _params(w_min=0.0, w_max=1.0, risk_free_rate=0.04),
    )

    np.testing.assert_allclose(
        compiled.A_eq.getrow(0).toarray().reshape(-1),
        [-0.01, 0.01, 0.0],
    )


def test_max_sharpe_invalid_scale_is_rejected():
    compiled = compile_portfolio_qp(small_returns_dict(), _params())
    invalid = np.zeros(compiled.n_variables)
    invalid[compiled.variable_slices["scale"].start] = 0.0

    with pytest.raises(QPCompilationError, match="c > 0"):
        compiled.recover_max_sharpe_weights(invalid)

    invalid[compiled.variable_slices["scale"].start] = -1.0
    with pytest.raises(QPCompilationError, match="c > 0"):
        compiled.recover_stock_weights(invalid)


def test_max_sharpe_feasibility_check_rejects_infeasible_box():
    with pytest.raises(QPCompilationError, match="no feasible"):
        compile_portfolio_qp(
            small_returns_dict(),
            _params(w_min=0.0, w_max=0.1),
        )
