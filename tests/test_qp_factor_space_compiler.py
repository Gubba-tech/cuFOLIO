import numpy as np
import pytest

from cufolio.exceptions import QPCompilationError
from cufolio.qp_formulations import compile_portfolio_qp
from cufolio.qp_parameters import QPParameters


def _factor_inputs():
    mapping = np.array(
        [
            [0.7, 0.0, 0.0],
            [0.3, 0.2, 0.0],
            [0.0, 0.5, 0.1],
            [0.0, 0.3, 0.4],
            [0.0, 0.0, 0.4],
            [0.0, 0.0, 0.1],
        ]
    )
    return {
        "mean": np.array([0.04, 0.03, 0.02]),
        "covariance": np.diag([0.04, 0.06, 0.08]),
    }, mapping


def _params(mapping, **overrides):
    options = {
        "mapping_mode": "factor_space",
        "V": mapping,
        "backend": "osqp",
    }
    options.update(overrides)
    return QPParameters(**options)


def test_factor_space_dimensions():
    returns_dict, mapping = _factor_inputs()
    compiled = compile_portfolio_qp(returns_dict, _params(mapping))

    assert compiled.mapping_mode == "factor_space"
    assert compiled.n_variables == 3
    assert compiled.stock_mapping.shape == (6, 3)
    np.testing.assert_allclose(
        compiled.A_eq.getrow(0).toarray().reshape(-1),
        np.ones(6) @ mapping,
    )


def test_factor_space_mean_variance_q_and_q():
    returns_dict, mapping = _factor_inputs()
    gamma = 2.5
    compiled = compile_portfolio_qp(
        returns_dict,
        _params(mapping, objective="mean_variance", risk_aversion=gamma),
    )

    np.testing.assert_allclose(compiled.Q.toarray(), returns_dict["covariance"])
    np.testing.assert_allclose(compiled.q, -gamma * returns_dict["mean"])
    assert compiled.q.shape == (mapping.shape[1],)


def test_factor_space_target_return_row():
    returns_dict, mapping = _factor_inputs()
    target = 0.025
    compiled = compile_portfolio_qp(
        returns_dict,
        _params(mapping, objective="target_return", target_return=target),
    )
    ineq_names = compiled.constraint_names[compiled.A_eq.shape[0] :]
    target_idx = ineq_names.index("target_return")

    np.testing.assert_allclose(
        compiled.A_ineq.getrow(target_idx).toarray().reshape(-1),
        -returns_dict["mean"],
    )
    assert compiled.b_ineq[target_idx] == pytest.approx(-target)


def test_factor_space_bounds_rows():
    returns_dict, mapping = _factor_inputs()
    compiled = compile_portfolio_qp(
        returns_dict,
        _params(mapping, w_min=-0.5, w_max=0.8),
    )
    ineq_names = compiled.constraint_names[compiled.A_eq.shape[0] :]
    upper_idx = ineq_names.index("upper_bound_2")
    lower_idx = ineq_names.index("lower_bound_2")

    np.testing.assert_allclose(
        compiled.A_ineq.getrow(upper_idx).toarray().reshape(-1),
        mapping[2],
    )
    np.testing.assert_allclose(
        compiled.A_ineq.getrow(lower_idx).toarray().reshape(-1),
        -mapping[2],
    )


def test_factor_space_invalid_shapes_rejected():
    returns_dict, mapping = _factor_inputs()
    with pytest.raises(QPCompilationError, match="stock_mapping"):
        compile_portfolio_qp(
            returns_dict,
            _params(mapping[:, :2]),
        )
    with pytest.raises(QPCompilationError, match="w_min"):
        compile_portfolio_qp(
            returns_dict,
            _params(mapping, w_min=np.zeros(5)),
        )
