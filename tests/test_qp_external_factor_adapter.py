import numpy as np
import pytest
from qp_test_utils import (
    assert_feasible_solution,
    assert_objective_gap_within,
    require_cuopt,
)

from cufolio.qp_backend import solve_compiled_qp_cuopt, solve_compiled_qp_osqp
from cufolio.qp_factor_workflows import build_external_factor_qp_data
from cufolio.qp_formulations import compile_portfolio_qp
from cufolio.qp_parameters import QPParameters


def _external_data():
    rng = np.random.default_rng(7)
    factor_returns = rng.normal(0.01, 0.02, size=(50, 3))
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
    stock_returns = factor_returns @ mapping.T + rng.normal(
        0.0, 0.002, size=(50, 6)
    )
    return build_external_factor_qp_data(
        factor_returns,
        mapping,
        stock_returns=stock_returns,
        tickers=[f"S{idx}" for idx in range(6)],
        factor_names=["F0", "F1", "F2"],
        model_name="toy_external",
    )


def _params(data, backend="osqp", **overrides):
    options = {
        "mapping_mode": "factor_space",
        "V": data.stock_mapping,
        "backend": backend,
    }
    options.update(overrides)
    return QPParameters(**options)


def test_external_factor_adapter_converts_metadata_and_inputs():
    data = _external_data()
    returns_dict = data.to_returns_dict()

    assert returns_dict["mean"].shape == (3,)
    assert returns_dict["covariance"].shape == (3, 3)
    assert returns_dict["stock_mapping"].shape == (6, 3)
    assert returns_dict["tickers"] == ["S0", "S1", "S2", "S3", "S4", "S5"]
    assert returns_dict["factor_names"] == ["F0", "F1", "F2"]
    assert returns_dict["model_name"] == "toy_external"


def test_external_factor_adapter_ordinary_qp():
    data = _external_data()
    compiled = compile_portfolio_qp(
        data.to_returns_dict(),
        _params(
            data,
            V=None,
            objective="mean_variance",
            risk_aversion=2.0,
        ),
    )
    solution = solve_compiled_qp_osqp(compiled)

    assert_feasible_solution(compiled, solution)
    np.testing.assert_allclose(
        compiled.recover_stock_weights(solution.x).sum(),
        1.0,
        atol=1e-6,
    )


def test_external_factor_adapter_max_sharpe_qp():
    data = _external_data()
    compiled = compile_portfolio_qp(
        data.to_returns_dict(),
        _params(data, objective="max_sharpe", w_min=-1.0, w_max=1.0),
    )
    solution = solve_compiled_qp_osqp(compiled)

    assert_feasible_solution(compiled, solution)
    factor_weights = compiled.recover_factor_weights(solution.x)
    np.testing.assert_allclose(
        compiled.recover_stock_weights(solution.x),
        data.stock_mapping @ factor_weights,
    )


@pytest.mark.gpu
def test_external_factor_adapter_cuopt_matches_osqp_when_available():
    require_cuopt()
    data = _external_data()
    osqp_compiled = compile_portfolio_qp(
        data.to_returns_dict(),
        _params(data, objective="max_sharpe", w_min=-1.0, w_max=1.0),
    )
    cuopt_compiled = compile_portfolio_qp(
        data.to_returns_dict(),
        _params(
            data,
            backend="cuopt",
            objective="max_sharpe",
            w_min=-1.0,
            w_max=1.0,
        ),
    )
    osqp_solution = solve_compiled_qp_osqp(osqp_compiled)
    cuopt_solution = solve_compiled_qp_cuopt(cuopt_compiled)

    assert_feasible_solution(osqp_compiled, osqp_solution)
    assert_feasible_solution(cuopt_compiled, cuopt_solution, tol=1e-5)
    assert_objective_gap_within(
        cuopt_compiled,
        cuopt_solution,
        osqp_solution,
        tol=5e-4,
    )
