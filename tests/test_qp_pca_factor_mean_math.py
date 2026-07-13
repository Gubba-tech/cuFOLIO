from __future__ import annotations

import numpy as np

from cufolio.qp_backend import solve_compiled_qp_osqp
from cufolio.qp_factor_workflows import build_pca_factor_qp_data
from cufolio.qp_formulations import compile_portfolio_qp
from cufolio.qp_parameters import QPParameters


def _returns(seed: int = 2026, observations: int = 80, assets: int = 12) -> np.ndarray:
    rng = np.random.default_rng(seed)
    means = np.linspace(0.002, 0.012, assets)
    return means + rng.normal(0.0, 0.01, size=(observations, assets))


def _max_sharpe_params(mapping: np.ndarray) -> QPParameters:
    return QPParameters(
        objective="max_sharpe",
        mapping_mode="factor_space",
        V=mapping,
        risk_free_rate=0.0,
        lambda_l1=1.7e-4,
        lambda_l2=1e-3,
        short_budget=None,
        w_min=-10.0,
        w_max=10.0,
        backend="osqp",
    )


def test_pca_factor_mean_uses_raw_returns_not_centered_scores():
    raw = _returns()
    data = build_pca_factor_qp_data(raw, n_components=4, center=True)

    expected = raw.mean(axis=0) @ data.stock_mapping
    np.testing.assert_allclose(data.factor_mean, expected, atol=1e-12)
    assert not np.allclose(data.factor_mean, 0.0)


def test_pca_factor_covariance_uses_projected_returns():
    raw = _returns(seed=2027)
    data = build_pca_factor_qp_data(raw, n_components=4, center=True)

    expected_returns = raw @ data.stock_mapping
    expected_covariance = np.cov(expected_returns, rowvar=False)
    np.testing.assert_allclose(data.factor_returns, expected_returns, atol=1e-12)
    np.testing.assert_allclose(data.factor_covariance, expected_covariance, atol=1e-12)


def test_pca_factor_sign_invariance_for_stock_weights():
    raw = _returns(seed=2028)
    data = build_pca_factor_qp_data(raw, n_components=4, center=True)
    flipped_mapping = data.stock_mapping.copy()
    flipped_mapping[:, 1] *= -1.0
    flipped_returns = raw @ flipped_mapping

    base = compile_portfolio_qp(
        data.to_returns_dict(),
        _max_sharpe_params(data.stock_mapping),
    )
    flipped = compile_portfolio_qp(
        {
            "mean": flipped_returns.mean(axis=0),
            "covariance": np.cov(flipped_returns, rowvar=False),
            "stock_mapping": flipped_mapping,
        },
        _max_sharpe_params(flipped_mapping),
    )
    base_solution = solve_compiled_qp_osqp(base)
    flipped_solution = solve_compiled_qp_osqp(flipped)

    np.testing.assert_allclose(
        base.recover_stock_weights(base_solution.x),
        flipped.recover_stock_weights(flipped_solution.x),
        atol=2e-5,
    )
