"""Synthetic PCA factor-space max-Sharpe demonstration."""

import argparse

import numpy as np
from _qp_example_utils import box_feasible_asset_count, solve_and_report

from cufolio.qp_factor_workflows import build_pca_factor_qp_data
from cufolio.qp_formulations import compile_portfolio_qp
from cufolio.qp_parameters import QPParameters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["osqp", "cuopt"], default="osqp")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--n-assets", type=int, default=13)
    parser.add_argument("--n-factors", type=int, default=3)
    args = parser.parse_args()
    n_assets = box_feasible_asset_count(args.n_assets)
    rng = np.random.default_rng(args.seed)
    market = rng.normal(0.001, 0.01, size=(60, 1))
    loadings = np.linspace(0.6, 1.4, n_assets).reshape(1, -1)
    stock_returns = 0.002 + market @ loadings + rng.normal(
        0.0, 0.004, size=(60, n_assets)
    )
    data = build_pca_factor_qp_data(
        stock_returns,
        n_components=args.n_factors,
        center=False,
    )
    params = QPParameters(
        mapping_mode="factor_space",
        V=data.stock_mapping,
        objective="max_sharpe",
        w_min=0.0,
        w_max=1.0,
        lambda_l1=0.01,
        lambda_l2=0.01,
        backend=args.backend,
    )
    compiled = compile_portfolio_qp(data.to_returns_dict(), params)
    solve_and_report(compiled, params, label="factor_space_pca")


if __name__ == "__main__":
    main()
