"""External factor adapter demo for RP-PCA/IPCA/AP-Trees outputs."""

import argparse

import numpy as np
from _qp_example_utils import solve_and_report

from cufolio.qp_factor_workflows import build_external_factor_qp_data
from cufolio.qp_formulations import compile_portfolio_qp
from cufolio.qp_parameters import QPParameters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["osqp", "cuopt"], default="osqp")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--n-assets", type=int, default=12)
    parser.add_argument("--n-factors", type=int, default=3)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    factor_returns = rng.normal(
        0.01, 0.02, size=(60, args.n_factors)
    )
    raw_mapping = rng.uniform(0.05, 1.0, size=(args.n_assets, args.n_factors))
    mapping = raw_mapping / raw_mapping.sum(axis=0, keepdims=True)
    stock_returns = factor_returns @ mapping.T + rng.normal(
        0.0, 0.002, size=(60, args.n_assets)
    )
    # This is the adapter path for externally computed RP-PCA/IPCA/AP-Trees
    # factor_returns plus stock_mapping V; no estimator is reimplemented here.
    data = build_external_factor_qp_data(
        factor_returns,
        mapping,
        stock_returns=stock_returns,
        model_name="external_demo",
    )
    for objective in ("mean_variance", "max_sharpe"):
        params = QPParameters(
            mapping_mode="factor_space",
            V=data.stock_mapping,
            objective=objective,
            risk_aversion=2.0,
            w_min=-1.0,
            w_max=1.0,
            backend=args.backend,
        )
        compiled = compile_portfolio_qp(data.to_returns_dict(), params)
        solve_and_report(compiled, params, label=f"external_{objective}")


if __name__ == "__main__":
    main()
