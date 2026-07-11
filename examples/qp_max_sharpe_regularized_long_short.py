"""Max-Sharpe QP with regularization and explicit long-short controls."""

import argparse

from _qp_example_utils import box_feasible_asset_count, solve, synthetic_returns

from cufolio.qp_parameters import QPParameters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["osqp", "cuopt"], default="osqp")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--n-assets", type=int, default=13)
    args = parser.parse_args()
    n_assets = box_feasible_asset_count(args.n_assets)

    solve(
        synthetic_returns(n_assets, args.seed),
        QPParameters(
            objective="max_sharpe",
            risk_free_rate=0.0,
            w_min=-0.08,
            w_max=0.08,
            short_budget=0.2,
            lambda_l1=1.7e-4,
            lambda_l2=1.0e-3,
            backend=args.backend,
        ),
        label="max_sharpe_regularized_long_short",
    )


if __name__ == "__main__":
    main()
