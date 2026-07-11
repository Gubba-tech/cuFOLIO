"""CPU-first mean-variance quickstart with a small l2 penalty."""

import argparse

from _qp_example_utils import solve, synthetic_returns

from cufolio.qp_parameters import QPParameters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["osqp", "cuopt"], default="osqp")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--n-assets", type=int, default=10)
    args = parser.parse_args()

    solve(
        synthetic_returns(args.n_assets, args.seed),
        QPParameters(
            objective="mean_variance",
            risk_aversion=2.0,
            lambda_l2=0.01,
            w_min=0.0,
            w_max=1.0,
            backend=args.backend,
        ),
        label="mean_variance",
    )


if __name__ == "__main__":
    main()
