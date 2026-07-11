"""CPU-first minimum-variance quickstart for the unified QP extension."""

import argparse

from _qp_example_utils import solve, synthetic_returns

from cufolio.qp_parameters import QPParameters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["osqp", "cuopt"], default="osqp")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-assets", type=int, default=10)
    args = parser.parse_args()

    returns_dict = synthetic_returns(args.n_assets, args.seed)
    solve(
        returns_dict,
        QPParameters(
            objective="min_variance",
            w_min=0.0,
            w_max=1.0,
            backend=args.backend,
        ),
        label="min_variance",
    )


if __name__ == "__main__":
    main()
