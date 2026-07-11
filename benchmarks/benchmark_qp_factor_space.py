"""Benchmark explicit factor-space PortOpt QP workflows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.qp_benchmark_utils import (  # noqa: E402
    collect_environment_metadata,
    create_run_dir,
    make_anchor_weights,
    make_external_factor_inputs,
    make_stock_returns,
    print_artifact_paths,
    run_problem_repeats,
    write_result_artifacts,
)
from cufolio.qp_factor_workflows import (  # noqa: E402
    build_external_factor_qp_data,
    build_pca_factor_qp_data,
)
from cufolio.qp_parameters import QPParameters  # noqa: E402

OBJECTIVES = ("mean_variance", "max_sharpe")
CASES = ("pca", "external", "full")


def _case_parameters(data, objective: str, case: str, previous, benchmark):
    if case == "full":
        options = {
            "w_min": -0.08,
            "w_max": 0.08,
            "short_budget": 0.2,
            "lambda_l1": 1.7e-4,
            "lambda_l2": 1e-3,
            "turnover_budget": 0.5,
            "previous_weights": previous,
            "benchmark_l1_budget": 0.5,
            "benchmark_weights": benchmark,
        }
    else:
        options = {"w_min": 0.0, "w_max": 1.0}
    if objective == "mean_variance":
        options["risk_aversion"] = 2.0
    return QPParameters(
        mapping_mode="factor_space",
        V=data.stock_mapping,
        objective=objective,
        backend="osqp",
        **options,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("osqp", "cuopt", "both"), default="osqp")
    parser.add_argument("--n-assets", nargs="+", type=int, default=[100, 500, 1000])
    parser.add_argument("--n-factors", nargs="+", type=int, default=[3, 6, 10])
    parser.add_argument("--objectives", nargs="+", choices=OBJECTIVES, default=list(OBJECTIVES))
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout-sec", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/benchmarks"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = create_run_dir(args.output_dir, prefix="factor_space")
    metadata = collect_environment_metadata(args.backend)
    rows = []
    for requested_n_assets in args.n_assets:
        if requested_n_assets < 1:
            raise SystemExit("--n-assets values must be positive")
        for n_factors in args.n_factors:
            if n_factors < 1:
                raise SystemExit("--n-factors values must be positive")
            effective_n_assets = max(
                requested_n_assets,
                13 if "full" in args.cases else 1,
            )
            data_seed = args.seed + effective_n_assets * 11 + n_factors
            for case in args.cases:
                if case == "pca":
                    if "full" in args.cases:
                        rng = np.random.default_rng(data_seed)
                        market = rng.normal(0.001, 0.01, size=(60, 1))
                        noise_scale = 0.0005 if "full" in args.cases else 0.004
                        stock_returns = 0.002 + market @ np.ones(
                            (1, effective_n_assets)
                        ) + rng.normal(
                            0.0,
                            noise_scale,
                            size=(60, effective_n_assets),
                        )
                    else:
                        stock_returns = make_stock_returns(
                            60,
                            effective_n_assets,
                            data_seed,
                            n_latent_factors=max(3, n_factors),
                        )
                    data = build_pca_factor_qp_data(
                        stock_returns,
                        n_components=n_factors,
                        center=False,
                    )
                elif case == "external":
                    factor_returns, mapping, stock_returns = make_external_factor_inputs(
                        60,
                        effective_n_assets,
                        n_factors,
                        data_seed,
                    )
                    data = build_external_factor_qp_data(
                        factor_returns,
                        mapping,
                        stock_returns=stock_returns,
                        model_name="external_benchmark",
                    )
                elif case == "full":
                    factor_returns, mapping, stock_returns = make_external_factor_inputs(
                        60,
                        effective_n_assets,
                        n_factors,
                        data_seed,
                    )
                    data = build_external_factor_qp_data(
                        factor_returns,
                        mapping,
                        stock_returns=stock_returns,
                        model_name="external_full_benchmark",
                    )
                else:
                    raise ValueError(f"Unsupported factor benchmark case: {case}")
                previous = make_anchor_weights(effective_n_assets, data_seed + 1)
                benchmark = make_anchor_weights(effective_n_assets, data_seed + 2)
                for objective in args.objectives:
                    if case == "full" and objective != "max_sharpe":
                        print(
                            f"skip: full case only applies to max_sharpe ({objective})"
                        )
                        continue
                    params = _case_parameters(
                        data,
                        objective,
                        case,
                        previous,
                        benchmark,
                    )
                    problem_id = (
                        f"factor:{case}:{objective}:n{effective_n_assets}:"
                        f"k{n_factors}:seed{data_seed}"
                    )
                    rows.extend(
                        run_problem_repeats(
                            returns_dict=data.to_returns_dict(),
                            params=params,
                            metadata=metadata,
                            benchmark_id="factor_space",
                            problem_id=problem_id,
                            requested_backend=args.backend,
                            objective=objective,
                            case=case,
                            repeat_count=args.repeats,
                            warmup_count=args.warmup,
                            seed=data_seed,
                            timeout_sec=args.timeout_sec,
                            factor_model=data.model_name,
                            n_time_observations=int(data.factor_returns.shape[0]),
                        )
                    )
    paths = write_result_artifacts(
        run_dir,
        "factor_space",
        rows,
        metadata,
        "Factor-Space PortOpt QP Benchmark",
    )
    print_artifact_paths(paths)


if __name__ == "__main__":
    main()
