"""Benchmark stock-level PortOpt QP workflows with explicit artifacts."""

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
    make_mean_vector,
    make_psd_covariance,
    print_artifact_paths,
    run_problem_repeats,
    write_result_artifacts,
)
from cufolio.qp_parameters import QPParameters  # noqa: E402

OBJECTIVES = ("min_variance", "mean_variance", "max_sharpe")
CASES = ("basic", "l2", "l1_l2", "long_short", "friction", "full")


def _case_parameters(
    objective: str,
    case: str,
    n_assets: int,
    previous_weights: np.ndarray,
    benchmark_weights: np.ndarray,
) -> QPParameters:
    if case == "basic":
        options = {"w_min": 0.0, "w_max": 1.0}
    elif case == "l2":
        options = {"w_min": 0.0, "w_max": 1.0, "lambda_l2": 1e-3}
    elif case == "l1_l2":
        options = {
            "w_min": 0.0,
            "w_max": 1.0,
            "lambda_l1": 1.7e-4,
            "lambda_l2": 1e-3,
        }
    elif case == "long_short":
        options = {
            "w_min": -0.08,
            "w_max": 0.08,
            "short_budget": 0.2,
        }
    elif case == "friction":
        options = {
            "w_min": 0.0,
            "w_max": 1.0,
            "turnover_budget": 0.5,
            "previous_weights": previous_weights,
            "benchmark_l1_budget": 0.5,
            "benchmark_weights": benchmark_weights,
            "lambda_tracking_error": 1e-3,
        }
    elif case == "full":
        options = {
            "w_min": -0.08,
            "w_max": 0.08,
            "short_budget": 0.2,
            "lambda_l1": 1.7e-4,
            "lambda_l2": 1e-3,
            "turnover_budget": 0.5,
            "previous_weights": previous_weights,
            "benchmark_l1_budget": 0.5,
            "benchmark_weights": benchmark_weights,
            "lambda_tracking_error": 1e-3,
        }
    else:
        raise ValueError(f"Unsupported stock benchmark case: {case}")
    if objective == "mean_variance":
        options["risk_aversion"] = 2.0
    return QPParameters(objective=objective, backend="osqp", **options)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("osqp", "cuopt", "both"), default="osqp")
    parser.add_argument("--n-assets", nargs="+", type=int, default=[50, 100, 250, 500, 1000])
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
    run_dir = create_run_dir(args.output_dir, prefix="stock_level")
    metadata = collect_environment_metadata(args.backend)
    rows = []
    for requested_n_assets in args.n_assets:
        if requested_n_assets < 1:
            raise SystemExit("--n-assets values must be positive")
        effective_n_assets = max(
            requested_n_assets,
            13 if any(case in args.cases for case in ("long_short", "full")) else 1,
        )
        data_seed = args.seed + effective_n_assets
        covariance = make_psd_covariance(effective_n_assets, data_seed)
        returns_dict = {
            "mean": make_mean_vector(effective_n_assets, data_seed + 1),
            "covariance": covariance,
            "tracking_covariance": covariance,
        }
        previous = make_anchor_weights(effective_n_assets, data_seed + 2)
        benchmark = make_anchor_weights(effective_n_assets, data_seed + 3)
        for objective in args.objectives:
            for case in args.cases:
                if case == "full" and objective != "max_sharpe":
                    print(
                        f"skip: full case only applies to max_sharpe ({objective})"
                    )
                    continue
                params = _case_parameters(
                    objective,
                    case,
                    effective_n_assets,
                    previous,
                    benchmark,
                )
                problem_id = (
                    f"stock:{case}:{objective}:n{effective_n_assets}:seed{data_seed}"
                )
                rows.extend(
                    run_problem_repeats(
                        returns_dict=returns_dict,
                        params=params,
                        metadata=metadata,
                        benchmark_id="stock_level",
                        problem_id=problem_id,
                        requested_backend=args.backend,
                        objective=objective,
                        case=case,
                        repeat_count=args.repeats,
                        warmup_count=args.warmup,
                        seed=data_seed,
                        timeout_sec=args.timeout_sec,
                    )
                )
    paths = write_result_artifacts(
        run_dir,
        "stock_level",
        rows,
        metadata,
        "Stock-Level PortOpt QP Benchmark",
    )
    print_artifact_paths(paths)


if __name__ == "__main__":
    main()
