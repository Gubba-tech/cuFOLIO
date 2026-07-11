"""Benchmark deterministic repeated QP solves over synthetic rolling windows."""

from __future__ import annotations

import argparse
import csv
import json
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
    make_stock_returns,
    print_artifact_paths,
    run_problem_once,
    write_result_artifacts,
)
from cufolio.qp_factor_workflows import build_pca_factor_qp_data  # noqa: E402
from cufolio.qp_parameters import QPParameters  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("osqp", "cuopt", "both"), default="osqp")
    parser.add_argument("--n-assets", nargs="+", type=int, default=[100, 250, 500])
    parser.add_argument("--n-factors", nargs="+", type=int, default=[3, 6])
    parser.add_argument("--n-windows", nargs="+", type=int, default=[10, 50, 100])
    parser.add_argument("--objective", choices=("max_sharpe",), default="max_sharpe")
    parser.add_argument("--mode", nargs="+", choices=("stock", "factor_pca"), default=["stock"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout-sec", type=float, default=1800.0)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/benchmarks"))
    return parser


def _aggregate_rows(rows: list[dict]) -> list[dict]:
    aggregate = []
    backends = sorted({row.get("backend") for row in rows})
    for backend in backends:
        backend_rows = [row for row in rows if row.get("backend") == backend]
        successful = [row for row in backend_rows if row.get("status") == "optimal"]
        def values(name):
            return [
                float(row[name])
                for row in successful
                if row.get(name) is not None
            ]
        times = values("total_time_sec")
        turnover = values("turnover")
        sharpe = values("sharpe")
        violations = values("max_constraint_violation")
        gaps = values("relative_objective_gap_vs_osqp")
        aggregate.append(
            {
                "backend": backend,
                "successful_windows": len(successful),
                "failed_windows": len(backend_rows) - len(successful),
                "total_windows": len(backend_rows),
                "total_compile_time_sec": sum(values("compile_time_sec")),
                "total_solve_time_sec": sum(values("solve_time_sec")),
                "total_time_sec": sum(times),
                "median_window_time_sec": float(np.median(times)) if times else None,
                "average_turnover": float(np.mean(turnover)) if turnover else None,
                "average_sharpe_proxy": float(np.mean(sharpe)) if sharpe else None,
                "max_constraint_violation_max": max(violations) if violations else None,
                "objective_gap_summary_vs_osqp": (
                    float(np.median(gaps)) if gaps else None
                ),
            }
        )
    return aggregate


def _write_aggregate_artifacts(run_dir: Path, rows: list[dict]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    csv_path = run_dir / "rolling_aggregate_summary.csv"
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    paths["aggregate_csv"] = csv_path
    json_path = run_dir / "rolling_aggregate_summary.json"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["aggregate_json"] = json_path
    markdown = [
        "# Rolling-Window QP Aggregate Summary",
        "",
        "Timing and ratios are scoped to this generated synthetic artifact.",
        "",
        "| backend | successful | failed | total_time_sec | median_window_time_sec | average_turnover | average_sharpe_proxy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        markdown.append(
            "| {backend} | {successful_windows} | {failed_windows} | {total_time_sec} | "
            "{median_window_time_sec} | {average_turnover} | {average_sharpe_proxy} |".format(
                **row
            )
        )
    markdown_path = run_dir / "rolling_aggregate_summary.md"
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    paths["aggregate_markdown"] = markdown_path
    return paths


def main() -> None:
    args = build_parser().parse_args()
    run_dir = create_run_dir(args.output_dir, prefix="rolling_windows")
    metadata = collect_environment_metadata(args.backend)
    rows = []
    aggregate_rows = []
    for n_assets in args.n_assets:
        if n_assets < 1:
            raise SystemExit("--n-assets values must be positive")
        for n_factors in args.n_factors:
            if n_factors < 1:
                raise SystemExit("--n-factors values must be positive")
            for mode in args.mode:
                for n_windows in args.n_windows:
                    previous = make_anchor_weights(n_assets, args.seed + n_assets)
                    benchmark = make_anchor_weights(
                        n_assets,
                        args.seed + n_assets + 1,
                    )
                    base_covariance = make_psd_covariance(n_assets, args.seed + 7)
                    for window in range(n_windows):
                        window_seed = (
                            args.seed + n_assets * 101 + n_factors * 13 + window
                        )
                        if mode == "stock":
                            rng = np.random.default_rng(window_seed)
                            mean = make_mean_vector(n_assets, window_seed)
                            mean += rng.normal(0.0, 0.001, size=n_assets)
                            perturbation = make_psd_covariance(
                                n_assets,
                                window_seed + 1,
                                factor_rank=min(16, n_assets),
                                diagonal_floor=0.002,
                            )
                            covariance = (
                                0.85 * base_covariance + 0.15 * perturbation
                            )
                            returns_dict = {
                                "mean": mean,
                                "covariance": covariance,
                                "tracking_covariance": covariance,
                            }
                            mapping_mode = "stock_space"
                            factor_model = None
                            n_time_observations = None
                        else:
                            effective_n_factors = min(n_factors, n_assets, 10)
                            stock_returns = make_stock_returns(
                                60,
                                n_assets,
                                window_seed,
                                n_latent_factors=max(3, effective_n_factors),
                            )
                            data = build_pca_factor_qp_data(
                                stock_returns,
                                n_components=effective_n_factors,
                                center=False,
                            )
                            returns_dict = data.to_returns_dict()
                            mapping_mode = "factor_space"
                            factor_model = data.model_name
                            n_time_observations = int(data.factor_returns.shape[0])
                        params_options = {
                            "objective": args.objective,
                            "w_min": 0.0,
                            "w_max": 1.0,
                            "turnover_budget": 0.5,
                            "previous_weights": previous,
                            "benchmark_l1_budget": 0.5,
                            "benchmark_weights": benchmark,
                            "lambda_tracking_error": 1e-3,
                            "backend": "osqp",
                        }
                        if mapping_mode == "factor_space":
                            params_options.update(
                                {
                                    "mapping_mode": "factor_space",
                                    "V": data.stock_mapping,
                                }
                            )
                        params = QPParameters(**params_options)
                        for repeat in range(max(1, args.repeats)):
                            problem_id = (
                                f"rolling:{mode}:n{n_assets}:k{n_factors}:"
                                f"windows{n_windows}:window{window}:seed{window_seed}"
                            )
                            window_rows, timed_results, compiled = run_problem_once(
                                returns_dict=returns_dict,
                                params=params,
                                metadata=metadata,
                                benchmark_id="rolling_windows",
                                problem_id=problem_id,
                                requested_backend=args.backend,
                                objective=args.objective,
                                case=mode,
                                repeat=repeat,
                                seed=window_seed,
                                timeout_sec=args.timeout_sec,
                                factor_model=factor_model,
                                n_time_observations=n_time_observations,
                                extra_fields={
                                    "window": window,
                                    "n_windows": n_windows,
                                    "requested_n_factors": n_factors,
                                },
                            )
                            rows.extend(window_rows)
                            candidate = timed_results.get("osqp")
                            if candidate is None or candidate.solution is None:
                                candidate = timed_results.get("cuopt")
                            if (
                                compiled is not None
                                and candidate is not None
                                and candidate.solution is not None
                                and candidate.status == "optimal"
                            ):
                                previous = compiled.recover_stock_weights(
                                    candidate.solution.x
                                )
                    config_rows = [
                        row
                        for row in rows
                        if row.get("n_assets") == n_assets
                        and row.get("case") == mode
                        and row.get("n_windows") == n_windows
                        and row.get("requested_n_factors") == n_factors
                    ]
                    for aggregate in _aggregate_rows(config_rows):
                        aggregate.update(
                            {
                                "mode": mode,
                                "n_assets": n_assets,
                                "n_factors": n_factors,
                                "n_windows": n_windows,
                            }
                        )
                        aggregate_rows.append(aggregate)
    paths = write_result_artifacts(
        run_dir,
        "rolling_windows",
        rows,
        metadata,
        "Rolling-Window PortOpt QP Benchmark",
    )
    paths.update(_write_aggregate_artifacts(run_dir, aggregate_rows))
    print_artifact_paths(paths)


if __name__ == "__main__":
    main()
