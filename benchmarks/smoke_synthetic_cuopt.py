# SPDX-License-Identifier: Apache-2.0
"""One-instance-per-family GPU canary for the registered synthetic benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from benchmarks.synthetic_solver_benchmark import (
    FAMILY_ORDER,
    generate_canonical_problem,
    solve_cuopt,
    solve_osqp,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "configs" / "synthetic_solver_benchmark.yaml").read_text(
            encoding="utf-8"
        )
    )
    correctness = config["correctness"]
    rows = []
    for family in FAMILY_ORDER:
        problem = generate_canonical_problem(config, family, 100, 11)
        osqp_result = solve_osqp(problem, config)
        cuopt_result = solve_cuopt(problem, config)
        if (
            osqp_result.objective_value is None
            or cuopt_result.objective_value is None
        ):
            raise RuntimeError(
                f"{family} canary returned no canonical objective: "
                f"OSQP={osqp_result.error_message!r}, "
                f"cuOpt={cuopt_result.error_message!r}"
            )
        gap = abs(osqp_result.objective_value - cuopt_result.objective_value)
        scale = max(
            1.0,
            abs(osqp_result.objective_value),
            abs(cuopt_result.objective_value),
        )
        limit = float(
            correctness["maximum_cross_solver_objective_gap_absolute"]
        ) + float(correctness["maximum_cross_solver_objective_gap_relative"]) * scale
        passed = (
            osqp_result.correctness_pass
            and cuopt_result.correctness_pass
            and gap <= limit
        )
        rows.append(
            {
                "problem_family": family,
                "canonical_sha256": problem.canonical_sha256,
                "osqp_raw_status": osqp_result.raw_status,
                "cuopt_raw_status": cuopt_result.raw_status,
                "osqp_objective": osqp_result.objective_value,
                "cuopt_objective": cuopt_result.objective_value,
                "canonical_objective_gap": gap,
                "objective_gap_limit": limit,
                "osqp_primal_violation": osqp_result.original_primal_violation,
                "cuopt_primal_violation": cuopt_result.original_primal_violation,
                "pass": passed,
            }
        )
        if not passed:
            raise RuntimeError(json.dumps(rows[-1], indent=2, sort_keys=True))
    print(json.dumps({"status": "pass", "rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
