"""Summarize generated QP benchmark CSV/JSONL artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _float(row: dict, name: str) -> float | None:
    value = row.get(name)
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _median(rows: list[dict], name: str) -> float | None:
    values = [value for row in rows if (value := _float(row, name)) is not None]
    return float(np.median(values)) if values else None


def _load_rows(input_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(input_dir.rglob("*_results.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row["_source_file"] = str(path)
                rows.append(row)
    return rows


def _comparison_ratios(rows: list[dict]) -> dict[tuple, tuple[float, float]]:
    grouped = defaultdict(dict)
    for row in rows:
        if row.get("status") != "optimal":
            continue
        grouped[(row.get("problem_id"), row.get("repeat"))][row.get("backend")] = row
    ratios: dict[tuple, tuple[float, float]] = {}
    for key, pair in grouped.items():
        osqp = pair.get("osqp")
        cuopt = pair.get("cuopt")
        if osqp is None or cuopt is None:
            continue
        osqp_total = _float(osqp, "total_time_sec")
        cuopt_total = _float(cuopt, "total_time_sec")
        osqp_solve = _float(osqp, "solve_time_sec")
        cuopt_solve = _float(cuopt, "solve_time_sec")
        if None in (osqp_total, cuopt_total, osqp_solve, cuopt_solve):
            continue
        ratios[key] = (
            osqp_total / cuopt_total if cuopt_total > 0 else np.nan,
            osqp_solve / cuopt_solve if cuopt_solve > 0 else np.nan,
        )
    return ratios


def summarize(rows: list[dict]) -> list[dict]:
    base_fields = (
        "benchmark_id",
        "objective",
        "case",
        "n_assets",
        "n_factors",
        "mapping_mode",
        "factor_model",
        "n_windows",
    )
    base_groups = defaultdict(list)
    for row in rows:
        base_groups[tuple(row.get(field, "") for field in base_fields)].append(row)
    ratios = _comparison_ratios(rows)
    summary_rows = []
    for base_key, base_rows in sorted(base_groups.items(), key=str):
        backend_groups = defaultdict(list)
        for row in base_rows:
            backend_groups[row.get("backend", "unknown")].append(row)
        for backend, backend_rows in sorted(backend_groups.items()):
            successful = [row for row in backend_rows if row.get("status") == "optimal"]
            group_ratios = [
                ratios[key]
                for key in ratios
                if any(
                    row.get("problem_id") == key[0]
                    and str(row.get("repeat")) == str(key[1])
                    for row in base_rows
                )
            ]
            total_ratios = [value[0] for value in group_ratios if np.isfinite(value[0])]
            solve_ratios = [value[1] for value in group_ratios if np.isfinite(value[1])]
            record = dict(zip(base_fields, base_key))
            record.update(
                {
                    "backend": backend,
                    "rows": len(backend_rows),
                    "successful_rows": len(successful),
                    "success_rate": len(successful) / len(backend_rows)
                    if backend_rows
                    else 0.0,
                    "median_total_time_sec": _median(successful, "total_time_sec"),
                    "median_solve_time_sec": _median(successful, "solve_time_sec"),
                    "median_compile_time_sec": _median(successful, "compile_time_sec"),
                    "median_max_constraint_violation": _median(
                        successful,
                        "max_constraint_violation",
                    ),
                    "median_relative_objective_gap_vs_osqp": _median(
                        successful,
                        "relative_objective_gap_vs_osqp",
                    ),
                    "observed_speed_ratio_total_osqp_over_cuopt": (
                        float(np.median(total_ratios))
                        if backend == "cuopt" and total_ratios
                        else None
                    ),
                    "observed_speed_ratio_solve_osqp_over_cuopt": (
                        float(np.median(solve_ratios))
                        if backend == "cuopt" and solve_ratios
                        else None
                    ),
                }
            )
            summary_rows.append(record)
    return summary_rows


def _write_summary(input_dir: Path, summary_rows: list[dict]) -> dict[str, Path]:
    fields = sorted({key for row in summary_rows for key in row})
    csv_path = input_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    json_path = input_dir / "summary.json"
    json_path.write_text(
        json.dumps(summary_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown = [
        "# QP Benchmark Summary",
        "",
        "These ratios are observed speed ratios in this generated artifact run.",
        "They are not universal speedup claims.",
        "",
        "| benchmark | backend | objective | case | n_assets | n_factors | median total sec | median solve sec | success rate | total ratio OSQP/cuOpt | solve ratio OSQP/cuOpt |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        markdown.append(
            "| {benchmark_id} | {backend} | {objective} | {case} | {n_assets} | "
            "{n_factors} | {median_total_time_sec} | {median_solve_time_sec} | "
            "{success_rate} | {observed_speed_ratio_total_osqp_over_cuopt} | "
            "{observed_speed_ratio_solve_osqp_over_cuopt} |".format(**row)
        )
    markdown_path = input_dir / "summary.md"
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return {"summary_csv": csv_path, "summary_json": json_path, "summary_md": markdown_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_dir = args.input_dir
    rows = _load_rows(input_dir)
    if not rows:
        raise SystemExit(f"No *_results.csv artifacts found under {input_dir}")
    summary_rows = summarize(rows)
    paths = _write_summary(input_dir, summary_rows)
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
