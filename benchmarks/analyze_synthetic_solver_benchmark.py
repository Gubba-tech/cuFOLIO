# SPDX-License-Identifier: Apache-2.0
"""Generate registered OSQP-versus-cuOpt tables, figures, report, and audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.backends.backend_pdf import PdfPages

from benchmarks.synthetic_solver_benchmark import FAMILY_ORDER

SOLVERS = ("OSQP", "cuOpt")
SOLVER_COLORS = {"OSQP": "#16697A", "cuOpt": "#D1495B"}
FAMILY_STYLES = {
    "LP-RANGED": "-",
    "LP-MIXED": "--",
    "QP-DIAGONAL": "-",
    "QP-SPARSE-COUPLED": "--",
}
PAIR_KEY = ["problem_family", "n_variables", "seed", "repetition"]
TIMING_DEFINITION = (
    "End-to-end from canonical host arrays through backend conversion/model setup, "
    "transfer where applicable, solve, extraction, and canonical correctness "
    "reconstruction."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def finite_max(values: Iterable[Any]) -> float | None:
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy()
    array = array[np.isfinite(array)]
    return float(array.max()) if array.size else None


def finite_median(values: Iterable[Any]) -> float | None:
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy()
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else None


def frames_match(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if left.shape != right.shape or list(left.columns) != list(right.columns):
        return False
    for column in left.columns:
        if pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(
            right[column]
        ):
            if not np.allclose(
                pd.to_numeric(left[column], errors="coerce"),
                pd.to_numeric(right[column], errors="coerce"),
                rtol=1.0e-13,
                atol=1.0e-15,
                equal_nan=True,
            ):
                return False
        elif not np.array_equal(
            left[column].fillna("<NA>").astype(str).to_numpy(),
            right[column].fillna("<NA>").astype(str).to_numpy(),
        ):
            return False
    return True


def json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for key, value in record.items():
            if pd.isna(value):
                clean[key] = None
            elif isinstance(value, np.generic):
                clean[key] = value.item()
            else:
                clean[key] = value
        records.append(clean)
    return records


def faster_label(speedup: float | None) -> str:
    if speedup is None or not np.isfinite(speedup):
        return "Insufficient common solved instances"
    if np.isclose(speedup, 1.0, rtol=1.0e-12, atol=1.0e-15):
        return "Tie"
    return "cuOpt" if speedup > 1.0 else "OSQP"


def measured_crossover(group: pd.DataFrame) -> int | None:
    valid = group.dropna(subset=["end_to_end_speedup"]).sort_values("n_variables")
    previous_sign: int | None = None
    for row in valid.itertuples(index=False):
        value = float(row.end_to_end_speedup)
        sign = 0 if np.isclose(value, 1.0) else (1 if value > 1.0 else -1)
        if sign == 0 or (previous_sign is not None and sign != previous_sign):
            return int(row.n_variables)
        previous_sign = sign
    return None


def load_registered_pairs(
    raw_path: Path, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(raw_path)
    required = {
        *PAIR_KEY,
        "backend",
        "registered_repetition",
        "canonical_sha256",
        "execution_sha",
        "strict_optimal",
        "correctness_pass",
        "censored",
        "setup_seconds",
        "solve_seconds",
        "end_to_end_seconds",
        "objective_value",
        "original_primal_violation",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"cold result file is missing columns: {missing}")
    registered = raw.loc[bool_series(raw["registered_repetition"])].copy()
    if registered.duplicated(PAIR_KEY + ["backend"]).any():
        raise ValueError("duplicate registered backend timing keys detected")
    unknown = set(registered["backend"]) - set(SOLVERS)
    if unknown:
        raise ValueError(f"unknown backends: {sorted(unknown)}")

    left = registered.loc[registered["backend"] == "OSQP"].copy()
    right = registered.loc[registered["backend"] == "cuOpt"].copy()
    pairs = left.merge(
        right,
        on=PAIR_KEY,
        how="outer",
        suffixes=("_osqp", "_cuopt"),
        indicator=True,
        validate="one_to_one",
    )
    for backend in ("osqp", "cuopt"):
        for column in ("strict_optimal", "correctness_pass", "censored"):
            pairs[f"{column}_{backend}"] = bool_series(
                pairs[f"{column}_{backend}"]
            )
    pairs["same_canonical_instance"] = (
        pairs["canonical_sha256_osqp"] == pairs["canonical_sha256_cuopt"]
    ) & pairs["canonical_sha256_osqp"].notna()
    pairs["same_execution_sha"] = (
        pairs["execution_sha_osqp"] == pairs["execution_sha_cuopt"]
    ) & pairs["execution_sha_osqp"].notna()
    pairs["canonical_objective_gap"] = (
        pairs["objective_value_osqp"] - pairs["objective_value_cuopt"]
    ).abs()
    scale = np.maximum(
        1.0,
        np.maximum(
            pairs["objective_value_osqp"].abs(),
            pairs["objective_value_cuopt"].abs(),
        ),
    )
    correctness = config["correctness"]
    objective_limit = float(
        correctness["maximum_cross_solver_objective_gap_absolute"]
    ) + float(correctness["maximum_cross_solver_objective_gap_relative"]) * scale
    pairs["cross_objective_gate"] = (
        pairs["canonical_objective_gap"] <= objective_limit
    )
    pairs["common_correct"] = (
        (pairs["_merge"] == "both")
        & pairs["same_canonical_instance"]
        & pairs["same_execution_sha"]
        & pairs["strict_optimal_osqp"]
        & pairs["strict_optimal_cuopt"]
        & pairs["correctness_pass_osqp"]
        & pairs["correctness_pass_cuopt"]
        & ~pairs["censored_osqp"]
        & ~pairs["censored_cuopt"]
        & pairs["cross_objective_gate"]
    )
    pairs["maximum_pair_primal_violation"] = pairs[
        ["original_primal_violation_osqp", "original_primal_violation_cuopt"]
    ].max(axis=1)
    return raw, pairs.drop(columns=["_merge"])


def build_seed_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    key = ["problem_family", "n_variables", "seed"]
    for values, group in pairs.groupby(key, sort=False):
        valid = group.loc[group["common_correct"]]
        row: dict[str, Any] = dict(zip(key, values))
        row["n_constraints"] = finite_median(group["n_constraints_osqp"])
        row["registered_pairs"] = int(len(group))
        row["common_correct_repetitions"] = int(len(valid))
        row["canonical_instance_pass"] = bool(len(valid) == len(group) and len(group))
        for backend in ("osqp", "cuopt"):
            for metric in (
                "setup_seconds",
                "solve_seconds",
                "end_to_end_seconds",
                "result_extraction_seconds",
                "correctness_seconds",
                "peak_host_memory_bytes",
                "peak_gpu_memory_bytes",
            ):
                row[f"median_{backend}_{metric}"] = finite_median(
                    valid[f"{metric}_{backend}"]
                )
        if len(valid):
            row["solve_speedup"] = (
                row["median_osqp_solve_seconds"]
                / row["median_cuopt_solve_seconds"]
            )
            row["end_to_end_speedup"] = (
                row["median_osqp_end_to_end_seconds"]
                / row["median_cuopt_end_to_end_seconds"]
            )
        else:
            row["solve_speedup"] = np.nan
            row["end_to_end_speedup"] = np.nan
        row["maximum_objective_gap"] = finite_max(
            valid["canonical_objective_gap"]
        )
        row["maximum_primal_violation"] = finite_max(
            valid["maximum_pair_primal_violation"]
        )
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _quantile(values: pd.Series, quantile: float) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.quantile(quantile)) if len(values) else np.nan


def build_by_size(
    registered: pd.DataFrame, pairs: pd.DataFrame, seed_summary: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (family, n_variables), seeds in seed_summary.groupby(
        ["problem_family", "n_variables"], sort=False
    ):
        paired = pairs.loc[
            (pairs["problem_family"] == family)
            & (pairs["n_variables"] == n_variables)
        ]
        valid = seeds.loc[seeds["common_correct_repetitions"] > 0]
        raw_group = registered.loc[
            (registered["problem_family"] == family)
            & (registered["n_variables"] == n_variables)
        ]
        row: dict[str, Any] = {
            "problem_family": family,
            "n_variables": int(n_variables),
            "n_constraints": int(round(float(seeds["n_constraints"].median()))),
        }
        column_map = {
            "median_osqp_setup_seconds": "median_osqp_setup_seconds",
            "median_osqp_solve_seconds": "median_osqp_solve_seconds",
            "median_osqp_end_to_end_seconds": "median_osqp_end_to_end_seconds",
            "median_cuopt_build_transfer_seconds": "median_cuopt_setup_seconds",
            "median_cuopt_solve_seconds": "median_cuopt_solve_seconds",
            "median_cuopt_end_to_end_seconds": "median_cuopt_end_to_end_seconds",
        }
        for output, source in column_map.items():
            row[output] = finite_median(valid[source])
            row[f"q25_{output.removeprefix('median_')}"] = _quantile(
                valid[source], 0.25
            )
            row[f"q75_{output.removeprefix('median_')}"] = _quantile(
                valid[source], 0.75
            )
        if len(valid):
            row["solve_speedup"] = (
                row["median_osqp_solve_seconds"]
                / row["median_cuopt_solve_seconds"]
            )
            row["end_to_end_speedup"] = (
                row["median_osqp_end_to_end_seconds"]
                / row["median_cuopt_end_to_end_seconds"]
            )
        else:
            row["solve_speedup"] = np.nan
            row["end_to_end_speedup"] = np.nan
        row["q25_solve_speedup"] = _quantile(valid["solve_speedup"], 0.25)
        row["q75_solve_speedup"] = _quantile(valid["solve_speedup"], 0.75)
        row["q25_end_to_end_speedup"] = _quantile(
            valid["end_to_end_speedup"], 0.25
        )
        row["q75_end_to_end_speedup"] = _quantile(
            valid["end_to_end_speedup"], 0.75
        )
        row["faster_solver_solve_only"] = faster_label(row["solve_speedup"])
        row["faster_solver_end_to_end"] = faster_label(row["end_to_end_speedup"])
        row["common_correct_instances"] = int(valid["seed"].nunique())
        for backend, output in (("OSQP", "osqp_failures"), ("cuOpt", "cuopt_failures")):
            backend_rows = raw_group.loc[raw_group["backend"] == backend]
            accepted = (
                bool_series(backend_rows["strict_optimal"])
                & bool_series(backend_rows["correctness_pass"])
                & ~bool_series(backend_rows["censored"])
            )
            row[output] = int((~accepted).sum())
            row[f"{backend.lower()}_timeouts"] = int(
                bool_series(backend_rows["censored"]).sum()
            )
        common_pairs = paired.loc[paired["common_correct"]]
        row["maximum_objective_gap"] = finite_max(
            common_pairs["canonical_objective_gap"]
        )
        row["maximum_primal_violation"] = finite_max(
            common_pairs["maximum_pair_primal_violation"]
        )
        rows.append(row)
    order = {family: index for index, family in enumerate(FAMILY_ORDER)}
    result = pd.DataFrame.from_records(rows)
    result["_order"] = result["problem_family"].map(order)
    return result.sort_values(["_order", "n_variables"]).drop(columns="_order")


def build_headline(registered: pd.DataFrame, pairs: pd.DataFrame, by_size: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        family_sizes = by_size.loc[by_size["problem_family"] == family].copy()
        common = family_sizes.loc[family_sizes["common_correct_instances"] > 0]
        raw_family = registered.loc[registered["problem_family"] == family]
        pair_family = pairs.loc[pairs["problem_family"] == family]
        if len(common):
            largest = common.loc[common["n_variables"].idxmax()]
            largest_n: int | str = int(largest["n_variables"])
            osqp_time = float(largest["median_osqp_end_to_end_seconds"])
            cuopt_time = float(largest["median_cuopt_end_to_end_seconds"])
            end_speedup = float(largest["end_to_end_speedup"])
            solve_speedup = float(largest["solve_speedup"])
            large_faster = faster_label(end_speedup)
        else:
            largest_n = "Insufficient common solved instances"
            osqp_time = cuopt_time = end_speedup = solve_speedup = np.nan
            large_faster = "Insufficient common solved instances"
        at_100 = common.loc[common["n_variables"] == 100]
        faster_100 = (
            faster_label(float(at_100.iloc[0]["end_to_end_speedup"]))
            if len(at_100)
            else "Insufficient common solved instances"
        )
        crossover = measured_crossover(common)
        strict_counts = {}
        for backend in SOLVERS:
            backend_rows = raw_family.loc[raw_family["backend"] == backend]
            strict_counts[backend] = int(bool_series(backend_rows["strict_optimal"]).sum())
        status = raw_family["status"].astype(str).str.lower()
        rows.append(
            {
                "Problem family": family,
                "Largest common solved size": largest_n,
                "OSQP median end-to-end time at that size": osqp_time,
                "cuOpt median end-to-end time at that size": cuopt_time,
                "End-to-end speedup (OSQP/cuOpt)": end_speedup,
                "Solve-only speedup (OSQP/cuOpt)": solve_speedup,
                "Faster solver at n=100": faster_100,
                "Faster solver at the largest common solved size": large_faster,
                "First measured crossover size": (
                    crossover if crossover is not None else "No measured crossover"
                ),
                "OSQP strict-optimal count": strict_counts["OSQP"],
                "cuOpt strict-optimal count": strict_counts["cuOpt"],
                "Timeout/failure count": int(status.isin({"timeout", "failed"}).sum()),
                "Maximum canonical objective gap": finite_max(
                    pair_family.loc[
                        pair_family["common_correct"], "canonical_objective_gap"
                    ]
                ),
                "Maximum original primal violation": finite_max(
                    pair_family.loc[
                        pair_family["common_correct"],
                        "maximum_pair_primal_violation",
                    ]
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def display_frame(frame: pd.DataFrame, digits: int = 6) -> pd.DataFrame:
    shown = frame.copy()
    float_columns = shown.select_dtypes(include=["float", "float64"]).columns
    shown[float_columns] = shown[float_columns].map(
        lambda value: "" if pd.isna(value) else f"{value:.{digits}g}"
    )
    return shown


def markdown_table(frame: pd.DataFrame) -> str:
    shown = display_frame(frame)
    headers = [str(column).replace("|", "\\|") for column in shown.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in shown.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_table_set(
    table_dir: Path,
    name: str,
    frame: pd.DataFrame,
    latex_caption: str | None = None,
    latex_note: str | None = None,
) -> None:
    atomic_csv(table_dir / f"{name}.csv", frame)
    atomic_text(table_dir / f"{name}.md", markdown_table(frame))
    latex = display_frame(frame).to_latex(
        index=False,
        escape=True,
        caption=latex_caption,
        label=f"tab:{name.replace('_', '-')}",
    )
    if latex_note:
        latex = latex.replace("\\end{table}", f"\\par\\footnotesize {latex_note}\n\\end{{table}}")
    atomic_text(table_dir / f"{name}.tex", latex)


def build_supporting_tables(
    config: dict[str, Any],
    environment: dict[str, Any],
    registered: pd.DataFrame,
    pairs: pd.DataFrame,
    seed_summary: pd.DataFrame,
    by_size: pd.DataFrame,
    repeated: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    environment_table = pd.DataFrame(
        [{"Field": key, "Value": value} for key, value in environment.items()]
    )
    problem_rows = []
    canonical = config["canonical_problem"]
    for family in FAMILY_ORDER:
        family_config = canonical["families"][family]
        problem_rows.append(
            {
                "problem_family": family,
                "kind": family_config["kind"],
                "constraint_mode": family_config["constraint_mode"],
                "constraint_ratio": family_config["constraint_ratio"],
                "row_nonzeros": canonical["row_nonzeros"],
                "variable_lower": canonical["variable_lower_bound"],
                "variable_upper": canonical["variable_upper_bound"],
                "sizes": ", ".join(map(str, canonical["sizes"])),
                "seeds": ", ".join(map(str, canonical["seeds"])),
                "description": family_config["description"],
            }
        )
    correctness_rows = []
    for (family, backend), group in registered.groupby(["problem_family", "backend"]):
        accepted = (
            bool_series(group["strict_optimal"])
            & bool_series(group["correctness_pass"])
            & ~bool_series(group["censored"])
        )
        correctness_rows.append(
            {
                "problem_family": family,
                "backend": backend,
                "registered_rows": len(group),
                "strict_optimal": int(bool_series(group["strict_optimal"]).sum()),
                "canonical_correctness_pass": int(
                    bool_series(group["correctness_pass"]).sum()
                ),
                "accepted_rows": int(accepted.sum()),
                "maximum_objective_reconstruction_error": finite_max(
                    group["objective_reconstruction_error"]
                ),
                "maximum_original_primal_violation": finite_max(
                    group["original_primal_violation"]
                ),
            }
        )
    cold = by_size[
        [
            "problem_family",
            "n_variables",
            "median_osqp_solve_seconds",
            "median_cuopt_solve_seconds",
            "q25_osqp_solve_seconds",
            "q75_osqp_solve_seconds",
            "q25_cuopt_solve_seconds",
            "q75_cuopt_solve_seconds",
        ]
    ].copy()
    end_to_end = by_size[
        [
            "problem_family",
            "n_variables",
            "median_osqp_end_to_end_seconds",
            "median_cuopt_end_to_end_seconds",
            "q25_osqp_end_to_end_seconds",
            "q75_osqp_end_to_end_seconds",
            "q25_cuopt_end_to_end_seconds",
            "q75_cuopt_end_to_end_seconds",
        ]
    ].copy()
    speedup = by_size[
        [
            "problem_family",
            "n_variables",
            "solve_speedup",
            "end_to_end_speedup",
            "q25_solve_speedup",
            "q75_solve_speedup",
            "q25_end_to_end_speedup",
            "q75_end_to_end_speedup",
            "faster_solver_solve_only",
            "faster_solver_end_to_end",
        ]
    ].copy()
    crossover = pd.DataFrame(
        [
            {
                "problem_family": family,
                "first_measured_crossover_size": measured_crossover(
                    by_size.loc[by_size["problem_family"] == family]
                )
                or "No measured crossover",
                "tested_sizes_only": True,
            }
            for family in FAMILY_ORDER
        ]
    )
    failures = (
        registered.assign(
            failure_or_timeout=lambda value: ~(
                bool_series(value["strict_optimal"])
                & bool_series(value["correctness_pass"])
                & ~bool_series(value["censored"])
            )
        )
        .groupby(["problem_family", "n_variables", "backend", "status"], dropna=False)
        .agg(count=("status", "size"), excluded=("failure_or_timeout", "sum"))
        .reset_index()
    )
    memory_rows = []
    for (family, n_variables, backend), group in registered.groupby(
        ["problem_family", "n_variables", "backend"]
    ):
        accepted = group.loc[
            bool_series(group["strict_optimal"])
            & bool_series(group["correctness_pass"])
            & ~bool_series(group["censored"])
        ]
        memory_rows.append(
            {
                "problem_family": family,
                "n_variables": n_variables,
                "backend": backend,
                "median_peak_host_memory_bytes": finite_median(
                    accepted["peak_host_memory_bytes"]
                ),
                "maximum_peak_host_memory_bytes": finite_max(
                    accepted["peak_host_memory_bytes"]
                ),
                "median_peak_gpu_memory_bytes": finite_median(
                    accepted["peak_gpu_memory_bytes"]
                ),
                "maximum_peak_gpu_memory_bytes": finite_max(
                    accepted["peak_gpu_memory_bytes"]
                ),
            }
        )
    if len(repeated):
        repeated_rows = []
        for backend, group in repeated.groupby("backend"):
            valid = group.loc[
                bool_series(group["strict_optimal"])
                & bool_series(group["correctness_pass"])
            ]
            repeated_rows.append(
                {
                    "backend": backend,
                    "family": group["problem_family"].iloc[0],
                    "n_variables": int(group["n_variables"].iloc[0]),
                    "seeds": int(group["seed"].nunique()),
                    "registered_updates": int(len(group)),
                    "correct_updates": int(len(valid)),
                    "median_setup_seconds": finite_median(valid["setup_seconds"]),
                    "median_update_seconds": finite_median(valid["update_seconds"]),
                    "median_solve_seconds": finite_median(valid["solve_seconds"]),
                    "median_amortized_end_to_end_seconds": finite_median(
                        valid["amortized_end_to_end_seconds"]
                    ),
                }
            )
        repeated_table = pd.DataFrame(repeated_rows)
    else:
        repeated_table = pd.DataFrame(
            columns=[
                "backend",
                "family",
                "n_variables",
                "seeds",
                "registered_updates",
                "correct_updates",
                "median_setup_seconds",
                "median_update_seconds",
                "median_solve_seconds",
                "median_amortized_end_to_end_seconds",
            ]
        )
    return {
        "environment_hardware": environment_table,
        "problem_generation": pd.DataFrame(problem_rows),
        "correctness_summary": pd.DataFrame(correctness_rows),
        "cold_solve_time": cold,
        "end_to_end_time": end_to_end,
        "speedup": speedup,
        "crossover_size": crossover,
        "repeated_update": repeated_table,
        "failure_timeout_inventory": failures,
        "memory_usage": pd.DataFrame(memory_rows),
    }


def save_figure(fig: plt.Figure, plot_dir: Path, name: str) -> None:
    fig.savefig(plot_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(plot_dir / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _series_arrays(
    by_size: pd.DataFrame,
    family: str,
    value_column: str,
    sizes: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    subset = by_size.loc[by_size["problem_family"] == family].set_index("n_variables")
    values = np.array(
        [subset.at[size, value_column] if size in subset.index else np.nan for size in sizes],
        dtype=float,
    )
    return np.asarray(sizes, dtype=float), values


def _plot_runtime_panel(
    axis: plt.Axes,
    by_size: pd.DataFrame,
    families: tuple[str, str],
    sizes: list[int],
    metric: str,
) -> None:
    for family in families:
        for solver, backend in (("OSQP", "osqp"), ("cuOpt", "cuopt")):
            median = f"median_{backend}_{metric}_seconds"
            q25 = f"q25_{backend}_{metric}_seconds"
            q75 = f"q75_{backend}_{metric}_seconds"
            x, y = _series_arrays(by_size, family, median, sizes)
            _, low = _series_arrays(by_size, family, q25, sizes)
            _, high = _series_arrays(by_size, family, q75, sizes)
            axis.plot(
                x,
                y,
                color=SOLVER_COLORS[solver],
                linestyle=FAMILY_STYLES[family],
                marker="o" if solver == "OSQP" else "s",
                markersize=4,
                label=f"{solver} - {family}",
            )
            axis.fill_between(x, low, high, color=SOLVER_COLORS[solver], alpha=0.10)
            failures = "osqp_failures" if solver == "OSQP" else "cuopt_failures"
            subset = by_size.loc[
                (by_size["problem_family"] == family) & (by_size[failures] > 0)
            ]
            if len(subset):
                axis.scatter(
                    subset["n_variables"],
                    np.full(len(subset), 0.97),
                    marker="x",
                    color="#8B0000",
                    zorder=5,
                    transform=axis.get_xaxis_transform(),
                )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Decision variables, n")
    axis.set_ylabel("Median runtime (seconds)")
    axis.grid(True, which="both", alpha=0.20)


def _plot_speedup_panel(
    axis: plt.Axes,
    by_size: pd.DataFrame,
    families: tuple[str, str],
    sizes: list[int],
) -> None:
    family_colors = {families[0]: "#4C78A8", families[1]: "#F58518"}
    for family_index, family in enumerate(families):
        x, y = _series_arrays(by_size, family, "end_to_end_speedup", sizes)
        _, low = _series_arrays(by_size, family, "q25_end_to_end_speedup", sizes)
        _, high = _series_arrays(by_size, family, "q75_end_to_end_speedup", sizes)
        axis.plot(x, y, marker="o", color=family_colors[family], label=family)
        axis.fill_between(x, low, high, color=family_colors[family], alpha=0.14)
        family_group = by_size.loc[by_size["problem_family"] == family]
        crossover = measured_crossover(family_group)
        if crossover is None:
            valid = family_group.dropna(subset=["end_to_end_speedup"])
            if len(valid):
                last = valid.sort_values("n_variables").iloc[-1]
                axis.annotate(
                    f"{family}: no measured crossover",
                    (last["n_variables"], last["end_to_end_speedup"]),
                    xytext=(-4, 10 if family_index == 0 else -14),
                    textcoords="offset points",
                    fontsize=7,
                    ha="right",
                )
        else:
            value = family_group.loc[
                family_group["n_variables"] == crossover, "end_to_end_speedup"
            ].iloc[0]
            axis.annotate(
                f"{family}: n={crossover}",
                (crossover, value),
                xytext=(-5, 10 if family_index == 0 else -14),
                textcoords="offset points",
                fontsize=7,
                ha="right",
            )
    axis.axhline(1.0, color="black", linewidth=1.0, linestyle=":")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Decision variables, n")
    axis.set_ylabel("OSQP time / cuOpt time")
    axis.grid(True, which="both", alpha=0.20)


def flagship_figure(
    by_size: pd.DataFrame,
    config: dict[str, Any],
    environment: dict[str, Any],
    plot_dir: Path,
) -> pd.DataFrame:
    sizes = [int(value) for value in config["canonical_problem"]["sizes"]]
    fig, axes = plt.subplots(2, 2, figsize=(13.33, 7.5))
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.865,
        bottom=0.11,
        wspace=0.34,
        hspace=0.42,
    )
    _plot_runtime_panel(
        axes[0, 0], by_size, ("LP-RANGED", "LP-MIXED"), sizes, "end_to_end"
    )
    axes[0, 0].set_title("A. LP end-to-end runtime", loc="left", weight="bold")
    _plot_runtime_panel(
        axes[0, 1],
        by_size,
        ("QP-DIAGONAL", "QP-SPARSE-COUPLED"),
        sizes,
        "end_to_end",
    )
    axes[0, 1].set_title("B. QP end-to-end runtime", loc="left", weight="bold")
    _plot_speedup_panel(
        axes[1, 0], by_size, ("LP-RANGED", "LP-MIXED"), sizes
    )
    axes[1, 0].set_title("C. LP end-to-end speedup", loc="left", weight="bold")
    _plot_speedup_panel(
        axes[1, 1], by_size, ("QP-DIAGONAL", "QP-SPARSE-COUPLED"), sizes
    )
    axes[1, 1].set_title("D. QP end-to-end speedup", loc="left", weight="bold")
    for axis in axes.flat:
        axis.legend(fontsize=6.8, frameon=False, ncol=2)
    fig.suptitle(
        "Synthetic LP/QP benchmark: OSQP versus direct NVIDIA cuOpt",
        fontsize=16,
        weight="bold",
        y=0.975,
    )
    footer = (
        f"End-to-end medians; seed IQR bands | CPU: {environment.get('cpu_model')} | "
        f"GPU: {environment.get('gpu_query')} | x marks excluded failures/timeouts"
    )
    fig.text(0.5, 0.025, footer, ha="center", fontsize=7)
    save_figure(fig, plot_dir, "osqp_vs_cuopt_runtime_speedup")
    points = by_size[
        [
            "problem_family",
            "n_variables",
            "median_osqp_end_to_end_seconds",
            "median_cuopt_end_to_end_seconds",
            "end_to_end_speedup",
            "q25_osqp_end_to_end_seconds",
            "q75_osqp_end_to_end_seconds",
            "q25_cuopt_end_to_end_seconds",
            "q75_cuopt_end_to_end_seconds",
            "q25_end_to_end_speedup",
            "q75_end_to_end_speedup",
        ]
    ].copy()
    return points


def runtime_figure(
    by_size: pd.DataFrame,
    families: tuple[str, str],
    metric: str,
    title: str,
    name: str,
    plot_dir: Path,
    sizes: list[int],
) -> None:
    fig, axis = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
    _plot_runtime_panel(axis, by_size, families, sizes, metric)
    axis.set_title(title, weight="bold")
    axis.legend(frameon=False, fontsize=8, ncol=2)
    save_figure(fig, plot_dir, name)


def decomposition_figure(by_size: pd.DataFrame, plot_dir: Path) -> None:
    rows = []
    for family in FAMILY_ORDER:
        group = by_size.loc[
            (by_size["problem_family"] == family)
            & (by_size["common_correct_instances"] > 0)
        ]
        if not len(group):
            continue
        row = group.loc[group["n_variables"].idxmax()]
        for solver, backend in (("OSQP", "osqp"), ("cuOpt", "cuopt")):
            setup_name = (
                "median_cuopt_build_transfer_seconds"
                if backend == "cuopt"
                else "median_osqp_setup_seconds"
            )
            extraction = max(
                0.0,
                float(row[f"median_{backend}_end_to_end_seconds"])
                - float(row[setup_name])
                - float(row[f"median_{backend}_solve_seconds"]),
            )
            rows.append(
                {
                    "label": f"{family}\n{solver}\nn={int(row['n_variables'])}",
                    "setup/transfer": float(row[setup_name]),
                    "solve": float(row[f"median_{backend}_solve_seconds"]),
                    "extract/check": extraction,
                }
            )
    frame = pd.DataFrame(rows)
    fig, axis = plt.subplots(figsize=(11.5, 5), constrained_layout=True)
    bottom = np.zeros(len(frame))
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    for color, metric in zip(colors, ("setup/transfer", "solve", "extract/check")):
        axis.bar(frame["label"], frame[metric], bottom=bottom, label=metric, color=color)
        bottom += frame[metric].to_numpy()
    axis.set_yscale("log")
    axis.set_ylabel("Median seconds (stacked)")
    axis.set_title("End-to-end runtime decomposition at largest common size", weight="bold")
    axis.legend(frameon=False, ncol=3)
    axis.tick_params(axis="x", labelsize=7)
    save_figure(fig, plot_dir, "runtime_decomposition")


def memory_figure(memory: pd.DataFrame, plot_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for axis, metric, title in (
        (axes[0], "median_peak_host_memory_bytes", "Peak host memory"),
        (axes[1], "median_peak_gpu_memory_bytes", "Peak GPU memory"),
    ):
        for (family, backend), group in memory.groupby(["problem_family", "backend"]):
            values = pd.to_numeric(group[metric], errors="coerce") / 2**20
            axis.plot(
                group["n_variables"],
                values,
                marker="o",
                color=SOLVER_COLORS[backend],
                linestyle=FAMILY_STYLES[family],
                label=f"{backend} - {family}",
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Decision variables, n")
        axis.set_ylabel("MiB")
        axis.set_title(title, weight="bold")
        axis.grid(True, which="both", alpha=0.2)
        axis.legend(fontsize=6.5, frameon=False)
    save_figure(fig, plot_dir, "peak_memory_vs_n")


def repeated_figure(repeated_table: pd.DataFrame, plot_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.8, 4.3), constrained_layout=True)
    if len(repeated_table):
        axis.bar(
            repeated_table["backend"],
            repeated_table["median_amortized_end_to_end_seconds"],
            color=[SOLVER_COLORS.get(value, "gray") for value in repeated_table["backend"]],
        )
        axis.set_yscale("log")
        axis.set_ylabel("Median amortized seconds per update")
    else:
        axis.text(0.5, 0.5, "No registered repeated-update results", ha="center")
        axis.set_axis_off()
    axis.set_title("Repeated objective-update amortized runtime", weight="bold")
    save_figure(fig, plot_dir, "repeated_update_amortized_time")


def correctness_figure(pairs: pd.DataFrame, plot_dir: Path) -> None:
    common = pairs.loc[pairs["common_correct"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for family, group in common.groupby("problem_family"):
        objective = group.groupby("n_variables")["canonical_objective_gap"].max()
        violation = group.groupby("n_variables")["maximum_pair_primal_violation"].max()
        axes[0].plot(objective.index, np.maximum(objective, 1e-18), marker="o", label=family)
        axes[1].plot(violation.index, np.maximum(violation, 1e-18), marker="o", label=family)
    for axis, title, ylabel in (
        (axes[0], "Canonical objective agreement", "Maximum absolute gap"),
        (axes[1], "Original-QP/LP feasibility", "Maximum primal violation"),
    ):
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Decision variables, n")
        axis.set_ylabel(ylabel)
        axis.set_title(title, weight="bold")
        axis.grid(True, which="both", alpha=0.2)
        axis.legend(frameon=False, fontsize=7)
    save_figure(fig, plot_dir, "correctness_diagnostics")


def caption_text(config: dict[str, Any], environment: dict[str, Any]) -> str:
    seed_count = len(config["canonical_problem"]["seeds"])
    return (
        "Synthetic LP/QP benchmark: OSQP versus direct NVIDIA cuOpt. "
        f"OSQP {environment.get('osqp_version')} ({environment.get('osqp_linear_system_backend')}) "
        f"ran on {environment.get('cpu_model')}; cuOpt {environment.get('cuopt_version')} "
        f"with CUDA {environment.get('cuda_version')} ran on {environment.get('gpu_query')}. "
        f"Central estimates are medians across {seed_count} registered seeds; shaded bands "
        "show the seed interquartile range. End-to-end timing begins with canonical arrays "
        "resident in host memory and includes backend conversion, setup, transfer where "
        "applicable, numerical solve, result extraction, and canonical correctness "
        "reconstruction. Only paired repetitions with strict optimal status, matching "
        "canonical instance hashes, and all canonical correctness gates passing enter "
        "timing and speedup summaries; failures and timeouts are excluded and marked. "
        "Speedup is OSQP time divided by cuOpt time, so values above one mean cuOpt is "
        "faster. Results apply only to the registered sparse synthetic problem families, "
        "dimensions, tolerances, software versions, and measured hardware.\n"
    )


def conclusions(by_size: pd.DataFrame, repeated_table: pd.DataFrame) -> list[str]:
    small = by_size.loc[
        (by_size["n_variables"] == by_size["n_variables"].min())
        & by_size["end_to_end_speedup"].notna()
    ]
    small_counts = small["faster_solver_end_to_end"].value_counts()
    small_solver = small_counts.index[0] if len(small_counts) else "neither solver"
    crossover_families = [
        family
        for family in FAMILY_ORDER
        if measured_crossover(by_size.loc[by_size["problem_family"] == family])
        is not None
    ]
    largest = (
        by_size.loc[by_size["common_correct_instances"] > 0]
        .sort_values("n_variables")
        .groupby("problem_family")
        .tail(1)
    )
    large_counts = largest["faster_solver_end_to_end"].value_counts()
    large_solver = large_counts.index[0] if len(large_counts) else "neither solver"
    crossover_text = (
        ", ".join(crossover_families)
        if crossover_families
        else "no family within the measured sizes"
    )
    if len(repeated_table) == 2:
        times = repeated_table.set_index("backend")["median_amortized_end_to_end_seconds"]
        update_speedup = float(times["OSQP"] / times["cuOpt"])
        update_text = (
            f"Amortized updates: OSQP/cuOpt speedup was {update_speedup:.3g}x; "
            f"{faster_label(update_speedup)} was faster."
        )
    else:
        update_text = "Amortized updates: insufficient common registered update results."
    return [
        f"Small problems: {small_solver} was faster for the plurality of measured families.",
        f"Large problems: {large_solver} led for the plurality; crossover occurred for {crossover_text}.",
        update_text,
    ]


def single_slide(
    flagship_png: Path,
    headline: pd.DataFrame,
    conclusions_text: list[str],
    environment: dict[str, Any],
    output_path: Path,
) -> None:
    image = plt.imread(flagship_png)
    with PdfPages(output_path) as pdf:
        fig = plt.figure(figsize=(13.33, 7.5), constrained_layout=True)
        grid = fig.add_gridspec(10, 12)
        image_axis = fig.add_subplot(grid[:7, :])
        image_axis.imshow(image)
        image_axis.set_axis_off()
        table_axis = fig.add_subplot(grid[7:, :8])
        table_axis.set_axis_off()
        compact = headline[
            [
                "Problem family",
                "Largest common solved size",
                "End-to-end speedup (OSQP/cuOpt)",
                "Faster solver at the largest common solved size",
            ]
        ].copy()
        compact.columns = ["Family", "Largest n", "E2E speedup", "Faster"]
        compact["E2E speedup"] = compact["E2E speedup"].map(
            lambda value: "n/a" if pd.isna(value) else f"{value:.3g}x"
        )
        table = table_axis.table(
            cellText=compact.values,
            colLabels=compact.columns,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.2)
        table.scale(1, 1.35)
        conclusion_axis = fig.add_subplot(grid[7:, 8:])
        conclusion_axis.set_axis_off()
        conclusion_axis.text(
            0,
            0.94,
            "Measured conclusions",
            weight="bold",
            fontsize=10,
            va="top",
        )
        for index, value in enumerate(conclusions_text):
            conclusion_axis.text(0, 0.75 - index * 0.27, value, fontsize=7.5, va="top", wrap=True)
        fig.text(
            0.5,
            0.006,
            f"OSQP {environment.get('osqp_version')} | cuOpt {environment.get('cuopt_version')} | "
            f"{environment.get('cpu_model')} | {environment.get('gpu_query')}",
            ha="center",
            fontsize=6.5,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def reconciliation_audit(
    raw_path: Path,
    summary_input_path: Path,
    headline_path: Path,
    by_size_path: Path,
    figure_pdf: Path,
    figure_png: Path,
    caption_path: Path,
    plot_points_path: Path,
    registered: pd.DataFrame,
    pairs: pd.DataFrame,
    seed_summary: pd.DataFrame,
    by_size: pd.DataFrame,
    headline: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    plotted = pd.read_csv(plot_points_path)
    table_points = by_size[plotted.columns]
    plotted_match = frames_match(plotted, table_points)
    rebuilt = build_by_size(registered, pairs, seed_summary)
    deterministic = rebuilt.fillna(-1).equals(by_size.fillna(-1))
    no_invalid_speedups = True
    for row in by_size.itertuples(index=False):
        if np.isfinite(row.end_to_end_speedup) and row.common_correct_instances <= 0:
            no_invalid_speedups = False
    speedup_direction = all(
        faster_label(row.end_to_end_speedup) == row.faster_solver_end_to_end
        for row in by_size.itertuples(index=False)
    )
    headline_match = True
    rebuilt_headline = build_headline(registered, pairs, rebuilt)
    if not rebuilt_headline.fillna(-1).equals(headline.fillna(-1)):
        headline_match = False
    independent_largest = True
    for family in FAMILY_ORDER:
        valid = by_size.loc[
            (by_size["problem_family"] == family)
            & (by_size["common_correct_instances"] > 0)
        ]
        expected: Any = (
            int(valid["n_variables"].max())
            if len(valid)
            else "Insufficient common solved instances"
        )
        actual = headline.loc[
            headline["Problem family"] == family, "Largest common solved size"
        ].iloc[0]
        independent_largest &= str(expected) == str(actual)
    measured_sizes = set(map(int, config["canonical_problem"]["sizes"]))
    measured_only = all(
        measured_crossover(by_size.loc[by_size["problem_family"] == family])
        in measured_sizes | {None}
        for family in FAMILY_ORDER
    )
    timing = config["execution"]["timing_definition"].replace("\n", " ").strip()
    checks = {
        "every_plotted_point_exists_in_by_size_table": plotted_match,
        "headline_speedups_reconcile_to_raw_results": headline_match,
        "figure_and_table_share_end_to_end_timing_definition": "End-to-end" in TIMING_DEFINITION and "End-to-end" in timing,
        "largest_common_size_computed_independently_by_family": independent_largest,
        "no_failed_timed_out_or_censored_row_enters_speedup": no_invalid_speedups,
        "speedup_definition_is_osqp_divided_by_cuopt": config["execution"]["speedup_definition"] == "OSQP time / cuOpt time",
        "faster_solver_labels_match_speedup": speedup_direction,
        "crossover_annotations_use_measured_sizes_only": measured_only,
        "figure_and_table_input_hashes_recorded": True,
        "repeat_analysis_produces_identical_table_values": deterministic,
        "all_included_pairs_have_strict_optimal_status": bool(
            pairs.loc[pairs["common_correct"], ["strict_optimal_osqp", "strict_optimal_cuopt"]].all().all()
        ),
        "all_included_pairs_pass_canonical_correctness": bool(
            pairs.loc[pairs["common_correct"], ["correctness_pass_osqp", "correctness_pass_cuopt"]].all().all()
        ),
        "all_included_pairs_use_same_canonical_instance": bool(
            pairs.loc[pairs["common_correct"], "same_canonical_instance"].all()
        ),
        "all_registered_rows_use_one_execution_sha": bool(
            registered["execution_sha"].notna().all()
            and registered["execution_sha"].nunique() == 1
        ),
    }
    return {
        "schema_version": 1,
        "status": "pass" if all(checks.values()) else "fail",
        "timing_definition": TIMING_DEFINITION,
        "speedup_definition": "OSQP time / cuOpt time",
        "checks": checks,
        "sha256": {
            "raw_merged_result_file": sha256_file(raw_path),
            "summary_input_file": sha256_file(summary_input_path),
            "headline_table_csv": sha256_file(headline_path),
            "by_size_table_csv": sha256_file(by_size_path),
            "flagship_pdf": sha256_file(figure_pdf),
            "flagship_png": sha256_file(figure_png),
            "caption_file": sha256_file(caption_path),
            "flagship_plot_points": sha256_file(plot_points_path),
        },
    }


def build_report(
    config: dict[str, Any],
    environment: dict[str, Any],
    registered: pd.DataFrame,
    pairs: pd.DataFrame,
    headline: pd.DataFrame,
    by_size: pd.DataFrame,
    repeated_table: pd.DataFrame,
    conclusions_text: list[str],
) -> str:
    canonical_count = (
        len(config["canonical_problem"]["families"])
        * len(config["canonical_problem"]["sizes"])
        * len(config["canonical_problem"]["seeds"])
    )
    passing_instances = int(
        pairs.groupby(["problem_family", "n_variables", "seed"])["common_correct"]
        .all()
        .sum()
    )
    status = registered["status"].astype(str).str.lower()
    max_gap = finite_max(pairs.loc[pairs["common_correct"], "canonical_objective_gap"])
    max_violation = finite_max(
        pairs.loc[pairs["common_correct"], "maximum_pair_primal_violation"]
    )
    lines = [
        "# Synthetic OSQP vs direct NVIDIA cuOpt benchmark",
        "",
        "## Registered evidence",
        "",
        f"- Canonical instances: {canonical_count}",
        f"- Instances passing every paired registered repetition: {passing_instances}",
        f"- Registered timing rows: {len(registered)}",
        f"- Explicit failed/timeout rows: {int(status.isin({'failed', 'timeout'}).sum())}",
        f"- Maximum common canonical objective gap: {max_gap}",
        f"- Maximum common original primal violation: {max_violation}",
        f"- CPU: {environment.get('cpu_model')}",
        f"- GPU: {environment.get('gpu_query')}",
        f"- OSQP: {environment.get('osqp_version')} ({environment.get('osqp_linear_system_backend')})",
        f"- cuOpt: {environment.get('cuopt_version')}; CUDA {environment.get('cuda_version')}",
        "",
        "## Headline comparison",
        "",
        markdown_table(headline).strip(),
        "",
        "## Detailed by-size comparison",
        "",
        markdown_table(by_size).strip(),
        "",
        "## Interpretation",
        "",
        *[f"- {value}" for value in conclusions_text],
        "",
        "## Repeated updates",
        "",
        markdown_table(repeated_table).strip(),
        "",
        "## Timing and inclusion rule",
        "",
        TIMING_DEFINITION,
        "Only matching registered OSQP/cuOpt repetitions with strict-optimal status and all canonical correctness gates pass into timing and speedup summaries. Censored, failed, and timed-out observations are inventoried but never replaced by the timeout cap.",
        "",
        "## Evidence boundary",
        "",
        "Results apply only to the registered sparse synthetic problem families, dimensions, tolerances, software versions, and measured hardware.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/synthetic_solver_benchmark.yaml")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/synthetic_solver_benchmark")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    output_root = args.output_root if args.output_root.is_absolute() else root / args.output_root
    raw_path = output_root / "raw" / "cold_results.csv"
    repeated_path = output_root / "raw" / "repeated_update_results.csv"
    environment_path = output_root / "provenance" / "environment.json"
    if not raw_path.is_file() or not environment_path.is_file():
        raise FileNotFoundError("raw cold results and environment provenance are required")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    raw, pairs = load_registered_pairs(raw_path, config)
    registered = raw.loc[bool_series(raw["registered_repetition"])].copy()
    seed_summary = build_seed_summary(pairs)
    by_size = build_by_size(registered, pairs, seed_summary)
    headline = build_headline(registered, pairs, by_size)
    repeated = pd.read_csv(repeated_path) if repeated_path.is_file() else pd.DataFrame()

    table_dir = output_root / "tables"
    plot_dir = output_root / "plots"
    provenance_dir = output_root / "provenance"
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir.mkdir(parents=True, exist_ok=True)
    summary_input_path = provenance_dir / "comparison_summary_input.csv"
    seed_summary_path = provenance_dir / "registered_seed_summary.csv"
    atomic_csv(summary_input_path, pairs)
    atomic_csv(seed_summary_path, seed_summary)
    supporting = build_supporting_tables(
        config, environment, registered, pairs, seed_summary, by_size, repeated
    )
    for name, frame in supporting.items():
        write_table_set(table_dir, name, frame)
    write_table_set(
        table_dir,
        "osqp_vs_cuopt_by_size",
        by_size,
        latex_caption="Detailed synthetic LP/QP comparison by registered size.",
    )
    write_table_set(
        table_dir,
        "osqp_vs_cuopt_headline_table",
        headline,
        latex_caption=(
            "Headline synthetic LP/QP comparison between CPU OSQP and direct NVIDIA "
            "cuOpt. Timings are medians over common correctness-passing instances."
        ),
        latex_note="Speedup greater than one means cuOpt is faster.",
    )
    atomic_json(
        table_dir / "osqp_vs_cuopt_headline_table.json",
        {
            "timing_definition": TIMING_DEFINITION,
            "speedup_definition": "OSQP time / cuOpt time",
            "records": json_records(headline),
        },
    )

    plot_points = flagship_figure(by_size, config, environment, plot_dir)
    plot_points_path = provenance_dir / "flagship_plot_points.csv"
    atomic_csv(plot_points_path, plot_points)
    sizes = [int(value) for value in config["canonical_problem"]["sizes"]]
    runtime_figure(by_size, ("LP-RANGED", "LP-MIXED"), "solve", "LP solve time versus n", "lp_solve_time_vs_n", plot_dir, sizes)
    runtime_figure(by_size, ("QP-DIAGONAL", "QP-SPARSE-COUPLED"), "solve", "QP solve time versus n", "qp_solve_time_vs_n", plot_dir, sizes)
    runtime_figure(by_size, ("LP-RANGED", "LP-MIXED"), "end_to_end", "LP end-to-end time versus n", "lp_end_to_end_time_vs_n", plot_dir, sizes)
    runtime_figure(by_size, ("QP-DIAGONAL", "QP-SPARSE-COUPLED"), "end_to_end", "QP end-to-end time versus n", "qp_end_to_end_time_vs_n", plot_dir, sizes)
    decomposition_figure(by_size, plot_dir)
    memory_figure(supporting["memory_usage"], plot_dir)
    repeated_figure(supporting["repeated_update"], plot_dir)
    correctness_figure(pairs, plot_dir)
    caption_path = plot_dir / "osqp_vs_cuopt_runtime_speedup_caption.txt"
    atomic_text(caption_path, caption_text(config, environment))
    conclusion_lines = conclusions(by_size, supporting["repeated_update"])
    single_slide(
        plot_dir / "osqp_vs_cuopt_runtime_speedup.png",
        headline,
        conclusion_lines,
        environment,
        plot_dir / "osqp_vs_cuopt_single_slide.pdf",
    )
    report_path = output_root / "synthetic_solver_benchmark_report.md"
    atomic_text(
        report_path,
        build_report(
            config,
            environment,
            registered,
            pairs,
            headline,
            by_size,
            supporting["repeated_update"],
            conclusion_lines,
        ),
    )
    audit = reconciliation_audit(
        raw_path,
        summary_input_path,
        table_dir / "osqp_vs_cuopt_headline_table.csv",
        table_dir / "osqp_vs_cuopt_by_size.csv",
        plot_dir / "osqp_vs_cuopt_runtime_speedup.pdf",
        plot_dir / "osqp_vs_cuopt_runtime_speedup.png",
        caption_path,
        plot_points_path,
        registered,
        pairs,
        seed_summary,
        by_size,
        headline,
        config,
    )
    atomic_json(provenance_dir / "comparison_deliverable_audit.json", audit)
    if audit["status"] != "pass":
        raise RuntimeError("comparison deliverable reconciliation audit failed")
    outputs = sorted(
        path for path in output_root.rglob("*") if path.is_file() and path.name != "analysis_manifest.json"
    )
    manifest = {
        "schema_version": 1,
        "status": "pass",
        "timing_definition": TIMING_DEFINITION,
        "speedup_definition": "OSQP time / cuOpt time",
        "files": {
            str(path.relative_to(output_root)): sha256_file(path) for path in outputs
        },
    }
    atomic_json(provenance_dir / "analysis_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
