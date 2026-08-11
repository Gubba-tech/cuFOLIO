from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from benchmarks import analyze_synthetic_solver_benchmark as analysis
from benchmarks.analyze_synthetic_solver_benchmark import (
    build_by_size,
    build_headline,
    build_seed_summary,
    faster_label,
    load_registered_pairs,
    measured_crossover,
)
from benchmarks.synthetic_solver_benchmark import (
    FAMILY_ORDER,
    anchor_violation,
    generate_canonical_problem,
    solve_osqp,
)


@pytest.fixture
def benchmark_config() -> dict:
    path = Path(__file__).parents[1] / "configs" / "synthetic_solver_benchmark.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_registered_generators_are_deterministic_feasible_and_bounded(
    benchmark_config: dict,
) -> None:
    for family in FAMILY_ORDER:
        first = generate_canonical_problem(benchmark_config, family, 31, 11)
        second = generate_canonical_problem(benchmark_config, family, 31, 11)
        assert first.canonical_sha256 == second.canonical_sha256
        assert anchor_violation(first) <= 1.0e-12
        assert np.all(first.variable_lower == -1.0)
        assert np.all(first.variable_upper == 1.0)
        assert (first.P - first.P.T).nnz == 0
        if family.startswith("LP-"):
            assert first.P.nnz == 0
        else:
            eigenvalue = np.linalg.eigvalsh(first.P.toarray()).min()
            assert eigenvalue > 0.0


def test_osqp_cpu_smoke_passes_all_registered_families(
    benchmark_config: dict,
) -> None:
    for family in FAMILY_ORDER:
        problem = generate_canonical_problem(benchmark_config, family, 50, 11)
        result = solve_osqp(problem, benchmark_config)
        assert result.raw_status == "solved"
        assert result.strict_optimal
        assert result.correctness_pass
        assert result.original_primal_violation <= 1.0e-6


def _raw_fixture() -> pd.DataFrame:
    rows = []
    for family_index, family in enumerate(FAMILY_ORDER):
        for n_variables in (100, 300):
            for seed in (11, 29):
                canonical_hash = f"{family}-{n_variables}-{seed}"
                for repetition in (0, 1):
                    for backend in ("OSQP", "cuOpt"):
                        cuopt = backend == "cuOpt"
                        base = (family_index + 1) * n_variables / 100_000
                        solve = base * (0.5 if cuopt else 1.0)
                        end_to_end = base * (
                            (1.5 if n_variables == 100 else 0.4) if cuopt else 1.0
                        )
                        strict = not (
                            family == "LP-MIXED"
                            and n_variables == 300
                            and seed == 29
                            and repetition == 1
                            and cuopt
                        )
                        rows.append(
                            {
                                "problem_family": family,
                                "n_variables": n_variables,
                                "n_constraints": n_variables,
                                "seed": seed,
                                "repetition": repetition,
                                "registered_repetition": True,
                                "backend": backend,
                                "canonical_sha256": canonical_hash,
                                "execution_sha": "fixture-execution-sha",
                                "status": "optimal" if strict else "failed",
                                "raw_status": (
                                    "Optimal" if cuopt else "solved"
                                )
                                if strict
                                else "failed",
                                "strict_optimal": strict,
                                "correctness_pass": strict,
                                "censored": False,
                                "setup_seconds": base * 0.2,
                                "solve_seconds": solve if strict else np.nan,
                                "reported_solve_seconds": solve if strict else np.nan,
                                "result_extraction_seconds": base * 0.05,
                                "correctness_seconds": base * 0.01,
                                "end_to_end_seconds": end_to_end if strict else np.nan,
                                "objective_value": 1.0 if strict else np.nan,
                                "reported_objective_value": 1.0 if strict else np.nan,
                                "objective_reconstruction_error": 0.0 if strict else np.nan,
                                "original_primal_violation": 1.0e-9 if strict else np.nan,
                                "peak_host_memory_bytes": 1_000_000,
                                "peak_gpu_memory_bytes": 2_000_000 if cuopt else 0,
                            }
                        )
    return pd.DataFrame(rows)


def test_reconciliation_uses_only_paired_correct_registered_rows(
    benchmark_config: dict, tmp_path: Path
) -> None:
    raw = _raw_fixture()
    raw_path = tmp_path / "cold_results.csv"
    raw.to_csv(raw_path, index=False)
    loaded, pairs = load_registered_pairs(raw_path, benchmark_config)
    registered = loaded.loc[loaded["registered_repetition"]].copy()
    seeds = build_seed_summary(pairs)
    by_size = build_by_size(registered, pairs, seeds)
    headline = build_headline(registered, pairs, by_size)

    assert len(by_size) == len(FAMILY_ORDER) * 2
    assert np.allclose(by_size["solve_speedup"], 2.0)
    assert set(by_size.loc[by_size["n_variables"] == 100, "faster_solver_end_to_end"]) == {
        "OSQP"
    }
    assert set(by_size.loc[by_size["n_variables"] == 300, "faster_solver_end_to_end"]) == {
        "cuOpt"
    }
    assert all(measured_crossover(by_size.loc[by_size["problem_family"] == family]) == 300 for family in FAMILY_ORDER)
    lp_mixed = by_size.loc[
        (by_size["problem_family"] == "LP-MIXED")
        & (by_size["n_variables"] == 300)
    ].iloc[0]
    assert lp_mixed.common_correct_instances == 2
    assert lp_mixed.cuopt_failures == 1
    assert headline["Largest common solved size"].tolist() == [300] * 4
    assert all(
        faster_label(value) == label
        for value, label in zip(
            by_size["end_to_end_speedup"],
            by_size["faster_solver_end_to_end"],
        )
    )


def test_resume_boolean_string_false_is_not_registered(
    benchmark_config: dict, tmp_path: Path
) -> None:
    raw = _raw_fixture()
    warmup = raw.iloc[[0]].copy()
    warmup["registered_repetition"] = "False"
    warmup["repetition"] = -1
    raw["registered_repetition"] = "True"
    raw = pd.concat([raw, warmup], ignore_index=True)
    path = tmp_path / "cold_results.csv"
    raw.to_csv(path, index=False)
    loaded, _pairs = load_registered_pairs(path, benchmark_config)
    assert len(loaded) == len(raw)


def test_analysis_generates_and_reconciles_required_deliverables(
    benchmark_config: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark_config["canonical_problem"]["sizes"] = [100, 300]
    benchmark_config["canonical_problem"]["seeds"] = [11, 29]
    benchmark_config["canonical_problem"]["registered_repetitions"] = 2
    benchmark_config["canonical_problem"]["warmup_repetitions"] = 0
    output_root = tmp_path / "artifacts"
    (output_root / "raw").mkdir(parents=True)
    (output_root / "provenance").mkdir(parents=True)
    _raw_fixture().to_csv(output_root / "raw" / "cold_results.csv", index=False)
    repeated_rows = []
    for backend, seconds in (("OSQP", 0.02), ("cuOpt", 0.01)):
        repeated_rows.append(
            {
                "problem_family": "QP-DIAGONAL",
                "n_variables": 3000,
                "n_constraints": 1500,
                "seed": 11,
                "update_index": 0,
                "backend": backend,
                "canonical_sha256": "update-fixture",
                "setup_seconds": 0.1,
                "update_seconds": 0.001,
                "solve_seconds": seconds,
                "result_extraction_seconds": 0.001,
                "correctness_seconds": 0.001,
                "amortized_end_to_end_seconds": seconds + 0.003,
                "raw_status": "solved" if backend == "OSQP" else "Optimal",
                "strict_optimal": True,
                "correctness_pass": True,
                "objective_value": 1.0,
                "reported_objective_value": 1.0,
                "objective_reconstruction_error": 0.0,
                "original_primal_violation": 1.0e-9,
            }
        )
    pd.DataFrame(repeated_rows).to_csv(
        output_root / "raw" / "repeated_update_results.csv", index=False
    )
    environment = {
        "cpu_model": "fixture CPU",
        "gpu_query": "fixture GPU",
        "osqp_version": "fixture",
        "osqp_linear_system_backend": "fixture",
        "cuopt_version": "fixture",
        "cuda_version": "fixture",
    }
    (output_root / "provenance" / "environment.json").write_text(
        json.dumps(environment), encoding="utf-8"
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(benchmark_config), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_synthetic_solver_benchmark.py",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
        ],
    )
    analysis.main()
    audit = json.loads(
        (output_root / "provenance" / "comparison_deliverable_audit.json").read_text()
    )
    assert audit["status"] == "pass"
    assert (output_root / "plots" / "osqp_vs_cuopt_runtime_speedup.pdf").is_file()
    assert (output_root / "plots" / "osqp_vs_cuopt_single_slide.pdf").is_file()
    assert (output_root / "tables" / "osqp_vs_cuopt_headline_table.json").is_file()
    assert len(list((output_root / "plots").glob("*.png"))) == 9
    assert len(list((output_root / "plots").glob("*.pdf"))) == 10
