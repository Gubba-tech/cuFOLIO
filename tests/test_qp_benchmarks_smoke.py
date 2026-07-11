import csv
import subprocess
import sys
from pathlib import Path

import pytest
from qp_test_utils import require_cuopt

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"


def _run(script, args, output_dir, timeout=180):
    result = subprocess.run(
        [sys.executable, str(BENCHMARKS / script), *args, "--output-dir", str(output_dir)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, (
        f"{script} failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    run_dirs = sorted(path for path in output_dir.iterdir() if path.is_dir())
    assert run_dirs, result.stdout
    return run_dirs[-1], result.stdout


def _assert_result_csv(path, expected_columns):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert expected_columns <= set(rows[0])
    assert any(row["status"] == "optimal" for row in rows)


def test_qp_benchmark_cpu_smoke(tmp_path):
    stock_dir, _ = _run(
        "benchmark_qp_stock_level.py",
        [
            "--backend",
            "osqp",
            "--n-assets",
            "10",
            "--objectives",
            "min_variance",
            "--cases",
            "basic",
            "--repeats",
            "1",
            "--warmup",
            "0",
        ],
        tmp_path,
    )
    _assert_result_csv(
        stock_dir / "stock_level_results.csv",
        {
            "compile_time_sec",
            "solver_build_time_sec",
            "solve_time_sec",
            "total_time_sec",
            "max_constraint_violation",
            "git_commit",
        },
    )

    factor_dir, _ = _run(
        "benchmark_qp_factor_space.py",
        [
            "--backend",
            "osqp",
            "--n-assets",
            "12",
            "--n-factors",
            "3",
            "--objectives",
            "mean_variance",
            "--cases",
            "pca",
            "--repeats",
            "1",
            "--warmup",
            "0",
        ],
        tmp_path,
    )
    _assert_result_csv(
        factor_dir / "factor_space_results.csv",
        {
            "mapping_mode",
            "factor_model",
            "n_time_observations",
            "stock_mapping_shape",
            "factor_covariance_condition_number",
        },
    )

    rolling_dir, _ = _run(
        "benchmark_qp_rolling_windows.py",
        [
            "--backend",
            "osqp",
            "--n-assets",
            "12",
            "--n-windows",
            "3",
            "--objective",
            "max_sharpe",
            "--mode",
            "stock",
            "--repeats",
            "1",
        ],
        tmp_path,
    )
    _assert_result_csv(
        rolling_dir / "rolling_windows_results.csv",
        {"window", "n_windows", "total_time_sec", "turnover"},
    )
    assert (rolling_dir / "rolling_aggregate_summary.csv").exists()

    summary = subprocess.run(
        [
            sys.executable,
            str(BENCHMARKS / "summarize_qp_benchmarks.py"),
            "--input-dir",
            str(stock_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert summary.returncode == 0, summary.stderr
    assert (stock_dir / "summary.csv").exists()
    assert (stock_dir / "summary.json").exists()
    assert (stock_dir / "summary.md").exists()


@pytest.mark.gpu
def test_qp_stock_benchmark_gpu_smoke(tmp_path):
    require_cuopt()
    run_dir, _ = _run(
        "benchmark_qp_stock_level.py",
        [
            "--backend",
            "cuopt",
            "--n-assets",
            "10",
            "--objectives",
            "min_variance",
            "--cases",
            "basic",
            "--repeats",
            "1",
            "--warmup",
            "0",
        ],
        tmp_path,
    )
    _assert_result_csv(run_dir / "stock_level_results.csv", {"solver_name"})


@pytest.mark.gpu
def test_qp_factor_benchmark_gpu_smoke(tmp_path):
    require_cuopt()
    run_dir, _ = _run(
        "benchmark_qp_factor_space.py",
        [
            "--backend",
            "cuopt",
            "--n-assets",
            "12",
            "--n-factors",
            "3",
            "--objectives",
            "max_sharpe",
            "--cases",
            "pca",
            "--repeats",
            "1",
            "--warmup",
            "0",
        ],
        tmp_path,
    )
    _assert_result_csv(
        run_dir / "factor_space_results.csv",
        {"mapping_mode", "factor_model", "solver_name"},
    )
