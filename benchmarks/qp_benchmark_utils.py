"""Shared data, timing, solver, metadata, and artifact helpers for QP runs."""

from __future__ import annotations

import csv
import importlib.metadata
import importlib.util
import json
import platform
import re
import signal
import socket
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from cufolio.exceptions import GPUBackendUnavailable
from cufolio.qp_backend import (
    CuOptQPBackend,
    QPSolution,
    max_constraint_violation,
    normalize_qp_status,
)


class BenchmarkTimeout(RuntimeError):
    """Raised when one benchmark backend exceeds its requested time limit."""


@dataclass
class TimedBackendResult:
    """A backend result with timing components and a stable failure record."""

    solution: QPSolution | None
    status: str
    raw_status: str | None
    solver_build_time_sec: float = 0.0
    solve_time_sec: float = 0.0
    solver_reported_solve_time_sec: float | None = None
    wall_clock_solve_time_sec: float = 0.0
    postprocess_time_sec: float = 0.0
    total_time_sec: float = 0.0
    error_message: str | None = None


RESULT_FIELDS = [
    "benchmark_id",
    "problem_id",
    "backend",
    "requested_backend",
    "objective",
    "case",
    "repeat",
    "warmup",
    "seed",
    "n_assets",
    "n_factors",
    "mapping_mode",
    "factor_model",
    "n_time_observations",
    "stock_mapping_shape",
    "factor_covariance_condition_number",
    "n_variables",
    "n_constraints",
    "n_nonzeros_Q",
    "n_nonzeros_A",
    "status",
    "raw_status",
    "solver_name",
    "compile_time_sec",
    "solver_build_time_sec",
    "solve_time_sec",
    "solver_reported_solve_time_sec",
    "wall_clock_solve_time_sec",
    "postprocess_time_sec",
    "total_time_sec",
    "objective_value",
    "solver_objective_value",
    "expected_return",
    "variance",
    "volatility",
    "sharpe",
    "gross_long",
    "gross_short",
    "turnover",
    "benchmark_l1_distance",
    "max_constraint_violation",
    "relative_objective_gap_vs_osqp",
    "weight_l2_distance_vs_osqp",
    "error_message",
    "timestamp",
    "git_commit",
    "git_branch",
    "hostname",
    "python_version",
    "platform",
    "numpy_version",
    "scipy_version",
    "pandas_version",
    "cvxpy_version",
    "cuopt_available",
    "cuopt_version",
    "cuda_version",
    "gpu_name",
    "backend_extra",
]


def utc_timestamp() -> str:
    """Return a filesystem-safe UTC timestamp with microsecond uniqueness."""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def create_run_dir(output_dir: str | Path, prefix: str | None = None) -> Path:
    """Create ``output_dir/<timestamp>`` for one benchmark invocation."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = utc_timestamp()
    if prefix:
        timestamp = f"{timestamp}_{prefix}"
    run_dir = root / timestamp
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


def _run_command(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _git_value(*args: str) -> str | None:
    return _run_command(["git", *args])


def _distribution_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _nvidia_metadata() -> tuple[str | None, str | None]:
    """Return the first visible GPU name and driver-reported CUDA version."""

    if _run_command(["nvidia-smi", "-L"]) is None:
        return None, None
    name_output = _run_command(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    )
    full_output = _run_command(["nvidia-smi"])
    gpu_name = name_output.splitlines()[0].strip() if name_output else None
    match = re.search(r"CUDA Version:\s*([0-9.]+)", full_output or "")
    return gpu_name, match.group(1) if match else None


def _installed_cuda_extra() -> str | None:
    if _distribution_version("cuopt-cu13") is not None:
        return "cuda13"
    if _distribution_version("cuopt-cu12") is not None:
        return "cuda12"
    if _distribution_version("cuopt-cu13-socp") is not None:
        return "cuda13-socp"
    return None


def collect_environment_metadata(backend: str) -> dict[str, Any]:
    """Collect reproducibility metadata without requiring a GPU runtime."""

    gpu_name, cuda_version = _nvidia_metadata()
    cuopt_version = _distribution_version(
        "cuopt-cu13",
        "cuopt-cu12",
        "cuopt-cu13-socp",
        "cuopt",
    )
    try:
        import cvxpy
        import numpy
        import pandas
        import scipy

        versions = {
            "numpy_version": numpy.__version__,
            "scipy_version": scipy.__version__,
            "pandas_version": pandas.__version__,
            "cvxpy_version": cvxpy.__version__,
        }
    except ImportError:
        versions = {
            "numpy_version": _distribution_version("numpy"),
            "scipy_version": _distribution_version("scipy"),
            "pandas_version": _distribution_version("pandas"),
            "cvxpy_version": _distribution_version("cvxpy"),
        }
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        **versions,
        "cuopt_available": importlib.util.find_spec("cuopt") is not None,
        "cuopt_version": cuopt_version,
        "cuda_version": cuda_version,
        "gpu_name": gpu_name,
        "backend": backend,
        "backend_extra": _installed_cuda_extra(),
    }


def make_psd_covariance(
    n_assets: int,
    seed: int,
    factor_rank: int | None = None,
    diagonal_floor: float = 0.02,
) -> np.ndarray:
    """Generate a deterministic positive-definite covariance matrix."""

    if n_assets < 1:
        raise ValueError("n_assets must be positive.")
    rank = min(n_assets, factor_rank or min(32, n_assets))
    rng = np.random.default_rng(seed)
    loadings = rng.normal(0.0, 0.04, size=(n_assets, rank))
    covariance = loadings @ loadings.T / max(rank, 1)
    covariance += diagonal_floor * np.eye(n_assets)
    return 0.5 * (covariance + covariance.T)


def make_mean_vector(n_assets: int, seed: int) -> np.ndarray:
    """Generate a deterministic positive-return mean vector."""

    rng = np.random.default_rng(seed)
    return 0.03 + rng.normal(0.0, 0.004, size=n_assets)


def make_anchor_weights(n_assets: int, seed: int) -> np.ndarray:
    """Generate positive, fully invested previous or benchmark weights."""

    rng = np.random.default_rng(seed)
    weights = 1.0 / n_assets + rng.normal(0.0, 0.001, size=n_assets)
    weights = np.maximum(weights, 1e-6)
    return weights / weights.sum()


def make_stock_returns(
    n_observations: int,
    n_assets: int,
    seed: int,
    n_latent_factors: int = 3,
) -> np.ndarray:
    """Generate deterministic synthetic stock returns for PCA workflows."""

    rng = np.random.default_rng(seed)
    factors = rng.normal(
        0.001,
        0.01,
        size=(n_observations, max(1, n_latent_factors)),
    )
    loadings = np.linspace(0.6, 1.4, n_assets).reshape(1, -1)
    if n_latent_factors > 1:
        extra = rng.normal(0.0, 0.15, size=(n_latent_factors - 1, n_assets))
        exposure = np.vstack([loadings, extra])
    else:
        exposure = loadings
    noise = rng.normal(0.0, 0.004, size=(n_observations, n_assets))
    return 0.002 + factors @ exposure + noise


def make_external_factor_inputs(
    n_observations: int,
    n_assets: int,
    n_factors: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate factor returns, a normalized mapping, and stock returns."""

    rng = np.random.default_rng(seed)
    factor_returns = rng.normal(0.01, 0.02, size=(n_observations, n_factors))
    raw_mapping = np.maximum(
        1.0 + rng.normal(0.0, 0.05, size=(n_assets, n_factors)),
        0.05,
    )
    mapping = raw_mapping / raw_mapping.sum(axis=0, keepdims=True)
    stock_returns = factor_returns @ mapping.T + rng.normal(
        0.0,
        0.002,
        size=(n_observations, n_assets),
    )
    return factor_returns, mapping, stock_returns


@contextmanager
def time_limit(seconds: float) -> Iterator[None]:
    """Apply a Unix wall-clock timeout to one benchmark operation."""

    if seconds <= 0 or not hasattr(signal, "setitimer"):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)

    def handle_timeout(_signum, _frame):
        raise BenchmarkTimeout(f"benchmark operation exceeded {seconds:g} seconds")

    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _build_osqp_problem(compiled):
    import cvxpy as cp

    x = cp.Variable(compiled.n_variables)
    objective = cp.Minimize(
        0.5 * cp.quad_form(x, cp.psd_wrap(compiled.Q.toarray())) + compiled.q @ x
    )
    constraints = []
    if compiled.A_eq.shape[0]:
        constraints.append(compiled.A_eq @ x == compiled.b_eq)
    if compiled.A_ineq.shape[0]:
        constraints.append(compiled.A_ineq @ x <= compiled.b_ineq)
    finite_lower = np.isfinite(compiled.lower)
    finite_upper = np.isfinite(compiled.upper)
    if finite_lower.any():
        constraints.append(x[finite_lower] >= compiled.lower[finite_lower])
    if finite_upper.any():
        constraints.append(x[finite_upper] <= compiled.upper[finite_upper])
    return cp.Problem(objective, constraints), x


def _timed_osqp(compiled) -> TimedBackendResult:
    build_start = time.perf_counter()
    problem, x = _build_osqp_problem(compiled)
    build_time = time.perf_counter() - build_start

    solve_start = time.perf_counter()
    split_names = {
        "pos",
        "neg",
        "turnover_pos",
        "turnover_neg",
        "benchmark_pos",
        "benchmark_neg",
    }
    max_iter = (
        2_000_000
        if split_names.intersection(compiled.variable_slices)
        else 100_000
    )
    problem.solve(
        solver="OSQP",
        eps_abs=1e-8,
        eps_rel=1e-8,
        max_iter=max_iter,
    )
    wall_solve = time.perf_counter() - solve_start
    raw_status = str(problem.status)
    if x.value is None:
        raise RuntimeError(f"OSQP failed to produce a solution. Status: {raw_status}")
    vector = np.asarray(x.value, dtype=float).reshape(-1)
    stats = getattr(problem, "solver_stats", None)
    reported = getattr(stats, "solve_time", None) if stats is not None else None
    post_start = time.perf_counter()
    solution = QPSolution(
        x=vector,
        status=normalize_qp_status(raw_status),
        solver_name="OSQP",
        objective_value=float(problem.value),
        solve_time=float(reported) if reported is not None else None,
        total_time=build_time + wall_solve,
        max_constraint_violation=max_constraint_violation(compiled, vector),
        variable_values_by_name=dict(zip(compiled.variable_names, vector)),
        raw_status=raw_status,
    )
    post_time = time.perf_counter() - post_start
    return TimedBackendResult(
        solution=solution,
        status=solution.status,
        raw_status=raw_status,
        solver_build_time_sec=build_time,
        solve_time_sec=wall_solve,
        solver_reported_solve_time_sec=solution.solve_time,
        wall_clock_solve_time_sec=wall_solve,
        postprocess_time_sec=post_time,
        total_time_sec=build_time + wall_solve + post_time,
    )


def _timed_cuopt(compiled) -> TimedBackendResult:
    if importlib.util.find_spec("cuopt") is None:
        raise GPUBackendUnavailable(
            "cuOpt GPU runtime unavailable; benchmark did not substitute OSQP."
        )

    from cuopt.linear_programming.problem import (
        CONTINUOUS,
        MINIMIZE,
        LinearExpression,
        Problem,
        QuadraticExpression,
    )
    from cuopt.linear_programming.solver_settings import SolverSettings

    build_start = time.perf_counter()
    problem = Problem("PortOpt Unified QP Benchmark")
    backend = CuOptQPBackend()
    variables = backend._add_variables(problem, compiled, CONTINUOUS)
    backend._add_linear_constraints(problem, compiled, variables, LinearExpression)
    objective_expr = backend._build_objective(
        compiled,
        variables,
        LinearExpression,
        QuadraticExpression,
    )
    problem.setObjective(objective_expr, sense=MINIMIZE)
    settings = SolverSettings()
    build_time = time.perf_counter() - build_start

    solve_start = time.perf_counter()
    problem.solve(settings)
    wall_solve = time.perf_counter() - solve_start
    raw_status = getattr(problem.Status, "name", str(problem.Status))
    status = normalize_qp_status(raw_status)
    if not backend._is_accepted_status(raw_status):
        raise RuntimeError(f"cuOpt failed to solve QP. Status: {raw_status}")

    post_start = time.perf_counter()
    vector = np.asarray([var.getValue() for var in variables], dtype=float)
    reported_value = getattr(problem, "SolveTime", None)
    reported = float(reported_value) if reported_value is not None else None
    solution = QPSolution(
        x=vector,
        status=status,
        solver_name="cuopt_qp",
        objective_value=float(problem.ObjValue),
        solve_time=reported,
        total_time=build_time + wall_solve,
        max_constraint_violation=max_constraint_violation(compiled, vector),
        variable_values_by_name=dict(zip(compiled.variable_names, vector)),
        raw_status=raw_status,
    )
    post_time = time.perf_counter() - post_start
    return TimedBackendResult(
        solution=solution,
        status=status,
        raw_status=raw_status,
        solver_build_time_sec=build_time,
        solve_time_sec=wall_solve,
        solver_reported_solve_time_sec=reported,
        wall_clock_solve_time_sec=wall_solve,
        postprocess_time_sec=post_time,
        total_time_sec=build_time + wall_solve + post_time,
    )


def solve_compiled_qp_timed(
    compiled,
    backend: str,
    timeout_sec: float,
) -> TimedBackendResult:
    """Solve one compiled QP with explicit timing and failure capture."""

    try:
        with time_limit(timeout_sec):
            if backend == "osqp":
                return _timed_osqp(compiled)
            if backend == "cuopt":
                return _timed_cuopt(compiled)
            raise ValueError(f"Unsupported benchmark backend: {backend}")
    except (GPUBackendUnavailable, ImportError) as exc:
        return TimedBackendResult(
            solution=None,
            status="skipped" if backend == "cuopt" else "error",
            raw_status=None,
            error_message=str(exc),
        )
    except Exception as exc:
        return TimedBackendResult(
            solution=None,
            status="timeout" if isinstance(exc, BenchmarkTimeout) else "error",
            raw_status=None,
            error_message=f"{type(exc).__name__}: {exc}",
        )


def factor_covariance_condition_number(compiled) -> float | None:
    if compiled.mapping_mode != "factor_space":
        return None
    try:
        return float(np.linalg.cond(np.asarray(compiled.covariance)))
    except np.linalg.LinAlgError:
        return float("inf")


def _metric_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _json_safe_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _metric_value(value) for key, value in row.items()}


def build_result_row(
    *,
    metadata: dict[str, Any],
    benchmark_id: str,
    problem_id: str,
    requested_backend: str,
    backend: str,
    objective: str,
    case: str,
    repeat: int,
    warmup: int,
    seed: int,
    params,
    compiled,
    compile_time_sec: float,
    timed: TimedBackendResult,
    baseline: TimedBackendResult | None = None,
    factor_model: str | None = None,
    n_time_observations: int | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one flat result row for CSV, JSONL, and summary consumers."""

    row: dict[str, Any] = {
        **metadata,
        **(extra_fields or {}),
        "benchmark_id": benchmark_id,
        "problem_id": problem_id,
        "backend": backend,
        "requested_backend": requested_backend,
        "objective": objective,
        "case": case,
        "repeat": repeat,
        "warmup": warmup,
        "seed": seed,
        "factor_model": factor_model,
        "n_time_observations": n_time_observations,
        "status": timed.status,
        "raw_status": timed.raw_status,
        "solver_name": None,
        "compile_time_sec": compile_time_sec,
        "solver_build_time_sec": timed.solver_build_time_sec,
        "solve_time_sec": timed.solve_time_sec,
        "solver_reported_solve_time_sec": timed.solver_reported_solve_time_sec,
        "wall_clock_solve_time_sec": timed.wall_clock_solve_time_sec,
        "postprocess_time_sec": timed.postprocess_time_sec,
        "total_time_sec": compile_time_sec + timed.total_time_sec,
        "error_message": timed.error_message,
        "relative_objective_gap_vs_osqp": None,
        "weight_l2_distance_vs_osqp": None,
    }
    if compiled is None:
        row.update(
            {
                "n_assets": getattr(params, "_benchmark_n_assets", None),
                "n_factors": getattr(params, "_benchmark_n_factors", None),
                "mapping_mode": getattr(params, "mapping_mode", None),
            }
        )
        return _json_safe_row(row)

    row.update(
        {
            "n_assets": int(compiled.stock_mapping.shape[0]),
            "n_factors": int(compiled.stock_mapping.shape[1])
            if compiled.mapping_mode == "factor_space"
            else 0,
            "mapping_mode": compiled.mapping_mode,
            "stock_mapping_shape": list(compiled.stock_mapping.shape),
            "factor_covariance_condition_number": factor_covariance_condition_number(
                compiled
            ),
            "n_variables": compiled.n_variables,
            "n_constraints": compiled.n_constraints,
            "n_nonzeros_Q": int(compiled.Q.nnz),
            "n_nonzeros_A": int(compiled.A.nnz),
        }
    )
    solution = timed.solution
    if solution is None:
        return _json_safe_row(row)

    weights = compiled.recover_stock_weights(solution.x)
    factor_weights = compiled.recover_factor_weights(solution.x)
    if compiled.mapping_mode == "factor_space":
        expected_return = float(compiled.mean @ factor_weights)
        variance = float(factor_weights @ compiled.covariance @ factor_weights)
    else:
        expected_return = float(compiled.mean @ weights)
        variance = float(weights @ compiled.covariance @ weights)
    volatility = float(np.sqrt(max(variance, 0.0)))
    excess_return = expected_return - params.risk_free_rate
    sharpe = excess_return / volatility if volatility > 0 else np.nan
    previous = getattr(params, "previous_weights", None)
    benchmark = getattr(params, "benchmark_weights", None)
    turnover = (
        float(np.abs(weights - np.asarray(previous, dtype=float)).sum())
        if previous is not None
        else None
    )
    benchmark_distance = (
        float(np.abs(weights - np.asarray(benchmark, dtype=float)).sum())
        if benchmark is not None
        else None
    )
    compiled_objective = compiled.objective_value(solution.x)
    row.update(
        {
            "solver_name": solution.solver_name,
            "objective_value": compiled_objective,
            "solver_objective_value": solution.objective_value,
            "expected_return": expected_return,
            "variance": variance,
            "volatility": volatility,
            "sharpe": sharpe,
            "gross_long": float(np.maximum(weights, 0.0).sum()),
            "gross_short": float(np.maximum(-weights, 0.0).sum()),
            "turnover": turnover,
            "benchmark_l1_distance": benchmark_distance,
            "max_constraint_violation": solution.max_constraint_violation,
        }
    )
    if baseline is not None and baseline.solution is not None:
        baseline_objective = compiled.objective_value(baseline.solution.x)
        row["relative_objective_gap_vs_osqp"] = abs(
            compiled_objective - baseline_objective
        ) / max(1.0, abs(baseline_objective))
        baseline_weights = compiled.recover_stock_weights(baseline.solution.x)
        row["weight_l2_distance_vs_osqp"] = float(
            np.linalg.norm(weights - baseline_weights)
        )
    return _json_safe_row(row)


def run_problem_repeats(
    *,
    returns_dict: dict[str, Any],
    params,
    metadata: dict[str, Any],
    benchmark_id: str,
    problem_id: str,
    requested_backend: str,
    objective: str,
    case: str,
    repeat_count: int,
    warmup_count: int,
    seed: int,
    timeout_sec: float,
    factor_model: str | None = None,
    n_time_observations: int | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Compile and solve one deterministic problem for requested backends."""

    if requested_backend == "both":
        backend_names = ["osqp", "cuopt"]
    else:
        backend_names = [requested_backend]
    rows: list[dict[str, Any]] = []

    def compile_once():
        from cufolio.qp_formulations import compile_portfolio_qp

        start = time.perf_counter()
        with time_limit(timeout_sec):
            compiled = compile_portfolio_qp(returns_dict, params)
        return compiled, time.perf_counter() - start

    for _ in range(max(0, warmup_count)):
        try:
            compiled, _ = compile_once()
        except Exception:
            continue
        for backend in backend_names:
            solve_compiled_qp_timed(compiled, backend, timeout_sec)

    for repeat in range(max(1, repeat_count)):
        try:
            compiled, compile_time = compile_once()
            compile_error = None
        except Exception as exc:
            compiled = None
            compile_time = 0.0
            compile_error = f"{type(exc).__name__}: {exc}"
        results: dict[str, TimedBackendResult] = {}
        if compiled is not None:
            for backend in backend_names:
                results[backend] = solve_compiled_qp_timed(
                    compiled,
                    backend,
                    timeout_sec,
                )
        else:
            for backend in backend_names:
                results[backend] = TimedBackendResult(
                    solution=None,
                    status="error",
                    raw_status=None,
                    error_message=compile_error,
                )
        baseline = results.get("osqp") if requested_backend == "both" else None
        for backend in backend_names:
            rows.append(
                build_result_row(
                    metadata=metadata,
                    benchmark_id=benchmark_id,
                    problem_id=problem_id,
                    requested_backend=requested_backend,
                    backend=backend,
                    objective=objective,
                    case=case,
                    repeat=repeat,
                    warmup=0,
                    seed=seed,
                    params=params,
                    compiled=compiled,
                    compile_time_sec=compile_time,
                    timed=results[backend],
                    baseline=baseline if backend == "cuopt" else None,
                    factor_model=factor_model,
                    n_time_observations=n_time_observations,
                    extra_fields=extra_fields,
                )
            )
    return rows


def run_problem_once(
    *,
    returns_dict: dict[str, Any],
    params,
    metadata: dict[str, Any],
    benchmark_id: str,
    problem_id: str,
    requested_backend: str,
    objective: str,
    case: str,
    repeat: int,
    seed: int,
    timeout_sec: float,
    factor_model: str | None = None,
    n_time_observations: int | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, TimedBackendResult], Any]:
    """Run one problem and return rows, timed results, and the compiled QP."""

    backend_names = ["osqp", "cuopt"] if requested_backend == "both" else [
        requested_backend
    ]
    try:
        from cufolio.qp_formulations import compile_portfolio_qp

        compile_start = time.perf_counter()
        with time_limit(timeout_sec):
            compiled = compile_portfolio_qp(returns_dict, params)
        compile_time = time.perf_counter() - compile_start
        compile_error = None
    except Exception as exc:
        compiled = None
        compile_time = 0.0
        compile_error = f"{type(exc).__name__}: {exc}"

    results: dict[str, TimedBackendResult] = {}
    if compiled is None:
        for backend in backend_names:
            results[backend] = TimedBackendResult(
                solution=None,
                status="error",
                raw_status=None,
                error_message=compile_error,
            )
    else:
        for backend in backend_names:
            results[backend] = solve_compiled_qp_timed(
                compiled,
                backend,
                timeout_sec,
            )

    baseline = results.get("osqp") if requested_backend == "both" else None
    rows = [
        build_result_row(
            metadata=metadata,
            benchmark_id=benchmark_id,
            problem_id=problem_id,
            requested_backend=requested_backend,
            backend=backend,
            objective=objective,
            case=case,
            repeat=repeat,
            warmup=0,
            seed=seed,
            params=params,
            compiled=compiled,
            compile_time_sec=compile_time,
            timed=results[backend],
            baseline=baseline if backend == "cuopt" else None,
            factor_model=factor_model,
            n_time_observations=n_time_observations,
            extra_fields=extra_fields,
        )
        for backend in backend_names
    ]
    return rows, results, compiled


def _csv_value(value: Any) -> str:
    value = _metric_value(value)
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def write_result_artifacts(
    run_dir: str | Path,
    stem: str,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    title: str,
) -> dict[str, Path]:
    """Write metadata, CSV, JSONL, and a compact Markdown run summary."""

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    metadata_path = run_path / "environment.json"
    metadata_path.write_text(
        json.dumps(_json_safe_row(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    jsonl_path = run_path / f"{stem}_results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe_row(row), sort_keys=True) + "\n")

    fields = list(RESULT_FIELDS)
    fields.extend(sorted({key for row in rows for key in row if key not in fields}))
    csv_path = run_path / f"{stem}_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})

    statuses: dict[str, int] = {}
    for row in rows:
        statuses[row.get("status", "unknown")] = statuses.get(
            row.get("status", "unknown"), 0
        ) + 1
    successful = [
        float(row["total_time_sec"])
        for row in rows
        if row.get("status") == "optimal" and row.get("total_time_sec") is not None
    ]
    lines = [f"# {title}", "", f"Rows: {len(rows)}", "", "## Status", ""]
    lines.append("| status | count |")
    lines.append("| --- | ---: |")
    for status, count in sorted(statuses.items()):
        lines.append(f"| {status} | {count} |")
    lines.extend(["", "## Timing", ""])
    lines.append("| metric | value |")
    lines.append("| --- | ---: |")
    if successful:
        lines.append(f"| median_total_time_sec | {float(np.median(successful)):.8g} |")
        lines.append(f"| min_total_time_sec | {min(successful):.8g} |")
        lines.append(f"| max_total_time_sec | {max(successful):.8g} |")
    else:
        lines.append("| median_total_time_sec | n/a |")
    lines.extend(
        [
            "",
            "Observed timing is scoped to this generated artifact. It is not a",
            "universal speedup claim.",
            "",
        ]
    )
    markdown_path = run_path / f"{stem}_summary.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "run_dir": run_path,
        "metadata": metadata_path,
        "csv": csv_path,
        "jsonl": jsonl_path,
        "markdown": markdown_path,
    }


def print_artifact_paths(paths: dict[str, Path]) -> None:
    """Print machine-readable artifact paths for shell jobs and smoke tests."""

    for key, path in paths.items():
        print(f"{key}: {path}")
