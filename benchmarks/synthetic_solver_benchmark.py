# SPDX-License-Identifier: Apache-2.0
"""Registered sparse synthetic LP/QP benchmark for native OSQP and cuOpt."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import yaml
from scipy import sparse

FAMILY_ORDER = (
    "LP-RANGED",
    "LP-MIXED",
    "QP-DIAGONAL",
    "QP-SPARSE-COUPLED",
)
FAMILY_SEED_OFFSETS = {
    "LP-RANGED": 10_000,
    "LP-MIXED": 20_000,
    "QP-DIAGONAL": 30_000,
    "QP-SPARSE-COUPLED": 40_000,
}


class BenchmarkTimeout(RuntimeError):
    """Raised when one registered backend operation exceeds its time limit."""


@dataclass(frozen=True)
class CanonicalProblem:
    """Sparse row-bounded canonical problem resident in host memory."""

    family: str
    seed: int
    P: sparse.csr_matrix
    q: np.ndarray
    A: sparse.csr_matrix
    row_lower: np.ndarray
    row_upper: np.ndarray
    variable_lower: np.ndarray
    variable_upper: np.ndarray
    anchor: np.ndarray
    canonical_sha256: str

    @property
    def n_variables(self) -> int:
        return int(self.q.size)

    @property
    def n_constraints(self) -> int:
        return int(self.A.shape[0])


@dataclass
class BackendMeasurement:
    """One backend timing and canonical correctness reconstruction."""

    backend: str
    status: str
    raw_status: str | None
    strict_optimal: bool
    correctness_pass: bool
    censored: bool
    setup_seconds: float | None
    solve_seconds: float | None
    reported_solve_seconds: float | None
    result_extraction_seconds: float | None
    correctness_seconds: float | None
    end_to_end_seconds: float | None
    objective_value: float | None
    reported_objective_value: float | None
    objective_reconstruction_error: float | None
    original_primal_violation: float | None
    peak_host_memory_bytes: int | None
    peak_gpu_memory_bytes: int | None
    error_type: str | None = None
    error_message: str | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _hash_array(digest: Any, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(str(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(array.view(np.uint8))


def _canonical_hash(
    family: str,
    seed: int,
    P: sparse.csr_matrix,
    q: np.ndarray,
    A: sparse.csr_matrix,
    row_lower: np.ndarray,
    row_upper: np.ndarray,
    variable_lower: np.ndarray,
    variable_upper: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(f"{family}|{seed}".encode("ascii"))
    for matrix in (P, A):
        matrix = matrix.tocsr()
        _hash_array(digest, matrix.data.astype(np.float64, copy=False))
        _hash_array(digest, matrix.indices.astype(np.int64, copy=False))
        _hash_array(digest, matrix.indptr.astype(np.int64, copy=False))
        digest.update(str(matrix.shape).encode("ascii"))
    for vector in (q, row_lower, row_upper, variable_lower, variable_upper):
        _hash_array(digest, vector.astype(np.float64, copy=False))
    return digest.hexdigest()


def _sparse_rows(
    rng: np.random.Generator,
    n_rows: int,
    n_variables: int,
    row_nonzeros: int,
) -> sparse.csr_matrix:
    width = min(row_nonzeros, n_variables)
    rows = np.repeat(np.arange(n_rows, dtype=np.int64), width)
    columns = np.empty(n_rows * width, dtype=np.int64)
    for row in range(n_rows):
        start = row * width
        columns[start : start + width] = rng.choice(
            n_variables, size=width, replace=False
        )
    data = rng.normal(0.0, 1.0 / np.sqrt(width), size=n_rows * width)
    return sparse.coo_matrix(
        (data, (rows, columns)), shape=(n_rows, n_variables)
    ).tocsr()


def _sparse_coupled_hessian(
    rng: np.random.Generator, n_variables: int
) -> sparse.csr_matrix:
    edge_count = 2 * n_variables
    left = rng.integers(0, n_variables, size=edge_count, dtype=np.int64)
    offsets = rng.integers(1, n_variables, size=edge_count, dtype=np.int64)
    right = (left + offsets) % n_variables
    values = rng.uniform(-0.05, 0.05, size=edge_count)
    off_diagonal = sparse.coo_matrix(
        (
            np.concatenate([values, values]),
            (np.concatenate([left, right]), np.concatenate([right, left])),
        ),
        shape=(n_variables, n_variables),
    ).tocsr()
    off_diagonal.setdiag(0.0)
    off_diagonal.eliminate_zeros()
    diagonal = 0.2 + np.asarray(np.abs(off_diagonal).sum(axis=1)).reshape(-1)
    diagonal += rng.uniform(0.05, 0.25, size=n_variables)
    return (off_diagonal + sparse.diags(diagonal, format="csr")).tocsr()


def generate_canonical_problem(
    config: dict[str, Any], family: str, n_variables: int, seed: int
) -> CanonicalProblem:
    """Generate one deterministic feasible and bounded sparse problem."""

    canonical = config["canonical_problem"]
    family_config = canonical["families"][family]
    rng = np.random.default_rng(
        FAMILY_SEED_OFFSETS[family] + int(seed) * 1_000_003 + int(n_variables)
    )
    n_constraints = max(1, int(round(family_config["constraint_ratio"] * n_variables)))
    A = _sparse_rows(
        rng,
        n_constraints,
        n_variables,
        int(canonical["row_nonzeros"]),
    )
    kind = family_config["kind"]
    mode = family_config["constraint_mode"]
    if kind == "lp" and mode == "ranged":
        # Make the known box optimum feasible so this family isolates ranged-row
        # build and solve costs without a pathological random LP active set.
        q = rng.normal(0.0, 1.0 / np.sqrt(n_variables), size=n_variables)
        anchor = -np.sign(q)
    else:
        anchor = rng.uniform(-0.15, 0.15, size=n_variables)
    center = np.asarray(A @ anchor).reshape(-1)
    slack = rng.uniform(0.3, 0.8, size=n_constraints)
    if not (kind == "lp" and mode == "ranged"):
        q = rng.normal(0.0, 1.0 / np.sqrt(n_variables), size=n_variables)
    if mode == "ranged":
        row_lower = center - slack
        row_upper = center + slack
    elif mode == "mixed":
        row_lower = np.full(n_constraints, -np.inf, dtype=np.float64)
        row_upper = np.full(n_constraints, np.inf, dtype=np.float64)
        equality_count = max(1, n_constraints // 10)
        remaining = n_constraints - equality_count
        upper_count = remaining // 2
        row_lower[:equality_count] = center[:equality_count]
        row_upper[:equality_count] = center[:equality_count]
        upper_stop = equality_count + upper_count
        row_upper[equality_count:upper_stop] = (
            center[equality_count:upper_stop] + slack[equality_count:upper_stop]
        )
        row_lower[upper_stop:] = center[upper_stop:] - slack[upper_stop:]
        if kind == "lp":
            weights = np.linspace(0.5, 1.5, equality_count)
            q = np.asarray(A[:equality_count].T @ weights).reshape(-1)
            q /= max(float(np.linalg.norm(q)), np.finfo(np.float64).eps)
    else:
        raise ValueError(f"Unknown constraint mode: {mode}")

    if kind == "lp":
        P = sparse.csr_matrix((n_variables, n_variables), dtype=np.float64)
    elif kind == "qp_diagonal":
        diagonal = rng.uniform(0.2, 1.2, size=n_variables)
        P = sparse.diags(diagonal, format="csr")
    elif kind == "qp_sparse_coupled":
        P = _sparse_coupled_hessian(rng, n_variables)
    else:
        raise ValueError(f"Unknown problem kind: {kind}")

    variable_lower = np.full(
        n_variables, float(canonical["variable_lower_bound"]), dtype=np.float64
    )
    variable_upper = np.full(
        n_variables, float(canonical["variable_upper_bound"]), dtype=np.float64
    )
    digest = _canonical_hash(
        family,
        seed,
        P,
        q,
        A,
        row_lower,
        row_upper,
        variable_lower,
        variable_upper,
    )
    return CanonicalProblem(
        family=family,
        seed=int(seed),
        P=P,
        q=q,
        A=A,
        row_lower=row_lower,
        row_upper=row_upper,
        variable_lower=variable_lower,
        variable_upper=variable_upper,
        anchor=anchor,
        canonical_sha256=digest,
    )


def canonical_objective(problem: CanonicalProblem, x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return float(0.5 * x @ (problem.P @ x) + problem.q @ x)


def original_primal_violation(
    problem: CanonicalProblem, x: np.ndarray
) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    activity = np.asarray(problem.A @ x).reshape(-1)
    violations = [0.0]
    finite = np.isfinite(problem.row_lower)
    if finite.any():
        violations.append(
            float(np.max(np.maximum(problem.row_lower[finite] - activity[finite], 0.0)))
        )
    finite = np.isfinite(problem.row_upper)
    if finite.any():
        violations.append(
            float(np.max(np.maximum(activity[finite] - problem.row_upper[finite], 0.0)))
        )
    violations.append(
        float(np.max(np.maximum(problem.variable_lower - x, 0.0)))
    )
    violations.append(
        float(np.max(np.maximum(x - problem.variable_upper, 0.0)))
    )
    return max(violations)


def anchor_violation(problem: CanonicalProblem) -> float:
    """Expose generator feasibility as a unit-testable invariant."""

    return original_primal_violation(problem, problem.anchor)


def _read_rss_bytes() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


class MemorySampler:
    """Poll current-process host and GPU memory during one backend call."""

    def __init__(self, interval_seconds: float = 0.01, track_gpu: bool = False):
        self.interval_seconds = interval_seconds
        self.track_gpu = track_gpu
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_host: int | None = None
        self.peak_gpu: int | None = None
        self._pynvml: Any = None
        self._handles: list[Any] = []
        self._cuda_runtime: Any = None

    def _initialize_gpu(self) -> None:
        if not self.track_gpu:
            return
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handles = [
                pynvml.nvmlDeviceGetHandleByIndex(index)
                for index in range(pynvml.nvmlDeviceGetCount())
            ]
        except Exception:
            self._pynvml = None
            self._handles = []
        if self._pynvml is None:
            try:
                from cuda.bindings import runtime

                error, _free, _total = runtime.cudaMemGetInfo()
                if int(error) == 0:
                    self._cuda_runtime = runtime
            except Exception:
                self._cuda_runtime = None

    def _gpu_memory(self) -> int | None:
        if self._pynvml is not None and self._handles:
            try:
                total = 0
                found = False
                for handle in self._handles:
                    processes = list(
                        self._pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    )
                    graphics = getattr(
                        self._pynvml, "nvmlDeviceGetGraphicsRunningProcesses", None
                    )
                    if graphics is not None:
                        processes.extend(graphics(handle))
                    for process in processes:
                        if (
                            int(process.pid) == os.getpid()
                            and process.usedGpuMemory is not None
                            and int(process.usedGpuMemory) >= 0
                        ):
                            total += int(process.usedGpuMemory)
                            found = True
                return total if found else 0
            except Exception:
                return None
        if self._cuda_runtime is not None:
            try:
                error, free, total = self._cuda_runtime.cudaMemGetInfo()
                if int(error) == 0:
                    return int(total - free)
            except Exception:
                return None
        return None

    def _sample(self) -> None:
        host = _read_rss_bytes()
        gpu = self._gpu_memory()
        if host is not None:
            self.peak_host = host if self.peak_host is None else max(self.peak_host, host)
        if gpu is not None:
            self.peak_gpu = gpu if self.peak_gpu is None else max(self.peak_gpu, gpu)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def __enter__(self) -> MemorySampler:
        self._initialize_gpu()
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._sample()


@contextmanager
def time_limit(seconds: float) -> Iterator[None]:
    if seconds <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGALRM)

    def _timeout(_signum, _frame):
        raise BenchmarkTimeout(f"operation exceeded {seconds:g} seconds")

    signal.signal(signal.SIGALRM, _timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _correctness(
    problem: CanonicalProblem,
    x: np.ndarray,
    reported_objective: float,
    config: dict[str, Any],
) -> tuple[float, float, float, bool, float]:
    start = time.perf_counter()
    objective = canonical_objective(problem, x)
    reconstruction = abs(objective - float(reported_objective))
    violation = original_primal_violation(problem, x)
    thresholds = config["correctness"]
    passed = (
        violation <= float(thresholds["maximum_original_primal_violation"])
        and reconstruction
        <= float(thresholds["maximum_objective_reconstruction_error"])
        * max(1.0, abs(objective))
    )
    return objective, reconstruction, violation, passed, time.perf_counter() - start


def solve_osqp(
    problem: CanonicalProblem, config: dict[str, Any]
) -> BackendMeasurement:
    timeout = float(config["canonical_problem"]["timeout_seconds"])
    sampler = MemorySampler(track_gpu=False)
    start = time.perf_counter()
    try:
        with sampler:
            start = time.perf_counter()
            with time_limit(timeout):
                return _solve_osqp_timed(problem, config, sampler, start)
    except Exception as exc:
        return BackendMeasurement(
            backend="OSQP",
            status="timeout" if isinstance(exc, BenchmarkTimeout) else "failed",
            raw_status=None,
            strict_optimal=False,
            correctness_pass=False,
            censored=isinstance(exc, BenchmarkTimeout),
            setup_seconds=None,
            solve_seconds=None,
            reported_solve_seconds=None,
            result_extraction_seconds=None,
            correctness_seconds=None,
            end_to_end_seconds=time.perf_counter() - start,
            objective_value=None,
            reported_objective_value=None,
            objective_reconstruction_error=None,
            original_primal_violation=None,
            peak_host_memory_bytes=sampler.peak_host,
            peak_gpu_memory_bytes=sampler.peak_gpu,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


def _solve_osqp_timed(
    problem: CanonicalProblem,
    config: dict[str, Any],
    sampler: MemorySampler,
    start: float,
) -> BackendMeasurement:
    import osqp

    settings = config["solver_contract"]["osqp"]
    timeout = float(config["canonical_problem"]["timeout_seconds"])
    setup_start = time.perf_counter()
    identity = sparse.eye(problem.n_variables, format="csc")
    backend_A = sparse.vstack([problem.A, identity], format="csc")
    lower = np.concatenate([problem.row_lower, problem.variable_lower])
    upper = np.concatenate([problem.row_upper, problem.variable_upper])
    solver = osqp.OSQP()
    solver.setup(
        P=sparse.triu(problem.P, format="csc"),
        q=np.asarray(problem.q, dtype=np.float64),
        A=backend_A,
        l=lower,
        u=upper,
        verbose=False,
        eps_abs=float(settings["eps_abs"]),
        eps_rel=float(settings["eps_rel"]),
        max_iter=int(settings["max_iter"]),
        polishing=bool(settings["polishing"]),
        adaptive_rho=bool(settings["adaptive_rho"]),
    )
    setup_seconds = time.perf_counter() - setup_start
    solve_start = time.perf_counter()
    result = solver.solve(raise_error=False)
    solve_seconds = time.perf_counter() - solve_start
    extraction_start = time.perf_counter()
    raw_status = str(result.info.status)
    if result.x is None:
        raise RuntimeError(f"OSQP returned no primal vector: {raw_status}")
    x = np.asarray(result.x, dtype=np.float64).reshape(-1)
    reported_objective = float(result.info.obj_val)
    extraction_seconds = time.perf_counter() - extraction_start
    objective, error, violation, passed, correctness_seconds = _correctness(
        problem, x, reported_objective, config
    )
    strict = raw_status == str(config["correctness"]["strict_osqp_raw_status"])
    elapsed = time.perf_counter() - start
    censored = elapsed > timeout
    return BackendMeasurement(
        backend="OSQP",
        status="timeout" if censored else ("optimal" if strict else raw_status),
        raw_status=raw_status,
        strict_optimal=strict and not censored,
        correctness_pass=strict and passed and not censored,
        censored=censored,
        setup_seconds=setup_seconds,
        solve_seconds=solve_seconds,
        reported_solve_seconds=float(result.info.solve_time),
        result_extraction_seconds=extraction_seconds,
        correctness_seconds=correctness_seconds,
        end_to_end_seconds=elapsed,
        objective_value=objective,
        reported_objective_value=reported_objective,
        objective_reconstruction_error=error,
        original_primal_violation=violation,
        peak_host_memory_bytes=sampler.peak_host,
        peak_gpu_memory_bytes=sampler.peak_gpu,
    )


def _build_cuopt_model(problem: CanonicalProblem, config: dict[str, Any]):
    from cuopt.linear_programming.problem import (
        CONTINUOUS,
        MINIMIZE,
        LinearExpression,
        Problem,
        QuadraticExpression,
    )
    from cuopt.linear_programming.solver_settings import SolverSettings

    model = Problem("Synthetic OSQP versus cuOpt benchmark")
    variables = [
        model.addVariable(
            lb=float(problem.variable_lower[index]),
            ub=float(problem.variable_upper[index]),
            vtype=CONTINUOUS,
            name=f"x_{index}",
        )
        for index in range(problem.n_variables)
    ]
    matrix = problem.A.tocsr()
    for row_index in range(problem.n_constraints):
        row = matrix.getrow(row_index).tocoo()
        expression = LinearExpression(
            [variables[int(index)] for index in row.col],
            [float(value) for value in row.data],
            0.0,
        )
        lower = float(problem.row_lower[row_index])
        upper = float(problem.row_upper[row_index])
        if np.isfinite(lower) and np.isfinite(upper) and lower == upper:
            model.addConstraint(expression == upper, name=f"row_{row_index}_eq")
        else:
            if np.isfinite(lower):
                model.addConstraint(expression >= lower, name=f"row_{row_index}_lo")
            if np.isfinite(upper):
                model.addConstraint(expression <= upper, name=f"row_{row_index}_hi")

    linear = LinearExpression(variables, problem.q.tolist(), 0.0)
    quadratic = None
    if problem.P.nnz:
        half = (0.5 * problem.P).tocoo()
        quadratic = QuadraticExpression(
            qvars1=[variables[int(row)] for row in half.row],
            qvars2=[variables[int(column)] for column in half.col],
            qcoefficients=[float(value) for value in half.data],
        )
        model.setObjective(quadratic + linear, sense=MINIMIZE)
    else:
        model.setObjective(linear, sense=MINIMIZE)
    settings = SolverSettings()
    cuopt_config = config["solver_contract"]["cuopt"]
    for parameter, value in cuopt_config.items():
        if parameter not in {"interface", "family_overrides"}:
            settings.set_parameter(parameter, value)
    for parameter, value in cuopt_config.get("family_overrides", {}).get(
        problem.family, {}
    ).items():
        settings.set_parameter(parameter, value)
    return model, variables, quadratic, settings, LinearExpression, MINIMIZE


def solve_cuopt(
    problem: CanonicalProblem, config: dict[str, Any]
) -> BackendMeasurement:
    timeout = float(config["canonical_problem"]["timeout_seconds"])
    sampler = MemorySampler(track_gpu=True)
    start = time.perf_counter()
    try:
        with sampler:
            start = time.perf_counter()
            with time_limit(timeout):
                return _solve_cuopt_timed(problem, config, sampler, start)
    except Exception as exc:
        return BackendMeasurement(
            backend="cuOpt",
            status="timeout" if isinstance(exc, BenchmarkTimeout) else "failed",
            raw_status=None,
            strict_optimal=False,
            correctness_pass=False,
            censored=isinstance(exc, BenchmarkTimeout),
            setup_seconds=None,
            solve_seconds=None,
            reported_solve_seconds=None,
            result_extraction_seconds=None,
            correctness_seconds=None,
            end_to_end_seconds=time.perf_counter() - start,
            objective_value=None,
            reported_objective_value=None,
            objective_reconstruction_error=None,
            original_primal_violation=None,
            peak_host_memory_bytes=sampler.peak_host,
            peak_gpu_memory_bytes=sampler.peak_gpu,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


def _solve_cuopt_timed(
    problem: CanonicalProblem,
    config: dict[str, Any],
    sampler: MemorySampler,
    start: float,
) -> BackendMeasurement:
    timeout = float(config["canonical_problem"]["timeout_seconds"])
    setup_start = time.perf_counter()
    model, variables, _quadratic, settings, _linear, _sense = _build_cuopt_model(
        problem, config
    )
    setup_seconds = time.perf_counter() - setup_start
    solve_start = time.perf_counter()
    model.solve(settings)
    solve_seconds = time.perf_counter() - solve_start
    extraction_start = time.perf_counter()
    raw_status = getattr(model.Status, "name", str(model.Status))
    x = np.asarray([variable.getValue() for variable in variables], dtype=float)
    reported_objective = float(model.ObjValue)
    extraction_seconds = time.perf_counter() - extraction_start
    objective, error, violation, passed, correctness_seconds = _correctness(
        problem, x, reported_objective, config
    )
    strict = raw_status == str(config["correctness"]["strict_cuopt_raw_status"])
    elapsed = time.perf_counter() - start
    censored = elapsed > timeout
    return BackendMeasurement(
        backend="cuOpt",
        status="timeout" if censored else ("optimal" if strict else raw_status),
        raw_status=raw_status,
        strict_optimal=strict and not censored,
        correctness_pass=strict and passed and not censored,
        censored=censored,
        setup_seconds=setup_seconds,
        solve_seconds=solve_seconds,
        reported_solve_seconds=float(getattr(model, "SolveTime", np.nan)),
        result_extraction_seconds=extraction_seconds,
        correctness_seconds=correctness_seconds,
        end_to_end_seconds=elapsed,
        objective_value=objective,
        reported_objective_value=reported_objective,
        objective_reconstruction_error=error,
        original_primal_violation=violation,
        peak_host_memory_bytes=sampler.peak_host,
        peak_gpu_memory_bytes=sampler.peak_gpu,
    )


def _measurement_row(
    problem: CanonicalProblem,
    measurement: BackendMeasurement,
    repetition: int,
    registered: bool,
) -> dict[str, Any]:
    return {
        "problem_family": problem.family,
        "n_variables": problem.n_variables,
        "n_constraints": problem.n_constraints,
        "seed": problem.seed,
        "repetition": repetition,
        "registered_repetition": registered,
        "backend": measurement.backend,
        "canonical_sha256": problem.canonical_sha256,
        **measurement.__dict__,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "host": socket.gethostname(),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "execution_sha": os.environ.get("EXPECTED_SHA"),
    }


def _package_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def collect_environment() -> dict[str, Any]:
    def command(*args: str) -> str | None:
        try:
            return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    import osqp

    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_model": command("bash", "-lc", "lscpu | awk -F: '/Model name/{gsub(/^ +/,\"\",$2); print $2; exit}'"),
        "cpu_count": os.cpu_count(),
        "gpu_query": command(
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
            "-i",
            "0",
        ),
        "cuda_version": command(
            "bash", "-lc", "nvidia-smi | sed -n '3p' | sed -n 's/.*CUDA Version: \\([^ ]*\\).*/\\1/p'"
        ),
        "osqp_version": _package_version("osqp"),
        "osqp_algebra": osqp.default_algebra(),
        "osqp_linear_system_backend": "QDLDL (OSQP builtin algebra)",
        "cuopt_version": _package_version("cuopt-cu13", "cuopt-cu12", "cuopt"),
        "numpy_version": _package_version("numpy"),
        "scipy_version": _package_version("scipy"),
        "pandas_version": _package_version("pandas"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def run_cold_benchmark(
    config: dict[str, Any], output_root: Path, backends: list[str]
) -> pd.DataFrame:
    raw_path = output_root / "raw" / "cold_results.csv"
    if raw_path.is_file():
        rows = pd.read_csv(raw_path).to_dict(orient="records")
    else:
        rows = []
    existing = {
        (
            row["problem_family"],
            int(row["n_variables"]),
            int(row["seed"]),
            str(row["backend"]),
            int(row["repetition"]),
            _as_bool(row["registered_repetition"]),
        )
        for row in rows
    }
    canonical = config["canonical_problem"]
    repetitions = int(canonical["registered_repetitions"])
    warmups = int(canonical["warmup_repetitions"])
    solver = {"OSQP": solve_osqp, "cuOpt": solve_cuopt}
    cuopt_measurements = 0
    restart_limit = int(
        config.get("execution", {}).get("max_cuopt_measurements_per_process", 0)
    )
    for backend in backends:
        for family in FAMILY_ORDER:
            for n_variables in canonical["sizes"]:
                for seed in canonical["seeds"]:
                    problem = generate_canonical_problem(
                        config, family, int(n_variables), int(seed)
                    )
                    if anchor_violation(problem) > 1.0e-12:
                        raise RuntimeError("canonical generator produced an infeasible anchor")
                    schedule = [
                        (-(index + 1), False) for index in range(warmups)
                    ] + [(index, True) for index in range(repetitions)]
                    for repetition, registered in schedule:
                        key = (
                            family,
                            int(n_variables),
                            int(seed),
                            backend,
                            repetition,
                            registered,
                        )
                        if key in existing:
                            continue
                        measurement = solver[backend](problem, config)
                        rows.append(
                            _measurement_row(
                                problem, measurement, repetition, registered
                            )
                        )
                        existing.add(key)
                        frame = pd.DataFrame.from_records(rows)
                        _atomic_csv(raw_path, frame)
                        print(
                            f"{backend} {family} n={n_variables} seed={seed} "
                            f"rep={repetition}: {measurement.status}",
                            flush=True,
                        )
                        if backend == "cuOpt":
                            cuopt_measurements += 1
                            if restart_limit and cuopt_measurements >= restart_limit:
                                print(
                                    "Restarting the cuOpt worker process after "
                                    f"{cuopt_measurements} persisted measurements",
                                    flush=True,
                                )
                                os.execv(
                                    sys.executable,
                                    [
                                        sys.executable,
                                        "-m",
                                        "benchmarks.synthetic_solver_benchmark",
                                        *sys.argv[1:],
                                    ],
                                )
    return pd.DataFrame.from_records(rows)


def _update_measurement_row(
    problem: CanonicalProblem,
    backend: str,
    seed: int,
    update_index: int,
    setup_seconds: float,
    update_seconds: float,
    solve_seconds: float,
    extraction_seconds: float,
    correctness_seconds: float,
    raw_status: str,
    objective: float,
    reported_objective: float,
    objective_error: float,
    violation: float,
    passed: bool,
) -> dict[str, Any]:
    return {
        "problem_family": problem.family,
        "n_variables": problem.n_variables,
        "n_constraints": problem.n_constraints,
        "seed": seed,
        "update_index": update_index,
        "backend": backend,
        "canonical_sha256": problem.canonical_sha256,
        "setup_seconds": setup_seconds,
        "update_seconds": update_seconds,
        "solve_seconds": solve_seconds,
        "result_extraction_seconds": extraction_seconds,
        "correctness_seconds": correctness_seconds,
        "amortized_end_to_end_seconds": update_seconds
        + solve_seconds
        + extraction_seconds
        + correctness_seconds,
        "raw_status": raw_status,
        "strict_optimal": passed,
        "correctness_pass": passed,
        "objective_value": objective,
        "reported_objective_value": reported_objective,
        "objective_reconstruction_error": objective_error,
        "original_primal_violation": violation,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "host": socket.gethostname(),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "execution_sha": os.environ.get("EXPECTED_SHA"),
    }


def run_repeated_updates(
    config: dict[str, Any], output_root: Path, backends: list[str]
) -> pd.DataFrame:
    output_path = output_root / "raw" / "repeated_update_results.csv"
    rows: list[dict[str, Any]] = []
    update_config = config["repeated_update"]
    family = str(update_config["family"])
    n_variables = int(update_config["n_variables"])
    updates = int(update_config["updates_per_seed"])
    scale = float(update_config["perturbation_scale"])
    thresholds = config["correctness"]

    for seed in update_config["seeds"]:
        base = generate_canonical_problem(config, family, n_variables, int(seed))
        perturb_rng = np.random.default_rng(900_000 + int(seed))
        q_values = [
            base.q
            + perturb_rng.normal(0.0, scale / np.sqrt(n_variables), n_variables)
            for _ in range(updates)
        ]
        if "OSQP" in backends:
            import osqp

            settings = config["solver_contract"]["osqp"]
            setup_start = time.perf_counter()
            identity = sparse.eye(n_variables, format="csc")
            backend_A = sparse.vstack([base.A, identity], format="csc")
            lower = np.concatenate([base.row_lower, base.variable_lower])
            upper = np.concatenate([base.row_upper, base.variable_upper])
            solver = osqp.OSQP()
            solver.setup(
                P=sparse.triu(base.P, format="csc"),
                q=base.q,
                A=backend_A,
                l=lower,
                u=upper,
                verbose=False,
                eps_abs=float(settings["eps_abs"]),
                eps_rel=float(settings["eps_rel"]),
                max_iter=int(settings["max_iter"]),
                polishing=bool(settings["polishing"]),
                adaptive_rho=bool(settings["adaptive_rho"]),
            )
            setup_seconds = time.perf_counter() - setup_start
            solver.solve(raise_error=False)
            for update_index, q_value in enumerate(q_values):
                update_start = time.perf_counter()
                solver.update(q=q_value)
                update_seconds = time.perf_counter() - update_start
                solve_start = time.perf_counter()
                result = solver.solve(raise_error=False)
                solve_seconds = time.perf_counter() - solve_start
                extraction_start = time.perf_counter()
                x = np.asarray(result.x, dtype=float)
                reported = float(result.info.obj_val)
                extraction_seconds = time.perf_counter() - extraction_start
                updated = replace(base, q=q_value)
                objective, error, violation, passed, correctness_seconds = _correctness(
                    updated, x, reported, config
                )
                raw_status = str(result.info.status)
                passed = passed and raw_status == thresholds["strict_osqp_raw_status"]
                rows.append(
                    _update_measurement_row(
                        updated,
                        "OSQP",
                        int(seed),
                        update_index,
                        setup_seconds,
                        update_seconds,
                        solve_seconds,
                        extraction_seconds,
                        correctness_seconds,
                        raw_status,
                        objective,
                        reported,
                        error,
                        violation,
                        passed,
                    )
                )
        if "cuOpt" in backends:
            setup_start = time.perf_counter()
            model, variables, quadratic, settings, linear_type, sense = (
                _build_cuopt_model(base, config)
            )
            setup_seconds = time.perf_counter() - setup_start
            model.solve(settings)
            for update_index, q_value in enumerate(q_values):
                update_start = time.perf_counter()
                linear = linear_type(variables, q_value.tolist(), 0.0)
                model.setObjective(
                    quadratic + linear if quadratic is not None else linear,
                    sense=sense,
                )
                update_seconds = time.perf_counter() - update_start
                solve_start = time.perf_counter()
                model.solve(settings)
                solve_seconds = time.perf_counter() - solve_start
                extraction_start = time.perf_counter()
                x = np.asarray([variable.getValue() for variable in variables])
                reported = float(model.ObjValue)
                extraction_seconds = time.perf_counter() - extraction_start
                updated = replace(base, q=q_value)
                objective, error, violation, passed, correctness_seconds = _correctness(
                    updated, x, reported, config
                )
                raw_status = getattr(model.Status, "name", str(model.Status))
                passed = passed and raw_status == thresholds["strict_cuopt_raw_status"]
                rows.append(
                    _update_measurement_row(
                        updated,
                        "cuOpt",
                        int(seed),
                        update_index,
                        setup_seconds,
                        update_seconds,
                        solve_seconds,
                        extraction_seconds,
                        correctness_seconds,
                        raw_status,
                        objective,
                        reported,
                        error,
                        violation,
                        passed,
                    )
                )
        _atomic_csv(output_path, pd.DataFrame.from_records(rows))
    return pd.DataFrame.from_records(rows)


def _git_sha(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/synthetic_solver_benchmark.yaml"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/synthetic_solver_benchmark"),
    )
    parser.add_argument(
        "--backend", choices=("both", "osqp", "cuopt"), default="both"
    )
    parser.add_argument("--skip-repeated-update", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    output_root = (
        args.output_root if args.output_root.is_absolute() else root / args.output_root
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    git_sha = _git_sha(root)
    expected_sha = os.environ.get("EXPECTED_SHA")
    if expected_sha and expected_sha != git_sha:
        raise RuntimeError("synthetic benchmark HEAD differs from EXPECTED_SHA")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    ).strip():
        raise RuntimeError("synthetic benchmark requires a clean immutable worktree")
    output_root.mkdir(parents=True, exist_ok=True)
    environment = collect_environment()
    environment["git_sha"] = git_sha
    environment["config_sha256"] = _sha256_file(config_path)
    _atomic_json(output_root / "provenance" / "environment.json", environment)
    backends = {
        "both": ["OSQP", "cuOpt"],
        "osqp": ["OSQP"],
        "cuopt": ["cuOpt"],
    }[args.backend]
    cold = run_cold_benchmark(config, output_root, backends)
    repeated = pd.DataFrame()
    if not args.skip_repeated_update:
        repeated = run_repeated_updates(config, output_root, backends)
    expected_cold = (
        len(config["canonical_problem"]["families"])
        * len(config["canonical_problem"]["sizes"])
        * len(config["canonical_problem"]["seeds"])
        * (
            int(config["canonical_problem"]["registered_repetitions"])
            + int(config["canonical_problem"]["warmup_repetitions"])
        )
        * len(backends)
    )
    manifest = {
        "schema_version": 1,
        "status": "pass" if len(cold) == expected_cold else "incomplete",
        "run_kind": "synthetic_osqp_direct_cuopt_raw_benchmark",
        "git_sha": git_sha,
        "config_sha256": _sha256_file(config_path),
        "canonical_instance_count": len(config["canonical_problem"]["families"])
        * len(config["canonical_problem"]["sizes"])
        * len(config["canonical_problem"]["seeds"]),
        "expected_cold_rows": expected_cold,
        "actual_cold_rows": len(cold),
        "repeated_update_rows": len(repeated),
        "backends": backends,
        "cold_results_sha256": _sha256_file(
            output_root / "raw" / "cold_results.csv"
        ),
        "repeated_update_results_sha256": (
            _sha256_file(output_root / "raw" / "repeated_update_results.csv")
            if not repeated.empty
            else None
        ),
        "environment_sha256": _sha256_file(
            output_root / "provenance" / "environment.json"
        ),
        "execution": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "host": socket.gethostname(),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "output_root": str(output_root),
        },
    }
    _atomic_json(output_root / "raw" / "run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
