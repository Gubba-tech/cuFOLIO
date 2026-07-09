# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backends for compiled PortOpt-style QP problems."""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass

import numpy as np

from .exceptions import GPUBackendUnavailable
from .qp_formulations import CompiledQP


@dataclass
class QPSolution:
    """Raw solution returned by a QP backend."""

    x: np.ndarray
    status: str
    solver: str
    objective_value: float
    solve_time: float | None
    total_time: float


def solve_compiled_qp_osqp(compiled: CompiledQP) -> QPSolution:
    """Solve a compiled QP with CVXPY's OSQP interface for validation."""

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

    problem = cp.Problem(objective, constraints)
    start = time.time()
    problem.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8, max_iter=100000)
    total_time = time.time() - start
    if x.value is None:
        raise RuntimeError(f"OSQP failed to produce a solution. Status: {problem.status}")

    stats = getattr(problem, "solver_stats", None)
    solve_time = getattr(stats, "solve_time", None) if stats is not None else None
    return QPSolution(
        x=np.asarray(x.value, dtype=float).reshape(-1),
        status=str(problem.status),
        solver="OSQP",
        objective_value=float(problem.value),
        solve_time=float(solve_time) if solve_time is not None else None,
        total_time=total_time,
    )


def solve_compiled_qp_cuopt(compiled: CompiledQP) -> QPSolution:
    """Guarded cuOpt backend placeholder for compiled sparse QPs."""

    if importlib.util.find_spec("cuopt") is None:
        raise GPUBackendUnavailable(
            "cuOpt GPU runtime unavailable; install a cuFOLIO CUDA extra and do "
            "not substitute a CPU solver for backend='cuopt'."
        )
    raise NotImplementedError(
        "Direct cuOpt compiled-QP execution will be added after the sparse compiler "
        "API stabilizes. backend='cuopt' must not fall back to CPU."
    )


def max_constraint_violation(compiled: CompiledQP, x: np.ndarray) -> float:
    """Return max absolute/equality/inequality/bound violation."""

    violations = [0.0]
    if compiled.A_eq.shape[0]:
        violations.append(float(np.max(np.abs(compiled.A_eq @ x - compiled.b_eq))))
    if compiled.A_ineq.shape[0]:
        violations.append(float(np.max(np.maximum(compiled.A_ineq @ x - compiled.b_ineq, 0.0))))
    finite_lower = np.isfinite(compiled.lower)
    finite_upper = np.isfinite(compiled.upper)
    if finite_lower.any():
        violations.append(float(np.max(np.maximum(compiled.lower[finite_lower] - x[finite_lower], 0.0))))
    if finite_upper.any():
        violations.append(float(np.max(np.maximum(x[finite_upper] - compiled.upper[finite_upper], 0.0))))
    return max(violations)
