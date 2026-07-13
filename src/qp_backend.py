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


def _is_cuda_unavailable_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "cudaerrorinsufficientdriver",
            "no cuda-capable device",
            "no devices were found",
            "driver version is insufficient",
        )
    )


@dataclass
class QPSolution:
    """Raw solution returned by a QP backend."""

    x: np.ndarray
    status: str
    solver_name: str
    objective_value: float
    solve_time: float | None
    total_time: float
    max_constraint_violation: float
    variable_values_by_name: dict[str, float]
    raw_status: str | None = None

    @property
    def solver(self) -> str:
        """Backward-compatible solver label used by the optimizer."""
        return self.solver_name


def normalize_qp_status(raw_status: str) -> str:
    """Normalize solver-specific QP statuses into a stable public label."""

    normalized = str(raw_status).strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        return "unknown"
    if "infeasible" in normalized:
        return "infeasible"
    if "unbounded" in normalized:
        return "unbounded"
    known_optimal = {
        "optimal",
        "optimal_inaccurate",
        "optimal_with_tolerance",
        "locally_optimal",
        "globally_optimal",
    }
    if normalized in known_optimal or normalized.startswith("optimal"):
        return "optimal"
    return normalized


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
    # Absolute-value split variables have flat auxiliary directions because
    # the split variables are constrained but do not belong to the QP objective.
    # Give OSQP's validation path enough iterations to resolve those directions
    # without changing the compiled objective or adding complementarity.
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
        solver=cp.OSQP,
        eps_abs=1e-8,
        eps_rel=1e-8,
        max_iter=max_iter,
    )
    total_time = time.time() - start
    if x.value is None:
        raise RuntimeError(
            f"OSQP failed to produce a solution. Status: {problem.status}"
        )

    stats = getattr(problem, "solver_stats", None)
    solve_time = getattr(stats, "solve_time", None) if stats is not None else None
    raw_status = str(problem.status)
    return QPSolution(
        x=np.asarray(x.value, dtype=float).reshape(-1),
        status=normalize_qp_status(raw_status),
        solver_name="OSQP",
        objective_value=float(problem.value),
        solve_time=float(solve_time) if solve_time is not None else None,
        total_time=total_time,
        max_constraint_violation=max_constraint_violation(
            compiled, np.asarray(x.value, dtype=float).reshape(-1)
        ),
        variable_values_by_name=dict(
            zip(
                compiled.variable_names,
                np.asarray(x.value, dtype=float).reshape(-1),
            )
        ),
        raw_status=raw_status,
    )


class CuOptQPBackend:
    """Direct cuOpt Python backend for compiled sparse portfolio QPs."""

    accepted_status_fragments = ("optimal",)

    def solve(
        self,
        compiled: CompiledQP,
        solver_settings: dict | None = None,
    ) -> QPSolution:
        """Translate ``CompiledQP`` into a cuOpt ``Problem`` and solve it."""

        if importlib.util.find_spec("cuopt") is None:
            raise GPUBackendUnavailable(
                "cuOpt GPU runtime unavailable; install a cuFOLIO CUDA extra and "
                "do not substitute a CPU solver for backend='cuopt'."
            )

        from cuopt.linear_programming.problem import (
            CONTINUOUS,
            MINIMIZE,
            LinearExpression,
            Problem,
            QuadraticExpression,
        )
        from cuopt.linear_programming.solver_settings import SolverSettings

        problem = Problem("PortOpt Unified QP")
        variables = self._add_variables(problem, compiled, CONTINUOUS)
        self._add_linear_constraints(problem, compiled, variables, LinearExpression)
        objective_expr = self._build_objective(
            compiled,
            variables,
            LinearExpression,
            QuadraticExpression,
        )
        problem.setObjective(objective_expr, sense=MINIMIZE)

        settings = SolverSettings()
        if solver_settings:
            for param, value in solver_settings.items():
                if param != "solver":
                    settings.set_parameter(param, value)

        total_start = time.time()
        try:
            problem.solve(settings)
        except RuntimeError as exc:
            if _is_cuda_unavailable_error(exc):
                raise GPUBackendUnavailable(
                    "cuOpt GPU runtime unavailable; CUDA driver/device is not "
                    "usable on this machine. Do not substitute a CPU solver."
                ) from exc
            raise
        total_time = time.time() - total_start

        raw_status = getattr(problem.Status, "name", str(problem.Status))
        status = normalize_qp_status(raw_status)
        if not self._is_accepted_status(raw_status):
            raise RuntimeError(f"cuOpt failed to solve QP. Status: {raw_status}")

        x = np.asarray([var.getValue() for var in variables], dtype=float)
        violation = max_constraint_violation(compiled, x)
        return QPSolution(
            x=x,
            status=status,
            solver_name="cuopt_qp",
            objective_value=float(problem.ObjValue),
            solve_time=float(getattr(problem, "SolveTime", np.nan)),
            total_time=total_time,
            max_constraint_violation=violation,
            variable_values_by_name=dict(zip(compiled.variable_names, x)),
            raw_status=raw_status,
        )

    def _add_variables(self, problem, compiled, continuous_type):
        variables = []
        for idx, name in enumerate(compiled.variable_names):
            var = problem.addVariable(
                lb=float(compiled.lower[idx]),
                ub=float(compiled.upper[idx]),
                vtype=continuous_type,
                name=name,
            )
            variables.append(var)
        return variables

    def _add_linear_constraints(
        self,
        problem,
        compiled: CompiledQP,
        variables,
        linear_expression,
    ) -> None:
        A = compiled.A.tocsr()
        row_lower = compiled.row_lower
        row_upper = compiled.row_upper
        for row_idx in range(A.shape[0]):
            expr = self._linear_expr_from_row(
                A.getrow(row_idx),
                variables,
                linear_expression,
            )
            lower = row_lower[row_idx]
            upper = row_upper[row_idx]
            if np.isfinite(lower) and np.isfinite(upper) and np.isclose(lower, upper):
                problem.addConstraint(expr == float(upper), name=f"qp_row_{row_idx}_eq")
            else:
                if np.isfinite(lower):
                    problem.addConstraint(
                        expr >= float(lower),
                        name=f"qp_row_{row_idx}_lower",
                    )
                if np.isfinite(upper):
                    problem.addConstraint(
                        expr <= float(upper),
                        name=f"qp_row_{row_idx}_upper",
                    )

    def _build_objective(
        self,
        compiled: CompiledQP,
        variables,
        linear_expression,
        quadratic_expression,
    ):
        # CompiledQP uses 0.5 * x.T @ Q @ x + q.T @ x.
        # cuOpt uses x.T @ Q_cuopt @ x + c.T @ x, so pass Q_cuopt = 0.5 * Q.
        quad_expr = self._quadratic_expression(
            0.5 * compiled.Q,
            variables,
            quadratic_expression,
        )
        lin_expr = linear_expression(
            variables,
            [float(value) for value in compiled.q],
            0.0,
        )
        return quad_expr + lin_expr

    def _quadratic_expression(self, q_matrix, variables, quadratic_expression):
        q_matrix = q_matrix.tocoo()
        if q_matrix.nnz:
            q_vars_1 = [variables[int(row)] for row in q_matrix.row]
            q_vars_2 = [variables[int(col)] for col in q_matrix.col]
            q_coefficients = [float(value) for value in q_matrix.data]
            try:
                return quadratic_expression(q_vars_1, q_vars_2, q_coefficients)
            except (TypeError, ValueError, AttributeError):
                pass

        dense = np.zeros((len(variables), len(variables)), dtype=float)
        if q_matrix.nnz:
            dense[q_matrix.row, q_matrix.col] = q_matrix.data
        return quadratic_expression(dense, variables)

    def _linear_expr_from_row(self, row, variables, linear_expression):
        row = row.tocoo()
        if row.nnz == 0:
            return linear_expression([variables[0]], [0.0], 0.0)
        return linear_expression(
            [variables[int(idx)] for idx in row.col],
            [float(value) for value in row.data],
            0.0,
        )

    def _is_accepted_status(self, status: str) -> bool:
        return normalize_qp_status(status) in self.accepted_status_fragments


def solve_compiled_qp_cuopt(
    compiled: CompiledQP,
    solver_settings: dict | None = None,
) -> QPSolution:
    """Solve a compiled QP with direct cuOpt Python API."""

    if importlib.util.find_spec("cuopt") is None:
        raise GPUBackendUnavailable(
            "cuOpt GPU runtime unavailable; install a cuFOLIO CUDA extra and do "
            "not substitute a CPU solver for backend='cuopt'."
        )
    return CuOptQPBackend().solve(compiled, solver_settings=solver_settings)


def max_constraint_violation(compiled: CompiledQP, x: np.ndarray) -> float:
    """Return max absolute/equality/inequality/bound violation."""

    violations = [0.0]
    if compiled.A_eq.shape[0]:
        violations.append(float(np.max(np.abs(compiled.A_eq @ x - compiled.b_eq))))
    if compiled.A_ineq.shape[0]:
        violations.append(
            float(np.max(np.maximum(compiled.A_ineq @ x - compiled.b_ineq, 0.0)))
        )
    finite_lower = np.isfinite(compiled.lower)
    finite_upper = np.isfinite(compiled.upper)
    if finite_lower.any():
        violations.append(
            float(
                np.max(
                    np.maximum(
                        compiled.lower[finite_lower] - x[finite_lower],
                        0.0,
                    )
                )
            )
        )
    if finite_upper.any():
        violations.append(
            float(
                np.max(
                    np.maximum(
                        x[finite_upper] - compiled.upper[finite_upper],
                        0.0,
                    )
                )
            )
        )
    return max(violations)
