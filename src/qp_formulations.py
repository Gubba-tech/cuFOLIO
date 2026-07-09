# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic sparse QP compiler for PortOpt-style portfolios."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .exceptions import QPCompilationError
from .qp_parameters import QPParameters


@dataclass(frozen=True)
class VariableSlice:
    """Named slice into the compiled decision vector."""

    start: int
    stop: int

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)

    @property
    def size(self) -> int:
        return self.stop - self.start


@dataclass
class CompiledQP:
    """Standard-form sparse QP plus recovery metadata."""

    Q: sparse.csr_matrix
    q: np.ndarray
    A_eq: sparse.csr_matrix
    b_eq: np.ndarray
    A_ineq: sparse.csr_matrix
    b_ineq: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    variable_slices: dict[str, VariableSlice]
    stock_mapping: sparse.csr_matrix
    mean: np.ndarray
    covariance: np.ndarray
    objective: str
    risk_free_rate: float
    constraint_names: list[str]

    @property
    def n_variables(self) -> int:
        return self.q.size

    @property
    def n_constraints(self) -> int:
        return self.A_eq.shape[0] + self.A_ineq.shape[0]

    def recover_stock_weights(self, x: np.ndarray) -> np.ndarray:
        decision = x[self.variable_slices["decision"].slice]
        stock_scaled = np.asarray(self.stock_mapping @ decision).reshape(-1)
        if self.objective == "max_sharpe":
            c_value = float(x[self.variable_slices["scale"].slice][0])
            if c_value <= 0:
                raise QPCompilationError(
                    f"max_sharpe recovery requires c > 0; got {c_value}."
                )
            return stock_scaled / c_value
        return stock_scaled

    def recover_factor_weights(self, x: np.ndarray) -> np.ndarray | None:
        if self.stock_mapping.shape[1] == self.stock_mapping.shape[0]:
            return None
        decision = x[self.variable_slices["decision"].slice]
        if self.objective == "max_sharpe":
            c_value = float(x[self.variable_slices["scale"].slice][0])
            if c_value <= 0:
                raise QPCompilationError(
                    f"max_sharpe recovery requires c > 0; got {c_value}."
                )
            return decision / c_value
        return decision


def compile_portfolio_qp(
    returns_dict: dict,
    params: QPParameters,
) -> CompiledQP:
    """Compile a portfolio problem into sparse QP standard form."""

    mean = _as_vector(returns_dict["mean"], "mean")
    covariance = _as_square_matrix(returns_dict["covariance"], "covariance")
    n_assets = mean.size
    if covariance.shape != (n_assets, n_assets):
        raise QPCompilationError(
            "covariance shape must match mean length; "
            f"got {covariance.shape} and {n_assets}."
        )
    covariance = _make_psd_symmetric(covariance)

    mapping = _mapping_matrix(params.V, n_assets)
    n_decision = mapping.shape[1]
    mean_decision = np.asarray(mapping.T @ mean).reshape(-1)
    cov_decision = np.asarray(mapping.T @ covariance @ mapping)

    slices: dict[str, VariableSlice] = {}
    cursor = 0
    slices["decision"] = VariableSlice(cursor, cursor + n_decision)
    cursor += n_decision

    if params.objective == "max_sharpe":
        slices["scale"] = VariableSlice(cursor, cursor + 1)
        cursor += 1

    needs_l1 = params.lambda_l1 > 0
    if needs_l1:
        slices["l1_pos"] = VariableSlice(cursor, cursor + n_assets)
        cursor += n_assets
        slices["l1_neg"] = VariableSlice(cursor, cursor + n_assets)
        cursor += n_assets

    needs_long_short = params.short_budget is not None
    if needs_long_short:
        slices["pos"] = VariableSlice(cursor, cursor + n_assets)
        cursor += n_assets
        slices["neg"] = VariableSlice(cursor, cursor + n_assets)
        cursor += n_assets

    needs_turnover = params.turnover_budget is not None
    if needs_turnover:
        _require_vector(params.previous_weights, n_assets, "previous_weights")
        slices["turnover_pos"] = VariableSlice(cursor, cursor + n_assets)
        cursor += n_assets
        slices["turnover_neg"] = VariableSlice(cursor, cursor + n_assets)
        cursor += n_assets

    needs_benchmark = params.benchmark_l1_budget is not None
    if needs_benchmark:
        _require_vector(params.benchmark_weights, n_assets, "benchmark_weights")
        slices["benchmark_pos"] = VariableSlice(cursor, cursor + n_assets)
        cursor += n_assets
        slices["benchmark_neg"] = VariableSlice(cursor, cursor + n_assets)
        cursor += n_assets

    n_variables = cursor
    Q = sparse.lil_matrix((n_variables, n_variables), dtype=float)
    q = np.zeros(n_variables, dtype=float)
    decision = slices["decision"].slice

    Q[decision, decision] = cov_decision
    if params.lambda_l2 > 0:
        Q[decision, decision] = (
            Q[decision, decision]
            + 2.0 * params.lambda_l2 * np.asarray(mapping.T @ mapping)
        )

    if params.lambda_tracking_error > 0:
        benchmark = _require_vector(
            params.benchmark_weights, n_assets, "benchmark_weights"
        )
        Q[decision, decision] = (
            Q[decision, decision]
            + 2.0
            * params.lambda_tracking_error
            * np.asarray(mapping.T @ covariance @ mapping)
        )
        q[decision] += -2.0 * params.lambda_tracking_error * np.asarray(
            mapping.T @ covariance @ benchmark
        ).reshape(-1)

    if params.objective == "mean_variance":
        q[decision] += -params.risk_aversion * mean_decision
    elif params.objective in {"min_variance", "target_return", "max_sharpe"}:
        pass
    else:
        raise QPCompilationError(f"Unsupported objective: {params.objective}")

    if needs_l1:
        q[slices["l1_pos"].slice] = params.lambda_l1
        q[slices["l1_neg"].slice] = params.lambda_l1

    lower = np.full(n_variables, -np.inf)
    upper = np.full(n_variables, np.inf)
    if params.objective == "max_sharpe":
        lower[slices["scale"].slice] = 1e-12
    for name in (
        "l1_pos",
        "l1_neg",
        "pos",
        "neg",
        "turnover_pos",
        "turnover_neg",
        "benchmark_pos",
        "benchmark_neg",
    ):
        if name in slices:
            lower[slices[name].slice] = 0.0

    eq_rows: list[np.ndarray] = []
    eq_rhs: list[float] = []
    ineq_rows: list[np.ndarray] = []
    ineq_rhs: list[float] = []
    names: list[str] = []

    stock_matrix = np.asarray(mapping)
    ones_stock = np.ones(n_assets)
    stock_sum_row = ones_stock @ stock_matrix

    if params.objective == "max_sharpe":
        excess = mean_decision - params.risk_free_rate * stock_sum_row
        if np.linalg.norm(excess, ord=np.inf) <= 1e-14:
            raise QPCompilationError("max_sharpe requires nonzero excess returns.")
        row = np.zeros(n_variables)
        row[decision] = excess
        eq_rows.append(row)
        eq_rhs.append(1.0)
        names.append("max_sharpe_excess_return")

        row = np.zeros(n_variables)
        row[decision] = stock_sum_row
        row[slices["scale"].slice] = -1.0
        eq_rows.append(row)
        eq_rhs.append(0.0)
        names.append("scaled_budget")
    else:
        row = np.zeros(n_variables)
        row[decision] = stock_sum_row
        eq_rows.append(row)
        eq_rhs.append(1.0)
        names.append("fully_invested")

    lower_w = _expand_bound(params.w_min, n_assets, "w_min")
    upper_w = _expand_bound(params.w_max, n_assets, "w_max")
    if np.any(lower_w > upper_w):
        raise QPCompilationError("w_min must be <= w_max for every asset.")

    _add_stock_bounds(
        ineq_rows,
        ineq_rhs,
        names,
        stock_matrix,
        slices,
        n_variables,
        lower_w,
        upper_w,
        scaled=params.objective == "max_sharpe",
    )

    if params.objective == "target_return":
        if params.target_return is None:
            raise QPCompilationError("target_return objective requires target_return.")
        row = np.zeros(n_variables)
        row[decision] = -mean_decision
        ineq_rows.append(row)
        ineq_rhs.append(-float(params.target_return))
        names.append("target_return")

    if needs_l1:
        _add_split_equalities(
            eq_rows,
            eq_rhs,
            names,
            stock_matrix,
            slices,
            n_variables,
            pos_name="l1_pos",
            neg_name="l1_neg",
            prefix="l1",
        )

    if needs_long_short:
        _add_split_equalities(
            eq_rows,
            eq_rhs,
            names,
            stock_matrix,
            slices,
            n_variables,
            pos_name="pos",
            neg_name="neg",
            prefix="long_short",
        )
        row = np.zeros(n_variables)
        row[slices["pos"].slice] = 1.0
        if params.objective == "max_sharpe":
            row[slices["scale"].slice] = -(1.0 + float(params.short_budget))
            rhs = 0.0
        else:
            rhs = 1.0 + float(params.short_budget)
        ineq_rows.append(row)
        ineq_rhs.append(rhs)
        names.append("long_budget")

        row = np.zeros(n_variables)
        row[slices["neg"].slice] = 1.0
        if params.objective == "max_sharpe":
            row[slices["scale"].slice] = -float(params.short_budget)
            rhs = 0.0
        else:
            rhs = float(params.short_budget)
        ineq_rows.append(row)
        ineq_rhs.append(rhs)
        names.append("short_budget")

    if needs_turnover:
        previous = _require_vector(params.previous_weights, n_assets, "previous_weights")
        _add_anchor_l1_constraint(
            eq_rows,
            eq_rhs,
            ineq_rows,
            ineq_rhs,
            names,
            stock_matrix,
            slices,
            n_variables,
            anchor=previous,
            budget=float(params.turnover_budget),
            pos_name="turnover_pos",
            neg_name="turnover_neg",
            prefix="turnover",
            scaled=params.objective == "max_sharpe",
        )

    if needs_benchmark:
        benchmark = _require_vector(
            params.benchmark_weights, n_assets, "benchmark_weights"
        )
        _add_anchor_l1_constraint(
            eq_rows,
            eq_rhs,
            ineq_rows,
            ineq_rhs,
            names,
            stock_matrix,
            slices,
            n_variables,
            anchor=benchmark,
            budget=float(params.benchmark_l1_budget),
            pos_name="benchmark_pos",
            neg_name="benchmark_neg",
            prefix="benchmark",
            scaled=params.objective == "max_sharpe",
        )

    _add_factor_exposure_constraints(
        ineq_rows, ineq_rhs, names, stock_matrix, slices, n_variables, params
    )

    return CompiledQP(
        Q=Q.tocsr(),
        q=q,
        A_eq=_rows_to_csr(eq_rows, n_variables),
        b_eq=np.asarray(eq_rhs, dtype=float),
        A_ineq=_rows_to_csr(ineq_rows, n_variables),
        b_ineq=np.asarray(ineq_rhs, dtype=float),
        lower=lower,
        upper=upper,
        variable_slices=slices,
        stock_mapping=sparse.csr_matrix(mapping),
        mean=mean,
        covariance=covariance,
        objective=params.objective,
        risk_free_rate=params.risk_free_rate,
        constraint_names=names,
    )


def _as_vector(value, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise QPCompilationError(f"{name} must be a non-empty finite vector.")
    return arr


def _as_square_matrix(value, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise QPCompilationError(f"{name} must be a square matrix.")
    if not np.all(np.isfinite(arr)):
        raise QPCompilationError(f"{name} must contain only finite values.")
    return arr


def _make_psd_symmetric(covariance: np.ndarray) -> np.ndarray:
    covariance = 0.5 * (covariance + covariance.T)
    min_eig = float(np.linalg.eigvalsh(covariance).min())
    if min_eig < -1e-10:
        raise QPCompilationError(
            "covariance must be positive semidefinite; "
            f"minimum eigenvalue is {min_eig:.3e}."
        )
    if min_eig < 0:
        covariance = covariance + np.eye(covariance.shape[0]) * (-min_eig + 1e-10)
    return covariance


def _mapping_matrix(value, n_assets: int) -> np.ndarray:
    if value is None:
        return np.eye(n_assets)
    mapping = np.asarray(value, dtype=float)
    if mapping.ndim != 2 or mapping.shape[0] != n_assets:
        raise QPCompilationError(
            f"V must have shape (n_assets, k); got {mapping.shape}."
        )
    if not np.all(np.isfinite(mapping)):
        raise QPCompilationError("V must contain only finite values.")
    return mapping


def _expand_bound(value, n_assets: int, name: str) -> np.ndarray:
    if value is None:
        if name == "w_min":
            return np.full(n_assets, -np.inf)
        return np.full(n_assets, np.inf)
    if isinstance(value, dict):
        if "others" not in value:
            raise QPCompilationError(f"{name} dict requires an 'others' key here.")
        return np.full(n_assets, float(value["others"]))
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n_assets, float(arr))
    arr = arr.reshape(-1)
    if arr.size != n_assets:
        raise QPCompilationError(f"{name} must have length {n_assets}.")
    return arr


def _require_vector(value, n_assets: int, name: str) -> np.ndarray:
    if value is None:
        raise QPCompilationError(f"{name} is required for this QP option.")
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != n_assets or not np.all(np.isfinite(arr)):
        raise QPCompilationError(f"{name} must be a finite vector of length {n_assets}.")
    return arr


def _rows_to_csr(rows: list[np.ndarray], n_variables: int) -> sparse.csr_matrix:
    if not rows:
        return sparse.csr_matrix((0, n_variables), dtype=float)
    return sparse.csr_matrix(np.vstack(rows))


def _add_stock_bounds(
    ineq_rows,
    ineq_rhs,
    names,
    stock_matrix,
    slices,
    n_variables,
    lower_w,
    upper_w,
    *,
    scaled: bool,
) -> None:
    for i in range(stock_matrix.shape[0]):
        if np.isfinite(upper_w[i]):
            row = np.zeros(n_variables)
            row[slices["decision"].slice] = stock_matrix[i]
            if scaled:
                row[slices["scale"].slice] = -upper_w[i]
                rhs = 0.0
            else:
                rhs = upper_w[i]
            ineq_rows.append(row)
            ineq_rhs.append(rhs)
            names.append(f"upper_bound_{i}")
        if np.isfinite(lower_w[i]):
            row = np.zeros(n_variables)
            row[slices["decision"].slice] = -stock_matrix[i]
            if scaled:
                row[slices["scale"].slice] = lower_w[i]
                rhs = 0.0
            else:
                rhs = -lower_w[i]
            ineq_rows.append(row)
            ineq_rhs.append(rhs)
            names.append(f"lower_bound_{i}")


def _add_split_equalities(
    eq_rows,
    eq_rhs,
    names,
    stock_matrix,
    slices,
    n_variables,
    *,
    pos_name: str,
    neg_name: str,
    prefix: str,
) -> None:
    for i in range(stock_matrix.shape[0]):
        row = np.zeros(n_variables)
        row[slices["decision"].slice] = stock_matrix[i]
        row[slices[pos_name].start + i] = -1.0
        row[slices[neg_name].start + i] = 1.0
        eq_rows.append(row)
        eq_rhs.append(0.0)
        names.append(f"{prefix}_split_{i}")


def _add_anchor_l1_constraint(
    eq_rows,
    eq_rhs,
    ineq_rows,
    ineq_rhs,
    names,
    stock_matrix,
    slices,
    n_variables,
    *,
    anchor: np.ndarray,
    budget: float,
    pos_name: str,
    neg_name: str,
    prefix: str,
    scaled: bool,
) -> None:
    for i in range(stock_matrix.shape[0]):
        row = np.zeros(n_variables)
        row[slices["decision"].slice] = stock_matrix[i]
        if scaled:
            row[slices["scale"].slice] = -anchor[i]
            rhs = 0.0
        else:
            rhs = anchor[i]
        row[slices[pos_name].start + i] = -1.0
        row[slices[neg_name].start + i] = 1.0
        eq_rows.append(row)
        eq_rhs.append(rhs)
        names.append(f"{prefix}_split_{i}")

    row = np.zeros(n_variables)
    row[slices[pos_name].slice] = 1.0
    row[slices[neg_name].slice] = 1.0
    if scaled:
        row[slices["scale"].slice] = -budget
        rhs = 0.0
    else:
        rhs = budget
    ineq_rows.append(row)
    ineq_rhs.append(rhs)
    names.append(f"{prefix}_budget")


def _add_factor_exposure_constraints(
    ineq_rows, ineq_rhs, names, stock_matrix, slices, n_variables, params
) -> None:
    B = params.factor_exposure_matrix
    if B is None:
        return
    B = np.asarray(B, dtype=float)
    if B.ndim != 2 or B.shape[0] != stock_matrix.shape[0]:
        raise QPCompilationError(
            "factor_exposure_matrix must have shape (n_assets, n_exposures)."
        )
    exposure_matrix = B.T @ stock_matrix
    lower = params.factor_exposure_lower
    upper = params.factor_exposure_upper
    if lower is None and upper is None:
        raise QPCompilationError(
            "factor exposure constraints require lower and/or upper bounds."
        )
    if lower is not None:
        lower = np.asarray(lower, dtype=float).reshape(-1)
    if upper is not None:
        upper = np.asarray(upper, dtype=float).reshape(-1)

    for j in range(exposure_matrix.shape[0]):
        if upper is not None and np.isfinite(upper[j]):
            row = np.zeros(n_variables)
            row[slices["decision"].slice] = exposure_matrix[j]
            if params.objective == "max_sharpe":
                row[slices["scale"].slice] = -upper[j]
                rhs = 0.0
            else:
                rhs = upper[j]
            ineq_rows.append(row)
            ineq_rhs.append(rhs)
            names.append(f"factor_exposure_upper_{j}")
        if lower is not None and np.isfinite(lower[j]):
            row = np.zeros(n_variables)
            row[slices["decision"].slice] = -exposure_matrix[j]
            if params.objective == "max_sharpe":
                row[slices["scale"].slice] = lower[j]
                rhs = 0.0
            else:
                rhs = -lower[j]
            ineq_rows.append(row)
            ineq_rhs.append(rhs)
            names.append(f"factor_exposure_lower_{j}")
