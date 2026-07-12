# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay saved paper-style QP matrices through the cuFOLIO compiler."""

from __future__ import annotations

import csv
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .exceptions import GPUBackendUnavailable
from .qp_backend import QPSolution, solve_compiled_qp_cuopt, solve_compiled_qp_osqp
from .qp_formulations import CompiledQP, compile_portfolio_qp
from .qp_parameters import QPParameters

REPLAY_SCHEMA_VERSION = "1.0"
_ARRAY_FIELDS = (
    "mean",
    "covariance",
    "factor_mean",
    "factor_covariance",
    "stock_mapping",
    "stock_covariance",
    "previous_weights",
    "benchmark_weights",
    "old_weights",
    "realized_next_returns",
    "w_min",
    "w_max",
)


def _optional_array(value: Any, name: str) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return None
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array.copy()


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if np.isfinite(result) else None


@dataclass
class PaperReplayWindow:
    """One saved matrix replay problem and its optional old result."""

    schema_version: str
    window_id: str
    rebalance_date: str
    model_name: str
    objective: str
    mapping_mode: str
    risk_free_rate: float
    lambda_l1: float
    lambda_l2: float
    short_budget: float | None
    w_min: np.ndarray | float
    w_max: np.ndarray | float
    target_return: float | None = None
    tickers: list[str] | None = None
    factor_names: list[str] | None = None
    mean: np.ndarray | None = None
    covariance: np.ndarray | None = None
    factor_mean: np.ndarray | None = None
    factor_covariance: np.ndarray | None = None
    stock_mapping: np.ndarray | None = None
    stock_covariance: np.ndarray | None = None
    previous_weights: np.ndarray | None = None
    turnover_budget: float | None = None
    benchmark_weights: np.ndarray | None = None
    benchmark_l1_budget: float | None = None
    lambda_tracking_error: float = 0.0
    old_weights: np.ndarray | None = None
    old_objective_value: float | None = None
    realized_next_returns: np.ndarray | None = None
    old_realized_portfolio_return: float | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != REPLAY_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported replay schema {self.schema_version!r}; "
                f"expected {REPLAY_SCHEMA_VERSION!r}."
            )
        if not self.window_id or not self.rebalance_date or not self.model_name:
            raise ValueError("window_id, rebalance_date, and model_name are required.")
        if self.objective not in {
            "min_variance",
            "mean_variance",
            "target_return",
            "max_sharpe",
        }:
            raise ValueError(f"Unsupported replay objective: {self.objective}")
        if self.mapping_mode not in {"stock_space", "factor_space"}:
            raise ValueError(f"Unsupported mapping_mode: {self.mapping_mode}")
        for name in (
            "mean",
            "covariance",
            "factor_mean",
            "factor_covariance",
            "stock_mapping",
            "stock_covariance",
            "previous_weights",
            "benchmark_weights",
            "old_weights",
            "realized_next_returns",
        ):
            setattr(self, name, _optional_array(getattr(self, name), name))
        self.risk_free_rate = float(self.risk_free_rate)
        self.lambda_l1 = float(self.lambda_l1)
        self.lambda_l2 = float(self.lambda_l2)
        self.lambda_tracking_error = float(self.lambda_tracking_error)
        for name in (
            "risk_free_rate",
            "lambda_l1",
            "lambda_l2",
            "lambda_tracking_error",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.short_budget is not None and self.short_budget < 0:
            raise ValueError("short_budget must be non-negative.")
        if self.turnover_budget is not None and self.turnover_budget < 0:
            raise ValueError("turnover_budget must be non-negative.")
        if self.benchmark_l1_budget is not None and self.benchmark_l1_budget < 0:
            raise ValueError("benchmark_l1_budget must be non-negative.")

        if self.mapping_mode == "stock_space":
            if self.mean is None or self.covariance is None:
                raise ValueError("stock_space replay requires mean and covariance.")
            self.mean = self.mean.reshape(-1)
            n_assets = self.mean.size
            if self.stock_mapping is None:
                self.stock_mapping = np.eye(n_assets)
            if self.covariance.shape != (n_assets, n_assets):
                raise ValueError("covariance must be square and match mean.")
        else:
            if self.factor_mean is None and self.mean is not None:
                self.factor_mean = self.mean.copy()
            if self.factor_covariance is None and self.covariance is not None:
                self.factor_covariance = self.covariance.copy()
            if self.factor_mean is None or self.factor_covariance is None:
                raise ValueError(
                    "factor_space replay requires factor_mean and factor_covariance."
                )
            if self.stock_mapping is None or self.stock_mapping.ndim != 2:
                raise ValueError("factor_space replay requires a two-dimensional V.")
            self.factor_mean = self.factor_mean.reshape(-1)
            n_assets, n_factors = self.stock_mapping.shape
            if self.factor_covariance.shape != (n_factors, n_factors):
                raise ValueError("factor_covariance must match V factor columns.")
            if self.factor_mean.size != n_factors:
                raise ValueError("factor_mean must match V factor columns.")

        self.stock_mapping = np.asarray(self.stock_mapping, dtype=float)
        n_assets = self.stock_mapping.shape[0]
        if self.stock_covariance is not None and self.stock_covariance.shape != (
            n_assets,
            n_assets,
        ):
            raise ValueError("stock_covariance must match the number of stocks.")
        if self.tickers is None:
            self.tickers = [f"asset_{idx}" for idx in range(n_assets)]
        if len(self.tickers) != n_assets:
            raise ValueError("tickers must match the number of stocks.")
        if self.factor_names is not None:
            n_factors = self.stock_mapping.shape[1]
            if len(self.factor_names) != n_factors:
                raise ValueError("factor_names must match the number of factors.")

        for name in (
            "previous_weights",
            "benchmark_weights",
            "old_weights",
            "realized_next_returns",
        ):
            value = getattr(self, name)
            if value is not None and value.size != n_assets:
                raise ValueError(f"{name} must match the number of stocks.")
        self.w_min = np.asarray(self.w_min, dtype=float)
        self.w_max = np.asarray(self.w_max, dtype=float)
        for name in ("w_min", "w_max"):
            value = getattr(self, name)
            if value.ndim > 1 or value.size not in {1, n_assets}:
                raise ValueError(f"{name} must be scalar or one value per stock.")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite.")
        if self.lambda_tracking_error > 0:
            if self.stock_covariance is None or self.benchmark_weights is None:
                raise ValueError(
                    "tracking error requires stock_covariance and benchmark_weights."
                )

    @property
    def n_assets(self) -> int:
        return int(self.stock_mapping.shape[0])

    @property
    def n_factors(self) -> int:
        return int(self.stock_mapping.shape[1])

    def to_returns_dict(self) -> dict[str, Any]:
        """Build the input dictionary expected by the QP compiler."""
        if self.mapping_mode == "factor_space":
            result = {
                "mean": self.factor_mean.copy(),
                "covariance": self.factor_covariance.copy(),
                "factor_mean": self.factor_mean.copy(),
                "factor_covariance": self.factor_covariance.copy(),
                "stock_mapping": self.stock_mapping.copy(),
            }
        else:
            result = {
                "mean": self.mean.copy(),
                "covariance": self.covariance.copy(),
                "stock_mapping": self.stock_mapping.copy(),
            }
        if self.stock_covariance is not None:
            result["stock_covariance"] = self.stock_covariance.copy()
        return result

    def to_parameters(self, backend: str) -> QPParameters:
        """Build explicit cuFOLIO parameters for this replay window."""
        return QPParameters(
            objective=self.objective,
            risk_free_rate=self.risk_free_rate,
            target_return=self.target_return,
            w_min=self.w_min.copy(),
            w_max=self.w_max.copy(),
            short_budget=self.short_budget,
            lambda_l1=self.lambda_l1,
            lambda_l2=self.lambda_l2,
            lambda_tracking_error=self.lambda_tracking_error,
            turnover_budget=self.turnover_budget,
            previous_weights=self.previous_weights,
            benchmark_weights=self.benchmark_weights,
            benchmark_l1_budget=self.benchmark_l1_budget,
            mapping_mode=self.mapping_mode,
            backend=backend,
            V=self.stock_mapping.copy(),
        )


@dataclass
class ReplaySolveResult:
    """Compiled problem and recovered weights from one backend solve."""

    window: PaperReplayWindow
    compiled: CompiledQP
    solution: QPSolution
    stock_weights: np.ndarray
    factor_weights: np.ndarray | None


def _metadata_for_window(window: PaperReplayWindow) -> dict[str, Any]:
    return {
        "schema_version": window.schema_version,
        "window_id": window.window_id,
        "rebalance_date": window.rebalance_date,
        "model_name": window.model_name,
        "objective": window.objective,
        "mapping_mode": window.mapping_mode,
        "risk_free_rate": window.risk_free_rate,
        "lambda_l1": window.lambda_l1,
        "lambda_l2": window.lambda_l2,
        "short_budget": window.short_budget,
        "target_return": window.target_return,
        "tickers": window.tickers,
        "factor_names": window.factor_names,
        "turnover_budget": window.turnover_budget,
        "benchmark_l1_budget": window.benchmark_l1_budget,
        "lambda_tracking_error": window.lambda_tracking_error,
        "old_objective_value": window.old_objective_value,
        "old_realized_portfolio_return": window.old_realized_portfolio_return,
        "notes": window.notes,
    }


def save_replay_window(window: PaperReplayWindow, path: str | Path) -> Path:
    """Save a replay window as a portable compressed NPZ artifact."""
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "metadata_json": np.asarray(json.dumps(_metadata_for_window(window))),
    }
    for name in _ARRAY_FIELDS:
        value = getattr(window, name)
        if value is not None:
            payload[name] = np.asarray(value)
    np.savez_compressed(path, **payload)
    return path


def _read_npz_value(data: Any, name: str) -> np.ndarray | None:
    if name not in data.files:
        return None
    value = np.asarray(data[name])
    return None if value.size == 0 else value


def load_replay_window(path: str | Path) -> PaperReplayWindow:
    """Load and validate one NPZ replay artifact."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        if "metadata_json" not in data.files:
            raise ValueError(f"{path} is missing metadata_json.")
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        arrays = {name: _read_npz_value(data, name) for name in _ARRAY_FIELDS}
    return PaperReplayWindow(**metadata, **arrays)


def solve_replay_window(
    window: PaperReplayWindow,
    backend: str = "osqp",
) -> ReplaySolveResult:
    """Solve one saved window using the explicitly requested backend."""
    if backend not in {"osqp", "cuopt"}:
        raise ValueError("backend must be 'osqp' or 'cuopt'.")
    compiled = compile_portfolio_qp(
        window.to_returns_dict(), window.to_parameters(backend)
    )
    if backend == "osqp":
        solution = solve_compiled_qp_osqp(compiled)
    else:
        solution = solve_compiled_qp_cuopt(compiled)
    stock_weights = compiled.recover_stock_weights(solution.x)
    factor_weights = compiled.recover_factor_weights(solution.x)
    return ReplaySolveResult(
        window=window,
        compiled=compiled,
        solution=solution,
        stock_weights=np.asarray(stock_weights, dtype=float),
        factor_weights=(
            None if factor_weights is None else np.asarray(factor_weights, dtype=float)
        ),
    )


def _weights_from_solution(value: Any) -> np.ndarray:
    if isinstance(value, ReplaySolveResult):
        return value.stock_weights
    if isinstance(value, QPSolution):
        return value.x
    if isinstance(value, dict):
        for key in ("stock_weights", "weights", "x"):
            if key in value:
                return np.asarray(value[key], dtype=float).reshape(-1)
    return np.asarray(value, dtype=float).reshape(-1)


def _violation(weights: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(
        max(
            np.max(lower - weights),
            np.max(weights - upper),
            0.0,
        )
    )


def compute_replay_diagnostics(
    old_solution: Any,
    new_solution: ReplaySolveResult | QPSolution | dict[str, Any],
    window: PaperReplayWindow,
) -> dict[str, Any]:
    """Compute parity, feasibility, and next-period return diagnostics."""
    if isinstance(new_solution, ReplaySolveResult):
        result = new_solution
        solution = result.solution
        weights = result.stock_weights
        compiled = result.compiled
    else:
        result = None
        solution = new_solution if isinstance(new_solution, QPSolution) else None
        weights = _weights_from_solution(new_solution)
        compiled = None

    old_weights = window.old_weights
    if old_solution is not None:
        old_weights = _weights_from_solution(old_solution)
    lower = np.broadcast_to(window.w_min, window.n_assets)
    upper = np.broadcast_to(window.w_max, window.n_assets)
    realized = (
        float(weights @ window.realized_next_returns)
        if window.realized_next_returns is not None
        else None
    )
    old_realized = window.old_realized_portfolio_return
    if old_realized is None and old_weights is not None and window.realized_next_returns is not None:
        old_realized = float(old_weights @ window.realized_next_returns)
    objective_value = (
        float(solution.objective_value)
        if solution is not None
        else None
    )
    if objective_value is None and compiled is not None:
        objective_value = compiled.objective_value(weights)
    old_objective = window.old_objective_value
    weight_distance = None
    weight_linf = None
    if old_weights is not None:
        weight_delta = weights - old_weights
        weight_distance = float(np.linalg.norm(weight_delta))
        weight_linf = float(np.max(np.abs(weight_delta)))
    turnover = None
    if window.previous_weights is not None:
        turnover = float(np.sum(np.abs(weights - window.previous_weights)))
    short_violation = max(float(-np.sum(np.minimum(weights, 0.0))) - float(window.short_budget or 0.0), 0.0)
    diagnostic = {
        "status": solution.status if solution is not None else "unknown",
        "raw_status": solution.raw_status if solution is not None else None,
        "objective_value": objective_value,
        "old_objective_value": old_objective,
        "objective_gap": (
            None
            if objective_value is None or old_objective is None
            else objective_value - old_objective
        ),
        "relative_objective_gap": (
            None
            if objective_value is None or old_objective is None
            else (objective_value - old_objective) / max(abs(old_objective), 1e-12)
        ),
        "weight_l2_distance": weight_distance,
        "weight_linf_distance": weight_linf,
        "stock_weight_l2_distance": weight_distance,
        "stock_weight_linf_distance": weight_linf,
        "sum_weights": float(np.sum(weights)),
        "gross_long": float(np.sum(np.maximum(weights, 0.0))),
        "gross_short": float(np.sum(np.maximum(-weights, 0.0))),
        "short_budget_violation": short_violation,
        "box_violation": _violation(weights, lower, upper),
        "max_constraint_violation": (
            float(solution.max_constraint_violation)
            if solution is not None
            else None
        ),
        "realized_next_return": realized,
        "old_realized_portfolio_return": old_realized,
        "return_difference_vs_old": (
            None if realized is None or old_realized is None else realized - old_realized
        ),
        "turnover": turnover,
        "solve_time": solution.solve_time if solution is not None else None,
        "total_time": solution.total_time if solution is not None else None,
    }
    return diagnostic


def _window_paths(input_dir: Path) -> list[Path]:
    return sorted(input_dir.rglob("*.npz"))


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=_json_default) + "\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value)!r}.")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, np.ndarray)):
        return json.dumps(np.asarray(value).tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _summary_markdown(rows: list[dict[str, Any]]) -> str:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["backend"]), str(row["model_name"])), []).append(row)
    lines = [
        "# Paper Replay Summary",
        "",
        "Rows are matrix-replay diagnostics. They are not full empirical replication results.",
        "",
        "| backend | model | rows | optimal | success rate | mean return | mean weight L2 distance | max constraint violation |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (backend, model), group in sorted(grouped.items()):
        successful = [row for row in group if row.get("status") == "optimal"]
        returns = [row["realized_next_return"] for row in successful if row.get("realized_next_return") is not None]
        distances = [row["weight_l2_distance"] for row in successful if row.get("weight_l2_distance") is not None]
        violations = [row["max_constraint_violation"] for row in successful if row.get("max_constraint_violation") is not None]
        def mean(values: list[float]) -> float | None:
            return float(np.mean(values)) if values else None
        lines.append(
            f"| {backend} | {model} | {len(group)} | {len(successful)} | "
            f"{len(successful) / len(group) if group else 0.0:.3f} | {mean(returns)} | "
            f"{mean(distances)} | {max(violations) if violations else None} |"
        )
    return "\n".join(lines) + "\n"


def _solve_replay_path(job: tuple[str, str, bool]) -> dict[str, Any]:
    """Solve one replay path in a worker process without changing QP semantics."""
    path_string, requested_backend, compare_old = job
    path = Path(path_string)
    window = load_replay_window(path)
    base = {
        "window_id": window.window_id,
        "rebalance_date": window.rebalance_date,
        "model_name": window.model_name,
        "objective": window.objective,
        "mapping_mode": window.mapping_mode,
        "backend": requested_backend,
        "source_path": str(path),
    }
    try:
        result = solve_replay_window(window, requested_backend)
    except GPUBackendUnavailable as exc:
        return {**base, "status": "skipped", "skip_reason": str(exc)}
    except Exception as exc:  # preserve one failed window for audit
        return {**base, "status": "failed", "error": str(exc)}
    old_solution = window.old_weights if compare_old else None
    return {**base, **compute_replay_diagnostics(old_solution, result, window)}


def _basic_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["backend"]), str(row["model_name"])), []).append(row)
    metrics = []
    for (backend, model), group in sorted(grouped.items()):
        successful = [row for row in group if row.get("status") == "optimal"]
        returns = [row["realized_next_return"] for row in successful if row.get("realized_next_return") is not None]
        distances = [row["weight_l2_distance"] for row in successful if row.get("weight_l2_distance") is not None]
        violations = [row["max_constraint_violation"] for row in successful if row.get("max_constraint_violation") is not None]
        metrics.append(
            {
                "backend": backend,
                "model_name": model,
                "rows": len(group),
                "successful_rows": len(successful),
                "success_rate": len(successful) / len(group) if group else 0.0,
                "mean_realized_next_return": float(np.mean(returns)) if returns else None,
                "mean_weight_l2_distance": float(np.mean(distances)) if distances else None,
                "max_constraint_violation": max(violations) if violations else None,
            }
        )
    return metrics


def run_replay_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    backend: str = "osqp",
    models: list[str] | None = None,
    max_windows: int | None = None,
    compare_old: bool = False,
    write_summary: bool = False,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """Replay all NPZ windows and write per-window diagnostics."""
    if backend not in {"osqp", "cuopt", "both"}:
        raise ValueError("backend must be osqp, cuopt, or both.")
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    paths = _window_paths(Path(input_dir))
    selected = []
    for path in paths:
        window = load_replay_window(path)
        if models and window.model_name not in models:
            continue
        selected.append((path, window))
    if max_windows is not None:
        selected = selected[:max_windows]
    backends = ["osqp", "cuopt"] if backend == "both" else [backend]
    jobs = [
        (str(path), requested_backend, compare_old)
        for path, _ in selected
        for requested_backend in backends
    ]
    if workers == 1:
        rows = [_solve_replay_path(job) for job in jobs]
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            rows = list(executor.map(_solve_replay_path, jobs))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "per_window_results.csv", rows)
    _write_jsonl(output / "per_window_results.jsonl", rows)
    _write_csv(output / "metrics.csv", _basic_metrics(rows))
    if write_summary:
        (output / "summary.md").write_text(_summary_markdown(rows), encoding="utf-8")
    return rows
