# NVIDIA QP Backend Route

Date: 2026-07-09

## Decision

The PortOpt unified QP extension will use the direct NVIDIA cuOpt Python API for production GPU solves.

This matches cuFOLIO's existing `MeanVariance` implementation in `src/mean_variance_optimizer.py`, which constructs cuOpt `Problem` objects with scalar variables, `LinearExpression`, and `QuadraticExpression`.

## Why Direct cuOpt Python API

- cuFOLIO already uses direct cuOpt Python API for quadratic Mean-Variance and variance-cap SOCP/QCQP workflows.
- The API exposes `QuadraticExpression`, which is the natural target for QP objectives.
- It keeps QP/SOCP code aligned with the existing cuFOLIO-native backend style.
- It avoids routing QP production solves through a second CVXPY modeling layer.

CVXPY plus `cp.CUOPT` remains part of the existing Mean-CVaR LP workflow, where cuFOLIO already uses CVXPY solver settings with PDLP.

## Current MVP Status

The QP MVP now includes:

- `QPParameters`
- deterministic sparse QP compiler
- explicit CPU validation backend named `osqp`
- direct cuOpt Python QP backend
- cuOpt backend guard with `GPUBackendUnavailable`
- `QuadraticPortfolioOptimizer` skeleton returning cuFOLIO-style `(result_row, Portfolio)`
- CPU OSQP validation tests and GPU cuOpt smoke tests

When cuOpt is unavailable, `backend="cuopt"` raises a clear error and does not fall back to CPU.

## Sprint 2 Validation Status

Sprint 2 validates the direct QP backend for:

- minimum variance
- mean variance
- target return
- l2-squared regularization

Validation compares the compiled `CompiledQP` objective and constraints against the explicit OSQP validation backend, and runs the same cases through cuOpt when a GPU/cuOpt runtime is available.

`l1`, long-short, max-Sharpe, and end-to-end `V` factor/managed-portfolio mapping remain next-step work and are not production-ready in the direct cuOpt QP backend yet. The generic `V.T @ V` l2 matrix increment is unit-tested, but that is not a production validation of full factor-mapping solves.

Do not claim QP GPU speedups yet. Speedup claims require dedicated benchmark scripts and saved CSV artifacts.

## CompiledQP Convention

`CompiledQP` represents:

```text
minimize 0.5 * x.T @ Q @ x + q.T @ x
```

cuOpt's quadratic objective uses:

```text
x.T @ Q_cuopt @ x + c.T @ x
```

Therefore the backend maps:

```text
Q_cuopt = 0.5 * Q
c = q
```

This convention is tested in `tests/test_qp_compiled_convention.py` and checked against the cuOpt objective value in `tests/test_qp_cuopt_backend.py`.

## CompiledQP to cuOpt Problem

The direct cuOpt backend translates `CompiledQP` as follows:

1. Create one scalar cuOpt variable for each `CompiledQP.variable_names` entry.
2. Always pass variable bounds explicitly. Do not rely on cuOpt defaults, because default-bound behavior can vary by API/version and portfolio QP variables may require negative lower bounds.
3. Build the quadratic objective from `0.5 * CompiledQP.Q`.
4. Build the linear objective from `CompiledQP.q`.
5. Translate `CompiledQP.A`, `row_lower`, and `row_upper` into cuOpt `LinearExpression` constraints:
   - equal finite lower/upper row bounds become `==`
   - finite lower-only rows become `>=`
   - finite upper-only rows become `<=`
   - two-sided rows become one lower and one upper constraint
6. Solve with cuOpt `SolverSettings`.
7. Return raw `x`, status, objective value, solve time, total time, max constraint violation, and per-variable values.

## CPU Validation Policy

CPU validation is allowed only when explicitly selecting:

```python
QPParameters(backend="osqp")
```

This path uses CVXPY's OSQP solver for correctness tests and numerical comparison. It must not be invoked automatically when a user selected `backend="cuopt"`.

## No-Fallback Rule

If a production QP solve requests cuOpt and the runtime is unavailable, cuFOLIO must raise `GPUBackendUnavailable`.

Do not silently use:

- OSQP
- CLARABEL
- SCS
- ECOS
- scipy optimizers
- any other CPU solver

This is required so reports and benchmarks cannot accidentally claim GPU results that were produced on CPU.

## Current Limitations

- Sprint 2 validation covers long-only minimum variance, mean variance, target return, and l2-squared regularization.
- `l1`, long-short, max-Sharpe, and end-to-end `V` factor/managed-portfolio mapping compile paths still need dedicated cuOpt validation tests before being called production-ready.
- Tracking-error hard constraints remain out of the MVP QP backend because they are QCQP/SOCP constraints, not ordinary QP. The MVP supports tracking error as a quadratic objective penalty.

## Validation Commands

CPU validation:

```bash
uv sync --extra dev
uv run python scripts/smoke_qp_env.py
uv run python -m compileall -q src tests scripts
uv run pytest tests/test_qp_compiled_convention.py -q
uv run pytest tests/test_qp_osqp_validation_backend.py -q
uv run pytest tests/test_qp_mean_variance.py -q
uv run pytest tests/test_qp_target_return.py -q
uv run pytest tests/test_qp_l2_regularization.py -q
uv run pytest tests/test_qp_backend_medium.py -q
uv run pytest -m "not gpu" -q
```

GPU validation on a cuOpt-capable B40/H200 node:

```bash
# Choose one CUDA extra from nvidia-smi output. Do not mix them.
uv sync --extra cuda12 --extra dev
# or
uv sync --extra cuda13 --extra dev

uv run pytest -m gpu tests/test_qp_cuopt_backend.py tests/test_qp_mean_variance.py tests/test_qp_target_return.py tests/test_qp_l2_regularization.py tests/test_qp_backend_medium.py -q
```

## Next Implementation Order

1. l1 regularization with positive/negative auxiliary variables.
2. Long-short constraints.
3. Max-Sharpe QP reparameterization and recovery.
4. End-to-end `V` factor/managed-portfolio mapping.
