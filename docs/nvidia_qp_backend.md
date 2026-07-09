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

The first QP commit adds:

- `QPParameters`
- deterministic sparse QP compiler
- explicit CPU validation backend named `osqp`
- cuOpt backend guard
- `QuadraticPortfolioOptimizer` skeleton returning cuFOLIO-style `(result_row, Portfolio)`

The current `backend="cuopt"` behavior intentionally raises a clear error when cuOpt is unavailable and does not fall back to CPU. Direct compiled-QP execution through cuOpt is the next implementation step after the sparse compiler API stabilizes.

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

## Next Backend Step

Translate `CompiledQP` into a cuOpt `Problem`:

1. Add one scalar cuOpt variable per compiled variable, with compiled lower/upper bounds.
2. Add equality rows from `A_eq x = b_eq` as `LinearExpression == rhs`.
3. Add inequality rows from `A_ineq x <= b_ineq` as `LinearExpression <= rhs`.
4. Add padded `QuadraticExpression` for `0.5 * x.T @ Q @ x`.
5. Add linear objective term `q.T @ x`.
6. Solve with `SolverSettings`.
7. Return raw `x`, status, objective value, solve time, and total time.

Tracking-error hard constraints remain out of the MVP QP backend because they are QCQP/SOCP constraints, not ordinary QP. The MVP supports tracking error as a quadratic objective penalty.
