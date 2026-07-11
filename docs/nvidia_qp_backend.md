# NVIDIA QP Backend Route

Date: 2026-07-11

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

At Sprint 2 close, `l1`, long-short, max-Sharpe, and end-to-end `V` factor/managed-portfolio mapping remained next-step work. Sprint 3 updates the l1 status below.

Do not claim QP GPU speedups yet. Speedup claims require dedicated benchmark scripts and saved CSV artifacts.

## Sprint 3 Validation Status

Sprint 3 validates l1 regularization using explicit positive/negative auxiliary variables:

```text
p = V @ x
p = y_plus - y_minus
y_plus >= 0
y_minus >= 0
q[y_plus] += lambda_l1
q[y_minus] += lambda_l1
```

No complementarity constraint is added; the l1 objective makes simultaneous positive and negative parts suboptimal.

Sprint 3 validates:

- stock-level l1 regularization
- stock-level l1 plus l2-squared regularization
- identity-mapping long-only fully invested l1 behavior
- gross-exposure behavior when short positions are allowed by direct bounds
- generic `V` compiler behavior for l1 auxiliary dimensions, equality rows, and l1/l2 matrix terms

Generic `V` validation in Sprint 3 is compiler and small-QP validation. It is not an end-to-end production validation of an IPCA/PCA/AP-Trees managed-portfolio workflow.

## Sprint 4 Validation Status

Sprint 4 validates long-short budgets with independent position splits:

```text
p = V @ x
p = pos - neg
pos >= 0
neg >= 0
1.T @ pos <= 1 + short_budget
1.T @ neg <= short_budget
```

The `pos`/`neg` variables are separate from the `l1_pos`/`l1_neg` regularization
variables. No complementarity constraint is added. For a fixed portfolio, any
additional simultaneous positive and negative split only tightens the budget
constraints.

Sprint 4 validates:

- long-short budget constraints with explicit position bounds
- stock-level minimum-variance, mean-variance, and target-return QPs
- stock-level l1, l2-squared, and l1 plus l2-squared QPs with long-short budgets
- `short_budget=0` equivalence to direct long-only bounds
- generic `V` compiler and small-QP long-short behavior
- deterministic medium-size (`N=20`) long-short QP cases

Validation provenance is recorded in
`docs/validation/sprint4_long_short_qp_validation.md`.

This validation makes no QP speedup claim.

## Sprint 5 Validation Status

Sprint 5 validates maximum Sharpe ratio through the linear reparameterization:

```text
w_tilde = c * w
(mu - rf * 1).T @ w_tilde = 1
1.T @ w_tilde - c = 0
w = w_tilde / c
```

The compiled QP remains:

```text
minimize 0.5 * w_tilde.T @ Sigma @ w_tilde
         + lambda_l1 * ||V @ w_tilde||_1
         + lambda_l2 * ||V @ w_tilde||_2^2
```

Scaled box and long-short constraints use the same positive `c` variable. The
compiler performs a linear pre-check that the supplied portfolio domain admits
strictly positive excess return before producing the max-Sharpe QP. Recovery
rejects non-finite or non-positive `c` values.

For generic `V`, this implementation uses stock-space expected returns mapped
consistently into decision space:

```text
excess_mu_stock.T @ V @ z_tilde = 1
1.T @ V @ z_tilde - c = 0
```

Sprint 5 validates:

- stock-level maximum-Sharpe QP and positive-scale recovery
- scaled equality and box constraints
- scaled long-short budgets
- max-Sharpe l1, l2-squared, and l1 plus l2-squared regularization
- generic `V` compiler and small-QP behavior
- deterministic medium-size (`N=20`) cases
- optimizer output for recovered weights, scale, excess return, status, and objective

Validation provenance is recorded in
`docs/validation/sprint5_max_sharpe_qp_validation.md`.

This validation makes no QP speedup claim.

## Sprint 6 Validation Status

Sprint 6 validates the remaining practical linear and quadratic-objective
constraints:

- total turnover budgets with `turnover_pos`/`turnover_neg`
- benchmark l1 exposure budgets with `benchmark_pos`/`benchmark_neg`
- linear factor exposure lower and upper bounds
- tracking-error quadratic objective penalties
- stock-level ordinary QPs and max-Sharpe scaled QPs
- generic `V` compiler and small-QP behavior

Turnover and benchmark auxiliary variables remain distinct from l1 and
long-short variables. Tracking error is implemented only as:

```text
lambda_tracking_error * (p - benchmark).T @ Sigma @ (p - benchmark)
```

with the corresponding QP matrix/vector terms. A hard tracking-error bound is
QCQP/SOCP, not an ordinary QP, and is not production-ready. Individual
per-asset turnover limits are future work; Sprint 6 validates only the total
turnover budget.

Validation provenance is recorded in
`docs/validation/sprint6_friction_constraints_validation.md`.

This validation makes no QP speedup claim.

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
- Sprint 3 validation covers l1 regularization and l1 plus l2-squared regularization.
- Individual per-asset turnover limits are not production-ready.
- End-to-end IPCA/PCA/AP-Trees factor workflows are not production-ready yet.
- Rolling-window benchmarks are not production-ready yet.
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
uv run pytest tests/test_qp_l1_regularization.py -q
uv run pytest tests/test_qp_l1_mapping.py -q
uv run pytest tests/test_qp_l1_l2_regularization.py -q
uv run pytest tests/test_qp_backend_medium_l1.py -q
uv run pytest tests/test_qp_long_short_constraints.py -q
uv run pytest tests/test_qp_long_short_solve.py -q
uv run pytest tests/test_qp_long_short_mapping.py -q
uv run pytest tests/test_qp_backend_medium_long_short.py -q
uv run pytest tests/test_qp_max_sharpe_reparameterization.py -q
uv run pytest tests/test_qp_max_sharpe_constraints.py -q
uv run pytest tests/test_qp_max_sharpe_regularization.py -q
uv run pytest tests/test_qp_max_sharpe_mapping.py -q
uv run pytest tests/test_qp_max_sharpe_validation.py -q
uv run pytest tests/test_qp_backend_medium_max_sharpe.py -q
uv run pytest tests/test_qp_turnover_constraints.py -q
uv run pytest tests/test_qp_turnover_solve.py -q
uv run pytest tests/test_qp_benchmark_constraints.py -q
uv run pytest tests/test_qp_benchmark_solve.py -q
uv run pytest tests/test_qp_turnover_benchmark_mapping.py -q
uv run pytest tests/test_qp_factor_exposure_constraints.py -q
uv run pytest tests/test_qp_tracking_error_penalty.py -q
uv run pytest tests/test_qp_backend_medium_friction_constraints.py -q
uv run pytest -m "not gpu" -q
```

GPU validation on a cuOpt-capable B40/H200 node:

```bash
# Choose one CUDA extra from nvidia-smi output. Do not mix them.
uv sync --extra cuda12 --extra dev
# or
uv sync --extra cuda13 --extra dev

uv run pytest -m gpu tests/test_qp_cuopt_backend.py tests/test_qp_mean_variance.py tests/test_qp_target_return.py tests/test_qp_l2_regularization.py tests/test_qp_backend_medium.py tests/test_qp_l1_regularization.py tests/test_qp_l1_mapping.py tests/test_qp_l1_l2_regularization.py tests/test_qp_backend_medium_l1.py tests/test_qp_long_short_constraints.py tests/test_qp_long_short_solve.py tests/test_qp_long_short_mapping.py tests/test_qp_backend_medium_long_short.py tests/test_qp_max_sharpe_reparameterization.py tests/test_qp_max_sharpe_constraints.py tests/test_qp_max_sharpe_regularization.py tests/test_qp_max_sharpe_mapping.py tests/test_qp_backend_medium_max_sharpe.py tests/test_qp_turnover_solve.py tests/test_qp_benchmark_solve.py tests/test_qp_turnover_benchmark_mapping.py tests/test_qp_factor_exposure_constraints.py tests/test_qp_tracking_error_penalty.py tests/test_qp_backend_medium_friction_constraints.py -q
```

## Next Implementation Order

1. End-to-end V factor workflows for PCA/RP-PCA/IPCA/AP-Trees.
2. cuFOLIO examples and notebooks.
3. Benchmark scripts and saved CSV/JSON artifacts.
4. README/project report speedup discussion only after benchmark artifacts exist.
