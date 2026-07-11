# PortOpt QP Extension in cuFOLIO

Advisor-facing technical report, Sprint 11, July 2026.

## 1. Executive Summary

This is a new fast project, not a revision of the original PortOpt paper. It
ports the PortOpt unified quadratic-programming framework into NVIDIA cuFOLIO
as a complementary optimization track. The extension adds a direct NVIDIA
cuOpt QP backend while keeping cuFOLIO's existing Mean-CVaR LP workflow intact.
It supports stock-level and true factor-space QP workflows, with OSQP retained
as an explicit CPU validation backend.

The validated scope includes minimum variance, mean variance, target return,
max-Sharpe reparameterization, regularization, long-short and friction
constraints, factor mappings, deterministic PCA factor data, and adapters for
externally supplied factor outputs.

## 2. Motivation

The original PortOpt work presented a unified QP interface for several
portfolio objectives and practical constraints. NVIDIA cuFOLIO provides a
GPU-accelerated portfolio scaffold centered on scenario-based Mean-CVaR LP
optimization. This project adds a complementary QP track for cases where a
quadratic risk model, expected-return objective, or factor covariance model is
the appropriate formulation.

The design keeps the two tracks explicit:

- Mean-CVaR remains the existing scenario-based LP workflow.
- PortOpt QP provides direct QP formulations and explicit backend selection.
- No QP backend silently falls back from `backend="cuopt"` to a CPU solver.

## 3. Relationship to the Original PortOpt Paper

The implementation follows the paper's portfolio modeling ideas while keeping
the cuFOLIO integration deliberately scoped:

| Paper concept | cuFOLIO QP extension |
| --- | --- |
| Minimum variance | Stock-level or factor-space quadratic risk minimization. |
| Mean variance | Expected return minus risk-aversion-weighted variance. |
| Max-Sharpe | Homogeneous reparameterization with a positive scale variable, followed by weight recovery. |
| l1 and l2^2 | Split-variable l1 penalties and quadratic `V.T @ V` regularization. |
| Long-short | Explicit nonnegative positive/negative portfolio splits and short budget. |
| `V` mapping | Stock weights are mapped from a primary variable with `p = V @ x`. |
| Rolling windows | Benchmark runners demonstrate repeated QP solves over synthetic windows. |

The external factor adapter accepts supplied factor returns and a stock mapping
matrix for RP-PCA, IPCA, or AP-Trees outputs. It does not implement or claim
full RP-PCA, IPCA, or AP-Trees estimation. Full CRSP/Compustat/IPCA/AP-Trees
empirical replication is not included.

## 4. Mathematical Formulation

### Objective convention

The internal `CompiledQP` convention is:

```text
0.5 * x.T @ Q @ x + q.T @ x
```

cuOpt represents the quadratic term as `x.T @ Q_cuopt @ x`, so the adapter
passes:

```text
Q_cuopt = 0.5 * Q
```

This factor is part of the backend contract and is tested independently.

### Regularization and splits

For an l1 term, the implementation uses positive and negative auxiliaries. The
negative part follows the paper convention:

```text
w_minus = -min(0, w) = max(-w, 0)
```

For l2-squared regularization, the compiled primary block receives:

```text
Q += 2 * lambda_l2 * V.T @ V
```

The factor of two is required by the leading `0.5` in the compiled objective.

### Max-Sharpe and factor-space rows

Max-Sharpe introduces `w_tilde` and `c`, enforces the excess-return equality,
and recovers portfolio weights as `w = w_tilde / c`. Stock-space and
factor-space rows use the same compiler contract. In factor space, the
risk-free row is:

```text
factor_mean.T @ z_tilde - risk_free_rate * c = 1
```

Stock constraints are applied through `V @ z`, while factor covariance and
factor mean remain in the factor-space objective.

### Constraints and tracking error

Long-short portfolios use `p = p_plus - p_minus` with nonnegative split
variables. Turnover and benchmark l1 exposure use separate split variables and
explicit budget rows. Linear factor exposures are compiled as paired upper and
lower linear inequalities.

For ordinary QPs, a tracking-error penalty is:

```text
lambda_te * (M @ x - b).T @ Sigma @ (M @ x - b)
```

For max-Sharpe, the scale must be homogeneous:

```text
lambda_te * (M @ x_tilde - c * b).T
    @ Sigma @ (M @ x_tilde - c * b)
```

The compiler adds the symmetric positive-semidefinite block over
`[x_tilde, c]`. It does not add the ordinary-QP linear benchmark term to the
max-Sharpe `q` vector. The full alignment review is in
[`docs/portopt_paper_math_audit.md`](../portopt_paper_math_audit.md).

Hard tracking-error limits are not represented as ordinary QP rows; they would
require a QCQP or SOCP treatment.

## 5. Implementation Architecture

- `QPParameters` in `src/qp_parameters.py` is the user-facing parameter model.
- `CompiledQP` and `compile_portfolio_qp` in `src/qp_formulations.py` build variables, objective blocks, rows, bounds, and recovery metadata.
- `OSQP` validation is implemented in `src/qp_backend.py` and is only used when explicitly requested.
- The direct cuOpt backend in `src/qp_backend.py` maps the compiled sparse problem to cuOpt and passes variable bounds explicitly.
- `QuadraticPortfolioOptimizer` in `src/qp_optimizer.py` returns cuFOLIO-compatible result rows and portfolios.
- `FactorModelQPData`, deterministic PCA construction, and external factor adapters are in `src/qp_factor_workflows.py`.
- `skills/portopt_qp/SKILL.md` records the implementation, validation, claim, and backend rules.

The benchmark runners under `benchmarks/` keep compilation, solver-build,
solve, postprocess, and total timing separate. Examples and notebooks default
to OSQP so that onboarding does not require a GPU; cuOpt is always an explicit
choice.

## 6. Validation Timeline

Sprint 1 has no independent validation record in this repository. The
available validation timeline is:

| Sprint | Scope | CPU result | GPU result | Tag |
| --- | --- | --- | --- | --- |
| 2 | Direct backend, basic objectives, l2^2 | 90 passed, 2 skipped | B40: 7 passed | `sprint2-green-d35ef15` |
| 3 | l1 and l1+l2^2 | 100 passed, 2 skipped | B40: 6 passed | `sprint3-green-27127c6` |
| 4 | Long-short budgets and mappings | 118 passed, 2 skipped | B40: 11 passed | `sprint4-green-4e5f945` |
| 5 | Max-Sharpe reparameterization | 147 passed, 2 skipped | B40: 10 passed | `sprint5-green-7fc1518` |
| 6 | Turnover, benchmark exposure, tracking penalty | 186 passed, 2 skipped | B40: 20 passed | `sprint6-green-dec0a52` |
| 7 | True factor-space and adapters | 205 passed, 2 skipped | B40: 4 passed | `sprint7-green-23e274a` |
| 8 | Examples, notebooks, product docs | 213 passed, 2 skipped | B40/H200 examples optimal; GPU smoke passed | `sprint8-green-a95c354` |
| 9 | Benchmark runners and artifact policy | 214 passed, 2 skipped | B40 smoke passed; H200 direct runners optimal, wrapper timeout not counted | `sprint9-green-d0ccbb8` |
| 10 | Homogeneous max-Sharpe tracking-error hardening | 223 passed, 2 skipped | B40: 11 passed | `sprint10-green-19eaa10` |

Sprint 10's documentation closeout commit `60ccd77` is after the validation
tag. The existing validation tag was not moved.

## 7. Examples and Notebooks

The six QP examples are:

1. `examples/qp_min_variance_quickstart.py`
2. `examples/qp_mean_variance_quickstart.py`
3. `examples/qp_max_sharpe_regularized_long_short.py`
4. `examples/qp_factor_space_pca_demo.py`
5. `examples/qp_external_factor_adapter_demo.py`
6. `examples/qp_vs_cvar_baseline_overview.py`

The six QP notebooks are under `notebooks/portopt_qp/`, covering an overview,
stock-level objectives, max-Sharpe regularization, PCA factor space, external
factor adapters, and QP versus Mean-CVaR.

All examples and notebooks use `backend="osqp"` by default. A cuOpt run must
select a CUDA extra and pass `--backend cuopt` explicitly; unavailable cuOpt
raises an error rather than falling back to CPU.

## 8. Benchmark Artifacts

Three benchmark paths are available:

- `benchmark_qp_stock_level.py` for stock-level cases;
- `benchmark_qp_factor_space.py` for PCA and external-factor cases;
- `benchmark_qp_rolling_windows.py` for repeated synthetic windows.

Sprint 9 CPU artifacts are in `artifacts/benchmarks/sprint9-cpu-final/` when
present. Sprint 10 math-patch artifacts are in
`artifacts/benchmarks/sprint10-math-patch/`. Generated raw artifacts are
gitignored and can be regenerated with the commands in
[`docs/qp_benchmarks.md`](../qp_benchmarks.md).

Sprint 10's B40 `backend=both` subset had successful OSQP and cuOpt rows for
the stock-level and factor-space cases. Its summary contains observed
OSQP/cuOpt ratios for that artifact run only. These values are not global QP
speedup claims and are not evidence of full-data or production performance.

## 9. Limitations

- No full CRSP/Compustat/IPCA/AP-Trees replication is included.
- No full IPCA estimator is implemented in this extension.
- RP-PCA, IPCA, and AP-Trees are accepted only through supplied factor outputs and mappings.
- A hard tracking-error constraint is QCQP/SOCP, not an ordinary QP row.
- Per-asset turnover limits are not production-ready unless separately implemented and validated.
- Benchmark inputs are deterministic synthetic/public data unless an artifact explicitly states otherwise.
- QP performance claims require reviewed, matching artifacts; no global QP speedup is claimed here.
- The existing Mean-CVaR LP workflow is outside this extension and remains unchanged.

## 10. Next Steps

1. Optional full IPCA/AP-Trees replication on a separately scoped data branch.
2. A rolling-window public-data demonstration with documented data provenance.
3. Review of B40/H200 benchmark artifacts before any performance statement.
4. A possible upstream PR to cuFOLIO.
5. A short workshop note or thesis-chapter appendix describing the QP track.

