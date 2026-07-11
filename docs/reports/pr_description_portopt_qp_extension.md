# Add PortOpt unified QP extension with direct cuOpt backend

## Summary

This change ports the PortOpt unified QP framework into cuFOLIO as a
complementary optimization track. It provides a direct cuOpt QP backend,
explicit OSQP validation, stock-level and factor-space workflows, examples,
notebooks, benchmark runners, and documentation.

The existing cuFOLIO Mean-CVaR LP workflow is unchanged.

## Motivation

cuFOLIO provides a GPU-accelerated scenario-based Mean-CVaR scaffold. PortOpt
also needs a unified QP path for variance, expected return, max-Sharpe,
regularization, practical portfolio constraints, and factor covariance models.
This extension makes that path explicit without conflating the two formulations.

## Main Features

- Minimum-variance, mean-variance, target-return, and max-Sharpe QPs.
- l1, l2^2, and combined l1+l2^2 regularization.
- Long-short, turnover, benchmark l1 exposure, and linear factor constraints.
- Ordinary and homogeneous max-Sharpe tracking-error penalties.
- Stock-space and true factor-space workflows through `V`.
- Deterministic PCA factor workflow and external factor adapter.
- Direct cuOpt backend with no CPU fallback for `backend="cuopt"`.
- CPU-first examples, notebooks, benchmark runners, and validation records.

## Mathematical Conventions

The compiled objective is:

```text
0.5 * x.T @ Q @ x + q.T @ x
```

The cuOpt adapter uses `Q_cuopt = 0.5 * Q`. The l1 split is
`w_minus = -min(0,w) = max(-w,0)`. l2^2 adds
`2 * lambda_l2 * V.T @ V` to the compiled `Q`. Max-Sharpe tracking error uses
the homogeneous form over `[x_tilde, c]` rather than inserting the ordinary
QP benchmark linear term into `q`.

## Backend Behavior

OSQP is used only when explicitly selected for CPU validation. cuOpt is used
only when explicitly selected for GPU solving. Variable bounds are always
passed explicitly, and an unavailable cuOpt runtime raises an error rather
than falling back to a CPU backend.

## Tests

The Sprint 11 CPU gate runs:

```bash
uv sync --extra dev
uv run python scripts/smoke_qp_env.py
uv run python -m compileall -q src tests scripts examples benchmarks
uv run pytest tests/test_qp_public_api.py -q
uv run pytest tests/test_qp_examples_smoke.py -q
uv run pytest tests/test_qp_benchmarks_smoke.py -q
uv run pytest -m "not gpu" -q
uv run ruff check src tests examples scripts benchmarks
```

## GPU Validation

Sprint 10 B40 validation used CUDA 13.2 and cuOpt 26.4.0. The targeted
max-Sharpe tracking-error and medium friction tests passed with `11 passed,
16 deselected`. The post-patch B40 stock-level and factor-space `backend=both`
benchmark subsets were all optimal. Sprint 11 does not rerun a GPU job; the
existing B40 results remain the cited GPU evidence.

## Benchmark Artifacts

The benchmark infrastructure covers stock-level, factor-space, and rolling
window runners. Sprint 9 CPU artifacts are under
`artifacts/benchmarks/sprint9-cpu-final/`; Sprint 10 math-patch artifacts are
under `artifacts/benchmarks/sprint10-math-patch/` when generated locally. Any
ratios are artifact-specific observations only and are not global speedup
claims.

## Limitations

This PR does not claim full CRSP/Compustat/IPCA/AP-Trees replication, a full
IPCA estimator, hard tracking-error constraints as ordinary QPs, production
per-asset turnover limits, or global QP speedups. Factor adapters consume
supplied factor outputs; they do not estimate the upstream research models.

## Reviewer Checklist

- [ ] No CPU fallback occurs when `backend="cuopt"`.
- [ ] Variable bounds are passed explicitly to cuOpt.
- [ ] `Q_cuopt = 0.5 * Q` is tested against the compiled convention.
- [ ] Ordinary and max-Sharpe tracking-error penalties are tested.
- [ ] Factor-space risk-free row and stock mapping are tested.
- [ ] Mean-CVaR LP workflow is untouched.
- [ ] No QP speedup claim is made without reviewed artifacts.
- [ ] No full IPCA/AP-Trees/CRSP/Compustat replication claim is made.

