# Sprint 10 Math Hardening Validation

Date: 2026-07-11

## Scope

Sprint 10 hardens the tracking-error mathematics before the advisor-facing
report. The max-Sharpe tracking-error penalty now uses the homogeneous scaled
form:

```text
lambda_te * (M @ x_tilde - c*b).T @ Sigma @ (M @ x_tilde - c*b)
```

Ordinary-QP tracking error is unchanged. The implementation keeps the
`0.5*x.T@Q*x + q.T@x` convention, adds a symmetric `[x_tilde, c]` block for
max-Sharpe, and does not add the ordinary-QP linear benchmark term to
max-Sharpe `q`.

## Provenance

Implementation commit:

```text
95180c5 Fix homogeneous max-Sharpe tracking error
```

The paper math audit is:

```text
docs/portopt_paper_math_audit.md
```

## CPU validation

Environment: Python 3.13.2, uv 0.11.28, OSQP validation backend.

```text
uv run python scripts/smoke_qp_env.py
    All checks passed

uv run python -m compileall -q src tests scripts examples benchmarks
    passed

uv run pytest tests/test_qp_tracking_error_penalty.py -q
    6 passed, 1 skipped

uv run pytest tests/test_qp_max_sharpe_tracking_error_penalty.py -q
    6 passed, 1 skipped

uv run pytest tests/test_qp_backend_medium_friction_constraints.py -q
    10 passed, 10 skipped

uv run pytest -m "not gpu" -q
    223 passed, 2 skipped, 66 deselected

uv run ruff check src tests examples scripts benchmarks
    passed
```

The GPU skips are expected on the login environment. The full CPU run retained
three pre-existing runtime/solver warnings and had no failures.

## B40 GPU validation

Slurm job `46085` used a B40 node with CUDA 13.2 from `nvidia-smi`,
`cuda13`, and `cuopt-cu13==26.4.0` in a job-local environment.

```text
uv run pytest -m gpu \
    tests/test_qp_max_sharpe_tracking_error_penalty.py \
    tests/test_qp_backend_medium_friction_constraints.py -q
    11 passed, 16 deselected
```

The same job ran `backend=both` post-patch benchmark subsets. Stock-level and
factor-space OSQP/cuOpt rows were all `optimal`; objective-gap and feasibility
fields were written to the artifact files.

## Post-patch benchmark artifacts

All post-patch results are under:

```text
artifacts/benchmarks/sprint10-math-patch/
```

CPU subset:

```text
stock_level_results.csv      4 rows, 4 optimal
factor_space_results.csv     2 rows, 2 optimal
```

B40 `backend=both` subset:

```text
stock_level_results.csv      8 rows, 8 optimal
factor_space_results.csv     2 rows, 2 optimal
```

The directory also contains `summary.csv`, `summary.json`, `summary.md`,
JSONL outputs, per-run Markdown summaries, and environment metadata. Any speed
ratios are scoped to this generated artifact run only.

## Math and paper alignment

- The ordinary tracking-error Q and q terms remain
  `2*lambda_te*M.T@Sigma@M` and `-2*lambda_te*M.T@Sigma@b`.
- Max-Sharpe uses the PSD block over `[M, -b]` and the scale variable.
- The l1 split is aligned with the paper:
  `w_minus = -min(0,w) = max(-w,0)`.
- Regularization coefficients are interpreted under the compiled `0.5` QP
  convention.
- No global QP speedup claim was added.
- No full CRSP/Compustat/IPCA/AP-Trees replication claim was added.
- The existing cuFOLIO Mean-CVaR LP workflow was untouched.
