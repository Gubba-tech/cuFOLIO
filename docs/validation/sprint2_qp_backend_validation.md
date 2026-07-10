# Sprint 2 QP Backend Validation

Date: 2026-07-10

## Scope

Sprint 2 validated the direct QP backend for:

- `min_variance`
- `mean_variance`
- `target_return`
- l2-squared regularization

Validated commit:

```text
d35ef15195cccccc9f1d38118b60144cfe300986 Validate Sprint 2 QP backends
```

## CPU Gate

CPU validation used a clean `uv` development environment:

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

Result:

```text
90 passed, 2 skipped, 8 deselected, 2 warnings
```

The warnings were existing runtime warnings in non-QP tests.

## GPU Gate

GPU validation ran on Slurm:

```text
JobID: 45433
Partition: debug-b40x4
Node: b40x4-02
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
Driver: 595.71.05
CUDA: 13.2
uv extra: cuda13
State: COMPLETED
ExitCode: 0:0
Elapsed: 00:00:41
```

GPU command:

```bash
uv sync --extra dev --extra cuda13
uv run python scripts/smoke_qp_env.py
uv run pytest -m gpu tests/test_qp_cuopt_backend.py tests/test_qp_mean_variance.py tests/test_qp_target_return.py tests/test_qp_l2_regularization.py tests/test_qp_backend_medium.py -q
```

Result:

```text
7 passed, 12 deselected
```

## Notes

- No QP speedup claim is made from Sprint 2 validation.
- The direct cuOpt backend preserves the `CompiledQP` objective convention: `0.5 * x.T @ Q @ x + q.T @ x`.
- The cuOpt mapping remains `Q_cuopt = 0.5 * Q`.
- `backend="cuopt"` must not silently fall back to OSQP, CVXPY, or any CPU solver.

Suggested tag command:

```bash
git tag sprint2-green-d35ef15 d35ef15195cccccc9f1d38118b60144cfe300986
git push gubba sprint2-green-d35ef15
```
