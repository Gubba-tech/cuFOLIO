# Sprint 9 QP Benchmark Validation

Date: 2026-07-11

## Scope

Sprint 9 adds reproducible benchmark runners and artifact handling for:

- stock-level min-variance, mean-variance, max-Sharpe, regularization,
  long-short, friction, and full cases;
- true factor-space PCA and external-factor adapter cases;
- synthetic rolling-window repeated solves in stock and factor modes;
- compile/build/solve/postprocess/total timing and OSQP/cuOpt diagnostics;
- CSV, JSONL, JSON, and Markdown artifact writers and summarizer;
- CPU/GPU run scripts, Slurm templates, smoke tests, and artifact policy docs.

No QP math convention, cuOpt scaling, no-fallback rule, or Mean-CVaR workflow
was changed.

## Provenance

Implementation commit:

```text
3aaefda Add Sprint 9 QP benchmark artifacts
```

Sprint 8 tag hygiene was preserved: `sprint8-green-a95c354` was not moved.
The Sprint 8 `e3af2f0` lint cleanup remains a post-validation commit. Sprint 9
uses the current branch HEAD as its implementation provenance.

## CPU validation

Environment:

```text
Python 3.13.2
uv 0.11.28
CUDA/cuOpt unavailable on login1
```

Passed:

```text
uv run pytest tests/test_qp_benchmarks_smoke.py -q
    1 passed, 2 skipped

uv run pytest -m "not gpu" -q
    214 passed, 2 skipped, 63 deselected

uv run ruff check src tests examples scripts benchmarks
    passed

uv run python -m compileall -q src tests scripts examples benchmarks
    passed
```

The post-implementation CPU benchmark artifacts are under:

```text
artifacts/benchmarks/sprint9-cpu-final/
```

The runners generated and summarized:

```text
stock_level_results.csv      24 rows, 24 optimal
factor_space_results.csv      8 rows,  8 optimal
rolling_windows_results.csv  20 rows, 20 optimal
summary.csv                  33 summary rows
```

The environment metadata in these artifacts records git commit
`3aaefda2adb07512e0d7b6e61d1637f1fbc3fad8`.

## GPU validation

CUDA 13 was selected from `nvidia-smi`; the job-local environments used
`cuopt-cu13==26.4.0` and did not modify `uv.lock`.

B40 job 46082, after implementation commit:

```text
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
GPU smoke: 2 passed, 1 deselected
```

The B40 smoke covered the stock-level and factor-space cuOpt benchmark tests.

H200 NVL job 46081 ran the direct benchmark runners (outside the pytest
wrapper) successfully:

```text
stock-level cuOpt benchmark: optimal
factor-space cuOpt benchmark: optimal
GPU: NVIDIA H200 NVL
cuOpt: 26.4.0
```

The H200 pytest wrapper job 46080 was cancelled after a subprocess timeout
with no result; it is not counted as a passing H200 smoke result. Its direct
runner artifacts are under `artifacts/benchmarks/sprint9-direct-46081/` and
record the equivalent pre-commit source hash `e3af2f0` in their environment
metadata. No GPU result was fabricated or replaced with OSQP.

## Artifact and claim policy

Each runner writes environment metadata, dimensions/sparsity, statuses,
compile/build/solve/postprocess/total timings, objective/risk metrics, and
constraint violations. `--backend both` runs OSQP first and cuOpt second on
the same compiled problem. A missing cuOpt runtime is recorded as `skipped`.

The generated summary uses `observed speed ratio in this benchmark run` only
for matching successful OSQP/cuOpt solves. No global QP speedup claim was
added to README. The benchmark data are deterministic synthetic/public data
and do not claim CRSP/Compustat/IPCA/AP-Trees replication.

## Tag

After this provenance commit, create and push:

```text
sprint9-green-<provenance_commit>
```
