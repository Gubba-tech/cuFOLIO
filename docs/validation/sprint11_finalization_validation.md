# Sprint 11 Finalization Validation

Date: 2026-07-11

## Scope

Sprint 11 packages the validated PortOpt unified QP extension for advisor
review. It adds the technical report, spoken demo script, PR description,
release checklist, benchmark interpretation note, README pointers, and final
skill workflow. No new optimization feature was added.

## Provenance

The documentation and advisor deliverables were created from this branch and
base HEAD before this validation record:

```text
f323acb Add Sprint 11 advisor deliverables
```

The existing Sprint 10 validation tag was not moved:

```text
sprint10-green-19eaa10  validation provenance tag
60ccd77                 post-validation documentation closeout
```

After this record is committed, its commit is the Sprint 11 provenance commit
and is tagged `sprint11-green-<provenance_commit>`.

## CPU Validation

Environment synchronization:

```text
uv sync --extra dev
    Resolved 181 packages; checked 90 packages
```

Passed:

```text
uv run python scripts/smoke_qp_env.py
    Python 3.13.2; numpy 2.2.6; scipy 1.16.3; pandas 2.3.2;
    pyarrow 22.0.0; pytest 9.0.3; cvxpy 1.9.2

uv run python -m compileall -q src tests scripts examples benchmarks
    passed

uv run pytest tests/test_qp_public_api.py -q
    1 passed

uv run pytest tests/test_qp_examples_smoke.py -q
    7 passed, 2 skipped

uv run pytest tests/test_qp_benchmarks_smoke.py -q
    1 passed, 2 skipped

uv run pytest -m "not gpu" -q
    223 passed, 2 skipped, 66 deselected, 3 warnings

uv run ruff check src tests examples scripts benchmarks
    All checks passed
```

The three warnings are pre-existing runtime/solver warnings and did not cause
test failures.

The three advisor demo commands also ran successfully with explicit OSQP:

```text
qp_min_variance_quickstart.py              status: optimal
qp_max_sharpe_regularized_long_short.py    status: optimal
qp_factor_space_pca_demo.py                status: optimal
```

## GPU Validation

Sprint 11 did not submit a new GPU job. The cited GPU evidence remains Sprint
10 B40 job 46085, run with CUDA 13.2 and cuOpt 26.4.0:

```text
uv run pytest -m gpu \
    tests/test_qp_max_sharpe_tracking_error_penalty.py \
    tests/test_qp_backend_medium_friction_constraints.py -q
    11 passed, 16 deselected
```

Sprint 10 B40 `backend=both` stock-level and factor-space benchmark subsets
were all `optimal`. No H200 Sprint 11 result is claimed.

## Deliverables

- `docs/reports/portopt_cufolio_qp_technical_report.md`
- `docs/reports/advisor_demo_script.md`
- `docs/reports/pr_description_portopt_qp_extension.md`
- `docs/reports/release_checklist.md`
- `docs/reports/benchmark_interpretation_note.md`
- `docs/portopt_paper_math_audit.md`
- `docs/qp_benchmarks.md`
- `skills/portopt_qp/SKILL.md`
- `.agents/skills/portopt_qp/SKILL.md` mirror

README now points to the advisor report, demo script, PR description, release
checklist, benchmark interpretation note, and paper math audit.

## Benchmark Artifacts

The interpretation note references:

```text
artifacts/benchmarks/sprint9-cpu-final/
artifacts/benchmarks/sprint10-math-patch/
```

The artifacts are gitignored generated outputs. The note documents how to
regenerate them and how to distinguish total time, solve time, skipped rows,
and failed/infeasible rows.

## Explicit Boundaries

- No global QP speedup claim is made.
- No full CRSP/Compustat/IPCA/AP-Trees replication claim is made.
- External factor adapters consume supplied factor outputs; they do not implement the upstream estimators.
- Hard tracking-error constraints are not presented as ordinary QP support.
- The existing cuFOLIO Mean-CVaR LP workflow is untouched.

