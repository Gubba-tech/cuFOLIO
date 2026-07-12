# Sprint 12 Paper Replay Validation

Date: 2026-07-11

## Scope

Sprint 12 adds a saved-matrix replay pilot for the original PortOpt/IPCA-style
QP inputs. It adds the artifact schema, loader, solver adapter, diagnostics,
export audit script, replay runner, metrics/plot summarizer, a synthetic PCA
factor-space fixture, and documentation. It does not reimplement IPCA or
modify the existing Mean-CVaR LP workflow.

## Provenance

Implementation commit:

```text
53cd02c Add paper-style QP replay pilot
```

After this validation record is committed, its commit is the Sprint 12
provenance commit and is tagged `sprint12-green-<provenance_commit>`.

Sprint 11 remains available at `sprint11-green-8125c93`. The QP conventions
remain `0.5*x.T@Q*x + q.T@x` and `Q_cuopt = 0.5*Q`; no cuOpt-to-CPU fallback
was added.

## Original PortOpt/IPCA Audit

The local path `../PortOpt_IPCA` was unavailable. The public repository was
cloned temporarily at source commit
`8b2a8a890364a435139bc74cefc8911f5cbfb129`. Its dry-run audit found:

```text
python_files=10
candidate_matrix_files=0
```

The audit located the old OSQP solver and max-Sharpe assembly in
`PortOpt_factor/optimizer/pyport.py`, PCA/RP-PCA factor moments in
`pca_rppca.py`, and IPCA `Gamma`, factor moments, mapping, and rolling-window
logic in `ipca.py`. The source uses `K=6`, `window=240`, `longShort=0.2`,
`maxAlloc=0.08`, `riskfree=0`, and logarithmic `g1/g2` grids from `1e-6` to
`5`.

No saved replay matrices, old weights, realized return paths, CRSP/Compustat
data, or Table 2 artifact were present. The requested local dry-run therefore
reported:

```text
PortOpt root not found: ../PortOpt_IPCA
No original files can be audited from this path.
dry_run=true; no replay windows were written
```

No original PortOpt/IPCA code was changed.

## CPU Validation

Environment setup:

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

uv run pytest tests/test_qp_paper_replay.py -q
    3 passed, 1 skipped

uv run pytest -m "not gpu" -q
    226 passed, 2 skipped, 67 deselected, 3 warnings

uv run ruff check src tests examples scripts benchmarks
    All checks passed
```

The three warnings are existing runtime/solver warnings and did not fail the
suite.

## Synthetic Replay

Tracked fixture:

```text
tests/fixtures/paper_replay/synthetic_pca_k3_window.npz
```

Commands:

```text
uv run python scripts/run_paper_replay.py \
    --input-dir tests/fixtures/paper_replay \
    --output-dir artifacts/paper_replay/synthetic_results \
    --backend osqp --write-summary
    replay_rows=1

uv run python scripts/summarize_paper_replay.py \
    --input-dir artifacts/paper_replay/synthetic_results --plots
    metric_rows=1
```

The single factor-space PCA-style max-Sharpe window solved to `optimal` with
fully invested weights, finite diagnostics, and maximum constraint violation
below `3e-8`. The output contains per-window CSV/JSONL, metrics, summary, and
the optional cumulative, underwater, monthly-return heatmap, weight-distance,
and constraint-violation plots.

## Backend and GPU Result

The CPU login environment has no cuOpt runtime. Running the replay runner with
`--backend cuopt` produced one row with:

```text
status=skipped
reason=cuOpt GPU runtime unavailable; ... do not substitute a CPU solver ...
```

This is the required no-fallback behavior. No Sprint 12 GPU job was run. The
existing Sprint 10 B40 validation remains the prior GPU evidence; no H200
Sprint 12 result is claimed.

## Artifacts and Claims

Generated replay outputs are under the gitignored directory:

```text
artifacts/paper_replay/
```

The schema, audit, Table 2 target checklist, and usage guide are under
`docs/paper_replay/`. Table 2 values are recorded as future targets only; they
are not matched by this synthetic run.

This sprint makes no claim of old-solution parity, full CRSP/Compustat/IPCA/
AP-Trees replication, or global QP speedup. Those claims require supplied old
matrices, exact rolling dates, realized returns, and the complete data/factor
construction pipeline. The existing Mean-CVaR LP workflow is untouched.

