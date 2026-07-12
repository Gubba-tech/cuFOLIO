# Sprint 12 Paper Replay

## Goal

Sprint 12 replays saved paper-style QP matrices through the new cuFOLIO PortOpt
QP framework. It asks whether the new OSQP path reproduces an old solution and
whether a direct cuOpt solve agrees with the new OSQP result when a GPU is
available.

Matrix replay validates the solver/formulation layer using paper-style inputs.
It is not full CRSP/Compustat/IPCA empirical replication unless the complete
data construction and factor-estimation pipeline is rerun.

## Audit Result

See [`portopt_ipca_audit.md`](portopt_ipca_audit.md). The public
`Gubba-tech/PortOpt_IPCA` clone contains the QP and factor-estimation source but
no saved replay matrices or raw data. It is not modified by Sprint 12.

## Export Old Inputs

First use dry-run audit mode. It lists Python functions/classes and candidate
matrix files without running the expensive estimator:

```bash
uv run python scripts/export_portopt_replay_windows.py \
  --portopt-root ../PortOpt_IPCA \
  --output-dir artifacts/paper_replay/windows \
  --models PCA IPCA --k-values 6 \
  --start-date 2005-01-31 --end-date 2005-12-31 --dry-run
```

If original saved matrices are available, write a window using the schema in
[`replay_artifact_schema.md`](replay_artifact_schema.md). For manually supplied
factor matrices, provide factor returns, `V`, factor covariance, and factor
mean files with `--factor-returns-file`, `--stock-mapping-file`,
`--factor-covariance-file`, and `--factor-mean-file`. Keep outputs under the
gitignored `artifacts/paper_replay/` directory and do not commit proprietary
CRSP/Compustat data.

## Run Replay

Run the synthetic or exported windows with OSQP:

```bash
uv run python scripts/run_paper_replay.py \
  --input-dir artifacts/paper_replay/windows \
  --output-dir artifacts/paper_replay/results/pilot \
  --backend osqp --compare-old --write-summary
```

For a paired GPU run, use `--backend both` on a cuOpt-capable machine. The
runner solves OSQP first and then cuOpt. A missing cuOpt runtime creates a
`skipped` row with a reason; it never substitutes OSQP for a requested cuOpt
solve.

## Summarize

```bash
uv run python scripts/summarize_paper_replay.py \
  --input-dir artifacts/paper_replay/results/pilot --plots
```

The summary writes `metrics.csv`, `summary.md`, and
`cumulative_returns.csv`, plus optional plots. Review statuses, objective gaps,
weight distances, constraint violations, and data provenance before comparing
returns.

## Interpreting Deviations

Differences can come from matrix ordering, factor sign/rotation, covariance
normalization, l1/l2 convention, max-Sharpe scaling, bounds, or the exact
rolling-window date set. The replay artifact must record those choices in
`notes`; do not treat a return-path difference as a solver bug until the
matrices and constraints are byte-for-byte aligned.

## Boundaries

- This sprint does not reimplement the IPCA estimator.
- No full CRSP/Compustat/IPCA/AP-Trees replication claim is made.
- No global QP speedup claim is made.
- The existing Mean-CVaR LP workflow is untouched.
- `CompiledQP` remains `0.5*x.T@Q*x + q.T@x`, and cuOpt receives `0.5*Q`.

