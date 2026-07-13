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

Sprint 13 adds the cleaned-data bridge. Start with
[`data_availability_checklist.md`](data_availability_checklist.md) and
[`cleaned_data_schema.md`](cleaned_data_schema.md). If the real inputs are not
available, see [`data_blockers_for_full_replication.md`](data_blockers_for_full_replication.md).

Sprint 14 uses the uploaded monthly stock-characteristic panel as the primary
empirical dataset. JKP is not used in Sprint 14. See
[`monthly_characteristic_panel_schema.md`](monthly_characteristic_panel_schema.md)
and [`monthly_panel_pilot.md`](monthly_panel_pilot.md) for the schema,
conversion, managed-portfolio, PCA, and QP pilot workflow.

The uploaded panel covers 2000 through 2022-12 with approximately 700 stocks
per month in the intended universe. Since the original paper uses a 20-year
lookback for a 2005 OOS start, exact 2005-start replication is impossible with
this date range. A shorter-lookback 2005 run is pilot-only; a 2020-start
20-year-lookback run has a shorter OOS period. Full IPCA and AP-Trees
replication, old-solution parity, and global QP speedup are not claimed.

## Cleaned Data Workflow

```bash
uv run python scripts/validate_paper_cleaned_data.py \
  --monthly-returns tests/fixtures/paper_data/monthly_returns.parquet \
  --characteristics tests/fixtures/paper_data/characteristics_monthly.parquet \
  --start-date 2004-01-31 --end-date 2005-12-31 \
  --lookback-months 12 \
  --report-path artifacts/paper_replay/synthetic_data_validation.md
uv run python scripts/build_amp_universe.py \
  --monthly-returns tests/fixtures/paper_data/monthly_returns.parquet \
  --characteristics tests/fixtures/paper_data/characteristics_monthly.parquet \
  --output-dir artifacts/paper_replay/synthetic_universe \
  --start-date 2004-01-31 --end-date 2005-12-31 --lookback-months 12
uv run python scripts/build_managed_portfolios.py \
  --monthly-returns tests/fixtures/paper_data/monthly_returns.parquet \
  --characteristics tests/fixtures/paper_data/characteristics_monthly.parquet \
  --universe-file artifacts/paper_replay/synthetic_universe/universe_by_date.parquet \
  --output-dir artifacts/paper_replay/synthetic_managed_portfolios \
  --lag-months 1 --start-date 2004-01-31 --end-date 2005-12-31
```

For a real pilot, use `--lookback-months 240`, `--k-values 6`, the documented
2005 dates, and approved lambda values. The exporter records that PCA `V`
maps factor weights to managed-portfolio weights; it must not label those as
individual stock weights.

## Sprint 15 Empirical Results

Sprint 15 runs the external `dfall_for_test.csv` panel through the monthly
managed-portfolio PCA workflow and records Table-2-style metrics, diagnostics,
plots, K sensitivity, lambda sensitivity, and a real cuOpt GPU execution. The
raw panel is not committed. Use the inventory and report for exact artifact
paths and claims boundaries:

- [`sprint14_artifact_inventory.md`](sprint14_artifact_inventory.md)
- [`monthly_panel_empirical_results.md`](monthly_panel_empirical_results.md)
- [`table2_targets.md`](table2_targets.md)
- [`../validation/sprint15_monthly_panel_empirical_results_validation.md`](../validation/sprint15_monthly_panel_empirical_results_validation.md)

The full 2005-2022 60-month run produces 215 realized-return windows because
the 2022-12 rebalance has no next-month return. The 2020-2022 240-month K=6
baseline is 35 windows; OSQP and both B40 and H200 cuOpt jobs solve all 35. The
2005 K sensitivity is a controlled 12-window sample, and the lambda grid is a
controlled 12-window sample per pair. Neither design is a full paper
replication.

## Sprint 16 PCA Audit And Grid

Sprint 16 corrects the PCA direction/mean separation, adds managed-portfolio
and per-window diagnostics, and runs a resumable K=2..6, 10x10 L1/L2 grid with
explicit variable bounds. See
[`monthly_panel_pca_grid_report.md`](monthly_panel_pca_grid_report.md) and
[`../validation/sprint16_monthly_panel_pca_grid_validation.md`](../validation/sprint16_monthly_panel_pca_grid_validation.md).

Summarize a saved run with optional matplotlib plots:

```bash
uv run python scripts/summarize_monthly_panel_results.py \
  --input-dir artifacts/paper_replay/results/<run_id> --plots
```

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

Export PCA replay windows from the managed portfolio table:

```bash
uv run python scripts/export_pca_replay_windows.py \
  --managed-portfolio-returns artifacts/paper_replay/synthetic_managed_portfolios/managed_portfolio_returns.parquet \
  --output-dir artifacts/paper_replay/synthetic_windows_pca \
  --k-values 3 --start-date 2005-01-31 --end-date 2005-12-31 \
  --lookback-months 12 --lambda-l1 0.01 --lambda-l2 0.01
```

Externally supplied IPCA/RP-PCA/AP-Trees outputs can use
`scripts/export_external_factor_replay_windows.py` with factor returns, a
mapping matrix, and asset/managed returns.

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
