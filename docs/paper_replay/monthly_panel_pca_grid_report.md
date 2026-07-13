# Monthly Panel PCA Grid Report

## Scope

This Sprint 16 report diagnoses the weak annualized Sharpe from the Sprint 15
monthly-panel PCA pilot and evaluates a paper-style sensitivity grid. The
experiment uses the uploaded `dfall_for_test.csv` through the existing managed
portfolio pipeline. It is empirical evidence for this data construction; it is
not a full replication of the paper, IPCA, AP-Trees, or an old solver result.

The raw panel remains external and is not committed. The managed portfolio
return artifact contains 330 characteristic-sorted portfolios, 275 months, and
zero missing managed returns in the diagnostic run.

## Math Audit

The compiled QP convention remains

```text
0.5 * x.T @ Q @ x + q.T @ x
```

and the cuOpt adapter receives `Q_cuopt = 0.5 * Q`. Variable bounds are passed
explicitly as `[-0.08, 0.08]`, with short budget `0.2`; no cuOpt default bounds
are used.

The PCA audit found an important estimator distinction. PCA directions must be
estimated from demeaned managed returns, but the optimizer must receive the
raw-return factor scores so that the factor mean is not silently set to zero.
The corrected implementation now computes

```text
V = PCA(demeaned managed returns)
factor_returns = managed_returns @ V
```

The factor mapping `V` maps factor weights to managed-portfolio weights. It does
not recover individual stock weights. The previous Sprint 15 exporter passed
`center=False`, so its PCA directions were estimated from uncentered returns;
those saved artifacts are retained as pre-audit results and are labeled that
way below. New grid and baseline artifacts use the corrected convention.

The audit is covered by
`tests/test_qp_pca_factor_mean_math.py`, including covariance consistency and
factor sign invariance for recovered managed weights.

## Data Diagnostics

The managed-portfolio diagnostic writes its full tables under
`artifacts/paper_replay/diagnostics/managed_portfolios/`:

- `managed_portfolio_summary.md`
- `managed_portfolio_return_stats.csv`
- `missingness_by_portfolio.csv`
- `covariance_condition_by_window.csv`
- `pca_explained_variance.csv`
- `pca_factor_mean_stats.csv`
- `portfolio_extreme_returns.csv`
- `portfolio_coverage.csv`
- `weight_concentration_by_portfolio.csv`

The full-sample PCA cumulative explained variance is approximately 0.8282,
0.8534, 0.8645, 0.8728, 0.8786, and 0.8967 for K=2, 3, 4, 5, 6, and 10.
The full-sample K=6 factor means are approximately
`[0.14434, -0.00486, 0.00138, -0.01602, -0.00510, -0.01142]`.
The minimum managed-portfolio membership count is 66, with median 88 and
maximum 105 assets per characteristic/bin group; no group is below the
five-asset threshold.

The per-window diagnostic script writes factor covariance eigenvalues and
condition numbers, factor means, max-Sharpe and scaled-budget training
diagnostics, exposure activity, and constraint violations to
`window_qp_diagnostics.json` and `window_qp_diagnostics.md`.

## Experiment Design

| Item | Value |
| --- | --- |
| Evaluation period | 2020-01-31 through 2022-12-31 requested |
| Realized windows | 35, ending 2022-11-30 because the next month is required |
| Lookback | 240 months |
| PCA dimensions | K=2, 3, 4, 5, 6 |
| L1 grid | `1e-6, 5.6e-6, 3.1e-5, 1.7e-4, 9.5e-4, 5.3e-3, 2.9e-2, 1.6e-1, 9e-1, 5.0` |
| L2 grid | Same 10 values |
| QP objective | Homogeneous max-Sharpe |
| Bounds | Explicit `w_min=-0.08`, `w_max=0.08` |
| Short budget | `0.2` |
| Weighting | Value-weighted managed portfolios |
| Backends | OSQP validation and direct cuOpt GPU execution |

The resumable grid runner is
`scripts/run_monthly_panel_pca_paper_grid.py`. The H200 submission wrapper is
`scripts/slurm_monthly_panel_pca_grid_h200.sh`; it can be resumed on another
GPU by pointing `GRID_OUTPUT_DIR` at the same artifact directory. The wrapper
restarts the Python process every 20 configurations because long cuOpt loops
can exhaust the process OpenMP thread resource even on an exclusive node.

## Results

The generated grid summary is the source of truth for the 500 configuration
comparison:

- B40 cuOpt grid: `artifacts/paper_replay/results/monthly_panel_pca_grid_2020_2022_240m_cuopt_b40/`
- H200 cuOpt grid: `artifacts/paper_replay/results/monthly_panel_pca_grid_2020_2022_240m_cuopt_h200/`
- CPU smoke grid: `artifacts/paper_replay/results/monthly_panel_pca_grid_cpu_smoke/`

Each completed grid directory contains `grid_results.csv`,
`grid_summary.md`, `last_completed.json`, per-configuration replay artifacts,
and (after post-processing) a `summary/` directory with ranked results,
heatmap data, and Matplotlib plots. The report intentionally does not copy
values from a partial checkpoint; use `grid_summary.md` after the Slurm job
metadata reports `run_status=0` and the row count is 500.

## Baseline Comparisons

`scripts/run_monthly_panel_pca_baselines.py` defines equal-weight, minimum
variance, mean-variance risk-aversion, unregularized PCA, L1-only, L2-only,
combined L1/L2, long-only, and long-short comparisons for both the 2020/240m
and 2005/60m designs. Its GPU wrapper is
`scripts/slurm_monthly_panel_pca_baselines_b40.sh`.

Infeasible or backend-failed windows are recorded as `status=failed` so that a
baseline comparison does not stop at the first problematic rolling window.
The baseline output is diagnostic and does not establish paper-level
performance.

## Interpretation Boundary

The Sprint 15 H200 result, annualized Sharpe approximately 0.486, is a short
35-window uploaded-panel pilot with a pre-audit PCA-direction convention. It
should not be compared numerically with the paper's approximately 1.85 Sharpe
without matching sample, managed portfolio construction, PCA estimator,
formation timing, risk-free treatment, objective, constraints, and evaluation
period. This Sprint can identify implementation/data effects and report
robustness on the uploaded panel; it cannot claim that the paper result has
been reproduced or explained by one single factor.

The existing Mean-CVaR workflow is unchanged. No global QP speedup claim is
made.
