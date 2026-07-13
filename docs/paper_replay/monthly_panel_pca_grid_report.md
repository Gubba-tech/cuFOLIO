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

The H200 job 46311 is submitted but remains pending on Slurm priority; no H200
rows are included in the completed B40 conclusions above.

Each completed grid directory contains `grid_results.csv`,
`grid_summary.md`, `last_completed.json`, per-configuration replay artifacts,
and (after post-processing) a `summary/` directory with ranked results,
heatmap data, and Matplotlib plots. The report intentionally does not copy
values from a partial checkpoint; use `grid_summary.md` after the Slurm job
metadata reports `run_status=0` and the row count is 500.

### Completed B40 Grid

The B40 cuOpt run is complete and is the current full-grid result:

| Item | Result |
| --- | ---: |
| Slurm job | 46312 on `b40x4-03` |
| Configurations | 500 / 500 |
| Windows per configuration | 35 / 35 optimal for every row |
| Failed or skipped windows | 0 |
| Maximum constraint violation | `2.1e-8` |
| Runtime | 5,941 seconds |
| cuOpt / CUDA extra | 26.04.000 / `cuda13` |

The best B40 row by annualized Sharpe is `K=2`, `lambda_l1=5.0`,
`lambda_l2=1e-6`: annualized Sharpe `0.558993`, CAGR `0.105144`, maximum
drawdown `-0.192737`, and Calmar `0.545531`. The best row for each K is:

| K | lambda_l1 | lambda_l2 | Annualized Sharpe | CAGR | Max drawdown |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 5.0 | 1e-6 | 0.558993 | 0.105144 | -0.192737 |
| 3 | 1e-6 | 5.0 | 0.555446 | 0.103736 | -0.197230 |
| 4 | 1e-6 | 5.0 | 0.557195 | 0.103827 | -0.196394 |
| 5 | 1e-6 | 5.0 | 0.556391 | 0.103562 | -0.196554 |
| 6 | 1e-6 | 5.0 | 0.556067 | 0.103442 | -0.196664 |

This is a descriptive result for the uploaded 35-window panel, not a paper
hyperparameter selection claim. The exact Sprint 15 pair
`lambda_l1=1.7e-4`, `lambda_l2=1e-3` is not in the current logarithmic grid;
the closest tested L2 value is `9.5e-4`. For K=6, that closest row has
annualized Sharpe `0.425782`, CAGR `0.070347`, and maximum drawdown
`-0.215637`.

### Constraint Sensitivity

Job 46325 tested the top B40 grid row over four short budgets
(`0.0, 0.1, 0.2, 0.5`) and four symmetric box bounds
(`+/-0.05, +/-0.08, +/-0.10, +/-0.20`). All 16 configurations completed
35/35 optimal windows. The box bounds were not active in any configuration.
The `short_budget=0` rows were effectively long-only and produced annualized
Sharpe `0.560529`; all positive short-budget rows produced `0.558993` with
average gross short exposure about `0.000240`. Thus the tested box bounds do
not explain the weak result, while allowing a small amount of short exposure
changes this selected row only marginally.

## Baseline Comparisons

`scripts/run_monthly_panel_pca_baselines.py` defines equal-weight, minimum
variance, mean-variance risk-aversion, unregularized PCA, L1-only, L2-only,
combined L1/L2, long-only, and long-short comparisons for both the 2020/240m
and 2005/60m designs. Its GPU wrapper is
`scripts/slurm_monthly_panel_pca_baselines_b40.sh`.

The B40 baseline job 46327 completed in resumable two-configuration chunks in
643 seconds. It wrote 22 design/config metric rows and 2,750 window rows with
cuOpt 26.04.000 and `run_status=0`. The 2020/240m design completed 35/35
windows for every configuration. The 2005/60m design recorded one failed
window for most PCA variants and three for the long-only variant; each failure
was the explicit `max_sharpe requires a feasible portfolio with strictly
positive excess return` compilation guard in the 2009 crisis window, and the
rows remain in the output.

Selected annualized Sharpe results are:

| Design | Equal weight | Min variance | MV RA=0.1 | PCA K6 L1+L2 | PCA K6 long-only |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2020/240m | 0.5691 | 0.7297 | 0.5779 | 0.4258 | 0.5677 |
| 2005/60m | 0.6622 | 0.8031 | 0.8436 | 0.7207 | 0.6769 |

The 2020/240m baseline confirms that the weak result is specific to the
regularized factor-space PCA configuration in this uploaded-panel setup; the
managed-portfolio min-variance and mean-variance controls perform differently
under the same rolling dates and explicit constraints. This is a diagnostic
comparison, not evidence that one alternative is the paper's estimator.

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
