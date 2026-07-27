# Current Model Description

## One-Sentence Definition

The current empirical model is a **rolling characteristic-managed-portfolio
PCA model followed by a factor-space homogeneous max-Sharpe quadratic program
(QP)**. It is a portfolio construction and optimization pipeline, not yet a
stock-level neural network, supervised return predictor, or full IPCA
estimator.

```text
monthly stock panel
  -> lagged characteristic sorts
  -> value-weighted managed portfolios
  -> rolling PCA of managed-portfolio returns
  -> factor mean/covariance
  -> regularized max-Sharpe QP
  -> managed-portfolio weights
  -> next-month realized return
```

The implementation is in:

- [`src/qp_monthly_panel.py`](../src/qp_monthly_panel.py): panel validation and
  managed-portfolio construction;
- [`scripts/export_pca_replay_windows.py`](../scripts/export_pca_replay_windows.py):
  rolling PCA windows;
- [`src/qp_factor_workflows.py`](../src/qp_factor_workflows.py): PCA factor
  mapping, factor means, and covariance;
- [`src/qp_formulations.py`](../src/qp_formulations.py): unified QP compiler;
- [`src/qp_paper_replay.py`](../src/qp_paper_replay.py): OSQP/cuOpt replay.

## Important Data Boundary

The current saved model experiments use the earlier external
`dfall_for_test.csv` panel, not the newly uploaded
`processed_data_may25.p` file.

| Item | Current model experiments |
| --- | --- |
| Source panel | External `dfall_for_test.csv` |
| Coverage | 2000-01-31 through 2022-12-31 |
| Rows | 241,347 |
| Unique assets | 1,089 |
| Characteristics available | 35; 33 default paper-style characteristics used |
| Managed portfolios | 33 characteristics x 10 bins = 330 |
| New May 2025 pickle | Described separately; not yet wired into this model or trained |

Therefore, the current model results must not be presented as results from the
new 1962-2025, 2.52-million-row panel. The new panel requires a separate
adapter, return-quality audit, timing confirmation, and a new run manifest.

## Stage 1: Managed-Portfolio Construction

For each characteristic `j` and formation month `t`:

1. Select the configured universe. The current paper-style runs use the
   available universe, with non-null non-negative market capitalization and a
   minimum of five assets per bin.
2. Read characteristic values at month `t`.
3. Rank the eligible assets cross-sectionally and split them into ten bins
   using `qcut`.
4. Weight each bin by month-`t` market capitalization. Equal weighting is
   supported, but value weighting is the current default.
5. Hold the membership fixed and realize the asset returns at month `t+1`.

For every characteristic this produces up to ten managed portfolios. The
result is a table of managed-portfolio returns, not individual-stock weights:

```text
managed_return[t+1, j_bin] = sum_i weight[t, i, j_bin] * stock_return[t+1, i]
```

The default current construction is therefore 330 managed portfolios. There
are no interactions between characteristics at this stage; each characteristic
creates its own independent decile portfolio family.

The timing contract is deliberately:

```text
sort on characteristic at t -> realize return at t+1
```

This avoids same-month return lookahead. The final raw month cannot produce a
realized next-month portfolio return and is excluded from the final replay
window.

## Stage 2: Rolling PCA

At each rebalance month, the model takes a rolling history of managed-portfolio
returns. With `T` history months and `M` complete managed portfolios, write the
history as:

```text
R in R^(T x M)
```

The current corrected PCA convention is:

1. Compute the column means of `R`.
2. Estimate PCA directions from demeaned returns:

   ```text
   R_centered = R - column_mean(R)
   R_centered = U S V^T
   V_K = first K right-singular-vector columns
   ```

3. Apply a deterministic sign convention to each component. The largest
   absolute loading is made positive so that factor signs are reproducible.
4. Project the **raw, uncentered** managed returns onto the PCA directions:

   ```text
   F = R @ V_K
   ```

5. Estimate the factor mean and covariance from `F`:

   ```text
   mu_F = mean(F, axis=0)
   Sigma_F = sample_covariance(F)
   ```

This distinction matters. Centering is used to estimate PCA directions, but
factor scores remain raw so that `mu_F` is not silently forced to zero. The
Sprint 15 artifacts used the earlier uncentered-direction convention and are
kept separate from the corrected Sprint 16 artifacts.

The PCA mapping has shape `(M, K)` and is called `V` in the QP layer. It maps
factor-space decisions to managed-portfolio space:

```text
p_scaled = V @ z
```

It does **not** map to individual stock weights. Individual stocks were already
aggregated into the managed portfolios in Stage 1.

## Stage 3: Factor-Space Max-Sharpe QP

The QP decision vector is a factor-space vector `z`, together with a positive
homogeneous scale variable `c` and auxiliary variables for L1 and long-short
constraints.

For the factor-space model:

```text
expected excess return = mu_F
covariance              = Sigma_F
managed mapping         = V
recovered weights       = p = (V @ z) / c
```

The max-Sharpe reparameterization fixes scaled excess return to one and lets
the solver minimize scaled variance. In conceptual form, the regularized
objective is:

```text
minimize    0.5 * z.T @ Sigma_F @ z
            + lambda_l2 * ||V @ z||_2^2
            + lambda_l1 * ||V @ z||_1
```

subject to the homogeneous return, budget, position, and shorting constraints.
The actual compiled problem includes split nonnegative variables for the L1
and long-short terms.

The canonical compiled objective is always:

```text
0.5 * x.T @ Q @ x + q.T @ x
```

For cuOpt, the adapter explicitly sends `Q_cuopt = 0.5 * Q` to match cuOpt's
quadratic convention. The OSQP and cuOpt paths consume the same compiled
problem and the same input matrices.

### Current Constraints

The current baseline uses:

| Constraint/parameter | Value | Meaning |
| --- | ---: | --- |
| Objective | Homogeneous `max_sharpe` | Maximize risk-adjusted excess return through the homogeneous QP reformulation |
| Risk-free rate | `0.0` | No separate risk-free subtraction in the current pilot |
| Short budget | `0.2` | Recovered gross short exposure is at most 20% |
| Implied long budget | `1.2` | Recovered gross long exposure is at most 120% |
| Per-managed-portfolio bound | `-0.08 <= p_i <= 0.08` | Explicit symmetric box bound |
| L1 regularization | configured per experiment | Penalizes absolute managed-portfolio exposure |
| L2 regularization | configured per experiment | Penalizes squared managed-portfolio exposure |
| Turnover penalty/budget | not used in current baseline | Available in the generic QP compiler, absent from the baseline replay |
| Benchmark tracking | not used in current baseline | Available in the generic QP compiler, absent from the baseline replay |

Variable bounds are always passed explicitly. The `-0.08` lower bound is
important because the model permits short positions; no cuOpt default bounds
are relied upon.

## Current Parameter Sets

There are two meanings of “current parameters” in the repository. They should
not be conflated.

### A. Established K=6 Baseline

This is the baseline used in the earlier B40/H200 pilot and in the default
monthly-panel pilot command:

| Parameter | Value |
| --- | ---: |
| Lookback | 240 months |
| PCA dimension | `K=6` |
| L1 | `1.7e-4` |
| L2 | `1e-3` |
| Bins | 10 |
| Managed weighting | Value weighted |
| Short budget | 0.2 |
| Bounds | `[-0.08, 0.08]` |
| Risk-free rate | 0.0 |
| OOS request | 2020-01 through 2022-12 |
| Realized windows | 35, ending 2022-11 |

The corrected Sprint 16 grid does not contain this exact L1/L2 pair. The old
Sprint 15 B40/H200 K=6 numbers used the pre-audit PCA-direction convention and
must be labeled as pre-audit pilot results.

### B. Current Completed B40 Grid Winner

The corrected full B40 grid tested `K=2..6` and a 10 x 10 logarithmic L1/L2
grid across the same 35-window 2020-2022 design. The best descriptive row by
annualized Sharpe was:

| Parameter | Selected row |
| --- | ---: |
| Lookback | 240 months |
| PCA dimension | `K=2` |
| L1 | `5.0` |
| L2 | `1e-6` |
| Backend | cuOpt on B40 |
| Windows | 35/35 optimal |
| Annualized Sharpe | 0.558993 |
| CAGR | 0.105144 |
| Maximum drawdown | -0.192737 |

This is the best row in a short 35-window uploaded-panel grid, not a final
paper hyperparameter selection. It should be called a **grid winner** or
**candidate configuration**, not the validated production model.

## Rolling Replay Procedure

For each requested rebalance date `t`:

1. Take the previous 240 months of managed returns ending at `t`.
2. Keep managed portfolios with complete history and a valid next-month return.
3. Fit the corrected PCA mapping `V` using only that rolling history.
4. Compute raw-return factor scores, `mu_F`, and `Sigma_F` using only that
   history.
5. Compile the factor-space max-Sharpe QP with the selected L1/L2 values and
   explicit bounds.
6. Solve through the requested backend.
7. Recover managed-portfolio weights `p = V @ z / c`.
8. Evaluate out of sample with the next month's managed returns:

   ```text
   portfolio_return[t+1] = p_t.T @ managed_returns[t+1]
   ```

9. Record solver status, objective, constraint violation, gross long/short
   exposure, weights, and realized next-month return.

The experiment does not randomly split rows. Every PCA fit, covariance estimate,
and QP solve is tied to a rolling time window.

## Backends And Reproducibility

| Backend | Role |
| --- | --- |
| OSQP | CPU reference and validation solver |
| cuOpt | Direct NVIDIA GPU execution on B40/H200 nodes |
| Fallback behavior | A requested cuOpt run is not silently replaced by OSQP |

The QP compilation is shared across backends. The replay records the requested
backend and keeps failed or skipped windows visible in the summary instead of
dropping them.

## What This Model Is Not

The current pipeline does not yet claim:

- full IPCA, RP-PCA, AP-Trees, or original-paper replication;
- stock-level individual asset weights after the PCA stage;
- a neural-network or other nonlinear return-prediction model;
- direct use of the new `processed_data_may25.p` panel;
- transaction-cost-adjusted performance in the baseline replay;
- turnover constraints or benchmark tracking in the baseline replay;
- a global GPU speedup claim;
- that the B40 grid winner is statistically selected or publication-ready.

The existing cuFOLIO Mean-CVaR workflow is separate and unchanged.

## Next Model Extension For The New Panel

The new `processed_data_may25.p` should not be appended to the current model
without an explicit bridge. The recommended sequence is:

1. Build a normalized adapter from `(eom, id, ret, me_company, prc, features)`
   to the current monthly-panel contract.
2. Resolve the extreme return records and confirm the forward-label definition.
3. Confirm whether the characteristic columns are already cross-sectional rank
   transformed and avoid double-ranking them.
4. Decide whether to preserve all 155 characteristic-style fields or first
   reproduce the existing 33-characteristic experiment.
5. Rebuild managed portfolios with the new panel and save a new manifest.
6. Re-run a no-regularization, K-sensitivity, and L1/L2 baseline before using
   the B40 grid winner as a candidate.
7. Add a true stock-level supervised/IPCA track only after the data contract is
   validated.

The current model is therefore a reliable QP/PCA replay framework and a useful
investment baseline. It is not yet the final training model for the new
1962-2025 dataset.
