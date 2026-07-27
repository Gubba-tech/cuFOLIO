# `processed_data_may25.p`: Data Description And Training Handoff

## Purpose

This document describes the uploaded file
`/lustre/nvwulf/home/weicdeng/PortOpt-IPCA-GPU/data/processed_data_may25.p`.
It is written as a handoff for GPT Pro to design the next training and portfolio
research plan. No model has been trained from this file yet.

The raw pickle is intentionally kept outside the Git repository. This document
records the audit results and the proposed data contract, but does not commit
the 3.3 GB source file.

## Executive Summary

| Item | Observed value |
| --- | --- |
| File format | pandas pickle, protocol 5 |
| File size | 3,467,958,781 bytes, approximately 3.3 GB |
| Rows | 2,523,680 stock-month observations |
| Columns | 170 |
| Date range | 1962-01-31 through 2025-05-31 |
| Monthly periods | 761; no missing month in the calendar sequence |
| Unique security identifier | 17,277 unique `id` values |
| Row key | One row per `(eom, id)`; duplicate key count is 0 |
| Original pandas index | Not unique; reset it before modeling |
| Python memory footprint | Approximately 3.79 GiB with pandas deep memory accounting |
| Geography | `excntry` is always `USA` |

The panel is a long US equity panel. It contains monthly security-level
returns, a large library of firm characteristics, market and liquidity fields,
and two explicit next-month return columns. The cross-section grows from a few
hundred stocks in the early 1960s to roughly 4,000-5,000 stocks per month in
recent decades.

The current file is substantially larger than the earlier
`dfall_for_test.csv` pilot panel. They must be treated as different datasets:

- `processed_data_may25.p`: 2.52 million rows, 1962-2025, approximately 155
  characteristic-style columns plus identifiers and returns.
- `dfall_for_test.csv`: the earlier managed-panel experiment and not the source
  of the statistics in this document.

## Loading The File

```python
import pandas as pd

path = "/lustre/nvwulf/home/weicdeng/PortOpt-IPCA-GPU/data/processed_data_may25.p"
df = pd.read_pickle(path)
df = df.reset_index(drop=True)
df["eom"] = pd.to_datetime(df["eom"])
```

The file is large enough that accidental copies can exceed the available host
memory. Prefer column selection, month-by-month processing, or a one-time
conversion to Parquet. Do not sort a full pandas copy unless the machine has
substantial memory headroom.

## Row And Time Structure

- `eom` is a month-end timestamp with 761 distinct monthly values.
- `id` is an integer security-level identifier. It is the only stable key that
  was validated in this audit.
- The pair `(eom, id)` is unique across all 2,523,680 rows.
- The original pandas index is repeated and has no modeling meaning.
- The number of active securities per month ranges from 301 to 5,175. The
  median is 3,559.
- The number of observed months per security ranges from 1 to 761; the median
  is 92. This is an unbalanced panel, not a fixed-stock panel.
- `ret` is missing in the first part of the sample and in a small number of
  other rows. `ret_*_lead1m` is missing at the final month and has a small
  amount of additional missingness.

Boundary observations should be handled explicitly. In particular, May 2025
has no next-month target, so it cannot be an ordinary realized-return OOS
 observation.

## Data Types And Identifiers

| Group | Columns | Audit result | Training guidance |
| --- | --- | --- | --- |
| Time/key | `eom`, `id` | datetime plus integer | Use `(eom, id)` as the observation key; reset the index |
| Security identifiers | `sedol`, `cusip`, `isin` | `sedol` is entirely missing; `cusip` and `isin` each have 16,715 non-null unique values and 20.9049% missingness | Do not use as numerical features; avoid one-hot encoding in the first model |
| Descriptive metadata | `conm`, `size_grp`, `excntry` | `conm` has 20.9049% missingness; `size_grp` has four categories and no missing values; `excntry` is constant `USA` | Treat `size_grp` as optional metadata or a controlled categorical feature; drop names and constant country |
| Raw size field | `me_company` | Non-null raw-looking market-equity scale; median 264.59 and maximum 125,176,381.35 | Use only after checking units and timing; prefer `log1p` and train-period transforms |
| Constant marker | `dummy_char` | Integer constant equal to 1 in every row | Drop from model features |

The identifier fields are useful for joins and diagnostics, but they can cause
memorization and survivorship artifacts. The first baseline should use only
economic characteristics and explicitly controlled size information.

## Return And Label Columns

| Column | Non-null rows | Valid-rate | Observed median | Observed maximum | Initial interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `ret` | 2,510,698 | 99.4856% | 0.005306 | 1,288,999.0 | Same-row realized return |
| `ret_exc` | 2,510,698 | 99.4856% | 0.002400 | 1,288,998.9999 | Same-row excess return candidate |
| `ret_local` | 2,510,698 | 99.4856% | 0.005306 | 1,288,999.0 | Same-row local return candidate |
| `ret_local_lead1m` | 2,519,463 | 99.8329% | 0.004061 | 220.126761 | Explicit next-month local-return candidate |
| `ret_exc_lead1m` | 2,519,463 | 99.8329% | 0.001233 | 220.126661 | Explicit next-month excess-return candidate |

The natural first supervised-learning contract is:

```text
features at month t  ->  ret_exc_lead1m at month t
```

`ret_local_lead1m` can be used as a parallel label for a local-return
experiment. However, the label semantics must be confirmed against the source
data dictionary before production use. A month-by-month key join gave the
following diagnostic:

| Candidate label | Compared with next month's same-row return | Exact matches | Mean absolute difference | Maximum absolute difference |
| --- | ---: | ---: | ---: | ---: |
| `ret_exc_lead1m` | next-month `ret_exc` | 2,403,796 / 2,487,267 = 96.6441% | 0.0005006 | 18.2053 |
| `ret_local_lead1m` | next-month `ret_local` | 2,468,661 / 2,487,267 = 99.2519% | 0.0004932 | 18.2054 |

This is strong evidence that the columns are forward-looking, but not evidence
that they are byte-for-byte shifted copies. The difference may reflect return
definition, corporate-action handling, delisting treatment, or source joins.
Do not silently replace the explicit lead columns with a locally shifted
version.

### Critical Return-Quality Finding

The return columns contain implausible extreme values for a monthly stock
return:

| Column | `abs(value) > 10` | `abs(value) > 100` | `abs(value) > 10,000` |
| --- | ---: | ---: | ---: |
| `ret` | 120 | 33 | 2 |
| `ret_exc` | 120 | 33 | 2 |
| `ret_local_lead1m` | 11 | 1 | 0 |
| `ret_exc_lead1m` | 11 | 1 | 0 |

The largest same-row observation is approximately `1,288,999`, and the
largest explicit forward label is approximately `220.13`. These observations
are not safe to pass directly into a regression loss, covariance estimate, or
portfolio metric. Before training, inspect the source records and determine
whether they are caused by stock splits, delistings, bad prices, special
distributions, or a return-unit/encoding problem. Then compare at least:

1. source-corrected returns, if the upstream data can be repaired;
2. a documented cross-sectional winsorization policy applied using the
   training sample only; and
3. a robust loss or bounded-label sensitivity run.

The policy must be applied before computing means, covariances, Sharpe ratios,
or portfolio weights. It must be recorded in the experiment metadata.

## Characteristic Library

The file contains approximately 155 characteristic-style numeric columns in
addition to keys, metadata, and return fields. The following groups are a
navigation aid based on column names, not an authoritative academic data
dictionary. Exact economic definitions, publication lags, and source vendors
must be confirmed before making paper-level claims.

### Accruals, investment, and growth

`cowc_gr1a`, `oaccruals_at`, `oaccruals_ni`, `seas_16_20an`, `taccruals_at`,
`taccruals_ni`, `capex_abn`, `debt_gr3`, `fnl_gr1a`, `ncol_gr1a`, `nfna_gr1a`,
`ni_ar1`, `noa_at`, `at_gr1`, `be_gr1a`, `capx_gr1`, `capx_gr2`, `capx_gr3`,
`coa_gr1a`, `col_gr1a`, `emp_gr1`, `inv_gr1`, `inv_gr1a`, `lnoa_gr1a`,
`ncoa_gr1a`, `nncoa_gr1a`, `noa_gr1a`, `ppeinv_gr1a`, `sale_gr1`, `sale_gr3`,
`saleq_gr1`, `dsale_dinv`, `dsale_drec`, `dsale_dsga`, `niq_at_chg1`,
`niq_be_chg1`, `ocf_at_chg1`, `sale_emp_gr1`, `tax_gr1a`, `dbnetis_at`,
`lti_gr1a`, `sti_gr1a`, `eqnetis_at`, `eqnpo_12m`, `netis_at`.

### Value, profitability, quality, and mispricing

`mispricing_mgmt`, `mispricing_perf`, `at_be`, `cash_at`, `netdebt_me`,
`rd_sale`, `rd5_at`, `tangibility`, `z_score`, `ebit_bev`, `ebit_sale`,
`f_score`, `ni_be`, `niq_be`, `o_score`, `ocf_at`, `ope_be`, `ope_bel1`,
`at_turnover`, `cop_at`, `cop_atl1`, `dgp_dsale`, `gp_at`, `gp_atl1`, `ni_inc8q`,
`niq_at`, `op_at`, `op_atl1`, `opex_at`, `qmj`, `qmj_growth`, `qmj_prof`,
`qmj_safety`, `sale_bev`, `pi_nix`, `be_me`, `bev_mev`, `debt_me`, `div12m_me`,
`ebitda_mev`, `fcf_me`, `ival_me`, `ni_me`, `ocf_me`, `sale_me`, `eqnpo_me`,
`eqpo_me`.

### Momentum, seasonality, and prior returns

`ret_60_12`, `ret_3_1`, `ret_6_1`, `ret_9_1`, `ret_12_1`, `ret_12_7`, `ret_1_0`,
`seas_2_5na`, `seas_1_1na`, `seas_1_1an`, `seas_2_5an`, `seas_6_10na`,
`seas_6_10an`, `seas_11_15an`, `seas_11_15na`, `seas_16_20na`, `resff3_6_1`,
`resff3_12_1`.

### Risk, liquidity, and trading

`median_turnover_3m_usd`, `avg_turnover_3m_usd`, `aliq_at`, `aliq_mat`,
`bidaskhl_21d`, `beta_60m`, `beta_dimson_21d`, `betabab_1260d`,
`betadown_252d`, `earnings_variability`, `ivol_capm_21d`, `ivol_capm_252d`,
`ivol_ff3_21d`, `ivol_hxz4_21d`, `ocfq_saleq_std`, `rmax1_21d`, `rmax5_21d`,
`rvol_21d`, `turnover_126d`, `zero_trades_21d`, `zero_trades_126d`,
`zero_trades_252d`, `prc_highprc_252d`, `dolvol_var_126d`,
`turnover_var_126d`, `corr_1260d`, `coskew_21d`, `ami_126d`, `dolvol_126d`,
`iskew_capm_21d`, `iskew_ff3_21d`, `iskew_hxz4_21d`, `rmax5_rvol_21d`,
`rskew_21d`, `eq_dur`.

### Size and market fields

`me_company` is the raw-looking company market-equity field. `market_equity`,
`prc`, `at_me`, and several ratio columns appear rank-like in the processed
file: spot checks have values bounded by approximately `[-0.5, 0.5]`, with a
median near zero. This strongly suggests that at least part of the input has
already undergone cross-sectional ranking or scaling. Confirm the processing
script before applying another rank transform. Double-ranking can erase the
intended scale and change the interpretation of the model.

## Missingness And Coverage

The observed missingness is concentrated in a small number of fields:

| Column/group | Missing rate | Comment |
| --- | ---: | --- |
| `sedol` | 100.0000% | Entirely missing; drop |
| `conm`, `cusip`, `isin` | 20.9049% | Identifier/name coverage issue |
| `median_turnover_3m_usd`, `avg_turnover_3m_usd` | 20.8826% | Likely limited trading-history coverage |
| `ret`, `ret_exc`, `ret_local` | 0.5144% | Same-row return missingness |
| `ret_*_lead1m` | 0.1671% | Boundary and source coverage |
| Most other inspected characteristics | 0% | Still verify source semantics and sentinels |

`size_grp` has the following row counts:

| Size group | Rows | Share |
| --- | ---: | ---: |
| `micro` | 1,074,952 | 42.593% |
| `small` | 670,300 | 26.559% |
| `large` | 487,001 | 19.300% |
| `mega` | 291,427 | 11.548% |

This universe is not neutral with respect to micro-cap stocks. A portfolio
experiment should report results both for the full universe and for an explicit
liquidity/size-filtered universe.

## Recommended Training Data Contract

### Candidate stock-level task

For each month `t`:

```text
X_t  = characteristics observable at the formation date t
y_t  = ret_exc_lead1m at t
group = eom
entity = id
```

The first baseline should use a clean, explicit feature list and exclude:

- `eom`, `id`, `sedol`, `cusip`, `isin`, and `conm`;
- `dummy_char` and constant `excntry`;
- same-period `ret`, `ret_exc`, and `ret_local`;
- both forward-return columns, unless one is the selected label;
- any field whose publication lag is not known to be available at month `t`.

`size_grp` may be used for universe filtering and diagnostics. Include it as a
model input only in a controlled ablation, because it can make the model
primarily a size/regime classifier.

### Feature preprocessing

1. Confirm whether the characteristic columns were transformed cross-sectionally
   at each month and whether the transformation used information available at
   the formation date.
2. Do not double-rank columns that are already centered rank scores.
3. For raw scale variables such as `me_company` and turnover, verify units and
   use `log1p` or a documented robust transform.
4. Fit every imputer, scaler, winsorization threshold, and feature selector on
   the training period only. Apply the frozen transform to validation and test.
5. Preserve a monthly cross-section identifier so that normalization never
   accidentally uses future months or the full sample.

### Target preprocessing

Before fitting any model:

1. trace the extreme return records to the source and classify the cause;
2. decide whether delisting and corporate-action returns belong in the target;
3. define an admissibility rule for impossible or economically implausible
   monthly returns;
4. compare raw, corrected, and robust/winsorized target variants;
5. keep the target rule identical across CPU, GPU, and QP replay runs.

Do not compute a covariance matrix from the unfiltered `ret` columns. A single
bad return can dominate a sample covariance and produce unstable QP weights.

## Time Splits And Leakage Control

Do not use a random row split. The panel is time ordered and highly repeated
by security. A reasonable first chronological design is:

| Split | Suggested period | Role |
| --- | --- | --- |
| Train | 1962-01 through 2004-12 | Fit model and preprocessing |
| Validation | 2005-01 through 2014-12 | Select hyperparameters and feature set |
| Test/OOS | 2015-01 through 2025-04 | Final untouched evaluation |

The exact split should be chosen together with the research question. An
expanding-window or rolling-window design is preferable for portfolio replay.
At each rebalance, use only data available before the rebalance date. Do not
fit PCA, IPCA, scalers, covariance estimators, or hyperparameters on the full
1962-2025 panel.

Because the label is one month forward, the final raw month is not a realized
test month. Keep a one-month buffer when constructing train/validation/test
artifacts and document whether boundary labels are dropped or shifted.

## Training Plan For GPT Pro To Evaluate

Please have GPT Pro compare these three nested tracks rather than jumping
directly to a large neural network:

| Track | Input | Output | Purpose |
| --- | --- | --- | --- |
| A. Stock-level supervised baseline | `X_t` | predicted `ret_exc_lead1m` | Establish predictability and feature sanity |
| B. IPCA/latent-factor model | `X_t` and next-month excess returns | factor loadings, factor returns, conditional expected returns | Test the intended asset-pricing structure |
| C. Managed portfolios plus QP | characteristic-managed returns and factor covariance | monthly constrained portfolio weights | Test the end-to-end investment objective in cuFOLIO |

For each track, compare a small baseline grid first: no regularization, ridge or
L2 regularization, L1 plus L2, and factor dimensions `K` in `{1, 2, 3, 5, 6,
10}`. Keep the initial feature set fixed while validating the data contract;
only then test the full characteristic library.

Report both prediction and investment metrics:

- monthly and annualized excess return, volatility, and Sharpe ratio;
- cumulative return, drawdown, turnover, and transaction-cost-adjusted return;
- cross-sectional rank correlation or information coefficient;
- number of valid assets and target coverage per month;
- constraint violations, solver status, and active bounds for QP runs;
- results by size group, decade, and high/low liquidity subset;
- sensitivity to the return-cleaning policy and to the universe filter.

The QP layer must receive an explicit, auditable expected-return vector and
covariance estimate. Variable bounds must always be passed explicitly,
including negative lower bounds for long-short portfolios. OSQP should remain
the reference solver, with cuOpt used as the GPU backend on B40/H200 machines.

## Open Questions Before Training

These questions should be answered from the upstream data dictionary or
processing code before treating the file as paper-ready:

1. What are the exact definitions and units of `ret`, `ret_exc`, and the two
   lead-return columns?
2. Why are there extreme same-month and forward returns, and how are splits,
   delistings, and special distributions represented?
3. Are the characteristic values rank-transformed per month? If so, what is the
   precise transform and what information set was used?
4. Does `id` remain stable through corporate events, name changes, and security
   changes?
5. Which fields have a publication lag, and what is the valid information date
   for accounting characteristics?
6. Should the target be excess return or local total return, and what is the
   risk-free-rate source?
7. Is the research universe all rows, only US common equity, or a size/liquidity
   filtered subset? What is the intended treatment of micro-cap stocks?
8. Is the objective stock-level return prediction, IPCA factor estimation, or
   portfolio Sharpe after QP optimization?
9. What transaction-cost, turnover, leverage, shorting, and position bounds
   should be used?
10. Which time period is reserved as the final publication-style OOS period?

## Copy/Paste Prompt For GPT Pro

```text
Use docs/data_description_processed_data_may25.md as the data contract. Design
a leakage-safe training and portfolio-research plan for this 1962-2025 monthly
US equity panel. Start by resolving the open questions, especially the extreme
return values and the semantics of ret_exc_lead1m. Compare: (A) stock-level
supervised prediction, (B) IPCA/latent-factor estimation, and (C) managed
portfolios followed by a constrained QP. Use chronological or rolling OOS
validation, never a random row split. Explain the exact feature exclusions,
target cleaning, monthly normalization, missing-data policy, size/liquidity
filters, PCA/IPCA fitting window, covariance construction, QP bounds, and
annualized evaluation metrics. Include a staged baseline plan before any large
neural model, plus the diagnostics needed to detect leakage, bad returns,
survivorship bias, and unstable portfolio weights.
```
