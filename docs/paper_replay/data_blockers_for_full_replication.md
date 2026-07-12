# Blockers for Full Paper Replication

Sprint 14 validates and runs a real monthly-panel PCA pilot from the uploaded
data. It does not remove the blockers for exact paper replication.

## Missing Inputs

- CRSP-style daily data and the paper's exact monthly data construction, prices, market caps, and stable identifiers.
- Compustat-style accounting characteristics with the paper's exact 33-characteristic panel, six-month lag, and missing-data rules.
- Exact AMP-style universe membership by rebalance date.
- Saved PCA/RP-PCA/IPCA/AP-Trees factor returns and mappings, or the authorized estimators and input data needed to generate them.
- Old per-window weights/objectives/realized returns for parity comparison.
- Benchmark return series and the exact lambda winners used for the paper target paths.

## What Was Completed

The uploaded monthly panel now validates as a complete 2000-01 through
2022-12 monthly panel with all listed characteristics. The Sprint 14 pilot
builds 330 characteristic/bin managed portfolios and runs PCA K=6 factor-space
QP windows through OSQP. A cuOpt request is recorded as `skipped` when the GPU
runtime is unavailable; it never falls back to OSQP.

The synthetic bridge validates the mechanics:

1. cleaned Parquet validation;
2. configurable market-cap/price/missingness universe selection;
3. lagged characteristic-sorted managed portfolios;
4. PCA factor-space replay-window export;
5. OSQP replay and summary metrics.

The real pilot is still pilot evidence, not full paper replication. The 2005
short-lookback design is not paper-comparable; the 2020 20-year-lookback design
has a short OOS period.

The synthetic outputs are not CRSP/Compustat data and do not establish paper
empirical replication. No proprietary data are committed.

## Handoff Needed

Provide the cleaned inputs under `data/private/` or
`artifacts/paper_replay/private/`, then run the documented 12-month PCA K=6
pilot. For IPCA/AP-Trees, provide externally generated factor artifacts or
authorize a separately scoped estimator/data implementation. Only after the
full window construction and realized-return path are matched should the
results be compared with the Table 2 targets.
