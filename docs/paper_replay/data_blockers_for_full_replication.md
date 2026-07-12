# Blockers for Full Paper Replication

Sprint 13 did not run a real 2005 PCA/IPCA pilot because no cleaned data or
saved old matrices are available in the current workspace.

## Missing Inputs

- CRSP-style daily and monthly returns, prices, market caps, and stable identifiers.
- Compustat-style accounting characteristics with the paper's 33-characteristic panel and six-month lag.
- Exact AMP-style universe membership by rebalance date.
- Saved PCA/RP-PCA/IPCA/AP-Trees factor returns and mappings, or the authorized estimators and input data needed to generate them.
- Old per-window weights/objectives/realized returns for parity comparison.
- Benchmark return series and the exact lambda winners used for the paper target paths.

## What Was Completed

The synthetic bridge validates the mechanics:

1. cleaned Parquet validation;
2. configurable market-cap/price/missingness universe selection;
3. lagged characteristic-sorted managed portfolios;
4. PCA factor-space replay-window export;
5. OSQP replay and summary metrics.

The synthetic outputs are not CRSP/Compustat data and do not establish paper
empirical replication. No proprietary data are committed.

## Handoff Needed

Provide the cleaned inputs under `data/private/` or
`artifacts/paper_replay/private/`, then run the documented 12-month PCA K=6
pilot. For IPCA/AP-Trees, provide externally generated factor artifacts or
authorize a separately scoped estimator/data implementation. Only after the
full window construction and realized-return path are matched should the
results be compared with the Table 2 targets.

