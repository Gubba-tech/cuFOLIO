# Paper Replay Artifact Schema

Schema version: `1.0`

## Storage

Each replay window is one compressed `.npz` file. JSON metadata is stored under
`metadata_json`; numeric fields are stored as named NumPy arrays. This format
is portable, explicit about matrix dimensions, and avoids Python object
pickles. Parquet may be used for a surrounding date/index table, but the QP
matrices remain `.npz` arrays.

Generated artifacts from proprietary data must remain under gitignored
directories such as `artifacts/paper_replay/`. Small deterministic synthetic
fixtures may be committed under `tests/fixtures/paper_replay/`.

## Metadata Fields

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | Must be `1.0`. |
| `window_id` | string | Stable per-window identifier. |
| `rebalance_date` | string | ISO date for the rebalance. |
| `model_name` | string | `PCA`, `RP-PCA`, `IPCA`, `AP-Trees`, or external model label. |
| `objective` | string | `min_variance`, `mean_variance`, `target_return`, or `max_sharpe`. |
| `mapping_mode` | string | `stock_space` or `factor_space`. |
| `risk_free_rate` | float | Risk-free rate used by the max-Sharpe row. |
| `lambda_l1` | float | l1 penalty coefficient. |
| `lambda_l2` | float | l2^2 penalty coefficient. |
| `lambda_tracking_error` | float | Optional tracking-error penalty coefficient. |
| `short_budget` | float or null | Gross short budget. |
| `target_return` | float or null | Target return when applicable. |
| `w_min` | array | Scalar or per-stock lower bound. |
| `w_max` | array | Scalar or per-stock upper bound. |
| `tickers` | list[string] | Stock ordering for all stock vectors and rows. |
| `factor_names` | list[string] or null | Factor ordering when factor space is used. |
| `turnover_budget` | float or null | Optional total turnover budget. |
| `benchmark_l1_budget` | float or null | Optional benchmark l1 budget. |
| `old_objective_value` | float or null | Objective reported by the old solver, if available. |
| `old_realized_portfolio_return` | float or null | Old next-period return, if already computed. |
| `notes` | string | Provenance, transformations, and missing-field notes. |

## Array Fields

| Field | Required | Shape |
| --- | --- | --- |
| `mean` | stock space | `(n_assets,)` |
| `covariance` | stock space | `(n_assets, n_assets)` |
| `factor_mean` | factor space | `(n_factors,)` |
| `factor_covariance` | factor space | `(n_factors, n_factors)` |
| `stock_mapping` | factor space, optional identity in stock space | `(n_assets, n_factors)` |
| `stock_covariance` | tracking-error penalty | `(n_assets, n_assets)` |
| `previous_weights` | turnover constraint | `(n_assets,)` |
| `benchmark_weights` | benchmark l1 or tracking error | `(n_assets,)` |
| `old_weights` | old-solution comparison | `(n_assets,)` stock weights |
| `realized_next_returns` | return-path comparison | `(n_assets,)` |

All numeric arrays must be finite. The stock ordering is defined by
`tickers`; factor ordering is defined by `factor_names`. For factor space,
stock weights are `p = V @ z`, and the QP mean/covariance are factor mean and
factor covariance. For tracking error, `stock_covariance` is required even in
factor space.

## Replay Contract

`load_replay_window` validates the schema and dimensions. `solve_replay_window`
builds `QPParameters` and `CompiledQP` without changing the stored inputs.
`compute_replay_diagnostics` reports status, objective gaps, weight distances,
budget and box violations, constraint violation, solve/total times, and the
realized next-period return. The old solution is diagnostic data only; it is not
used to alter the new optimization problem.

