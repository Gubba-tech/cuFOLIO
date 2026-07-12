# Cleaned Paper Data Schema

Parquet is the preferred exchange format. Dates are parsed to month-end
timestamps. All `asset_id` values must be stable across tables and all rows
must be ordered by `date, asset_id` before export.

## Monthly Returns

`monthly_returns.parquet`:

| Column | Type | Rule |
| --- | --- | --- |
| `date` | date | Monthly observation date. |
| `asset_id` | string/int | Stable identifier, e.g. CRSP `permno`. |
| `ticker` | string | Human-readable identifier. |
| `ret` | float | Adjusted total return; finite when observed. |
| `price` | float | Price used for the universe filter. |
| `market_cap` | float | Nonnegative market capitalization. |

`daily_returns.parquet` has the same columns at daily frequency and is used for
daily stock-level covariance benchmarks.

## Characteristics

`characteristics_monthly.parquet` may be wide:

```text
date, asset_id, ticker, char_1, ..., char_33
```

or long:

```text
date, asset_id, ticker, characteristic_name, value
```

The data bridge accepts either form and converts long data to wide form. The
characteristic values must document their accounting release lag; the paper
setup uses a six-month lag when sorting stocks.

## Benchmarks and Old Outputs

`benchmark_returns.parquet`:

```text
date, benchmark_name, ret
```

Optional `old_weights.parquet`:

```text
rebalance_date, model_name, k_factors, lambda_l1, lambda_l2,
asset_id, ticker, weight
```

Optional `old_returns.parquet`:

```text
date, model_name, k_factors, lambda_l1, lambda_l2, portfolio_return
```

## Validation and Privacy

Run `scripts/validate_paper_cleaned_data.py` before building a universe. It
checks required columns, date parsing, finite returns, nonnegative market caps,
duplicates, month coverage, asset counts, characteristic coverage, missingness,
and the first valid lookback/OOS date.

Real data belongs under `data/private/` or
`artifacts/paper_replay/private/`, both excluded by `.gitignore`. Only small
synthetic Parquet fixtures may be committed under `tests/fixtures/paper_data/`.

