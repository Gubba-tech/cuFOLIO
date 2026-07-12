# Monthly Panel PCA Pilot

Sprint 14 uses the uploaded monthly panel as the primary empirical dataset.
JKP is not used in this sprint. The panel covers 2000-01 through 2022-12 and
has approximately 700 stocks per month in the intended universe. The exact
observed coverage is recorded by the validator rather than assumed.

The workflow is:

```text
monthly panel
  -> validation and cleaned Parquet
  -> characteristic-sorted managed portfolios
  -> PCA factor-space replay windows
  -> max-Sharpe + l1/l2 + long-short QP
  -> OSQP and optional cuOpt replay
  -> diagnostics and summary
```

## Environment

```bash
uv sync --extra dev
```

For GPU execution, select exactly one CUDA extra matching `nvidia-smi`:

```bash
uv sync --extra cuda12 --extra dev
# or
uv sync --extra cuda13 --extra dev
```

## Validate and Convert

Keep the uploaded file outside git, for example under `data/private/` or use
an external path directly:

```bash
uv run python scripts/validate_monthly_characteristic_panel.py \
  --input data/private/monthly_characteristic_panel.tsv \
  --output-dir artifacts/paper_replay/monthly_panel_validation \
  --format tsv --sep auto

uv run python scripts/convert_monthly_characteristic_panel.py \
  --input data/private/monthly_characteristic_panel.tsv \
  --output artifacts/paper_replay/processed/monthly_characteristic_panel.parquet \
  --format tsv --sep auto --drop-missing-ret
```

The converter writes metadata beside the Parquet file and records that the
source data are not committed.

## Managed Portfolios

The builder uses month-t characteristics and month-t+1 realized returns:

```bash
uv run python scripts/build_paper_managed_portfolios.py \
  --monthly-panel artifacts/paper_replay/processed/monthly_characteristic_panel.parquet \
  --output-dir artifacts/paper_replay/managed_portfolios_monthly_panel \
  --characteristics all --n-bins 10 --weighting value \
  --min-assets-per-bin 5
```

With all 33 default characteristics and ten bins, the expected maximum is 330
managed portfolios. These are managed-portfolio returns; they are not
individual stock weights.

## Recommended Pilots

### Pilot A: 2005 Short Lookback

This gives a longer evaluation period but is not paper-comparable because the
panel does not contain the 20 years before 2005.

```bash
uv run python scripts/run_monthly_panel_pca_pilot.py \
  --monthly-panel data/private/monthly_characteristic_panel.tsv \
  --output-dir artifacts/paper_replay/results/monthly_panel_pca_k6_2005_short_lookback \
  --start-date 2005-01-31 --end-date 2022-12-31 \
  --lookback-months 60 --k-values 6 --characteristics all \
  --n-bins 10 --weighting value \
  --lambda-l1 1.7e-4 --lambda-l2 1e-3 \
  --short-budget 0.2 --w-min -0.08 --w-max 0.08 \
  --backend both --allow-short-lookback --assume-characteristics-lagged
```

Every result from this design must be labeled **2005 short-lookback pilot,
not full paper replication**.

### Pilot B: 2020 Twenty-Year Lookback

This is closer to the paper's lookback length but has a short OOS period:

```bash
uv run python scripts/run_monthly_panel_pca_pilot.py \
  --monthly-panel data/private/monthly_characteristic_panel.tsv \
  --output-dir artifacts/paper_replay/results/monthly_panel_pca_k6_2020_20y_lookback \
  --start-date 2020-01-31 --end-date 2022-12-31 \
  --lookback-months 240 --k-values 6 --characteristics all \
  --n-bins 10 --weighting value \
  --lambda-l1 1.7e-4 --lambda-l2 1e-3 \
  --short-budget 0.2 --w-min -0.08 --w-max 0.08 \
  --backend both --assume-characteristics-lagged
```

This design must be labeled **20-year-lookback pilot with short OOS period**.
Because managed returns use a one-month formation lag, the final replay window
with realized returns may end one month before the raw panel's final date.

`backend=both` records separate OSQP and cuOpt rows. If cuOpt is unavailable,
the cuOpt rows are `skipped`; OSQP is never substituted for a requested cuOpt
solve.

## Claims Boundary

- Full original paper replication is not claimed.
- Full IPCA replication is not claimed.
- Full AP-Trees replication is not claimed.
- Old-solution parity is not claimed unless old weights/objectives are supplied.
- No global QP speedup is claimed.
- The Mean-CVaR LP workflow is untouched.

## Stock-Level Optional Pilot

`scripts/run_monthly_panel_stock_qp_pilot.py` is a solver demonstration only.
Stock-level max-Sharpe based on sample means is noisy and should not be expected
to match the factor-model or managed-portfolio results.
