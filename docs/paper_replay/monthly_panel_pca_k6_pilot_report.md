# Monthly Panel PCA K=6 Pilot Report

## Scope

This report records Sprint 14 experiments using the uploaded monthly
stock-characteristic panel as the primary empirical dataset. The source is an
external/private file and is not committed or copied into this repository.
The reported experiments are PCA managed-portfolio pilots, not full paper
replication.

JKP is not used in Sprint 14.

## Data And Construction

| Item | Value |
| --- | --- |
| Data path used | External uploaded monthly panel; exact private path omitted |
| Raw date range | 2000-01-31 through 2022-12-31 |
| Number of raw months | 276 |
| Raw rows | 241,347 |
| Full-panel average stocks/month | 874.45 |
| OOS-panel average stocks/month, 2020-2022 | 692.44 |
| Characteristics used | 33 default paper-style characteristics |
| Managed portfolios | 330 (33 characteristics x 10 bins) |
| Weighting | Value-weighted |
| Timing | Sort at month t, realize return at month t+1 |
| Model | PCA, K=6, factor-space mapping |
| Objective | Homogeneous max-Sharpe |
| Risk-free rate | 0.0 |
| Short budget | 0.2 |
| Bounds | `w_min=-0.08`, `w_max=0.08` |
| L1 penalty | `1.7e-4` |
| L2 penalty | `1e-3` |

`V` maps PCA factor weights to managed-portfolio weights. It does not map to
individual stock weights.

## Pilot A: 2005 Short Lookback

This controlled run used `lookback_months=60`, the 2005-01 through 2022-12
requested OOS range, and 12 replay windows (`--max-windows 12`) to validate the
pilot path without claiming a complete 2005-2022 run.

| Backend | Windows | Status counts | Annualized return | Annualized volatility | Sharpe | Max drawdown | Max constraint violation |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| OSQP | 12 | 12 optimal | 0.16446 | 0.10076 | 1.56646 | -0.04584 | 8.53e-7 |
| cuOpt | 12 | 12 skipped | N/A | N/A | N/A | N/A | N/A |

This is a **short-lookback pilot only**, not paper-comparable and not full
replication. The cuOpt rows were skipped because the runtime was unavailable;
OSQP was not substituted for cuOpt.

## Pilot B: 2020 Twenty-Year Lookback

This run used `lookback_months=240`, K=6, and the 2020-01 through 2022-12
requested OOS range. The managed-return timing contract produced 35 windows
through 2022-11 because a realized t+1 return is required.

| Backend | Windows | Status counts | Annualized return | Annualized volatility | Sharpe | Max drawdown | Max constraint violation |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| OSQP | 35 | 35 optimal | 0.08436 | 0.21306 | 0.48597 | -0.21618 | 1.18e-6 |
| cuOpt | 35 | 35 skipped | N/A | N/A | N/A | N/A | N/A |

This is a **20-year-lookback pilot with a short OOS period**. It is not full
paper replication.

## Artifacts

The generated artifacts are gitignored:

- `artifacts/paper_replay/monthly_panel_validation/`
- `artifacts/paper_replay/results/monthly_panel_pca_k6_2005_short_lookback/`
- `artifacts/paper_replay/results/monthly_panel_pca_k6_2020_20y_lookback/`

Each result directory contains the cleaned Parquet output, managed portfolio
tables, PCA windows, per-window backend statuses, and a pilot manifest. The
raw uploaded panel is not included.

## Claims Boundary

- Full original paper replication is not claimed.
- Full IPCA replication is not claimed.
- Full AP-Trees replication is not claimed.
- Old-solution parity is not claimed because old weights/objectives were not supplied.
- No global QP speedup is claimed.
- Mean-CVaR LP functionality was not changed.
