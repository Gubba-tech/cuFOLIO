# Paper Replay Data Availability Checklist

Status vocabulary: `available`, `missing`, `blocked`, `synthetic-only`.

| Category | Required inputs | Status | Path | Owner/contact | Notes |
| --- | --- | --- | --- | --- | --- |
| Daily stock data | Adjusted daily returns, prices, market cap, `permno`/ticker/date | missing | Not present in cuFOLIO or the public PortOpt clone | Data owner | Needed for daily stock-level shrinkage/min-variance benchmarks. |
| Monthly stock data | Adjusted monthly returns, prices, market cap, identifiers | synthetic-only | `tests/fixtures/paper_data/monthly_returns.parquet` | Test fixture | The fixture is deterministic and is not CRSP data. |
| Compustat/characteristics | 33 firm characteristics, accounting variables, six-month lag, missing-data rules | synthetic-only | `tests/fixtures/paper_data/characteristics_monthly.parquet` | Test fixture | Three synthetic wide characteristics only. |
| AMP universe | Large/mid-cap coverage, price filter, missing-data filter, rolling universe | synthetic-only | `artifacts/paper_replay/synthetic_universe/` | Test fixture | Configurable approximation; no exact AMP universe source is available. |
| PCA/RP-PCA managed returns | 330 managed portfolios or equivalent saved matrices | synthetic-only | `artifacts/paper_replay/synthetic_managed_portfolios/` | Test fixture | Synthetic builder supports configurable characteristic count and bins. |
| PCA mapping | PCA eigenvectors/mapping `V` by window | synthetic-only | `artifacts/paper_replay/synthetic_windows_pca/` | Test fixture | Generated from synthetic managed returns. |
| RP-PCA mapping | RP-PCA factors and mapping `V` | missing | No saved artifacts | Data owner | Must be supplied externally. |
| IPCA factors | IPCA factor returns and mapping `V` | missing | No saved artifacts | Data owner | Full estimator/data bridge is not included in this sprint. |
| AP-Trees | Managed portfolio returns and mapping | missing | No saved artifacts | Data owner | The public PortOpt clone contains no complete AP-Trees artifact. |
| Benchmarks | S&P 500 or other benchmark returns | missing | No private/public input supplied | Data owner | Required for comparable benchmark path. |
| Old paper outputs | Old weights, objectives, returns, winners, Table 2 artifacts | missing | No saved outputs in `Gubba-tech/PortOpt_IPCA` | Data owner | Without these, old-solution parity cannot be tested. |

## Required Real-Data Handoff

To run the first real PCA K=6 pilot, provide cleaned Parquet inputs for monthly
returns and characteristics, a documented universe policy, and the selected
lambda values. For old-solution parity, also provide per-window old weights or
objectives with the same asset/factor ordering. Keep those files under
`data/private/` or `artifacts/paper_replay/private/`; do not commit them.

