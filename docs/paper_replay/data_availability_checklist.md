# Paper Replay Data Availability Checklist

Status vocabulary: `available`, `missing`, `blocked`, `synthetic-only`.

| Category | Required inputs | Status | Path | Owner/contact | Notes |
| --- | --- | --- | --- | --- | --- |
| Daily stock data | Adjusted daily returns, prices, market cap, `permno`/ticker/date | missing | Not present in cuFOLIO or the public PortOpt clone | Data owner | Needed for daily stock-level shrinkage/min-variance benchmarks. |
| Monthly stock data | Adjusted monthly returns, prices, market cap, identifiers | available externally | External uploaded panel; not committed | Data owner | Validated Sprint 14 panel covers 2000-01 through 2022-12. |
| Compustat/characteristics | 33 firm characteristics, accounting variables, six-month lag, missing-data rules | available externally | External uploaded panel; not committed | Data owner | All listed uploaded characteristics detected; timing/lag assumptions remain documented pilot assumptions. |
| AMP universe | Large/mid-cap coverage, price filter, missing-data filter, rolling universe | pilot approximation | `artifacts/paper_replay/results/monthly_panel_pca_k6_2020_20y_lookback/` | Data owner | Sprint 14 uses configurable all/AMP-style selection; exact paper universe is not claimed. |
| PCA managed returns | 330 managed portfolios or equivalent saved matrices | pilot available | `artifacts/paper_replay/results/monthly_panel_pca_k6_2020_20y_lookback/` | Validation artifact | Built from the uploaded panel with 33 characteristics and 10 bins. |
| PCA mapping | PCA eigenvectors/mapping `V` by window | pilot available | `artifacts/paper_replay/results/monthly_panel_pca_k6_2020_20y_lookback/windows_pca/` | Validation artifact | `V` maps PCA factor weights to managed-portfolio weights. |
| RP-PCA mapping | RP-PCA factors and mapping `V` | missing | No saved artifacts | Data owner | Must be supplied externally. |
| IPCA factors | IPCA factor returns and mapping `V` | missing | No saved artifacts | Data owner | Full estimator/data bridge is not included in this sprint. |
| AP-Trees | Managed portfolio returns and mapping | missing | No saved artifacts | Data owner | The public PortOpt clone contains no complete AP-Trees artifact. |
| Benchmarks | S&P 500 or other benchmark returns | missing | No private/public input supplied | Data owner | Required for comparable benchmark path. |
| Old paper outputs | Old weights, objectives, returns, winners, Table 2 artifacts | missing | No saved outputs in `Gubba-tech/PortOpt_IPCA` | Data owner | Without these, old-solution parity cannot be tested. |

## Sprint 14 Real-Data Validation

The uploaded monthly panel is the primary empirical dataset for Sprint 14 and
has a real-data validation artifact under
`artifacts/paper_replay/monthly_panel_validation/`. The 20-year-lookback K=6
pilot produced 35 OSQP windows with optimal status. The paired cuOpt requests
were recorded as skipped because the runtime was unavailable; no CPU fallback
was used.

## Required Real-Data Handoff

For future reruns, provide or point to cleaned Parquet inputs for monthly
returns and characteristics, a documented universe policy, and the selected
lambda values. For old-solution parity, also provide per-window old weights or
objectives with the same asset/factor ordering. Keep those files under
`data/private/` or `artifacts/paper_replay/private/`; do not commit them.
