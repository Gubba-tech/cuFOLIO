# Table 2 Replay Targets

Source: Deng, Polak, Safikhani, and Shah, *A Unified Framework for Fast
Large-Scale Portfolio Optimization*, Table 2, arXiv:2303.12751 / Data Science
in Science 3(1), 2295539.

The paper reports a monthly rolling-window exercise from 2005-01-31 through
2022-12-31. The target rows below are recorded for future replay review. They
are not results produced by this cuFOLIO sprint.

| Method | CAGR | Sharpe | Smart Sharpe | Max drawdown | Volatility (ann.) | Calmar |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S&P 500 | 0.07 | 0.50 | 0.45 | -0.52 | 0.15 | 0.13 |
| Sample Cov min Var | 0.08 | 0.67 | 0.61 | -0.35 | 0.12 | 0.22 |
| L&W-NLS min Var | 0.08 | 0.73 | 0.66 | -0.38 | 0.12 | 0.22 |
| AP-Trees max SR + l1+l2^2 | 0.31 | 1.80 | 1.75 | -0.18 | 0.16 | 1.70 |
| PCA max SR + l1+l2^2 | 0.35 | 1.85 | 1.67 | -0.41 | 0.17 | 0.86 |
| RP-PCA max SR + l1+l2^2 | 0.59 | 3.40 | 3.29 | -0.09 | 0.14 | 6.96 |
| IPCA max SR | 1.24 | 3.74 | 3.73 | -0.22 | 0.23 | 5.52 |
| IPCA max SR + l1+l2^2 | 2.10 | 4.91 | 4.69 | -0.12 | 0.25 | 17.39 |

Additional reported targets include Omega, longest drawdown, information ratio,
skew, kurtosis, expected yearly return, beta, alpha, and correlation. The
source table is the target definition; Sprint 12 does not require matching all
rows or metrics. A replay report must state whether it uses the same universe,
factor estimator, date alignment, regularization grid, and realized return
construction before making any comparison.

## Sprint 13 Comparability

No Sprint 13 result is comparable to Table 2 yet. The available run is a
synthetic managed-portfolio pipeline with a short lookback and a small
characteristic set. It is useful for testing the data bridge, but it does not
use the full 20-year window, exact 330 managed portfolios, AMP universe,
CRSP/Compustat panel, IPCA estimator, or exact paper hyperparameter winners.

## Sprint 14/15 Monthly-Panel Comparison

The uploaded `dfall_for_test.csv` panel is an empirical input for the new
cuFOLIO PortOpt QP workflow. The rows below are shown side by side for
orientation only. They are **not a replication comparison** because the
monthly-panel pilots do not use the paper's exact 20-year 2005 lookback,
universe construction, factor estimator, or full OOS alignment.

| Design | Lookback | Windows | CAGR | Sharpe | Max drawdown | Volatility | Comparability |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Paper PCA max SR + l1+l2^2 | 240 months, 2005-2022 | paper Table 2 | 0.35 | 1.85 | -0.41 | 0.17 | Paper target |
| Monthly panel K=6 pilot, 2005 short lookback | 60 months | 12 Sprint 14 windows | 0.1645 | 1.5665 | -0.0458 | 0.1008 | Not comparable; pilot-only |
| Monthly panel K=6 pilot, 2020 lookback | 240 months | 35 Sprint 14 windows | 0.0844 | 0.4860 | -0.2162 | 0.2131 | Not comparable; short OOS |

Sprint 15 adds the completed runs below. The B40 row is a direct cuOpt result;
the OSQP row is the CPU validation for the same 2020 design.

| Design | Backend | Lookback | Windows | CAGR | Sharpe | Max drawdown | Volatility | Comparability |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Paper PCA max SR + l1+l2^2 | paper | 240 months, 2005-2022 | paper Table 2 | 0.35 | 1.85 | -0.41 | 0.17 | Paper target |
| Monthly panel K=6 | OSQP | 60 months, 2005-2022 | 215 | 0.1047 | 0.7314 | -0.4610 | 0.1527 | Not comparable; shorter lookback, 2 `user_limit` windows |
| Monthly panel K=6 | OSQP | 240 months, 2020-2022 | 35 | 0.0844 | 0.4860 | -0.2162 | 0.2131 | Not comparable; short OOS |
| Monthly panel K=6 | cuOpt/B40 | 240 months, 2020-2022 | 35 | 0.0844 | 0.4860 | -0.2162 | 0.2131 | Not comparable; short OOS and different data pipeline |

The Sprint 15 K and lambda sensitivity artifacts are under
`artifacts/paper_replay/results/monthly_panel_pca_k_grid_*` and
`artifacts/paper_replay/results/monthly_panel_pca_lambda_grid_*`. Do not
interpret a higher or lower pilot Sharpe as evidence of replication,
old-solution parity, or a QP speedup.
