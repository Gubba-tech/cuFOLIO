# Sprint 15 Monthly-Panel Empirical Results Validation

## Provenance

- Branch: `feature/portopt-unified-qp-cuopt`
- Repository: `Gubba-tech/cuFOLIO`
- Implementation commit: `TBD before the implementation commit`
- Provenance commit: `TBD after the provenance commit`
- Raw data: external `/lustre/nvwulf/home/weicdeng/PortOpt-IPCA-GPU/data/dfall_for_test.csv`
- Raw data committed: no
- Sprint 14 base tag: `sprint14-green-13f0de8`

## Data Validation

The real panel validator passed with 241,347 rows, 1,089 unique assets, and
276 complete months from 2000-01-31 through 2022-12-31. It reports an average
of 874.45 stocks per month, with 664 minimum, 877.5 median, and 1,048 maximum.
Thirty-five characteristics are available and the 33 default paper-style
characteristics are used. The managed-portfolio builder produced 330
characteristic-sort portfolios.

The raw data and generated outputs remain outside the git index. The data
start in 2000, so an exact 2005-start 240-month lookback is infeasible. The
2020-start 240-month run is therefore short OOS, and the 2005 60-month run is
pilot-only.

## Experiments

| Experiment | Output | Result |
| --- | --- | --- |
| 2005-2022, K=6, 60m, OSQP | `artifacts/paper_replay/results/monthly_panel_pca_k6_2005_2022_60m/` | 215 windows; 213 optimal, 2 `user_limit` |
| 2020-2022, K=6, 240m, OSQP | `artifacts/paper_replay/results/monthly_panel_pca_k6_2020_2022_240m/` | 35/35 optimal |
| K=2..6, 2005 60m, OSQP | `artifacts/paper_replay/results/monthly_panel_pca_k_grid_2005_2022_60m/` | 12/12 optimal for every K; controlled sample |
| K=2..6, 2020 240m, OSQP | `artifacts/paper_replay/results/monthly_panel_pca_k_grid_2020_2022_240m/` | 35/35 optimal for every K |
| K=6, 2020 240m, 4x4 lambda grid, OSQP | `artifacts/paper_replay/results/monthly_panel_pca_lambda_grid_2020_2022_240m/` | 12/12 optimal per pair; controlled sample |

The K sensitivity uses `lambda_l1=1.7e-4` and `lambda_l2=1e-3`. The lambda
grid uses L1 `[0, 1e-6, 1.7e-4, 1e-3]` and L2 `[0, 1e-6, 1e-3, 2.9e-2]`.
The final 2022-12 rebalance is excluded from realized-return metrics because
the panel ends in 2022-12.

## GPU Validation

The B40 job was submitted as Slurm job `46129` and completed successfully:

- Node: `b40x4-02`
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- CUDA report: 13.2
- cuOpt: `cuopt-cu13==26.4.0`
- Runtime: 162 seconds
- Output: `artifacts/paper_replay/results/monthly_panel_pca_k6_2020_2022_240m_cuopt_b40/`
- Status: 35 optimal, 0 failed, 0 skipped
- Maximum constraint violation: `1.945e-10`

The H200 job `46128` remained pending for priority and was not used. No OSQP
rerun was used as a fallback for the B40 cuOpt request.

## Metrics And Plots

The baseline metrics are:

| Design/backend | CAGR | Annual vol. | Sharpe | Max drawdown | Optimal |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2005 60m OSQP | 0.104721 | 0.152717 | 0.731418 | -0.460989 | 213/215 |
| 2020 240m OSQP | 0.084365 | 0.213060 | 0.485968 | -0.216178 | 35/35 |
| 2020 240m cuOpt/B40 | 0.084365 | 0.213061 | 0.485969 | -0.216178 | 35/35 |

Each baseline output includes `metrics_table.csv`, `metrics_table.md`,
`cumulative_returns.csv`, `monthly_returns.csv`,
`constraint_diagnostics.csv`, and `weights_summary.csv`. Matplotlib-only
plots include cumulative returns, underwater, monthly-return heatmap,
constraint violation, gross exposure, K sensitivity, and lambda sensitivity.

## CPU Validation Commands

The following commands are required for the implementation commit:

```bash
uv sync --extra dev
uv run python scripts/smoke_qp_env.py
uv run python -m compileall -q src tests scripts examples benchmarks
uv run pytest tests/test_qp_monthly_panel_empirical_summary.py -q
uv run pytest tests/test_qp_monthly_panel_grid.py -q
uv run pytest -m "not gpu" -q
uv run ruff check src tests examples scripts benchmarks
```

## Claims Boundary

- This is not full original paper replication.
- Full IPCA, RP-PCA, and AP-Trees replication is not claimed.
- Old-solution parity is not claimed without old weights and objectives.
- No global QP speedup is claimed.
- The existing Mean-CVaR LP workflow is untouched.
- The compiled QP convention remains `0.5*x.T@Q@x + q.T@x`; cuOpt receives
  `0.5*Q`, and variable bounds are passed explicitly.
