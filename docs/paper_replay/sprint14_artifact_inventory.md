# Sprint 14 Artifact Inventory

The artifacts below were generated from the external real monthly panel. The
raw `dfall_for_test.csv` file is not committed to this repository. Generated
results under `artifacts/paper_replay/` are gitignored.

| Artifact | Status | Windows | OSQP optimal | cuOpt skipped | Managed portfolios | Characteristics | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `artifacts/paper_replay/monthly_panel_validation/` | present, gitignored | N/A | N/A | N/A | N/A | 35 detected; 33 default | 2000-01 through 2022-12; 276 months |
| `artifacts/paper_replay/managed_portfolios_monthly_panel/` | present, gitignored | N/A | N/A | N/A | 330 | 33 default | Full managed-return table and membership Parquet |
| `artifacts/paper_replay/windows_pca_monthly_panel/` | absent | N/A | N/A | N/A | N/A | N/A | Windows are stored under each run's `windows_pca/` directory instead |
| `artifacts/paper_replay/results/monthly_panel_pca_k6_2005_short_lookback/` | present, gitignored | 12 | 12 | 12 | 330 | 33 default | 60-month pilot-only run |
| `artifacts/paper_replay/results/monthly_panel_pca_k6_2020_20y_lookback/` | present, gitignored | 35 | 35 | 35 | 330 | 33 default | 240-month lookback, short OOS |

## Common Provenance

- Source: external `/lustre/.../PortOpt-IPCA-GPU/data/dfall_for_test.csv`.
- Raw date range: 2000-01-31 through 2022-12-31.
- Raw rows: 241,347.
- Default method: value-weighted characteristic-sorted deciles.
- PCA mapping `V` maps factor weights to managed-portfolio weights.
- `backend=cuopt` was never replaced with OSQP. Login-node cuOpt requests are
  recorded as skipped because no cuOpt runtime was installed.
- These artifacts are empirical pilot evidence, not full paper replication.

## Sprint 15 Outputs

Sprint 15 reuses the validated Sprint 14 panel and managed-portfolio artifacts
and adds the following gitignored result directories:

| Artifact | Windows | Backend/status | Notes |
| --- | ---: | --- | --- |
| `artifacts/paper_replay/results/monthly_panel_pca_k6_2005_2022_60m/` | 215 | OSQP: 213 optimal, 2 `user_limit` | Full 2005-2022 60-month pilot; final realized return is 2022-11 |
| `artifacts/paper_replay/results/monthly_panel_pca_k6_2020_2022_240m/` | 35 | OSQP: 35 optimal | 240-month lookback, short OOS |
| `artifacts/paper_replay/results/monthly_panel_pca_k6_2020_2022_240m_cuopt_b40/` | 35 | cuOpt/B40: 35 optimal | Slurm job 46129 on `b40x4-02` |
| `artifacts/paper_replay/results/monthly_panel_pca_k6_2020_2022_240m_cuopt_h200/` | 35 | cuOpt/H200: 35 optimal | Slurm job 46128 on `h200x8-04` |
| `artifacts/paper_replay/results/monthly_panel_pca_k_grid_2005_2022_60m/` | 12 per K | OSQP: 12/12 per K | Controlled K=2..6 sensitivity |
| `artifacts/paper_replay/results/monthly_panel_pca_k_grid_2020_2022_240m/` | 35 per K | OSQP: 35/35 per K | Full short-OOS K=2..6 sensitivity |
| `artifacts/paper_replay/results/monthly_panel_pca_lambda_grid_2020_2022_240m/` | 12 per pair | OSQP: 12/12 per pair | Controlled 4x4 lambda grid |

The B40 and H200 metadata record the node, GPU, CUDA report, cuOpt version,
status, and runtime in `artifacts/paper_replay/gpu_logs/monthly-panel-b40-46129.metadata`
and `artifacts/paper_replay/gpu_logs/monthly-panel-h200-46128.metadata`. See
[`monthly_panel_empirical_results.md`](monthly_panel_empirical_results.md) for
metrics, plots, and comparison caveats.
