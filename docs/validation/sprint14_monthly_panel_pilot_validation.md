# Sprint 14 Monthly Panel Pilot Validation

## Provenance

| Item | Value |
| --- | --- |
| Branch | `feature/portopt-unified-qp-cuopt` |
| Implementation commit | To be filled with the Sprint 14 implementation commit |
| Real-data source | External uploaded monthly panel; not committed |
| Source data committed | No |
| JKP touched | No |
| Mean-CVaR changed | No |

## Real Panel

- Raw date range: 2000-01-31 through 2022-12-31.
- Raw month count: 276.
- Raw row count: 241,347.
- Full-panel average stocks per month: 874.45.
- All 35 listed uploaded characteristics detected, including the two preserved extras `beme_adj` and `c2d`.
- Default paper-style set used: 33 characteristics.
- Exact 2005-start 240-month lookback: impossible.
- Earliest raw-panel 240-month OOS date: 2020-01-31.

## CPU Validation

The following checks passed:

```text
uv run python -m compileall -q src tests scripts examples benchmarks
uv run pytest tests/test_qp_monthly_panel_loader.py -q                 2 passed
uv run pytest tests/test_qp_monthly_panel_managed_portfolios.py -q     1 passed
uv run pytest tests/test_qp_monthly_panel_pca_pilot.py -q              2 passed
uv run pytest tests/test_qp_pca_replay_export.py tests/test_qp_paper_data_schema.py tests/test_qp_paper_replay.py tests/test_qp_compiled_convention.py tests/test_qp_osqp_validation_backend.py -q
                                                                    13 passed, 1 skipped
uv run ruff check ...                                                   passed
```

The focused Sprint 14 suite covers validator reports, Parquet conversion,
characteristic mapping, managed portfolio sorting/weights, PCA windows, an
end-to-end synthetic pilot, and cuOpt skip-without-fallback behavior.

## Real Pilots

### 2005 Short-Lookback Pilot

- Lookback: 60 months.
- K: 6.
- Managed portfolios: 330.
- Windows attempted: 12 controlled windows.
- OSQP: 12 optimal.
- cuOpt: 12 skipped because the GPU runtime was unavailable.
- Pilot label: short-lookback pilot only; not full paper replication.
- Artifact: `artifacts/paper_replay/results/monthly_panel_pca_k6_2005_short_lookback/`.

### 2020 Twenty-Year-Lookback Pilot

- Lookback: 240 months.
- K: 6.
- Managed portfolios: 330.
- Windows attempted: 35, through 2022-11 because t+1 realized returns are required.
- OSQP: 35 optimal.
- cuOpt: 35 skipped because the GPU runtime was unavailable.
- Pilot label: 20-year-lookback pilot with short OOS period.
- Artifact: `artifacts/paper_replay/results/monthly_panel_pca_k6_2020_20y_lookback/`.

## Explicit Non-Claims

- No full paper replication claim.
- No old-solution parity claim because old weights/objectives are unavailable.
- No full IPCA claim.
- No full AP-Trees claim.
- No global QP speedup claim.
- No CPU fallback for `backend="cuopt"`.
- Mean-CVaR LP remains untouched.
