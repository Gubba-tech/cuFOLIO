# Sprint 16 Monthly Panel PCA Grid Validation

## Scope

This validation covers the PCA mean/covariance audit, managed-portfolio
diagnostics, paper-style K/lambda grid tooling, baseline tooling, and GPU job
provenance. The raw `dfall_for_test.csv` file is external and is not part of
the repository.

## Automated Checks

The focused Sprint 16 suite includes:

```text
tests/test_qp_pca_factor_mean_math.py
tests/test_qp_monthly_panel_diagnostics.py
tests/test_qp_monthly_panel_pca_grid.py
tests/test_qp_monthly_panel_pca_grid_summary.py
tests/test_qp_monthly_panel_pca_baselines.py
```

The checks cover raw-return factor means, projected covariance, factor sign
invariance, diagnostics artifacts, resumable grid execution, summary ranking
and heatmap data, and failed-window baseline handling.

## Commands

```bash
uv sync --extra dev
uv run python scripts/smoke_qp_env.py
uv run python -m compileall -q src tests scripts
uv run pytest tests/test_qp_compiled_convention.py -q
uv run pytest tests/test_qp_osqp_validation_backend.py -q
uv run pytest -m "not gpu" -q
uv run ruff check src tests scripts
```

On CUDA 13 machines, use only the matching extra:

```bash
uv sync --extra cuda13 --extra dev
uv run pytest -m gpu tests/test_qp_cuopt_backend.py -q
```

## GPU Provenance

| Job | Intended result | Status |
| --- | --- | --- |
| 46312 | Full K=2..6, 10x10 cuOpt grid on B40 | resumable Slurm run, chunk-isolated |
| 46311 | Full K=2..6, 10x10 cuOpt grid on H200 | queued/running Slurm run |
| 46313 | GPU baseline comparison after B40 grid | dependent Slurm run |
| 46314 | Top-ranked B40 constraint sensitivity | dependent Slurm run |

The job wrappers record node, GPU name, driver/CUDA report, cuOpt version,
partition, runtime, output directory, and exit status in
`artifacts/paper_replay/gpu_logs/*.metadata`. A grid is accepted only when its
metadata reports `run_status=0` and `grid_results.csv` contains 500 data rows.

## Claims Boundary

The validation demonstrates a corrected and reproducible QP/PCA workflow on the
uploaded panel. It does not establish full paper replication, old-solution
parity, or a general GPU speedup claim. The QP objective convention and
explicit variable-bound requirement remain unchanged.
