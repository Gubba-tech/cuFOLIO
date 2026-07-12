# Sprint 13 Real-Data Bridge Validation

Date: 2026-07-11

## Scope

Sprint 13 builds the cleaned-data bridge needed for a paper-style empirical
pilot: Parquet validation, configurable AMP-style universe selection, lagged
managed portfolios, PCA replay-window export, external factor export, and
synthetic end-to-end tests. No existing Mean-CVaR LP or QP convention changed.

## Provenance

Implementation commit:

```text
10966c6 Add cleaned paper data bridge
```

After this validation record is committed, its commit is the Sprint 13
provenance commit and is tagged `sprint13-green-<provenance_commit>`.

Sprint 12 remains available at `sprint12-green-65118d3`.

## Data Availability

No real cleaned CRSP/Compustat-style data were available in the workspace. The
public `Gubba-tech/PortOpt_IPCA` clone contained source code but no exportable
matrices or raw data. The Sprint 13 pilot therefore used only committed,
deterministic synthetic Parquet fixtures:

```text
tests/fixtures/paper_data/monthly_returns.parquet
tests/fixtures/paper_data/characteristics_monthly.parquet
```

The full status is in:

```text
docs/paper_replay/data_availability_checklist.md
docs/paper_replay/data_blockers_for_full_replication.md
```

No real PCA K=6 2005-2022 pilot ran. No IPCA external replay ran. No
proprietary data were committed.

## CPU Validation

```text
uv sync --extra dev
    Resolved 181 packages; checked 90 packages

uv run python scripts/smoke_qp_env.py
    Python 3.13.2; numpy 2.2.6; scipy 1.16.3; pandas 2.3.2;
    pyarrow 22.0.0; pytest 9.0.3; cvxpy 1.9.2

uv run python -m compileall -q src tests scripts examples benchmarks
    passed

uv run pytest tests/test_qp_paper_replay.py -q
    3 passed, 1 skipped

uv run pytest tests/test_qp_paper_data_schema.py -q
    3 passed

uv run pytest tests/test_qp_pca_replay_export.py -q
    1 passed

uv run pytest -m "not gpu" -q
    230 passed, 2 skipped, 67 deselected, 3 warnings

uv run ruff check src tests examples scripts benchmarks
    All checks passed
```

The three warnings are pre-existing runtime/solver warnings and did not fail
the suite.

## Synthetic Pipeline

The exact synthetic bridge commands completed with:

```text
validate_paper_cleaned_data.py
    status=pass, issues=0

build_amp_universe.py
    dates=24, selected_rows=432

build_managed_portfolios.py
    rows=720, portfolios=30

export_pca_replay_windows.py
    windows_written=11, K=3, lookback=12 months

run_paper_replay.py
    replay_rows=11, backend=osqp

summarize_paper_replay.py
    metric_rows=1, plots written
```

The synthetic PCA exporter records that `V` maps factor weights to
managed-portfolio weights. It does not label those weights as individual stock
weights. Generated outputs are under the gitignored directory:

```text
artifacts/paper_replay/synthetic_data_validation.md
artifacts/paper_replay/synthetic_universe/
artifacts/paper_replay/synthetic_managed_portfolios/
artifacts/paper_replay/synthetic_windows_pca/
artifacts/paper_replay/synthetic_results_pca/
```

## GPU and Claims

No Sprint 13 GPU run was submitted. The direct cuOpt backend and no-fallback
rule remain validated by earlier B40 work. This sprint makes no global QP
speedup claim, no old-solution parity claim, and no full CRSP/Compustat/IPCA/
AP-Trees replication claim. The existing Mean-CVaR LP workflow is untouched.

