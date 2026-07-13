# cuFOLIO: GPU-Accelerated Portfolio Optimization

cuFOLIO is NVIDIA's portfolio optimization developer example. It includes the
original scenario-based Mean-CVaR workflow and, on this branch, a complementary
PortOpt unified quadratic-programming (QP) extension.

The QP extension is designed for portfolio research and backend validation. It
supports a CPU reference path through OSQP and an explicit NVIDIA cuOpt path on
CUDA-enabled machines. The two workflows share portfolio conventions, but the
QP implementation is not a replacement for the existing Mean-CVaR example.

## Scope At A Glance

| Track | Formulation | Backend | Current role |
| --- | --- | --- | --- |
| cuFOLIO baseline | Scenario-based Mean-CVaR LP | Existing cuFOLIO workflow | Original developer example |
| PortOpt extension | Convex portfolio QP and homogeneous max-Sharpe QP | OSQP validation, direct cuOpt | Research and backend integration |
| Paper replay | Cleaned monthly data, managed portfolios, PCA windows | Reproducible scripts | Sprint 14 uploaded-panel pilot; full paper replication is not claimed |

The QP work is intentionally scoped. It does not claim full paper replication,
hard tracking constraints, per-asset turnover limits, or a global QP speedup.
Existing upstream Mean-CVaR performance statements are separate from the QP
extension and should not be interpreted as QP benchmarks.

## Quick Start

The recommended development environment uses `uv`:

```bash
uv sync --extra dev

uv run python scripts/smoke_qp_env.py
uv run python examples/qp_min_variance_quickstart.py --backend osqp
uv run python examples/qp_max_sharpe_regularized_long_short.py --backend osqp
uv run python examples/qp_factor_space_pca_demo.py --backend osqp
```

For HPC bootstrap instructions, including the no-`uv` fallback, see
[`docs/dev_environment.md`](docs/dev_environment.md).

## GPU QP Backend

Use a CUDA extra that matches the CUDA version reported by `nvidia-smi`. Do not
install both CUDA extras in the same environment.

```bash
# Choose exactly one:
uv sync --extra cuda12 --extra dev
# or:
uv sync --extra cuda13 --extra dev

uv run python examples/qp_min_variance_quickstart.py --backend cuopt
uv run python examples/qp_max_sharpe_regularized_long_short.py --backend cuopt
```

The backend is selected explicitly with `--backend cuopt`. A missing or
incompatible cuOpt installation is reported as an error; the QP path does not
silently fall back to another solver.

## Unified QP Extension

The unified QP interface covers:

- minimum variance, mean-variance, target-return, and homogeneous max-Sharpe objectives;
- L1, L2, and combined regularization;
- long-short portfolios, turnover penalties, and benchmark L1 penalties;
- factor-exposure penalties and factor-tracking penalties;
- factor-space covariance through `V`, with deterministic PCA and external-factor adapters;
- OSQP validation and direct cuOpt execution using the same compiled problem convention.

The canonical quadratic objective is:

```text
0.5 * x.T @ Q @ x + q.T @ x
```

When a compiled problem is sent to cuOpt, the adapter uses the solver's
quadratic convention explicitly (`Q_cuopt = 0.5 * Q`). Variable bounds are
always passed explicitly, including negative lower bounds for long-short
portfolios.

Useful examples:

| Example | Purpose |
| --- | --- |
| [`qp_min_variance_quickstart.py`](examples/qp_min_variance_quickstart.py) | Minimum-variance QP |
| [`qp_mean_variance_quickstart.py`](examples/qp_mean_variance_quickstart.py) | Mean-variance QP |
| [`qp_max_sharpe_regularized_long_short.py`](examples/qp_max_sharpe_regularized_long_short.py) | Regularized long-short max-Sharpe QP |
| [`qp_factor_space_pca_demo.py`](examples/qp_factor_space_pca_demo.py) | Factor-space PCA covariance |
| [`qp_external_factor_adapter_demo.py`](examples/qp_external_factor_adapter_demo.py) | External factor adapter |
| [`qp_vs_cvar_baseline_overview.py`](examples/qp_vs_cvar_baseline_overview.py) | QP and Mean-CVaR scope comparison |

The detailed interface and backend notes are in
[`docs/portopt_qp_quickstart.md`](docs/portopt_qp_quickstart.md) and
[`docs/portopt_qp_examples.md`](docs/portopt_qp_examples.md).

## Paper Replay Workflow

The replay workflow is organized as a data contract followed by deterministic
portfolio construction and QP execution. The repository contains synthetic
fixtures for pipeline validation. Licensed or proprietary real data is not
committed to the repository.

Start by validating cleaned inputs:

```bash
uv run python scripts/validate_paper_cleaned_data.py \
  --monthly-returns <monthly_returns.parquet> \
  --characteristics <characteristics_monthly.parquet> \
  --report-path artifacts/paper_replay/cleaned_data_validation.json
```

The complete sequence is documented in
[`docs/paper_replay/README.md`](docs/paper_replay/README.md):

1. validate the cleaned monthly-return and characteristic tables;
2. build the characteristic-sorted asset-management portfolios;
3. build the managed-portfolio universe;
4. export PCA replay windows;
5. run the paper replay with the selected QP backend;
6. summarize results and record provenance.

See the data contract in
[`docs/paper_replay/cleaned_data_schema.md`](docs/paper_replay/cleaned_data_schema.md),
the availability checklist in
[`docs/paper_replay/data_availability_checklist.md`](docs/paper_replay/data_availability_checklist.md),
and the blocker report in
[`docs/paper_replay/data_blockers_for_full_replication.md`](docs/paper_replay/data_blockers_for_full_replication.md).

### Sprint 14 Uploaded-Panel Pilot

The uploaded monthly stock-characteristic panel is the primary empirical
dataset for Sprint 14. JKP is out of scope. The panel covers 2000 through
2022-12 with approximately 700 stocks per month in the intended universe.

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

Exact 2005-start 20-year-lookback replication is impossible with a panel that
starts in 2000. A 2005 short-lookback result is pilot-only; a 2020
20-year-lookback result has a short OOS period. Full IPCA/AP-Trees replication,
old-solution parity, and global QP speedup are not claimed. See
[`docs/paper_replay/monthly_characteristic_panel_schema.md`](docs/paper_replay/monthly_characteristic_panel_schema.md),
[`docs/paper_replay/monthly_panel_pilot.md`](docs/paper_replay/monthly_panel_pilot.md),
and [`docs/validation/sprint14_monthly_panel_pilot_validation.md`](docs/validation/sprint14_monthly_panel_pilot_validation.md).

### Sprint 15 Empirical Monthly-Panel Results

Sprint 15 extends the uploaded-panel pilot into a full 2005-2022 60-month
experiment, a 2020-2022 240-month experiment, K=2..6 sensitivity, a 4x4
lambda sensitivity grid, and a direct cuOpt GPU run. The external
`dfall_for_test.csv` is not committed. The empirical report, artifact inventory,
and generated output locations are documented in
[`docs/paper_replay/monthly_panel_empirical_results.md`](docs/paper_replay/monthly_panel_empirical_results.md)
and [`docs/paper_replay/sprint14_artifact_inventory.md`](docs/paper_replay/sprint14_artifact_inventory.md).

The 2005 run has 215 realized-return windows because the final 2022-12
rebalance has no next-month return. The 2020 K=6 baseline has 35/35 optimal
OSQP windows and both the B40 and H200 cuOpt runs have 35/35 optimal windows.
These are
uploaded-panel empirical results, not full paper replication, old-solution
parity, or global speedup claims.

For a saved result directory, generate Table-2-style metrics and plots with:

```bash
uv run python scripts/summarize_monthly_panel_results.py \
  --input-dir artifacts/paper_replay/results/<run_id> --plots
```

## Validation

The CPU validation suite can be run without a GPU:

```bash
uv sync --extra dev
uv run python scripts/smoke_qp_env.py
uv run python -m compileall -q src tests scripts examples benchmarks
uv run pytest tests/test_qp_compiled_convention.py -q
uv run pytest tests/test_qp_osqp_validation_backend.py -q
uv run pytest -m "not gpu" -q
uv run ruff check src tests examples scripts benchmarks
```

On a CUDA-enabled machine, after selecting the matching CUDA extra:

```bash
uv run pytest -m gpu tests/test_qp_cuopt_backend.py -q
```

Validation and provenance reports:

- [`Sprint 13 real-data bridge validation`](docs/validation/sprint13_real_data_bridge_validation.md)
- [`Sprint 12 paper replay validation`](docs/validation/sprint12_paper_replay_validation.md)
- [`Sprint 15 monthly-panel empirical validation`](docs/validation/sprint15_monthly_panel_empirical_results_validation.md)
- [`QP technical report`](docs/reports/portopt_cufolio_qp_technical_report.md)
- [`Benchmark interpretation note`](docs/reports/benchmark_interpretation_note.md)

## Original cuFOLIO Demo

The original Streamlit rebalancing demo remains available:

```bash
uv run streamlit run demo/rebalancing_streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

For the original demo's data setup and Docker workflow, see
[`docs/`](docs/) and the upstream
[NVIDIA-AI-Blueprints/cuFOLIO repository](https://github.com/NVIDIA-AI-Blueprints/cuFOLIO).

## Research Notes

- [`PortOpt paper math audit`](docs/portopt_paper_math_audit.md)
- [`Advisor demo script`](docs/reports/advisor_demo_script.md)
- [`QP release checklist`](docs/reports/release_checklist.md)
- [`Sprint 11 validation report`](docs/validation/sprint11_finalization_validation.md)

The mathematical source project for the portfolio formulations is
[Gubba-tech/PortOpt_IPCA](https://github.com/Gubba-tech/PortOpt_IPCA). This
repository contains the cuFOLIO integration, backend adapters, examples, and
validation artifacts.

## Contributing

Before opening a pull request:

```bash
uv run python -m compileall -q src tests scripts examples benchmarks
uv run pytest -m "not gpu" -q
uv run ruff check src tests examples scripts benchmarks
```

GPU-specific changes should include the CUDA extra used and the output of the
cuOpt-marked test suite. Keep benchmark claims tied to the exact problem class,
backend, hardware, and dataset used.

## References

- [NVIDIA cuOpt documentation](https://docs.nvidia.com/cuopt/)
- [OSQP documentation](https://osqp.org/docs/)
- [PortOpt_IPCA](https://github.com/Gubba-tech/PortOpt_IPCA)

## License

See [`LICENSE`](LICENSE).
