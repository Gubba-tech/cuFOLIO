# Sprint 8 Examples and Product Docs Validation

Date: 2026-07-11

## Scope

Sprint 8 adds the product/demo layer for the unified PortOpt QP extension:

- public `cufolio` exports for QP parameters, optimizer, factor data/adapters,
  and QP exceptions;
- six CPU-first Python examples under `examples/`;
- six lightweight notebooks under `notebooks/portopt_qp/`;
- quickstart, examples, QP-vs-CVaR, factor-space, and limitations docs;
- README separation between the existing Mean-CVaR performance narrative and
  the new QP extension;
- CPU CI workflow and an example/API smoke test suite.

This sprint does not change the QP objective convention, cuOpt scaling, or
backend selection policy. `CompiledQP` remains
`0.5 * x.T @ Q @ x + q.T @ x`; cuOpt receives `Q_cuopt = 0.5 * Q`; and
`backend="cuopt"` never falls back to a CPU solver.

## Provenance

Implementation commit:

```text
c1528d8 Add Sprint 8 QP examples and product docs
```

This validation record is committed separately after the implementation
commit. The final tag is `sprint8-green-<provenance-commit>`.

## CPU validation

On the HPC login environment:

```text
python 3.13.2
uv 0.11.28
```

Passed:

```text
uv run python scripts/smoke_qp_env.py                         All checks passed
uv run python -m compileall -q src tests scripts examples     passed
uv run pytest tests/test_qp_public_api.py tests/test_qp_examples_smoke.py -q
                                                               8 passed, 2 skipped
uv run pytest -m "not gpu" -q                                  213 passed,
                                                               2 skipped,
                                                               61 deselected
uv run ruff check src tests examples scripts                    passed
git diff --check                                                passed
```

The two focused skips are the two GPU-marked example tests. The full CPU
result contains three pre-existing runtime warnings and no failures.

## GPU validation

The same explicit `--backend cuopt` examples and GPU smoke tests were run in
Slurm job-local environments after selecting `cuda13` from `nvidia-smi`.

```text
B40: NVIDIA RTX PRO 6000 Blackwell Server Edition, cuOpt 26.4.0
     min-variance, max-Sharpe, factor-space PCA: optimal
     GPU smoke: 2 passed, 7 deselected

H200: NVIDIA H200 NVL, cuOpt 26.4.0
      min-variance, max-Sharpe, factor-space PCA: optimal
      GPU smoke: 2 passed, 7 deselected
```

The GPU jobs used separate temporary environments and did not modify the
repository lockfile. No CPU fallback was used.

## Product artifacts

Examples:

```text
examples/qp_min_variance_quickstart.py
examples/qp_mean_variance_quickstart.py
examples/qp_max_sharpe_regularized_long_short.py
examples/qp_factor_space_pca_demo.py
examples/qp_external_factor_adapter_demo.py
examples/qp_vs_cvar_baseline_overview.py
```

Notebooks:

```text
notebooks/portopt_qp/00_qp_extension_overview.ipynb
notebooks/portopt_qp/01_stock_min_mean_variance_qp.ipynb
notebooks/portopt_qp/02_stock_max_sharpe_l1_l2_long_short.ipynb
notebooks/portopt_qp/03_factor_space_pca_qp.ipynb
notebooks/portopt_qp/04_external_factor_adapter_for_ipca_rppca_aptrees.ipynb
notebooks/portopt_qp/05_qp_vs_cvar_baseline.ipynb
```

## Claims and boundaries

- No QP speedup claim is made. Speedup discussion requires dedicated benchmark
  scripts and saved artifacts.
- The external factor adapter accepts supplied `factor_returns` and stock
  mapping `V`; it does not claim full RP-PCA, IPCA, or AP-Trees estimation.
- Full CRSP/Compustat/IPCA/AP-Trees replication is not claimed.
- Mean-CVaR remains a separate scenario-based LP workflow; the QP extension is
  complementary.
