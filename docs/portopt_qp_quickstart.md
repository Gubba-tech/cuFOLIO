# PortOpt Unified QP Quickstart

This extension adds a direct QP workflow to cuFOLIO. It supports stock-space
portfolio QPs and explicit factor-space QPs while leaving the existing
Mean-CVaR workflow unchanged.

## Install

CPU validation uses OSQP:

```bash
uv sync --extra dev
```

On a GPU host, choose exactly one CUDA extra based on `nvidia-smi`:

```bash
uv sync --extra cuda12 --extra dev
# or
uv sync --extra cuda13 --extra dev
```

## Run the examples

All examples default to `--backend osqp`. Selecting `--backend cuopt` is an
explicit GPU request; an unavailable cuOpt runtime raises an error rather than
falling back to a CPU solver.

```bash
uv run python examples/qp_min_variance_quickstart.py --backend osqp
uv run python examples/qp_mean_variance_quickstart.py --backend osqp
uv run python examples/qp_max_sharpe_regularized_long_short.py --backend osqp
uv run python examples/qp_factor_space_pca_demo.py --backend osqp
uv run python examples/qp_external_factor_adapter_demo.py --backend osqp
uv run python examples/qp_vs_cvar_baseline_overview.py
```

For an explicit cuOpt run:

```bash
uv run python examples/qp_min_variance_quickstart.py --backend cuopt
uv run python examples/qp_max_sharpe_regularized_long_short.py --backend cuopt
```

The examples print the solver status, compiled objective, and maximum
constraint violation. The max-Sharpe example also prints the positive scale,
excess return, and recovered weight sum. Factor-space examples print factor,
stock, and mapping shapes.

## Public API

The main imports are available from `cufolio`:

```python
from cufolio import QPParameters, QuadraticPortfolioOptimizer
from cufolio import build_external_factor_qp_data, build_pca_factor_qp_data
```

Use `mapping_mode="factor_space"` explicitly when the supplied mean and
covariance are factor statistics. External RP-PCA, IPCA, or AP-Trees outputs
can use `build_external_factor_qp_data(factor_returns, stock_mapping)`.

## Math Conventions

Compiled objectives use `0.5*x.T@Q*x + q.T@x`; regularization coefficients are
interpreted under this convention. The l1 split uses
`w_minus = -min(0,w) = max(-w,0)`. Max-Sharpe tracking error uses the scaled
homogeneous form `(p_tilde - c*b).T@Sigma@(p_tilde - c*b)`, not the ordinary
unscaled benchmark anchor. The full paper-alignment audit is in
[`docs/portopt_paper_math_audit.md`](portopt_paper_math_audit.md).

## Validation

```bash
uv run python scripts/smoke_qp_env.py
uv run python -m compileall -q src tests scripts examples
uv run pytest tests/test_qp_examples_smoke.py -q
uv run pytest -m "not gpu" -q
```

GPU validation is separate:

```bash
uv run pytest -m gpu tests/test_qp_examples_smoke.py -q
```

This quickstart makes no QP speedup claim and does not provide a complete
CRSP/Compustat/IPCA/AP-Trees replication.
