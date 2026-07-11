# PortOpt QP Examples and Notebooks

The product layer contains small, deterministic examples that exercise the
public QP workflow with synthetic or supplied in-memory data. They are
CPU-first and use `osqp` by default. Use `--backend cuopt` only when cuOpt is
installed on a compatible GPU host.

## Python examples

| Example | Demonstrates |
| --- | --- |
| `qp_min_variance_quickstart.py` | Fully invested stock-space minimum variance |
| `qp_mean_variance_quickstart.py` | Mean-variance objective with l2 regularization |
| `qp_max_sharpe_regularized_long_short.py` | Max-Sharpe reparameterization, l1/l2, and long-short budgets |
| `qp_factor_space_pca_demo.py` | Deterministic PCA factor-space QP with stock mapping |
| `qp_external_factor_adapter_demo.py` | External factor returns and mapping for RP-PCA/IPCA/AP-Trees outputs |
| `qp_vs_cvar_baseline_overview.py` | Conceptual comparison of the QP and existing Mean-CVaR workflows |

Run one example with:

```bash
uv run python examples/qp_max_sharpe_regularized_long_short.py \
    --backend osqp --seed 17 --n-assets 13
```

Replace `osqp` with `cuopt` only for an explicit GPU solve. There is no
automatic CPU fallback. The examples use feasible default dimensions for the
requested box bounds; smaller dimensions are raised when necessary so the
full-investment constraints remain feasible.

## Notebooks

The lightweight notebooks in `notebooks/portopt_qp/` mirror the examples:

- `00_qp_extension_overview.ipynb`
- `01_stock_min_mean_variance_qp.ipynb`
- `02_stock_max_sharpe_l1_l2_long_short.ipynb`
- `03_factor_space_pca_qp.ipynb`
- `04_external_factor_adapter_for_ipca_rppca_aptrees.ipynb`
- `05_qp_vs_cvar_baseline.ipynb`

They are intentionally compact and keep `backend="osqp"` explicit in the
code cells. Change that value to `"cuopt"` only in a CUDA-enabled environment.

## Output contract

The runnable examples print `status`, `objective`, and
`max_constraint_violation`. Max-Sharpe additionally reports `c_scale`,
`excess_return`, and `recovered_weights_sum`. Factor-space workflows report
the factor, stock, and mapping shapes. The baseline overview is explanatory
and reports `objective: not_solved` by design.

These examples are validation and onboarding artifacts, not performance
benchmarks. No QP speedup is claimed from them.
