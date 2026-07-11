# Factor-Space QP Design

The unified QP compiler has two explicit mapping modes. `QPParameters.mapping_mode`
defaults to `stock_space`; factor-space behavior must be selected explicitly with
`mapping_mode="factor_space"`.

## Stock-Space Reduced QP

Stock-space inputs use:

```text
mean_stock: N
covariance_stock: N x N
V: N x K
decision: z in R^K
```

The compiler reduces the stock objective into decision space:

```text
Q = V.T @ covariance_stock @ V
q_mean = -gamma * V.T @ mean_stock
p = V @ z
```

This is the existing reduced formulation and remains the default for backward
compatibility.

## True Factor-Space QP

Factor-space inputs use factor statistics directly:

```python
returns_dict = {
    "factor_mean": factor_mean,
    "factor_covariance": factor_covariance,
    "stock_mapping": V,
}
params = QPParameters(mapping_mode="factor_space", V=V, backend="osqp")
```

The equivalent `mean`/`covariance` keys are also accepted when the explicit
mapping mode is selected. Here `factor_mean` has length `K`, `factor_covariance`
is `K x K`, `V` is `N x K`, and:

```text
Q = factor_covariance
q_mean = -gamma * factor_mean
p = V @ z
1_N.T @ p = 1
```

Stock box, long-short, turnover, benchmark l1, and linear factor exposure
constraints all operate on `p`. l1 and l2-squared regularization also operate
on `V @ z`.

For factor-space max-Sharpe, the excess-return equality is:

```text
factor_mean.T @ z_tilde - risk_free_rate * c = 1
1_N.T @ V @ z_tilde - c = 0
z = z_tilde / c
stock_weights = V @ z
```

The risk-free term is multiplied by invested capital `c`; it is not inferred
from the factor-weight sum.

## Tracking Error

Factor-space tracking-error penalties require an explicit stock covariance under
`returns_dict["stock_covariance"]` or `returns_dict["tracking_covariance"]`:

```text
Q += 2 * lambda_tracking_error * V.T @ Sigma_stock @ V
q += -2 * lambda_tracking_error * V.T @ Sigma_stock @ benchmark
```

The factor covariance is not silently reused as a stock tracking covariance.

## Workflow Adapters

`build_external_factor_qp_data` accepts externally estimated factor returns and
`V`, so RP-PCA, IPCA, and AP-Trees implementations can provide their outputs
without being reimplemented here. `build_pca_factor_qp_data` provides a
deterministic PCA workflow for public or synthetic return matrices.

This sprint does not claim full CRSP/Compustat replication, RP-PCA/IPCA
replication, AP-Trees replication, or QP speedups.
