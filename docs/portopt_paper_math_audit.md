# PortOpt Paper Math Audit

## 1. Scope

This audit checks the QP mathematics used by the cuFOLIO PortOpt extension
against the published PortOpt paper. It documents implementation choices,
normalizations, and notation notes; it does not revise the paper.

The audit covers the compiled objective, regularization, max-Sharpe
reparameterization, tracking-error penalty, factor-space risk-free adjustment,
and linear portfolio constraints. It does not claim a full CRSP/Compustat,
IPCA, RP-PCA, or AP-Trees research-pipeline replication.

## 2. QP Convention

The implementation uses:

```text
0.5 * x.T @ Q @ x + q.T @ x
```

Some paper formulas write a pure quadratic term as `x.T @ A @ x` without the
leading `0.5`. Under the implementation convention, a penalty written as
`lambda * x.T @ A @ x` contributes:

```text
Q += 2 * lambda * A
```

This factor of two is intentional. Regularization coefficients in the code are
interpreted under the `0.5*x.T@Q*x` convention. When comparing with formulas
that omit the `0.5` risk term, penalty coefficients may need rescaling.

The direct cuOpt API uses:

```text
x.T @ Q_cuopt @ x + c.T @ x
```

Therefore the backend maps:

```text
Q_cuopt = 0.5 * Q
c = q
```

## 3. l1 Positive/Negative Split

The paper and implementation are aligned. For any portfolio vector `w`:

```text
w_plus  = max(0, w)
w_minus = -min(0, w) = max(-w, 0)
w       = w_plus - w_minus
||w||_1 = 1.T @ (w_plus + w_minus)
```

No complementarity constraint is needed. If both split variables have positive
mass for the same coordinate, reducing both by their common minimum preserves
their difference and improves or preserves the nonnegative l1 objective.

The paper's `w_minus` definition is not wrong; it is mathematically equivalent
to the implementation's nonnegative negative-part variable.

## 4. l2 Factor Regularization

For a factor mapping `V`, the factor-space l2 penalty is:

```text
||V @ w||_2^2 = w.T @ V.T @ V @ w
```

The compiler adds:

```text
Q += 2 * lambda_l2 * V.T @ V
```

under the `0.5*x.T@Q*x` convention. If this expression is written in an
eigenbasis of a factor covariance `R_f`, the basis rotation must be explicit
and dimensionally compatible. The implementation uses the direct matrix form
to avoid an ambiguous basis convention.

## 5. Max-Sharpe Stock-Space Reparameterization

Stock-space max-Sharpe uses:

```text
w_tilde = c * w
(mu - rf*1).T @ w_tilde = 1
1.T @ w_tilde = c
w = w_tilde / c
```

The scale variable is constrained positive. Box, long-short, turnover,
benchmark l1, and linear exposure constraints use the scaled stock exposure
`p_tilde` and are recovered only after dividing by `c`.

## 6. Max-Sharpe Factor-Space Risk-Free Adjustment

In true factor-space mode:

```text
p = V @ z
1_N.T @ V @ z = 1
```

The scaled excess-return row is therefore:

```text
factor_mean.T @ z_tilde - rf * c = 1
```

It is not generally:

```text
(factor_mean - rf*1_K).T @ z_tilde = 1
```

unless factor weights themselves sum to `c`. In the paper's empirical setup
`rf=0`, so this distinction does not change those reported cases.

## 7. Max-Sharpe Regularization Interpretation

The regularized max-Sharpe QP uses homogeneous scaled penalties:

```text
lambda_l1 * ||V @ z_tilde||_1
lambda_l2 * ||V @ z_tilde||_2^2
```

These are the scaled counterparts of penalties on recovered portfolio
exposures. Coefficients are interpreted using the implementation's
`0.5*x.T@Q*x` convention.

## 8. Max-Sharpe Tracking-Error Penalty

For ordinary QP objectives, tracking error is:

```text
lambda_te * (M @ x - b).T @ Sigma @ (M @ x - b)
```

For max-Sharpe, the decision is scaled and the correct homogeneous penalty is:

```text
lambda_te * (M @ x_tilde - c*b).T @ Sigma @ (M @ x_tilde - c*b)
```

The implementation adds the corresponding `[x_tilde, c]` quadratic block:

```text
Q_xx += 2 * lambda_te * M.T @ Sigma @ M
Q_xc += -2 * lambda_te * M.T @ Sigma @ b
Q_cx += -2 * lambda_te * b.T @ Sigma @ M
Q_cc += 2 * lambda_te * b.T @ Sigma @ b
```

It does not add the ordinary-QP linear benchmark term to `q_x` in the
max-Sharpe case. Since `M @ x_tilde - c*b = c * (M @ x - b)`, the penalty is
homogeneous in the reparameterization and remains a convex QP. The block is
positive semidefinite because it is:

```text
[M, -b].T @ Sigma @ [M, -b]
```

when `Sigma` is positive semidefinite.

## 9. Constraints

The implementation supports box bounds, long-short budgets, total turnover,
benchmark l1 exposure, linear factor exposure, and tracking-error objective
penalties.

For max-Sharpe, stock constraints use:

```text
p_tilde = c * p
lower_i * c <= p_tilde_i <= upper_i * c
||p_tilde - c*previous_weights||_1 <= c*turnover_budget
||p_tilde - c*benchmark_weights||_1 <= c*benchmark_l1_budget
```

A hard tracking-error bound is not compiled as an ordinary QP because it is a
QCQP/SOCP constraint. Tracking error is supported here as a quadratic
objective penalty.

## 10. Out of Scope

- hard tracking-error constraints as ordinary QPs;
- per-asset turnover limits;
- full CRSP/Compustat/IPCA/AP-Trees replication;
- AP-Trees/IPCA estimators beyond supplied external factor adapters;
- global QP speedup claims. Benchmark ratios remain artifact-specific.
