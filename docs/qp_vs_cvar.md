# Unified QP and Mean-CVaR

cuFOLIO now contains two complementary portfolio optimization routes.

## Existing Mean-CVaR route

The existing Mean-CVaR workflow is scenario-based and uses a linear program.
It models tail risk through auxiliary variables and a confidence level over
return scenarios. It is the appropriate route when the portfolio decision is
driven by empirical or simulated tail-loss scenarios.

## Unified QP route

The PortOpt extension compiles a quadratic portfolio problem with the
convention:

```text
0.5 * x.T @ Q @ x + q.T @ x
```

It supports minimum variance, mean variance, target return, max Sharpe through
a positive-scale reparameterization, l1/l2 regularization, long-short
budgets, turnover and benchmark l1 budgets, linear factor exposure bounds,
tracking-error objective penalties, and explicit factor-space mappings.

The direct cuOpt backend maps this convention to cuOpt using `Q_cuopt = 0.5 *
Q`. CPU correctness validation uses OSQP only when `backend="osqp"` is selected.

## Choosing a route

- Use Mean-CVaR for scenario-based tail-risk optimization and the existing
  rebalancing workflow.
- Use unified QP for covariance/variance objectives, Sharpe-style objectives,
  quadratic regularization, and the supported linear portfolio constraints.
- Use both as complementary baselines when comparing variance-oriented and
  tail-risk-oriented portfolio decisions.

The QP route does not replace Mean-CVaR, and the examples do not claim a QP
speedup over the existing workflow. A hard tracking-error bound remains
outside the ordinary QP backend because it is a QCQP/SOCP constraint.
