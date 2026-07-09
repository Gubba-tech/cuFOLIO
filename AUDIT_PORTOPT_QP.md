# PortOpt Unified QP Extension Audit

Date: 2026-07-09

Working branch: `feature/portopt-unified-qp-cuopt`

Primary target repository: NVIDIA cuFOLIO (`NVIDIA-AI-Blueprints/cuFOLIO`)

Reference source repository: PortOpt_IPCA (`Gubba-tech/PortOpt_IPCA`)

## 1. Scope

This audit treats cuFOLIO as the baseline project. The goal is to add the PortOpt paper's unified QP framework as a first-class cuFOLIO extension, not to create a standalone replacement library.

Existing cuFOLIO Mean-CVaR LP and Mean-Variance SOCP/QCQP workflows must remain intact. The new work should add QP-specific modules, parameters, backends, tests, docs, benchmarks, notebooks, and a new agent skill without breaking current CVaR or rebalancing APIs.

## 2. NVIDIA Skills Review

Files reviewed in cuFOLIO:

- `AGENTS.md`
- `skills/cufolio/SKILL.md`
- `skills/cufolio/references/workflows/agent_recipes.md`

Files reviewed in `NVIDIA/skills`:

- `skills/cufolio/SKILL.md`

Findings:

- The cuFOLIO repository copy of `skills/cufolio/SKILL.md` is newer than the external `NVIDIA/skills` copy. It documents both Mean-CVaR and Mean-Variance SOCP/QCQP workflows.
- The external `NVIDIA/skills` copy documents Mean-CVaR only and does not include the newer Mean-Variance SOCP/QCQP guidance.
- No root-level `AGENTS.md` was found in the external `NVIDIA/skills` clone; the relevant agent entry point is the cuFOLIO repository's `AGENTS.md`.
- Both skill versions emphasize the same critical rule: use NVIDIA cuOpt for GPU production solves and do not silently substitute CPU solvers.

## 3. Current cuFOLIO Package Structure

cuFOLIO uses a flat `src/` package directory mapped to package name `cufolio` in `pyproject.toml`:

```text
src/
├── __init__.py
├── backtest.py
├── base_optimizer.py
├── base_parameters.py
├── cvar_data.py
├── cvar_optimizer.py
├── cvar_parameters.py
├── cvar_utils.py
├── mean_variance_optimizer.py
├── mean_variance_parameters.py
├── portfolio.py
├── rebalance.py
├── scenario_generation.py
├── settings.py
└── utils.py
```

Supporting areas:

- `tests/`: CPU/unit tests and skill tests.
- `tests/benchmarks/`: current workflow benchmark tests.
- `skills/cufolio/`: current cuFOLIO skill, evals, benchmark description.
- `notebooks/`: existing CVaR/Mean-Variance/rebalancing notebooks.
- `demo/`: Streamlit demo.
- `data/`: data notes; default price CSV is gitignored.

`pyproject.toml` declares:

- Package name: `cufolio`
- Python: `>=3.11`
- Core deps: `numpy`, `pandas`, `cvxpy`, `scikit-learn`, `seaborn`, `yfinance`, `pydantic`
- Extras:
  - `dev`
  - `cuda12`
  - `cuda13`
  - `cuda13-socp`
- NVIDIA index for `cuml-cu*` and `cuopt-cu*`.

## 4. Current Optimizer Architecture

### Base Optimizer

File: `src/base_optimizer.py`

`BaseOptimizer` owns the common optimization lifecycle:

- Stores `returns_dict`, `tickers`, `n_assets`, covariance, regime name/range, previous portfolio state, and deep-copied params.
- Converts weight bounds from float/dict/array to per-asset `np.ndarray` via `_update_weight_constraints`.
- Dispatches setup through `_setup_optimization_problem`:
  - `api_settings.api == "cvxpy"` calls `_setup_cvxpy_problem()` and `_assign_cvxpy_parameter_values()`.
  - `api_settings.api == "cuopt_python"` calls `_validate_cuopt_setup()` and `_setup_cuopt_problem()`.
- Dispatches solving through `solve_optimization_problem(...)`:
  - CVXPY path requires explicit solver settings.
  - cuOpt path calls subclass `_solve_cuopt_problem(...)`.
- Wraps solved weights/cash into `Portfolio(name=..., tickers=..., weights=..., cash=..., time_range=...)`.

Implication for PortOpt QP:

- The new `QuadraticPortfolioOptimizer` should subclass `BaseOptimizer`.
- It should reuse `ApiSettings(api="cuopt_python")` for production GPU solves.
- CPU validation should use `ApiSettings(api="cvxpy")` or a separate explicit backend flag, never an automatic fallback when GPU was requested.
- Result rows should remain `pd.Series` with cuFOLIO-style flat columns.

### Settings

File: `src/settings.py`

Relevant models:

- `ApiSettings`
  - `api: Literal["cvxpy", "cuopt_python"]`
  - weight/cash constraint mode for CVXPY
  - risk-aversion scaling
  - optional pickle path
- `ReturnsComputeSettings`
  - LOG/LINEAR/ABSOLUTE/PNL returns
- `ScenarioGenerationSettings` and `KDESettings`
  - CVaR scenario generation, including GPU KDE

Implication:

- QP should not require CVaR scenarios.
- It should consume the flat `returns_dict` produced by `utils.calculate_returns(...)`.
- Any QP-specific backend choice beyond current API options should live in `QPParameters` or be mapped cleanly onto `ApiSettings`.

## 5. Current Mean-CVaR LP Implementation

Files:

- `src/cvar_parameters.py`
- `src/cvar_data.py`
- `src/cvar_optimizer.py`
- `src/cvar_utils.py`

Current Mean-CVaR formulation:

- Variables:
  - asset weights `w`
  - cash `c`
  - CVaR auxiliaries `u_j`
  - VaR threshold `t`
  - optional positive/negative auxiliaries for leverage/turnover
  - optional integer variables for cardinality
- Objective:
  - minimize `risk_aversion * CVaR - expected_return`, or
  - maximize expected return with a hard CVaR limit.
- Constraints:
  - `sum(w) + c = 1`
  - weight/cash bounds
  - scenario loss constraints
  - leverage as `norm1(w) <= L_tar`
  - optional turnover as `norm1(w - w_prev) <= T_tar`
  - optional cardinality and group constraints

Backends:

- CVXPY path in `_setup_cvxpy_problem`.
- Direct cuOpt Python API path in `_setup_cuopt_problem`, using:
  - `cuopt.linear_programming.problem.Problem`
  - `LinearExpression`
  - `CONTINUOUS` and optional `INTEGER`
  - `MINIMIZE`/`MAXIMIZE`
- The cuOpt path manually creates scalar variables and row constraints in loops.

Scenario generation:

- `cvar_utils.generate_cvar_data(...)` mutates/returns `returns_dict` with `"cvar_data"`.
- KDE GPU path uses lazy imports of `cuml` and `cuml.neighbors`.

Result shape:

- `_result_columns = ["regime", "solver", "solve time", "return", "CVaR", "obj"]`
- `solve_optimization_problem(...)` returns `(result_row, Portfolio)`.

Implication:

- PortOpt QP should not modify the CVaR code path.
- Comparison examples should instantiate CVaR exactly as today, then instantiate the new QP optimizer separately.

## 6. Current Mean-Variance / SOCP / QCQP Implementation

Files:

- `src/mean_variance_parameters.py`
- `src/mean_variance_optimizer.py`

Current capabilities:

- Basic mean-variance objective:
  - CVXPY: minimize `risk_aversion * w.T @ Sigma @ w - mean.T @ w`
  - cuOpt: same form with `QuadraticExpression`
- Optional hard variance cap:
  - CVXPY: `portfolio_variance <= var_limit`
  - cuOpt: quadratic constraint via `QuadraticExpression`
  - documented as SOCP/QCQP, not ordinary QP.
- Constraints:
  - `sum(w) + c = 1`
  - weight/cash bounds
  - leverage via `norm1(w) <= L_tar` or `w_pos/w_neg`
  - optional turnover
  - optional group constraints
- Cardinality is explicitly rejected for cuOpt Mean-Variance and not implemented for CVXPY.

cuOpt backend details:

- Uses direct cuOpt Python API, not CVXPY + `cp.CUOPT`.
- Imports from `cuopt.linear_programming.problem`:
  - `Problem`
  - `LinearExpression`
  - `QuadraticExpression`
  - `CONTINUOUS`
  - `MINIMIZE`, `MAXIMIZE`
- Adds one scalar cuOpt variable per weight and per auxiliary.
- Pads quadratic matrices to `problem.NumVariables x problem.NumVariables`.
- Solves with `cuopt.linear_programming.solver_settings.SolverSettings`.

Result shape:

- `_result_columns = ["regime", "solver", "solve time", "return", "variance", "obj"]`

Implication:

- The PortOpt QP cuOpt backend should prefer direct cuOpt Python API, matching `MeanVariance`.
- CVXPY + `cp.CUOPT` may remain relevant for CVaR LP, but direct cuOpt is the native route already present for QP/QCQP.
- Tracking-error hard constraints must be treated as QCQP/SOCP, consistent with current variance-cap handling. For MVP QP, tracking error should be an objective penalty only.

## 7. returns_dict, Portfolio, Backtest, and Rebalance APIs

### returns_dict

Created by `utils.calculate_returns(...)`.

Flat keys:

- `"return_type"`
- `"returns"`: pandas DataFrame
- `"regime"`: dict with `"name"` and `"range"`
- `"dates"`
- `"mean"`: `np.ndarray`
- `"covariance"`: `np.ndarray`
- `"tickers"`: list

CVaR adds:

- `"cvar_data"`: `CvarData(mean, R, p)`

Implication:

- QP consumes `"mean"`, `"covariance"`, and `"tickers"`.
- QP factor mapping should accept user-supplied `V` or factor-estimator outputs without changing the base returns_dict shape.

### Portfolio

File: `src/portfolio.py`

`Portfolio(name="", tickers=None, weights=None, cash=0.0, time_range=None)` stores:

- `name`
- `tickers`
- `weights`
- `cash`
- `time_range`

It also provides:

- `print_clean(...)`
- self-financing checks
- expected return / variance helpers
- serialization and plotting helpers

Implication:

- QP output should always return a standard `Portfolio` with stock-level weights.
- Factor-space QP should additionally expose factor weights and mapping metadata on the optimizer/result row, but the returned `Portfolio.weights` should be stock weights aligned to `returns_dict["tickers"]`.

### Backtest

File: `src/backtest.py`

Primary entry:

- `portfolio_backtester(test_portfolio, returns_dict, risk_free_rate=0.0, test_method="historical", benchmark_portfolios=None)`
- `backtest_against_benchmarks(...)` returns `(backtest_results, ax)`.

Metrics include:

- returns
- cumulative returns
- portfolio name
- mean portfolio return
- Sharpe
- Sortino
- max drawdown

Implication:

- QP examples and notebooks should reuse `portfolio_backtester` directly.
- No separate QP backtest loop should be invented.

### Rebalance

File: `src/rebalance.py`

Primary class:

- `rebalance_portfolio(...)`
- `re_optimize(...)` returns `(results_dataframe, re_optimize_dates, cumulative_portfolio_value)`.

Current rebalance implementation is CVaR-specific: it takes `cvar_params`, generates CVaR data, and instantiates `cvar_optimizer.CVaR`.

Implication:

- Rebalancing reuse is not plug-in generic yet.
- QP rebalancing can either:
  1. Add a generic optimizer hook to `rebalance_portfolio`, or
  2. Add a QP-specific rebalancing wrapper that preserves the same return format.
- The less invasive path is to add an optional optimizer factory while keeping existing CVaR defaults unchanged.

## 8. PortOpt_IPCA Relevant Functions

Reference repository audited at `/lustre/nvwulf/home/weicdeng/PortOpt-IPCA-GPU`.

### Data and Managed Portfolios

File: `PortOpt_factor/data_processing/data_helper.py`

- `single_sort(df, characteristics)`
  - Builds characteristic decile managed-portfolio returns.
- `triple_sort(df, char1, char2, char3)`
  - Builds characteristic triple-sort managed-portfolio returns.
- `preprocess_data(...)`
  - Proprietary/local-data preprocessing around Asness universe membership.

AP-Trees:

- No explicit AP-Tree implementation was found in the source repository.
- Current code only contains sort helpers that can be used as managed-portfolio building blocks.

### Imputation

File: `PortOpt_factor/data_processing/imputation.py`

- `backward_cross_section_imputation(C)`
- `my_inverse(A)`
- rolling sum helpers

These are useful for paper replication but should not be in the first QP compiler MVP.

### QP Optimizer

File: `PortOpt_factor/optimizer/pyport.py`

Key functions:

- `portfolio_optimization(...)`
  - Main OSQP wrapper.
  - Handles minimum variance, target-return mean-variance, max-Sharpe reparameterization, factor mapping, long-only/long-short, l1/l2 regularization, turnover, benchmark exposure, and factor exposure.
- `constrain_matrix(...)`
  - Builds OSQP `A, l, u` linear constraints.
- `sigMat_expend(...)`
  - Expands covariance matrix for auxiliary variables and factor/max-Sharpe variants.
- `penalty_vector(...)`
  - Builds linear objective terms.
- `sigMatShrinkage(...)`
  - Adds l2-style covariance augmentation.
- `nearestPD(...)`, `isPD(...)`
  - Positive-definite repair/check.
- `Dmat(...)`
  - Ordering constraint matrix.

Issues to avoid copying directly:

- Returns equal weights when `not w_opt.all()`, which treats legitimate zero weights as failure.
- Does not return/check OSQP status.
- Uses ambiguous truthiness checks on arrays/matrices.
- Mixes formulation compilation, solver call, and recovery in one large function.
- Uses OSQP sparse form; cuFOLIO direct cuOpt backend uses scalar variable/expression construction.

### PCA / RP-PCA / IPCA

Files:

- `PortOpt_factor/optimizer/pca_rppca.py`
- `PortOpt_factor/optimizer/ipca.py`

Functions:

- `PCA_factor(...)`
- `PRPCA_factor(...)`
- `RPPCAOOS(...)`
- `IPCA_factor(...)`
- `IPCAOOS(...)`

Findings:

- `RPPCAOOS` and `IPCAOOS` embed rolling-window backtesting and hyperparameter grid search inside the factor model functions.
- `ipca.py` duplicates PCA/RP-PCA functions from `pca_rppca.py`.
- `IPCA_factor` returns `Gamma`, `Factors`, next-period returns/features, last features, and CUSIPs; it does not expose a clean cuFOLIO-ready factor mapping object yet.

Implication:

- For QP MVP, support generic user-supplied `V` first.
- Add PCA/RP-PCA/IPCA adapters later that return:
  - factor expected returns `mu_f`
  - factor covariance `Sigma_f`
  - mapping matrix `V` such that stock weights `p = V @ z`
  - tickers aligned to cuFOLIO returns_dict

## 9. QP Extension Integration Plan

Suggested cuFOLIO-native files:

```text
src/exceptions.py
src/qp_parameters.py
src/qp_formulations.py
src/qp_constraints.py
src/qp_regularization.py
src/qp_backend.py
src/qp_optimizer.py
```

Recommended responsibilities:

- `exceptions.py`
  - `GPUBackendUnavailable`
  - `QPCompilationError`
- `qp_parameters.py`
  - Pydantic `QPParameters` model with objective, backend, bounds, regularization, turnover, benchmark, factor exposure, and optional `V`.
- `qp_formulations.py`
  - Deterministic sparse QP compiler:
    - min variance
    - mean variance
    - target return
    - max Sharpe reparameterization
    - factor-space max Sharpe
  - Data classes for compiled QP matrices, variable slices, and recovery metadata.
- `qp_regularization.py`
  - l2 matrix update
  - l1 auxiliary-variable construction
- `qp_constraints.py`
  - budget, bounds, long-short splits, turnover, benchmark l1, factor exposure, scaled max-Sharpe constraints.
- `qp_backend.py`
  - direct cuOpt Python backend for production GPU solve.
  - explicit OSQP/CVXPY backend for tests only.
  - no silent CPU fallback.
- `qp_optimizer.py`
  - `QuadraticPortfolioOptimizer(BaseOptimizer)` matching cuFOLIO solve style and returning `(result_row, Portfolio)`.

## 10. QP Compiler Design Notes

Internal standard form should be deterministic and testable:

```text
minimize 0.5 * x.T @ Q @ x + q.T @ x
subject to A_eq x = b_eq
           A_ineq x <= b_ineq
           lower <= x <= upper
```

Use `scipy.sparse.csr_matrix` for CPU assembly and tests unless direct cuOpt construction needs dense/padded matrices.

Required metadata:

- variable names and slices
- original asset count `N`
- factor count `K` if factor-space
- objective type
- whether max-Sharpe recovery is required
- mapping matrix shape
- constraint names/row slices

For cuOpt direct API, translate compiled form into:

- scalar variables with bounds
- one `LinearExpression` per equality/inequality row
- padded `QuadraticExpression` for objective

For CPU validation, solve the compiled form through OSQP or CVXPY explicitly selected by tests/benchmarks.

## 11. Solver Backend Choice

Recommended GPU route:

- Use direct cuOpt Python API.
- Rationale:
  - cuFOLIO already uses this route for Mean-Variance QP/QCQP in `mean_variance_optimizer.py`.
  - It supports `QuadraticExpression`.
  - It avoids introducing a separate CVXPY + `cp.CUOPT` route for QP while CVaR already has its own CVXPY LP workflow.

Guardrail:

- If `import cuopt` fails for `backend="cuopt"` / `api="cuopt_python"`, raise `GPUBackendUnavailable`.
- Do not call OSQP/CVXPY from the GPU backend.
- Do not catch cuOpt-unavailable errors and continue with CPU results.

## 12. Result Row Requirements

The new QP result row should include the requested fields while preserving cuFOLIO's flat `pd.Series` style:

- `regime`
- `solver`
- `objective`
- `status`
- `solve_time`
- `total_time`
- `objective_value`
- `expected_return`
- `variance`
- `volatility`
- `sharpe`
- `lambda_l1`
- `lambda_l2`
- `short_budget`
- `turnover`
- `max_constraint_violation`
- optional factor fields:
  - `factor_weights`
  - `stock_weights`
  - `mapping_matrix_shape`

The returned `Portfolio` should contain stock-level weights and cash, even for factor-space solves.

## 13. Tests to Add First

CPU-only initial tests:

- `tests/test_qp_min_variance.py`
- `tests/test_qp_mean_variance.py`
- `tests/test_qp_target_return.py`
- `tests/test_qp_max_sharpe_reparameterization.py`
- `tests/test_qp_l1_regularization.py`
- `tests/test_qp_l2_regularization.py`
- `tests/test_qp_long_short_constraints.py`
- `tests/test_qp_turnover_constraints.py`
- `tests/test_qp_benchmark_constraints.py`
- `tests/test_qp_factor_mapping.py`
- `tests/test_qp_cuopt_guard.py`

Use small deterministic fixtures:

- `N=5`
- PSD covariance matrix
- non-degenerate expected returns
- feasible long-only and long-short settings
- small synthetic mapping `V`

GPU tests:

- Mark with `pytest.mark.gpu`.
- Skip only when cuOpt is unavailable.
- Skip message: `"cuOpt GPU runtime unavailable; QP GPU test skipped."`

## 14. Refactor Risks

- cuFOLIO's `rebalance_portfolio` is CVaR-specific today; QP rebalancing requires a careful extension point.
- cuOpt direct Python API setup is scalar/loop-based; a generic sparse QP compiler must still be translated into cuOpt expressions.
- Max-Sharpe reparameterization changes variable recovery and all constraints must be scaled consistently.
- Factor-space QP must keep stock-level `Portfolio.weights` aligned with `returns_dict["tickers"]`.
- l1 regularization and long-short constraints both introduce positive/negative splits; variable naming/slices must avoid collisions.
- Tracking-error hard constraints are QCQP/SOCP, not QP. They should be an objective penalty in the MVP.
- PortOpt source contains several numerical shortcuts and status-handling gaps that should be replaced by explicit validation.
- Documentation must avoid claiming GPU speedups until benchmark scripts generate artifact CSV/JSON/Markdown outputs.

## 15. Immediate Next Step

Implement a minimal QP skeleton in small commits:

1. Add `exceptions.py`, `qp_parameters.py`, and a no-solve `QuadraticPortfolioOptimizer` skeleton.
2. Add a sparse compiler for long-only minimum variance with fully invested constraint.
3. Add CPU validation tests for compiler dimensions and OSQP/CVXPY solution.
4. Add direct cuOpt backend guard and GPU skip test.
5. Extend objective support one case at a time.
