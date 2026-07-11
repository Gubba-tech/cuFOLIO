# Unified QP Benchmark Design

Sprint 9 benchmarks the existing PortOpt QP extension without changing its
mathematical conventions or backend selection behavior. The default data are
deterministic synthetic/public in-memory data; no CRSP, Compustat, IPCA, or
AP-Trees data pipeline is introduced.

## Scope

The benchmark suite measures three paths:

1. stock-level minimum variance, mean variance, and max-Sharpe QPs;
2. true factor-space QPs using deterministic PCA data or externally supplied
   factor returns and mapping `V`;
3. repeated rolling-window solves with synthetic monthly-style data.

The existing Mean-CVaR LP workflow is not modified or included in the QP
timing comparison.

## Problem generators and cases

`benchmarks/qp_benchmark_utils.py` provides deterministic PSD covariance,
positive mean, anchor-weight, stock-return, and external-factor generators.
Each artifact records the seed and git commit.

Stock-level cases are:

- `basic`: fully invested, long-only box bounds;
- `l2`: positive l2-squared regularization;
- `l1_l2`: positive l1 and l2-squared regularization;
- `long_short`: `w_min=-0.08`, `w_max=0.08`, `short_budget=0.2`;
- `friction`: total turnover and benchmark l1 budgets plus a tracking-error
  objective penalty;
- `full`: max-Sharpe with long-short, l1/l2, friction, and tracking penalty.

Factor-space cases are deterministic PCA, external factor adapter, and a full
max-Sharpe case using supplied factor returns and mapping `V`. Rolling mode
uses synthetic stock or PCA factor data and updates the next window's previous
weights from the prior successful solution. Solver warm starts are not used.

## Solver comparison policy

`--backend both` always runs OSQP first and then cuOpt on the same compiled
problem inputs. A cuOpt-unavailable row is recorded as `skipped`; OSQP is never
used as a replacement for a requested cuOpt result. A failed OSQP baseline is
retained as a failed row and cuOpt is still attempted.

The compiled objective remains:

```text
0.5 * x.T @ Q @ x + q.T @ x
```

The direct cuOpt path continues to use `Q_cuopt = 0.5 * Q`. The benchmark
runner never changes `backend="cuopt"` into a CPU solve.

## Timing definitions

Every result row contains:

- `compile_time_sec`: time to compile portfolio inputs into `CompiledQP`;
- `solver_build_time_sec`: time to construct the CVXPY/OSQP or direct cuOpt
  problem object;
- `solve_time_sec`: wall-clock time spent in the backend solve call;
- `solver_reported_solve_time_sec`: solver-reported time when available;
- `wall_clock_solve_time_sec`: measured backend solve wall time;
- `postprocess_time_sec`: solution extraction, recovery, and diagnostics;
- `total_time_sec`: compile plus build plus solve plus postprocess.

For cuOpt both solver-reported and wall-clock solve times are recorded. For
OSQP the wall-clock CVXPY/OSQP solve call is recorded, together with OSQP's
reported time when available. Total time is the primary end-to-end comparison
when compile and build overheads matter.

## Artifact schema

Each timestamped run directory contains:

- `environment.json` with Python, package, git, CUDA, GPU, and backend metadata;
- `*_results.csv` and `*_results.jsonl` with dimensions, sparsity, statuses,
  timing components, objective/return/risk metrics, and constraint diagnostics;
- `*_summary.md` with a small run-local status/timing summary;
- rolling runs additionally contain aggregate CSV/JSON/Markdown summaries.

The summarizer writes `summary.csv`, `summary.json`, and `summary.md`. It only
reports an observed OSQP/cuOpt speed ratio when both backends succeeded for the
same problem and repeat.

## No-speedup-claim policy

Generated summaries use the phrase `observed speed ratio in this benchmark
run`. They do not establish a universal speedup, a product-wide performance
claim, or a claim about real financial datasets. README performance language
must remain separate until generated artifacts are reviewed.
