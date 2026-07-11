# QP Benchmark Interpretation Note

## Scope

The benchmark suite measures the PortOpt QP extension on deterministic
synthetic/public inputs. It is a reproducibility and backend-comparison tool,
not a claim about all QP workloads or real financial datasets.

The available benchmark paths are:

- stock-level cases in `benchmarks/benchmark_qp_stock_level.py`;
- factor-space PCA and external-factor cases in
  `benchmarks/benchmark_qp_factor_space.py`;
- repeated stock or factor solves in
  `benchmarks/benchmark_qp_rolling_windows.py`.

## Existing Artifacts

The Sprint 9 CPU summary is:

```text
artifacts/benchmarks/sprint9-cpu-final/summary.md
```

It contains successful OSQP-only stock, factor-space, and rolling-window
records, so it establishes runner coverage but no OSQP/cuOpt ratio.

The Sprint 10 math-patch summary is:

```text
artifacts/benchmarks/sprint10-math-patch/summary.md
```

Its B40 `backend=both` metadata records an NVIDIA RTX PRO 6000 Blackwell
Server Edition, CUDA 13.2, and cuOpt 26.4.0. The stock-level and factor-space
rows were all optimal. For example, the summary contains the following
artifact-scoped observations, where the ratio is OSQP divided by cuOpt:

| Case | Dimensions | Total ratio | Solve ratio |
| --- | --- | ---: | ---: |
| Stock friction | 50 assets | 0.0864 | 0.0477 |
| Stock friction | 100 assets | 0.5015 | 0.5666 |
| Stock full | 50 assets | 0.1711 | 0.1112 |
| Stock full | 100 assets | 0.1431 | 0.0816 |
| Factor full | 100 assets, 3 factors | 0.1323 | 0.0668 |

These values are observations in one generated B40 artifact run. They are not
global speedup claims, and they should not be extrapolated to H200, production
portfolios, or real-data workloads. Sprint 9 has no reviewed local GPU-vs-CPU
summary in the checkout; its H200 direct-runner result is documented in the
Sprint 9 validation record rather than merged into this comparison table.

## How to Regenerate

Install the CPU environment:

```bash
uv sync --extra dev
```

Recreate a compact Sprint 9-style CPU artifact:

```bash
uv run python benchmarks/benchmark_qp_stock_level.py \
  --backend osqp --n-assets 50 100 \
  --objectives min_variance mean_variance max_sharpe \
  --cases basic l2 l1_l2 long_short --repeats 1 --warmup 0 \
  --output-dir artifacts/benchmarks/<run>
uv run python benchmarks/benchmark_qp_factor_space.py \
  --backend osqp --n-assets 100 --n-factors 3 6 \
  --objectives mean_variance max_sharpe --cases pca external \
  --repeats 1 --warmup 0 --output-dir artifacts/benchmarks/<run>
uv run python benchmarks/summarize_qp_benchmarks.py \
  --input-dir artifacts/benchmarks/<run>
```

On a cuOpt-capable node, select one matching CUDA extra and use
`--backend both` for a paired comparison. Do not mix CUDA extras. Generated
raw artifacts are ignored by git; keep the environment metadata with any
reviewed summary.

## Reading `summary.md`

The summary groups successful rows by benchmark, objective, case, and
dimensions. `success rate` is the fraction of rows with status `optimal`.
Review status, maximum constraint violation, objective agreement, and the
environment metadata before reading a timing ratio.

## `solve_time` versus `total_time`

- `solve_time_sec` measures the backend solve call and is useful for isolating solver work.
- `total_time_sec` includes compilation, solver object construction, solve, and postprocessing.
- `compile_time_sec`, `solver_build_time_sec`, and `postprocess_time_sec` explain why the two measures differ.
- `solver_reported_solve_time_sec` is an optional backend-reported value and is not a substitute for wall-clock timing.

For an end-to-end user workflow, total time is the more relevant measure. For
solver-kernel analysis, solve time is useful, but it should not be interpreted
without build and compile overhead.

## Comparing OSQP and cuOpt

`--backend both` compiles the same problem inputs, runs OSQP first, and then
runs cuOpt. The summarizer emits a ratio only for matching successful
OSQP/cuOpt rows. A ratio below one means the OSQP time was lower than the cuOpt
time for that artifact row; it does not mean either backend is universally
faster.

Compile and build overhead matters especially for small problems. A GPU solver
can have a strong backend solve path while total time is dominated by setup,
data transfer, or problem construction. This is why both timing columns and
the component timings must be reviewed.

## Failed, Infeasible, and Skipped Rows

Check the row `status`, `raw_status`, and constraint diagnostics. An
`infeasible` or failed row is not a timing datapoint. A `skipped` cuOpt row
usually means the runtime was unavailable in the selected environment; OSQP
must not be substituted for it. The environment JSON identifies CUDA, GPU,
cuOpt availability, git commit, and backend selection.

No global QP speedup claim should be made from these artifacts. Any future
performance statement must name the artifact, hardware, dimensions, backend
versions, success status, and whether it refers to solve or total time.

