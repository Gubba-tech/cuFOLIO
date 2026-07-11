# Unified QP Benchmarks

Sprint 9 adds reproducible benchmark runners for the PortOpt unified QP
extension. The default inputs are deterministic synthetic/public data and do
not require CRSP, Compustat, IPCA, or AP-Trees replication.

## CPU run

Install the CPU validation environment and run the moderate suite:

```bash
uv sync --extra dev
bash scripts/run_qp_benchmarks_cpu.sh
```

The script writes timestamped directories under `artifacts/benchmarks/`. A
single runner can be invoked directly for a smaller run:

```bash
uv run python benchmarks/benchmark_qp_stock_level.py \
    --backend osqp --n-assets 50 100 \
    --objectives min_variance mean_variance max_sharpe \
    --cases basic l2 l1_l2 long_short --repeats 1 --warmup 0
```

Factor-space and rolling examples:

```bash
uv run python benchmarks/benchmark_qp_factor_space.py \
    --backend osqp --n-assets 100 --n-factors 3 6 \
    --objectives mean_variance max_sharpe --cases pca external \
    --repeats 1 --warmup 0

uv run python benchmarks/benchmark_qp_rolling_windows.py \
    --backend osqp --n-assets 100 --n-windows 10 \
    --objective max_sharpe --mode stock --repeats 1
```

## GPU run

On a B40 or H200 node, choose one CUDA extra from `nvidia-smi` and do not mix
extras:

```bash
uv sync --extra cuda13 --extra dev
bash scripts/run_qp_benchmarks_gpu_b40.sh
```

The H200 script has the same artifact contract:

```bash
bash scripts/run_qp_benchmarks_gpu_h200.sh
```

Slurm templates capture `nvidia-smi`, request one GPU, install the selected
CUDA 13 extra, run the benchmark, summarize it, and create an artifact
tarball. Adjust the partition or CUDA extra for a different cluster:

```bash
sbatch scripts/slurm_qp_benchmark_b40.sh
sbatch scripts/slurm_qp_benchmark_h200.sh
```

## Summaries and interpretation

Point the summarizer at one timestamped run directory:

```bash
uv run python benchmarks/summarize_qp_benchmarks.py \
    --input-dir artifacts/benchmarks/<timestamp>
```

Use `total_time_sec` for the end-to-end path and inspect compile/build/solve
components separately. `solve_time_sec` is wall-clock backend solve time;
`solver_reported_solve_time_sec` is included when the backend exposes it.
Feasibility and objective-agreement fields must be reviewed alongside timing.

The summary's speed ratios are labeled `observed speed ratio in this benchmark
run` and are only produced for matching successful OSQP/cuOpt problems. They
are artifact-specific observations, not universal speedup claims. No README
QP speedup statement is added by this sprint.

Large generated artifacts are ignored by default. Small reviewed summaries may
be placed under `artifacts/benchmarks/reviewed/`.

Benchmark objective and regularization values follow the implementation's
`0.5*x.T@Q*x + q.T@x` convention. Math alignment, including the homogeneous
max-Sharpe tracking-error penalty, is documented in
[`docs/portopt_paper_math_audit.md`](portopt_paper_math_audit.md).
