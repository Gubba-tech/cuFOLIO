#!/usr/bin/env bash
set -euo pipefail

output_dir="${1:-artifacts/benchmarks}"

uv run python benchmarks/benchmark_qp_stock_level.py \
    --backend both \
    --n-assets 50 100 250 \
    --objectives min_variance mean_variance max_sharpe \
    --cases basic l2 l1_l2 long_short full \
    --repeats 3 \
    --warmup 1 \
    --output-dir "${output_dir}"

uv run python benchmarks/benchmark_qp_factor_space.py \
    --backend both \
    --n-assets 100 500 \
    --n-factors 3 6 \
    --objectives mean_variance max_sharpe \
    --cases pca external full \
    --repeats 3 \
    --warmup 1 \
    --output-dir "${output_dir}"

uv run python benchmarks/benchmark_qp_rolling_windows.py \
    --backend both \
    --n-assets 100 250 \
    --n-windows 10 50 \
    --objective max_sharpe \
    --mode stock factor_pca \
    --repeats 1 \
    --output-dir "${output_dir}"
