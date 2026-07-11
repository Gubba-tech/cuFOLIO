#!/usr/bin/env bash
set -euo pipefail

output_dir="${1:-artifacts/benchmarks}"

uv sync --extra dev

uv run python benchmarks/benchmark_qp_stock_level.py \
    --backend osqp \
    --n-assets 50 100 250 \
    --objectives min_variance mean_variance max_sharpe \
    --cases basic l2 l1_l2 long_short \
    --repeats 1 \
    --warmup 0 \
    --output-dir "${output_dir}"

uv run python benchmarks/benchmark_qp_factor_space.py \
    --backend osqp \
    --n-assets 100 500 \
    --n-factors 3 6 \
    --objectives mean_variance max_sharpe \
    --cases pca external \
    --repeats 1 \
    --warmup 0 \
    --output-dir "${output_dir}"

uv run python benchmarks/benchmark_qp_rolling_windows.py \
    --backend osqp \
    --n-assets 100 \
    --n-windows 10 \
    --objective max_sharpe \
    --mode stock \
    --repeats 1 \
    --output-dir "${output_dir}"
