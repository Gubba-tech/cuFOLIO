#!/usr/bin/env bash
#SBATCH --job-name=synthetic-osqp-cuopt
#SBATCH --partition=b40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/nvwulf/scratch/weicdeng/cuFOLIO_synthetic_benchmark_logs_30d_20260810/%x-%j.out
#SBATCH --error=/lustre/nvwulf/scratch/weicdeng/cuFOLIO_synthetic_benchmark_logs_30d_20260810/%x-%j.err

set -euo pipefail

: "${BENCHMARK_WORKTREE:?Set BENCHMARK_WORKTREE to the immutable detached worktree}"
: "${EXPECTED_SHA:?Set EXPECTED_SHA to the registered benchmark commit}"
: "${BENCHMARK_OUTPUT_ROOT:?Set BENCHMARK_OUTPUT_ROOT to the shared scratch output}"

PYTHON_BIN="${BENCHMARK_PYTHON:-/lustre/nvwulf/home/weicdeng/cuFOLIO/.venv/bin/python}"
cd "${BENCHMARK_WORKTREE}"

ACTUAL_SHA="$(git rev-parse HEAD)"
test "${ACTUAL_SHA}" = "${EXPECTED_SHA}"
test -z "$(git status --porcelain)"
test -x "${PYTHON_BIN}"

export EXPECTED_SHA
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "registered_sha=${ACTUAL_SHA}"
echo "output_root=${BENCHMARK_OUTPUT_ROOT}"
echo "python=${PYTHON_BIN}"
scontrol show job "${SLURM_JOB_ID}"
nvidia-smi

"${PYTHON_BIN}" benchmarks/synthetic_solver_benchmark.py \
  --config configs/synthetic_solver_benchmark.yaml \
  --output-root "${BENCHMARK_OUTPUT_ROOT}"

"${PYTHON_BIN}" benchmarks/analyze_synthetic_solver_benchmark.py \
  --config configs/synthetic_solver_benchmark.yaml \
  --output-root "${BENCHMARK_OUTPUT_ROOT}"
