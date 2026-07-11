#!/usr/bin/env bash
#SBATCH --job-name=cufolio-s9-qp-h200
#SBATCH --partition=h200x8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=artifacts/benchmarks/slurm-h200-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p "artifacts/benchmarks/slurm-${SLURM_JOB_ID}"
nvidia-smi | tee "artifacts/benchmarks/slurm-${SLURM_JOB_ID}/nvidia-smi.txt"

# Select the extra from nvidia-smi before submitting if the cluster differs.
uv sync --extra cuda13 --extra dev
bash scripts/run_qp_benchmarks_gpu_h200.sh "artifacts/benchmarks/slurm-${SLURM_JOB_ID}"
uv run python benchmarks/summarize_qp_benchmarks.py \
    --input-dir "artifacts/benchmarks/slurm-${SLURM_JOB_ID}"
tar -czf "artifacts/benchmarks/qp-benchmark-h200-${SLURM_JOB_ID}.tar.gz" \
    -C artifacts/benchmarks "slurm-${SLURM_JOB_ID}"
