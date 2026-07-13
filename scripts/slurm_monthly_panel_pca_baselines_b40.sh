#!/usr/bin/env bash
#SBATCH --job-name=cufolio-s16-baselines-b40
#SBATCH --partition=b40x4-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --exclusive
#SBATCH --output=artifacts/paper_replay/gpu_logs/monthly-panel-pca-baselines-b40-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

log_dir="artifacts/paper_replay/gpu_logs"
output_dir="${BASELINE_OUTPUT_DIR:-artifacts/paper_replay/results/monthly_panel_pca_baselines_cuopt_b40}"
returns_path="${MANAGED_RETURNS_PATH:-artifacts/paper_replay/managed_portfolios_monthly_panel/managed_portfolio_returns.parquet}"
mkdir -p "$log_dir" "$output_dir"
start_epoch="$(date +%s)"
status=0
cuopt_version="unknown"

set +e
{
    echo "slurm_job_id=${SLURM_JOB_ID:-unknown}"
    echo "slurm_node=${SLURMD_NODENAME:-unknown}"
    echo "managed_returns_path=$returns_path"
    nvidia-smi
    echo "gpu_query=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)"
    uv sync --extra cuda13 --extra dev
    cuopt_version="$(uv run python -c 'import cuopt; print(getattr(cuopt, "__version__", "unknown"))')"
    echo "cuopt_version=$cuopt_version"
    uv run python scripts/run_monthly_panel_pca_baselines.py \
        --managed-portfolio-returns "$returns_path" \
        --output-dir "$output_dir" \
        --backend cuopt \
        --workers 1
    status=$?
} 2>&1 | tee "$log_dir/monthly-panel-pca-baselines-b40-${SLURM_JOB_ID:-local}.log"
pipeline_status=${PIPESTATUS[0]}
if [[ "$pipeline_status" != "0" ]]; then
    status="$pipeline_status"
fi
set -e

end_epoch="$(date +%s)"
cat > "$log_dir/monthly-panel-pca-baselines-b40-${SLURM_JOB_ID:-local}.metadata" <<EOF
slurm_job_id=${SLURM_JOB_ID:-unknown}
slurm_node=${SLURMD_NODENAME:-unknown}
gpu_type=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unavailable)
cuda_report=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([^ ]*\).*/\1/p' | head -1 || true)
cuopt_version=$cuopt_version
run_status=$status
runtime_seconds=$((end_epoch - start_epoch))
partition=${SLURM_JOB_PARTITION:-unknown}
output_dir=$output_dir
EOF
exit "$status"
