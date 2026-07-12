#!/usr/bin/env bash
#SBATCH --job-name=cufolio-s15-pca-h200
#SBATCH --partition=h200x8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=artifacts/paper_replay/gpu_logs/monthly-panel-h200-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PATH="$HOME/.local/bin:$PATH"

log_dir="artifacts/paper_replay/gpu_logs"
run_dir="artifacts/paper_replay/results/monthly_panel_pca_k6_2020_2022_240m_cuopt_h200"
mkdir -p "$log_dir"
start_epoch="$(date +%s)"
status=0

set +e
{
    echo "slurm_job_id=${SLURM_JOB_ID:-unknown}"
    echo "slurm_node=${SLURMD_NODENAME:-unknown}"
    nvidia-smi
    echo "gpu_query=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)"
    uv sync --extra cuda13 --extra dev
    cuopt_version="$(uv run python -c 'import cuopt; print(getattr(cuopt, "__version__", "unknown"))')"
    echo "cuopt_version=$cuopt_version"
    uv run python scripts/run_monthly_panel_pca_pilot.py \
        --monthly-panel /lustre/nvwulf/home/weicdeng/PortOpt-IPCA-GPU/data/dfall_for_test.csv \
        --output-dir "$run_dir" \
        --start-date 2020-01-31 \
        --end-date 2022-12-31 \
        --lookback-months 240 \
        --k-values 6 \
        --characteristics all \
        --n-bins 10 \
        --weighting value \
        --lambda-l1 1.7e-4 \
        --lambda-l2 1e-3 \
        --short-budget 0.2 \
        --w-min -0.08 \
        --w-max 0.08 \
        --backend cuopt \
        --assume-characteristics-lagged
    if [[ "${RUN_SHORT_LOOKBACK:-0}" == "1" ]]; then
        uv run python scripts/run_monthly_panel_pca_pilot.py \
            --monthly-panel /lustre/nvwulf/home/weicdeng/PortOpt-IPCA-GPU/data/dfall_for_test.csv \
            --output-dir artifacts/paper_replay/results/monthly_panel_pca_k6_2005_2022_60m_cuopt_h200 \
            --start-date 2005-01-31 --end-date 2022-12-31 \
            --lookback-months 60 --k-values 6 --characteristics all \
            --n-bins 10 --weighting value --lambda-l1 1.7e-4 --lambda-l2 1e-3 \
            --short-budget 0.2 --w-min -0.08 --w-max 0.08 \
            --backend cuopt --allow-short-lookback --assume-characteristics-lagged
    fi
} 2>&1 | tee "$log_dir/monthly-panel-h200-${SLURM_JOB_ID:-local}.log"
status=${PIPESTATUS[0]}
set -e

end_epoch="$(date +%s)"
cat > "$log_dir/monthly-panel-h200-${SLURM_JOB_ID:-local}.metadata" <<EOF
slurm_job_id=${SLURM_JOB_ID:-unknown}
slurm_node=${SLURMD_NODENAME:-unknown}
gpu_type=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unavailable)
cuda_report=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([^ ]*\).*/\1/p' | head -1 || true)
cuopt_version=${cuopt_version:-unknown}
cuopt_run_status=$status
runtime_seconds=$((end_epoch - start_epoch))
output_dir=$run_dir
EOF
exit "$status"
