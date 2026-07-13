#!/usr/bin/env bash
#SBATCH --job-name=cufolio-s16-pca-grid-h200
#SBATCH --partition=h200x8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --exclusive
#SBATCH --output=artifacts/paper_replay/gpu_logs/monthly-panel-pca-grid-h200-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

log_dir="artifacts/paper_replay/gpu_logs"
run_dir="${GRID_OUTPUT_DIR:-artifacts/paper_replay/results/monthly_panel_pca_grid_2020_2022_240m_cuopt_h200}"
data_path="${MONTHLY_PANEL_DATA:-/lustre/nvwulf/home/weicdeng/PortOpt-IPCA-GPU/data/dfall_for_test.csv}"
gpu_tag="${GPU_TAG:-h200}"
cuda_extra="${CUDA_EXTRA:-cuda13}"
read -r -a k_values <<< "${K_VALUES:-2 3 4 5 6}"
chunk_size="${GRID_CHUNK_SIZE:-20}"
total_combinations=$(( ${#k_values[@]} * 10 * 10 ))
mkdir -p "$log_dir" "$run_dir"

start_epoch="$(date +%s)"
status=0
cuopt_version="unknown"

set +e
{
    echo "slurm_job_id=${SLURM_JOB_ID:-unknown}"
    echo "slurm_node=${SLURMD_NODENAME:-unknown}"
    echo "k_values=${k_values[*]}"
    echo "grid_chunk_size=$chunk_size"
    echo "total_combinations=$total_combinations"
    echo "grid_output_dir=$run_dir"
    echo "monthly_panel_data=$data_path"
    nvidia-smi
    echo "gpu_query=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)"
    uv sync --extra "$cuda_extra" --extra dev
    cuopt_version="$(uv run python -c 'import cuopt; print(getattr(cuopt, "__version__", "unknown"))')"
    echo "cuopt_version=$cuopt_version"
    for ((offset = 0; offset < total_combinations; offset += chunk_size)); do
        echo "starting_combination_offset=$offset"
        uv run python scripts/run_monthly_panel_pca_paper_grid.py \
            --monthly-panel "$data_path" \
            --output-dir "$run_dir" \
            --start-date 2020-01-31 \
            --end-date 2022-12-31 \
            --lookback-months 240 \
            --k-values "${k_values[@]}" \
            --lambda-l1-grid paper \
            --lambda-l2-grid paper \
            --n-bins 10 \
            --weighting value \
            --short-budget 0.2 \
            --w-min -0.08 \
            --w-max 0.08 \
            --risk-free-rate 0.0 \
            --backend cuopt \
            --assume-characteristics-lagged \
            --resume \
            --skip-existing \
            --max-combinations "$chunk_size" \
            --combination-offset "$offset" \
            --write-summary
        status=$?
        if [[ "$status" != "0" ]]; then
            break
        fi
    done
    if [[ "$status" == "0" ]]; then
        uv run python scripts/summarize_monthly_panel_pca_grid.py \
            --grid-results "$run_dir/grid_results.csv" \
            --output-dir "$run_dir/summary"
        status=$?
    fi
} 2>&1 | tee "$log_dir/monthly-panel-pca-grid-${gpu_tag}-${SLURM_JOB_ID:-local}.log"
pipeline_status=${PIPESTATUS[0]}
if [[ "$pipeline_status" != "0" ]]; then
    status="$pipeline_status"
fi
set -e

end_epoch="$(date +%s)"
cat > "$log_dir/monthly-panel-pca-grid-${gpu_tag}-${SLURM_JOB_ID:-local}.metadata" <<EOF
slurm_job_id=${SLURM_JOB_ID:-unknown}
slurm_node=${SLURMD_NODENAME:-unknown}
gpu_type=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unavailable)
cuda_report=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([^ ]*\).*/\1/p' | head -1 || true)
cuopt_version=$cuopt_version
run_status=$status
runtime_seconds=$((end_epoch - start_epoch))
partition=${SLURM_JOB_PARTITION:-unknown}
cuda_extra=$cuda_extra
gpu_tag=$gpu_tag
output_dir=$run_dir
EOF
exit "$status"
