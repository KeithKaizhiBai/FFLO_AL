#!/bin/bash
#SBATCH --job-name=stagev4_select
#SBATCH --partition=NV_H100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --exclude=gpuh01,gpuh14
#SBATCH --output=slurm-select-%j.out

set -euo pipefail

export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export PYTHONUTF8=1
export PYTHONIOENCODING="utf-8"

host_name="$(hostname 2>/dev/null || true)"
node_name="${SLURMD_NODENAME:-${host_name%%.*}}"
if [ "${node_name}" = "gpuh01" ] || [ "${host_name%%.*}" = "gpuh01" ]; then
  echo "[error] refusing to run Stage V-v4 selection on gpuh01" >&2
  exit 42
fi

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}"
OUTPUT_ROOT="ML_Phase_StageV_V4_GraphAcq"
RUN_ID="stagev_v4_phase_structure_graph_acquisition_3d_v1"
CONFIG_JSON="${CONFIG_JSON:-configs/stagev_v4_production.json}"
ITER="${ITER:?ITER must be set}"
WORLD_SIZE="${WORLD_SIZE:-8}"
SELECT_MODE="${SELECT_MODE:?SELECT_MODE must be seed or acquisition}"
PARTITION_STRATEGY="${PARTITION_STRATEGY:-cost_aware}"
SELECT_DEVICE="${SELECT_DEVICE:-cpu}"
RUN_DIR="${OUTPUT_ROOT}/active_runs/${RUN_ID}"
TAG="$(printf "%03d" "${ITER}")"
export OUTPUT_ROOT RUN_ID

cd "${PROJECT_DIR}"

echo "stage=selection"
echo "run_id=${RUN_ID}"
echo "iteration=${ITER}"
echo "select_mode=${SELECT_MODE}"
echo "world_size=${WORLD_SIZE}"
echo "hostname=${host_name}"
echo "slurmd_nodename=${SLURMD_NODENAME:-}"
nvidia-smi || true

if [ "${SELECT_MODE}" = "seed" ]; then
  "${PYTHON_BIN}" scripts/stagev_v4_select.py     --config "${CONFIG_JSON}"     --mode seed     --iteration "${ITER}"     --output-root "${OUTPUT_ROOT}"     --run-id "${RUN_ID}"     --world-size "${WORLD_SIZE}"     --partition-strategy "${PARTITION_STRATEGY}"
elif [ "${SELECT_MODE}" = "acquisition" ]; then
  "${PYTHON_BIN}" scripts/stagev_v4_select.py     --config "${CONFIG_JSON}"     --mode acquisition     --iteration "${ITER}"     --output-root "${OUTPUT_ROOT}"     --run-id "${RUN_ID}"     --dataset "${RUN_DIR}/dataset_iter${TAG}.npz"     --world-size "${WORLD_SIZE}"     --partition-strategy "${PARTITION_STRATEGY}"     --device "${SELECT_DEVICE}"
else
  echo "[error] invalid SELECT_MODE=${SELECT_MODE}" >&2
  exit 43
fi
