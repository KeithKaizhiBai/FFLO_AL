#!/bin/bash
#SBATCH --job-name=stageiv3d_exact
#SBATCH --partition=NV_H100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --exclude=gpuh01
#SBATCH --output=slurm-%A_%a.out

set -euo pipefail

STAGEIV_EXPECTED_OUTPUT_ROOT="ML_Phase_StageIV_Topology3D"
STAGEIV_EXPECTED_RUN_ID="active_phase_topology_3d_t_ja_mu_from_scratch_v1"
STAGEIV_EXPECTED_CONFIG_JSON="configs/stageiv_3d_production.json"

if [ -n "${OUTPUT_ROOT:-}" ] && [ "${OUTPUT_ROOT}" != "${STAGEIV_EXPECTED_OUTPUT_ROOT}" ]; then
  echo "[error] refusing stale OUTPUT_ROOT=${OUTPUT_ROOT}; expected ${STAGEIV_EXPECTED_OUTPUT_ROOT}" >&2
  exit 31
fi
if [ -n "${RUN_ID:-}" ] && [ "${RUN_ID}" != "${STAGEIV_EXPECTED_RUN_ID}" ]; then
  echo "[error] refusing stale RUN_ID=${RUN_ID}; expected ${STAGEIV_EXPECTED_RUN_ID}" >&2
  exit 32
fi
if [ -n "${CONFIG_JSON:-}" ] && [ "${CONFIG_JSON}" != "${STAGEIV_EXPECTED_CONFIG_JSON}" ]; then
  echo "[error] refusing stale CONFIG_JSON=${CONFIG_JSON}; expected ${STAGEIV_EXPECTED_CONFIG_JSON}" >&2
  exit 33
fi

OUTPUT_ROOT="${STAGEIV_EXPECTED_OUTPUT_ROOT}"
RUN_ID="${STAGEIV_EXPECTED_RUN_ID}"
CONFIG_JSON="${STAGEIV_EXPECTED_CONFIG_JSON}"
export OUTPUT_ROOT RUN_ID CONFIG_JSON

export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export PYTHONUTF8=1
export PYTHONIOENCODING="utf-8"

host_name="$(hostname 2>/dev/null || true)"
if [ "${host_name%%.*}" = "gpuh01" ]; then
  echo "[error] refusing to run on excluded node gpuh01" >&2
  exit 42
fi

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}"
ITER="${ITER:?ITER must be set}"
WORLD_SIZE="${WORLD_SIZE:-8}"
RANK="${SLURM_ARRAY_TASK_ID:-0}"
RUN_DIR="${OUTPUT_ROOT}/active_runs/${RUN_ID}"

cd "${PROJECT_DIR}"

echo "run_id=${RUN_ID}"
echo "iteration=${ITER}"
echo "rank=${RANK}"
echo "world_size=${WORLD_SIZE}"
echo "hostname=${host_name}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true
"${PYTHON_BIN}" - <<'PY'
import torch
print("torch.__version__=", torch.__version__)
print("torch.version.cuda=", torch.version.cuda)
print("torch.cuda.is_available()=", torch.cuda.is_available())
PY

"${PYTHON_BIN}" -m ml_phase.exact_oracle   --active-root "${OUTPUT_ROOT}/active_runs"   --run-id "${RUN_ID}"   --iteration "${ITER}"   --rank "${RANK}"   --world-size "${WORLD_SIZE}"   --device cuda:0   --save-every 1   --oracle-mode robust_incremental   --enable-q-expansion   --enable-incremental-q-expansion   --enable-delta-refinement   --allow-ambiguous-output   --enable-local-box-instrumentation   --enable-basin-clustering   --enable-selective-refinement   --max-refined-minima 3   --max-optional-refined-basins 3   --no-mandatory-basins-can-exceed-cap   --high-risk-overflow-policy rank_and_cap   --max-edge-risk-basins 1   --max-delta-near-eps-basins 2   --max-near-degenerate-basins 2   --enable-topology-classification   --topology-gap-nk "${TOPOLOGY_GAP_NK:-2048}"   --topology-gap-backend "${TOPOLOGY_GAP_BACKEND:-gpu}"   --topology-gap-tol-rel "${TOPOLOGY_GAP_TOL_REL:-1e-8}"   --topology-gap-tol-abs "${TOPOLOGY_GAP_TOL_ABS:-0.0}"   --topology-gap-k-chunk "${TOPOLOGY_GAP_K_CHUNK:-512}"
