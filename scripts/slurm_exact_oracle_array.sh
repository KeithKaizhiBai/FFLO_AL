#!/bin/bash
#SBATCH --job-name=al_exact
#SBATCH --partition=NV_H100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-7

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
CONDA_ENV="${CONDA_ENV:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_MODULE="${CUDA_MODULE:-compiler/cuda/cuda-12.8.1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-ML_Phase}"

cd "${PROJECT_DIR}"

if command -v module >/dev/null 2>&1; then
  module purge || true
  if [ -n "${CUDA_MODULE}" ]; then
    module load "${CUDA_MODULE}" || true
  fi
fi

if [ -n "${CONDA_ENV}" ] && [ "${PYTHON_BIN}" = "python" ]; then
  PYTHON_BIN="${HOME}/.conda/envs/${CONDA_ENV}/bin/python"
fi

RUN_ID="${RUN_ID:?set RUN_ID}"
ITER="${ITER:?set ITER}"
WORLD_SIZE="${WORLD_SIZE:-${SLURM_ARRAY_TASK_COUNT}}"
RANK="${SLURM_ARRAY_TASK_ID}"
Q_EXPANSION_FLAG="${Q_EXPANSION_FLAG:---enable-q-expansion}"
DELTA_REFINEMENT_FLAG="${DELTA_REFINEMENT_FLAG:---enable-delta-refinement}"
Q_EXPAND_FACTOR="${Q_EXPAND_FACTOR:-1.5}"
Q_EXPAND_PAD_STEPS="${Q_EXPAND_PAD_STEPS:-50}"
Q_MAX_ABS="${Q_MAX_ABS:-3.141592653589793}"
MAX_Q_REFINEMENTS="${MAX_Q_REFINEMENTS:-3}"
DELTA_REFINE_HALF_WIDTH="${DELTA_REFINE_HALF_WIDTH:-0.03}"
N_DELTA_REFINED="${N_DELTA_REFINED:-300}"
MAX_DELTA_REFINEMENTS="${MAX_DELTA_REFINEMENTS:-2}"
POSITIVE_DELTA_GAP_TOL="${POSITIVE_DELTA_GAP_TOL:-1e-8}"
ORACLE_MODE="${ORACLE_MODE:-robust_al}"
INCREMENTAL_Q_EXPANSION_FLAG="${INCREMENTAL_Q_EXPANSION_FLAG:-}"
LOCAL_BOX_INSTRUMENTATION_FLAG="${LOCAL_BOX_INSTRUMENTATION_FLAG:-}"
ENABLE_BASIN_CLUSTERING_FLAG="${ENABLE_BASIN_CLUSTERING_FLAG:-}"
ENABLE_SELECTIVE_REFINEMENT_FLAG="${ENABLE_SELECTIVE_REFINEMENT_FLAG:-}"
MAX_REFINED_MINIMA="${MAX_REFINED_MINIMA:-6}"
MAX_OPTIONAL_REFINED_BASINS="${MAX_OPTIONAL_REFINED_BASINS:-3}"
MANDATORY_BASINS_CAN_EXCEED_CAP_FLAG="${MANDATORY_BASINS_CAN_EXCEED_CAP_FLAG:---mandatory-basins-can-exceed-cap}"
HIGH_RISK_OVERFLOW_POLICY="${HIGH_RISK_OVERFLOW_POLICY:-keep_all}"
MAX_EDGE_RISK_BASINS="${MAX_EDGE_RISK_BASINS:-1}"
MAX_DELTA_NEAR_EPS_BASINS="${MAX_DELTA_NEAR_EPS_BASINS:-2}"
MAX_NEAR_DEGENERATE_BASINS="${MAX_NEAR_DEGENERATE_BASINS:-2}"
ENERGY_WINDOW_PRUNING_FLAG="${ENERGY_WINDOW_PRUNING_FLAG:-}"
TOPOLOGY_CLASSIFICATION_FLAG="${TOPOLOGY_CLASSIFICATION_FLAG:-}"
TOPOLOGY_GAP_NK="${TOPOLOGY_GAP_NK:-2048}"
TOPOLOGY_GAP_BACKEND="${TOPOLOGY_GAP_BACKEND:-gpu}"
TOPOLOGY_GAP_TOL_REL="${TOPOLOGY_GAP_TOL_REL:-1e-8}"
TOPOLOGY_GAP_TOL_ABS="${TOPOLOGY_GAP_TOL_ABS:-0.0}"
TOPOLOGY_GAP_K_CHUNK="${TOPOLOGY_GAP_K_CHUNK:-512}"

ITER_DIR="${OUTPUT_ROOT}/active_runs/${RUN_ID}/iter$(printf '%03d' "${ITER}")"
RANK_LOG_DIR="${RANK_LOG_DIR:-${ITER_DIR}/rank_logs}"
mkdir -p "${RANK_LOG_DIR}"
ENV_SNAPSHOT="${RANK_LOG_DIR}/rank$(printf '%03d' "${RANK}")_env_snapshot.txt"
SHARD_META="${ITER_DIR}/exact_shard_rank$(printf '%03d' "${RANK}")_of$(printf '%03d' "${WORLD_SIZE}").json"

{
  echo "timestamp=$(date -Is)"
  echo "hostname=$(hostname)"
  echo "run_id=${RUN_ID}"
  echo "iteration=${ITER}"
  echo "rank=${RANK}"
  echo "world_size=${WORLD_SIZE}"
  echo "oracle_mode=${ORACLE_MODE}"
  echo "incremental_q_expansion_flag=${INCREMENTAL_Q_EXPANSION_FLAG:-N/A}"
  echo "local_box_instrumentation_flag=${LOCAL_BOX_INSTRUMENTATION_FLAG:-N/A}"
  echo "enable_basin_clustering_flag=${ENABLE_BASIN_CLUSTERING_FLAG:-N/A}"
  echo "enable_selective_refinement_flag=${ENABLE_SELECTIVE_REFINEMENT_FLAG:-N/A}"
  echo "max_refined_minima=${MAX_REFINED_MINIMA}"
  echo "max_total_refined_basins=${MAX_REFINED_MINIMA}"
  echo "max_optional_refined_basins=${MAX_OPTIONAL_REFINED_BASINS}"
  echo "mandatory_basins_can_exceed_cap_flag=${MANDATORY_BASINS_CAN_EXCEED_CAP_FLAG}"
  echo "high_risk_overflow_policy=${HIGH_RISK_OVERFLOW_POLICY}"
  echo "max_edge_risk_basins=${MAX_EDGE_RISK_BASINS}"
  echo "max_delta_near_eps_basins=${MAX_DELTA_NEAR_EPS_BASINS}"
  echo "max_near_degenerate_basins=${MAX_NEAR_DEGENERATE_BASINS}"
  echo "energy_window_pruning_flag=${ENERGY_WINDOW_PRUNING_FLAG:-N/A}"
  echo "topology_classification_flag=${TOPOLOGY_CLASSIFICATION_FLAG:-N/A}"
  echo "topology_gap_nk=${TOPOLOGY_GAP_NK}"
  echo "topology_gap_backend=${TOPOLOGY_GAP_BACKEND}"
  echo "topology_gap_tol_rel=${TOPOLOGY_GAP_TOL_REL}"
  echo "topology_gap_tol_abs=${TOPOLOGY_GAP_TOL_ABS}"
  echo "topology_gap_k_chunk=${TOPOLOGY_GAP_K_CHUNK}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "python_bin=${PYTHON_BIN}"
  echo "output_root=${OUTPUT_ROOT}"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
  "${PYTHON_BIN}" - <<'PY'
import torch
print("torch_version=" + str(torch.__version__))
print("torch_cuda_available=" + str(torch.cuda.is_available()))
print("torch_cuda_version=" + str(torch.version.cuda))
PY
} > "${ENV_SNAPSHOT}" 2>&1

"${PYTHON_BIN}" -m ml_phase.exact_oracle \
  --run-id "${RUN_ID}" \
  --iteration "${ITER}" \
  --rank "${RANK}" \
  --world-size "${WORLD_SIZE}" \
  --active-root "${OUTPUT_ROOT}/active_runs" \
  --device cuda:0 \
  ${Q_EXPANSION_FLAG} \
  --q-expand-factor "${Q_EXPAND_FACTOR}" \
  --q-expand-pad-steps "${Q_EXPAND_PAD_STEPS}" \
  --q-max-abs "${Q_MAX_ABS}" \
  --max-q-refinements "${MAX_Q_REFINEMENTS}" \
  ${DELTA_REFINEMENT_FLAG} \
  --delta-refine-half-width "${DELTA_REFINE_HALF_WIDTH}" \
  --n-delta-refined "${N_DELTA_REFINED}" \
  --max-delta-refinements "${MAX_DELTA_REFINEMENTS}" \
  --positive-delta-gap-tol "${POSITIVE_DELTA_GAP_TOL}" \
  --oracle-mode "${ORACLE_MODE}" \
  ${INCREMENTAL_Q_EXPANSION_FLAG} \
  ${LOCAL_BOX_INSTRUMENTATION_FLAG} \
  ${ENABLE_BASIN_CLUSTERING_FLAG} \
  ${ENABLE_SELECTIVE_REFINEMENT_FLAG} \
  --max-refined-minima "${MAX_REFINED_MINIMA}" \
  --max-optional-refined-basins "${MAX_OPTIONAL_REFINED_BASINS}" \
  ${MANDATORY_BASINS_CAN_EXCEED_CAP_FLAG} \
  --high-risk-overflow-policy "${HIGH_RISK_OVERFLOW_POLICY}" \
  --max-edge-risk-basins "${MAX_EDGE_RISK_BASINS}" \
  --max-delta-near-eps-basins "${MAX_DELTA_NEAR_EPS_BASINS}" \
  --max-near-degenerate-basins "${MAX_NEAR_DEGENERATE_BASINS}" \
  ${ENERGY_WINDOW_PRUNING_FLAG} \
  ${TOPOLOGY_CLASSIFICATION_FLAG} \
  --topology-gap-nk "${TOPOLOGY_GAP_NK}" \
  --topology-gap-backend "${TOPOLOGY_GAP_BACKEND}" \
  --topology-gap-tol-rel "${TOPOLOGY_GAP_TOL_REL}" \
  --topology-gap-tol-abs "${TOPOLOGY_GAP_TOL_ABS}" \
  --topology-gap-k-chunk "${TOPOLOGY_GAP_K_CHUNK}"

{
  echo "rank_end_timestamp=$(date -Is)"
  if [ -f "${SHARD_META}" ]; then
    "${PYTHON_BIN}" - "${SHARD_META}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
fields = [
    "elapsed_sec",
    "n_points",
    "trusted_exact_count",
    "q_expanded_count",
    "delta_refined_count",
    "q_unresolved_count",
    "delta_unresolved_count",
    "selected_refine_target_count_sum",
    "local_refinement_runtime_sec_sum",
    "point_total_runtime_sec_sum",
    "total_q_points_evaluated",
    "total_estimated_grid_evaluations",
    "local_box_summary_file",
    "output_file",
]
for field in fields:
    print(f"{field}={data.get(field, '')}")
PY
  else
    echo "shard_meta_missing=${SHARD_META}"
  fi
} >> "${ENV_SNAPSHOT}" 2>&1
