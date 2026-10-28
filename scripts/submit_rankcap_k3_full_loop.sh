#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}"
RUN_ID="${RUN_ID:-active_boundary_discovery_rankcap_k3_full_loop_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-ML_Phase_512_RankCapK3_FullLoop}"
WORLD_SIZE="${WORLD_SIZE:-8}"
EXCLUDE_NODES="${EXCLUDE_NODES:-gpuh01}"
N_ITERS="${N_ITERS:-31}"

if [ "${CONFIRM_FULL_LOOP:-0}" != "1" ]; then
  echo "[error] full-loop submission is guarded. Set CONFIRM_FULL_LOOP=1 after the 5-iteration validation passes." >&2
  exit 1
fi

cd "${PROJECT_DIR}"

run_dir="${OUTPUT_ROOT}/active_runs/${RUN_ID}"
if [ -e "${run_dir}" ]; then
  echo "[error] run directory already exists; refusing to overwrite: ${run_dir}" >&2
  exit 1
fi

echo "[submit] package=rankcap_k3_full_loop"
echo "[submit] run_id=${RUN_ID}"
echo "[submit] output_root=${OUTPUT_ROOT}"
echo "[submit] n_iters=${N_ITERS}"
echo "[submit] semantics=iter000 random seed + full acquisition-selected loop through N_ITERS-1"
echo "[submit] world_size=${WORLD_SIZE}"
echo "[submit] exclude_nodes=${EXCLUDE_NODES}"

env \
  PROJECT_DIR="${PROJECT_DIR}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  RUN_ID="${RUN_ID}" \
  RUN_MODE="discovery" \
  START_ITER="0" \
  N_ITERS="${N_ITERS}" \
  CANDIDATE_DOMAIN_MODE="full" \
  ACQUISITION_PROFILE="full" \
  ORACLE_MODE="robust_incremental" \
  INCREMENTAL_Q_EXPANSION_FLAG="--enable-incremental-q-expansion" \
  LOCAL_BOX_INSTRUMENTATION_FLAG="--enable-local-box-instrumentation" \
  ENABLE_BASIN_CLUSTERING_FLAG="--enable-basin-clustering" \
  ENABLE_SELECTIVE_REFINEMENT_FLAG="--enable-selective-refinement" \
  MAX_REFINED_MINIMA="3" \
  MAX_OPTIONAL_REFINED_BASINS="3" \
  MANDATORY_BASINS_CAN_EXCEED_CAP_FLAG="--no-mandatory-basins-can-exceed-cap" \
  HIGH_RISK_OVERFLOW_POLICY="rank_and_cap" \
  MAX_EDGE_RISK_BASINS="1" \
  MAX_DELTA_NEAR_EPS_BASINS="2" \
  MAX_NEAR_DEGENERATE_BASINS="2" \
  ENERGY_WINDOW_PRUNING_FLAG="" \
  WORLD_SIZE="${WORLD_SIZE}" \
  PARTITION_STRATEGY="round_robin" \
  EXCLUDE_NODES="${EXCLUDE_NODES}" \
  ENABLE_EARLY_STOP="${ENABLE_EARLY_STOP:-1}" \
  ENABLE_STOP_CONTROLLER="${ENABLE_STOP_CONTROLLER:-1}" \
  STOP_MIN_ITERATIONS="${STOP_MIN_ITERATIONS:-5}" \
  STOP_PATIENCE="${STOP_PATIENCE:-4}" \
  STOP_MAX_ITERATIONS="${STOP_MAX_ITERATIONS:-${N_ITERS}}" \
  POINTS_PER_ITER="${POINTS_PER_ITER:-256}" \
  INITIAL_SEED_SIZE="${INITIAL_SEED_SIZE:-512}" \
  BATCH_SIZE_MAX="${BATCH_SIZE_MAX:-256}" \
  BATCH_SIZE_MIN="${BATCH_SIZE_MIN:-0}" \
  BATCH_SIZE_MIN_BEFORE_MIN_ITER="${BATCH_SIZE_MIN_BEFORE_MIN_ITER:-64}" \
  BATCH_SIZE_MIN_AFTER_MIN_ITER="${BATCH_SIZE_MIN_AFTER_MIN_ITER:-0}" \
  N_ENSEMBLE="${N_ENSEMBLE:-5}" \
  REG_EPOCHS="${REG_EPOCHS:-240}" \
  CLS_EPOCHS="${CLS_EPOCHS:-240}" \
  BATCH_SIZE="${BATCH_SIZE:-512}" \
  bash hpc_active_loop.sh

echo "[collect] build full-loop report"
"${PYTHON_BIN}" scripts/run_rankcap_k3_active_loop_package.py \
  --mode collect \
  --package-kind full \
  --world-size "${WORLD_SIZE}" \
  --n-iters "${N_ITERS}" \
  --create-archive

echo "[done] return archive: ${OUTPUT_ROOT}/rankcap_k3_full_loop_results.tar.gz"
