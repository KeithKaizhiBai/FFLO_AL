#!/bin/bash
set -euo pipefail

export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export PYTHONUTF8=1
export PYTHONIOENCODING="utf-8"

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-ML_Phase_512_TopoTrivial_FullLoop}"
RUN_ID="${RUN_ID:-active_phase_topology_from_scratch_full_loop_v1}"
RUN_DIR="${OUTPUT_ROOT}/active_runs/${RUN_ID}"
N_ITERS="${N_ITERS:-51}"
WORLD_SIZE="${WORLD_SIZE:-8}"
EXCLUDE_NODES="${EXCLUDE_NODES:-gpuh01}"

if [ "${CONFIRM_TOPO_FULL_LOOP:-0}" != "1" ]; then
  echo "[error] set CONFIRM_TOPO_FULL_LOOP=1 to submit the production run." >&2
  exit 2
fi

if [ -e "${RUN_DIR}" ]; then
  echo "[error] run directory already exists: ${RUN_DIR}" >&2
  exit 4
fi

echo "[submit] project_dir=${PROJECT_DIR}"
echo "[submit] run_id=${RUN_ID}"
echo "[submit] output_root=${OUTPUT_ROOT}"
echo "[submit] n_iters=${N_ITERS}"
echo "[submit] world_size=${WORLD_SIZE}"
echo "[submit] exclude_nodes=${EXCLUDE_NODES}"

cd "${PROJECT_DIR}"

RUN_MODE="discovery" \
CANDIDATE_DOMAIN_MODE="full" \
INITIALIZATION="sobol_scrambled" \
INITIAL_SEED_SIZE="512" \
BATCH_SIZE_MAX="256" \
BATCH_SIZE_MIN="0" \
POINTS_PER_ITER="256" \
N_ITERS="${N_ITERS}" \
WORLD_SIZE="${WORLD_SIZE}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
RUN_ID="${RUN_ID}" \
PYTHON_BIN="${PYTHON_BIN}" \
ORACLE_MODE="robust_incremental" \
INCREMENTAL_Q_EXPANSION_FLAG="--enable-incremental-q-expansion" \
LOCAL_BOX_INSTRUMENTATION_FLAG="--enable-local-box-instrumentation" \
ENABLE_BASIN_CLUSTERING_FLAG="--enable-basin-clustering" \
ENABLE_SELECTIVE_REFINEMENT_FLAG="--enable-selective-refinement" \
MAX_REFINED_MINIMA="3" \
MAX_OPTIONAL_REFINED_BASINS="3" \
MANDATORY_BASINS_CAN_EXCEED_CAP_FLAG="" \
HIGH_RISK_OVERFLOW_POLICY="rank_and_cap" \
MAX_EDGE_RISK_BASINS="1" \
MAX_DELTA_NEAR_EPS_BASINS="2" \
MAX_NEAR_DEGENERATE_BASINS="2" \
ACQUISITION_PROFILE="topo_trivial" \
TOPOLOGY_CLASSIFICATION_FLAG="--enable-topology-classification" \
TOPOLOGY_GAP_NK="${TOPOLOGY_GAP_NK:-2048}" \
TOPOLOGY_GAP_BACKEND="${TOPOLOGY_GAP_BACKEND:-gpu}" \
TOPOLOGY_GAP_TOL_REL="${TOPOLOGY_GAP_TOL_REL:-1e-8}" \
TOPOLOGY_GAP_TOL_ABS="${TOPOLOGY_GAP_TOL_ABS:-0.0}" \
TOPOLOGY_GAP_K_CHUNK="${TOPOLOGY_GAP_K_CHUNK:-512}" \
EXCLUDE_NODES="${EXCLUDE_NODES}" \
STOP_SURPRISE_MODE="trusted" \
TRUSTED_SURPRISE_MIN_DENOMINATOR="64" \
TRUSTED_SURPRISE_MIN_FRACTION="0.25" \
STOP_MAX_ITERATIONS="${N_ITERS}" \
bash hpc_active_loop.sh
