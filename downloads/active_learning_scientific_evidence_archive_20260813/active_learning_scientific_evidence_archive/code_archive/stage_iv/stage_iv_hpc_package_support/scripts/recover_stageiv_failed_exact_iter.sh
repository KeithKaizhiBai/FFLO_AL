#!/bin/bash
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

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}"
RUN_DIR="${OUTPUT_ROOT}/active_runs/${RUN_ID}"
WORLD_SIZE="${WORLD_SIZE:-8}"
EXCLUDE_NODES="${EXCLUDE_NODES:-gpuh01}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"
POSITIVE_DELTA_GAP_TOL="${POSITIVE_DELTA_GAP_TOL:-1e-8}"

if [ -z "${ITER:-}" ]; then
  echo "[error] set ITER to the failed exact iteration, for example ITER=0." >&2
  exit 2
fi
if [ -z "${FAILED_RANKS:-}" ]; then
  echo "[error] set FAILED_RANKS to a Slurm array list, for example FAILED_RANKS=3 or FAILED_RANKS=1,4." >&2
  exit 2
fi

cd "${PROJECT_DIR}"

TAG="$(printf "%03d" "${ITER}")"
NEXT="$(printf "%03d" "$((ITER + 1))")"

echo "[recover] iteration=${TAG}"
echo "[recover] failed_ranks=${FAILED_RANKS}"
echo "[recover] run_dir=${RUN_DIR}"

if [ ! -d "${RUN_DIR}/iter${TAG}" ]; then
  echo "[error] missing iteration directory: ${RUN_DIR}/iter${TAG}" >&2
  exit 3
fi

export PROJECT_DIR PYTHON_BIN OUTPUT_ROOT RUN_ID ITER WORLD_SIZE
JOB_ID="$(sbatch --parsable --exclude="${EXCLUDE_NODES}" --array="${FAILED_RANKS}" scripts/slurm_stageiv_exact_array.sh)"
echo "[submit] Stage IV recovery exact iter ${TAG}: job ${JOB_ID}"

while true; do
  active="$(squeue -h -j "${JOB_ID}" -o "%i|%T" 2>/dev/null | paste -sd, - || true)"
  if [ -z "${active}" ]; then
    break
  fi
  echo "[wait] Stage IV recovery iter ${TAG}: squeue active: ${active}"
  sleep "${SLEEP_SECONDS}"
done

failures="$(sacct -j "${JOB_ID}" --format=JobID,State,ExitCode --parsable2 | awk -F'|' 'NR>1 && $1 !~ /\.batch|\.extern/ && $2 !~ /COMPLETED/ {print $1":"$2":"$3}' | paste -sd, -)"
if [ -n "${failures}" ]; then
  echo "[error] Stage IV recovery iter ${TAG}: job ${JOB_ID} failed: ${failures}" >&2
  exit 5
fi

missing=""
for rank in $(seq 0 "$((WORLD_SIZE - 1))"); do
  shard="${RUN_DIR}/iter${TAG}/exact_shard_rank$(printf "%03d" "${rank}")_of$(printf "%03d" "${WORLD_SIZE}").npz"
  if [ ! -s "${shard}" ]; then
    missing="${missing} ${rank}"
  fi
done
if [ -n "${missing}" ]; then
  echo "[error] still missing exact shard ranks:${missing}" >&2
  exit 6
fi

"${PYTHON_BIN}" -m ml_phase.hpc   --merge   --run-dir "${RUN_DIR}"   --iteration "${ITER}"   --world-size "${WORLD_SIZE}"   --positive-delta-gap-tol "${POSITIVE_DELTA_GAP_TOL}"

"${PYTHON_BIN}" -m ml_phase.append_trusted   --dataset "${RUN_DIR}/dataset_iter${TAG}.npz"   --trusted-exact "${RUN_DIR}/iter${TAG}/exact_trusted_iter${TAG}.npz"   --output-npz "${RUN_DIR}/dataset_iter${NEXT}.npz"   --output-csv "${RUN_DIR}/dataset_iter${NEXT}.csv"   --output-root "${OUTPUT_ROOT}"

echo "[done] recovered Stage IV iter ${TAG}; wrote dataset_iter${NEXT}"
echo "[next] continue with: START_ITER=$((ITER + 1)) bash scripts/resume_stageiv_3d_full_loop.sh"
