#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-ML_Phase_StageV_V4_GraphAcq}"
RUN_ID="${RUN_ID:-stagev_v4_phase_structure_graph_acquisition_3d_v1}"
RUN_DIR="${OUTPUT_ROOT}/active_runs/${RUN_ID}"
JOB_ID="${JOB_ID:?Set JOB_ID to the exact-array job that is already running.}"
ITER="${ITER:-0}"
WORLD_SIZE="${WORLD_SIZE:-8}"
CONFIG_JSON="${CONFIG_JSON:-configs/stagev_v4_production.json}"
POSITIVE_DELTA_GAP_TOL="${POSITIVE_DELTA_GAP_TOL:-1e-8}"
SBATCH_BIN="${SBATCH_BIN:-/opt/gridview/slurm-2.3.3/bin/sbatch}"
SQUEUE_BIN="${SQUEUE_BIN:-/opt/gridview/slurm-2.3.3/bin/squeue}"
SACCT_BIN="${SACCT_BIN:-/opt/gridview/slurm-2.3.3/bin/sacct}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"

cd "${PROJECT_DIR}"

TAG="$(printf "%03d" "${ITER}")"
NEXT="$(printf "%03d" "$((ITER + 1))")"

echo "[info] watching existing Stage V-v4 exact job ${JOB_ID} for iter ${TAG}"
while true; do
  active="$("${SQUEUE_BIN}" -h -j "${JOB_ID}" -o "%i|%T" 2>/dev/null | paste -sd, - || true)"
  if [ -z "${active}" ]; then
    break
  fi
  echo "[wait] Stage V-v4 exact iter ${TAG}: squeue active: ${active}"
  sleep "${SLEEP_SECONDS}"
done

sleep 5
failures="$("${SACCT_BIN}" -j "${JOB_ID}" --format=JobID,State,ExitCode --parsable2 | awk -F'|' 'NR>1 && $1 !~ /\.batch|\.extern/ && $2 !~ /COMPLETED/ {print $1":"$2":"$3}' | paste -sd, -)"
if [ -n "${failures}" ]; then
  echo "[error] Stage V-v4 exact iter ${TAG}: job ${JOB_ID} failed: ${failures}" >&2
  exit 5
fi

if [ ! -f "${RUN_DIR}/dataset_iter${NEXT}.npz" ]; then
  "${PYTHON_BIN}" -m ml_phase.hpc \
    --merge \
    --run-dir "${RUN_DIR}" \
    --iteration "${ITER}" \
    --world-size "${WORLD_SIZE}" \
    --positive-delta-gap-tol "${POSITIVE_DELTA_GAP_TOL}"

  "${PYTHON_BIN}" -m ml_phase.append_trusted \
    --dataset "${RUN_DIR}/dataset_iter${TAG}.npz" \
    --trusted-exact "${RUN_DIR}/iter${TAG}/exact_trusted_iter${TAG}.npz" \
    --output-npz "${RUN_DIR}/dataset_iter${NEXT}.npz" \
    --output-csv "${RUN_DIR}/dataset_iter${NEXT}.csv" \
    --output-root "${OUTPUT_ROOT}"

  "${PYTHON_BIN}" scripts/stagev_v4_update_reward.py \
    --config "${CONFIG_JSON}" \
    --output-root "${OUTPUT_ROOT}" \
    --run-id "${RUN_ID}" \
    --iteration "${ITER}"
else
  echo "[skip] dataset_iter${NEXT}.npz already exists"
fi

echo "[done] Stage V-v4 iter ${TAG}; wrote dataset_iter${NEXT}"
export CONFIRM_STAGEV4_PRODUCTION="${CONFIRM_STAGEV4_PRODUCTION:-1}"
export START_ITER="$((ITER + 1))"
export PROJECT_DIR PYTHON_BIN OUTPUT_ROOT RUN_ID WORLD_SIZE CONFIG_JSON SBATCH_BIN SQUEUE_BIN SACCT_BIN SLEEP_SECONDS
exec bash scripts/submit_stagev_v4_full_loop.sh
