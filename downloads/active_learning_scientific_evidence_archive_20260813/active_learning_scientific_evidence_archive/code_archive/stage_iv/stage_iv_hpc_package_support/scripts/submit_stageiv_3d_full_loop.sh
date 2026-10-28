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
START_ITER="${START_ITER:-0}"
MAX_ACQUISITION_BATCHES="${MAX_ACQUISITION_BATCHES:-24}"
FINAL_EXACT_ITER="${FINAL_EXACT_ITER:-$MAX_ACQUISITION_BATCHES}"
PARTITION_STRATEGY="${PARTITION_STRATEGY:-cost_aware}"
SELECT_DEVICE="${SELECT_DEVICE:-cpu}"
EXCLUDE_NODES="${EXCLUDE_NODES:-gpuh01}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"
POSITIVE_DELTA_GAP_TOL="${POSITIVE_DELTA_GAP_TOL:-1e-8}"

if [ "${CONFIRM_STAGEIV_FULL_LOOP:-0}" != "1" ]; then
  echo "[error] set CONFIRM_STAGEIV_FULL_LOOP=1 to submit the production Stage IV full loop." >&2
  exit 2
fi

cd "${PROJECT_DIR}"

if [ "${START_ITER}" = "0" ] && [ -e "${RUN_DIR}" ]; then
  echo "[error] run directory already exists: ${RUN_DIR}" >&2
  echo "Set START_ITER>0 only if you are intentionally resuming a completed prefix." >&2
  exit 4
fi

mkdir -p "${OUTPUT_ROOT}/logs"
"${PYTHON_BIN}" scripts/stageiv_3d_preflight.py   --config "${CONFIG_JSON}"   --output-dir "${OUTPUT_ROOT}/preflight"   --n-ja "${PREFLIGHT_N_JA:-48}"   --n-mu "${PREFLIGHT_N_MU:-64}"   --n-k "${PREFLIGHT_N_K:-1024}"

wait_for_job() {
  local job_id="$1"
  local iter_tag="$2"
  while true; do
    active="$(squeue -h -j "${job_id}" -o "%i|%T" 2>/dev/null | paste -sd, - || true)"
    if [ -z "${active}" ]; then
      break
    fi
    echo "[wait] Stage IV exact iter ${iter_tag}: squeue active: ${active}"
    sleep "${SLEEP_SECONDS}"
  done
  failures="$(sacct -j "${job_id}" --format=JobID,State,ExitCode --parsable2 | awk -F'|' 'NR>1 && $1 !~ /\.batch|\.extern/ && $2 !~ /COMPLETED/ {print $1":"$2":"$3}' | paste -sd, -)"
  if [ -n "${failures}" ]; then
    echo "[error] Stage IV exact iter ${iter_tag}: job ${job_id} failed: ${failures}" >&2
    exit 5
  fi
}

for ITER in $(seq "${START_ITER}" "${FINAL_EXACT_ITER}"); do
  TAG="$(printf "%03d" "${ITER}")"
  NEXT="$(printf "%03d" "$((ITER + 1))")"
  if [ "${ITER}" = "0" ]; then
    "${PYTHON_BIN}" scripts/stageiv_3d_select.py       --config "${CONFIG_JSON}"       --mode seed       --iteration "${ITER}"       --output-root "${OUTPUT_ROOT}"       --run-id "${RUN_ID}"       --world-size "${WORLD_SIZE}"       --partition-strategy "${PARTITION_STRATEGY}"
  else
    "${PYTHON_BIN}" scripts/stageiv_3d_select.py       --config "${CONFIG_JSON}"       --mode acquisition       --iteration "${ITER}"       --output-root "${OUTPUT_ROOT}"       --run-id "${RUN_ID}"       --dataset "${RUN_DIR}/dataset_iter${TAG}.npz"       --world-size "${WORLD_SIZE}"       --partition-strategy "${PARTITION_STRATEGY}"       --device "${SELECT_DEVICE}"
  fi

  export PROJECT_DIR PYTHON_BIN OUTPUT_ROOT RUN_ID ITER WORLD_SIZE
  JOB_ID="$(sbatch --parsable --exclude="${EXCLUDE_NODES}" --array=0-$((WORLD_SIZE - 1)) scripts/slurm_stageiv_exact_array.sh)"
  echo "[submit] Stage IV exact iter ${TAG}: job ${JOB_ID}"
  wait_for_job "${JOB_ID}" "${TAG}"

  "${PYTHON_BIN}" -m ml_phase.hpc     --merge     --run-dir "${RUN_DIR}"     --iteration "${ITER}"     --world-size "${WORLD_SIZE}"     --positive-delta-gap-tol "${POSITIVE_DELTA_GAP_TOL}"

  "${PYTHON_BIN}" -m ml_phase.append_trusted     --dataset "${RUN_DIR}/dataset_iter${TAG}.npz"     --trusted-exact "${RUN_DIR}/iter${TAG}/exact_trusted_iter${TAG}.npz"     --output-npz "${RUN_DIR}/dataset_iter${NEXT}.npz"     --output-csv "${RUN_DIR}/dataset_iter${NEXT}.csv"     --output-root "${OUTPUT_ROOT}"

  echo "[done] Stage IV iter ${TAG}; wrote dataset_iter${NEXT}"
done

echo "[done] Stage IV full loop completed through exact iter $(printf "%03d" "${FINAL_EXACT_ITER}")"
