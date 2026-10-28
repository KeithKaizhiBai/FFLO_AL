#!/bin/bash
set -euo pipefail

export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export PYTHONUTF8=1
export PYTHONIOENCODING="utf-8"

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}"
OUTPUT_ROOT="ML_Phase_StageV_V4_GraphAcq"
RUN_ID="stagev_v4_phase_structure_graph_acquisition_3d_v1"
CONFIG_JSON="${CONFIG_JSON:-configs/stagev_v4_production.json}"
RUN_DIR="${OUTPUT_ROOT}/active_runs/${RUN_ID}"
WORLD_SIZE="${WORLD_SIZE:-8}"
START_ITER="${START_ITER:-0}"
FINAL_EXACT_ITER="${FINAL_EXACT_ITER:-96}"
PARTITION_STRATEGY="${PARTITION_STRATEGY:-cost_aware}"
SELECT_DEVICE="${SELECT_DEVICE:-cpu}"
EXCLUDE_NODES="${EXCLUDE_NODES:-gpuh01,gpuh14}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"
POSITIVE_DELTA_GAP_TOL="${POSITIVE_DELTA_GAP_TOL:-1e-8}"
SBATCH_BIN="${SBATCH_BIN:-/opt/gridview/slurm-2.3.3/bin/sbatch}"
SQUEUE_BIN="${SQUEUE_BIN:-/opt/gridview/slurm-2.3.3/bin/squeue}"
SACCT_BIN="${SACCT_BIN:-/opt/gridview/slurm-2.3.3/bin/sacct}"

if [ "${CONFIRM_STAGEV4_PRODUCTION:-0}" != "1" ]; then
  echo "[error] set CONFIRM_STAGEV4_PRODUCTION=1 to submit Stage V-v4." >&2
  exit 2
fi

cd "${PROJECT_DIR}"

if [ "${START_ITER}" = "0" ] && [ -e "${RUN_DIR}" ] && [ ! -f "${RUN_DIR}/iter000/selected_points.csv" ]; then
  echo "[error] run directory already exists: ${RUN_DIR}" >&2
  echo "Set START_ITER>0 only for intentional resume." >&2
  exit 4
elif [ "${START_ITER}" = "0" ] && [ -e "${RUN_DIR}" ] && [ -f "${RUN_DIR}/iter000/selected_points.csv" ]; then
  echo "[warn] existing seed selection detected; Stage V-v4 will resume from exact iter 000 without overwriting selection"
fi

mkdir -p "${OUTPUT_ROOT}/logs"
"${PYTHON_BIN}" scripts/stagev_v4_preflight.py   --config "${CONFIG_JSON}"   --output-dir "${OUTPUT_ROOT}/preflight"

wait_for_job() {
  local job_id="$1"
  local iter_tag="$2"
  local expected_path="${3:-}"
  while true; do
    active="$("${SQUEUE_BIN}" -h -j "${job_id}" -o "%i|%T" 2>/dev/null | paste -sd, - || true)"
    if [ -z "${active}" ]; then
      break
    fi
    echo "[wait] Stage V-v4 ${iter_tag}: squeue active: ${active}"
    sleep "${SLEEP_SECONDS}"
  done
  # Slurm accounting can briefly report RUNNING:0:0 after squeue clears.
  sleep 5
  failures="$("${SACCT_BIN}" -j "${job_id}" --format=JobID,State,ExitCode --parsable2 | awk -F'|' 'NR>1 && $1 !~ /\.batch|\.extern/ && $2 !~ /COMPLETED/ {print $1":"$2":"$3}' | paste -sd, -)"
  if [ -n "${failures}" ] && [ -n "${expected_path}" ] && [ -e "${expected_path}" ]; then
    echo "[warn] Stage V-v4 ${iter_tag}: transient sacct state ignored because expected output exists: ${expected_path}"
    failures=""
  fi
  if [ -n "${failures}" ]; then
    echo "[error] Stage V-v4 ${iter_tag}: job ${job_id} failed: ${failures}" >&2
    exit 5
  fi
}

for ITER in $(seq "${START_ITER}" "${FINAL_EXACT_ITER}"); do
  TAG="$(printf "%03d" "${ITER}")"
  NEXT="$(printf "%03d" "$((ITER + 1))")"
  if [ "${ITER}" = "0" ]; then
    SELECT_MODE="seed"
  else
    SELECT_MODE="acquisition"
  fi
  export PROJECT_DIR PYTHON_BIN OUTPUT_ROOT RUN_ID ITER WORLD_SIZE CONFIG_JSON PARTITION_STRATEGY SELECT_DEVICE SELECT_MODE
  if [ -f "${RUN_DIR}/iter${TAG}/selected_points.csv" ]; then
    echo "[skip] Stage V-v4 selection iter ${TAG} already exists"
  else
    SELECT_JOB_ID="$("${SBATCH_BIN}" --parsable --exclude="${EXCLUDE_NODES}" scripts/slurm_stagev_v4_select.sh)"
    echo "[submit] Stage V-v4 selection iter ${TAG}: job ${SELECT_JOB_ID}"
    wait_for_job "${SELECT_JOB_ID}" "selection iter ${TAG}" "${RUN_DIR}/iter${TAG}/selected_points.csv"
  fi

  if [ -f "${RUN_DIR}/dataset_iter${NEXT}.npz" ]; then
    echo "[skip] Stage V-v4 exact/append iter ${TAG} already produced dataset_iter${NEXT}"
    continue
  fi

  export PROJECT_DIR PYTHON_BIN OUTPUT_ROOT RUN_ID ITER WORLD_SIZE
  JOB_ID="$("${SBATCH_BIN}" --parsable --exclude="${EXCLUDE_NODES}" --array=0-$((WORLD_SIZE - 1)) scripts/slurm_stagev_v4_exact_array.sh)"
  echo "[submit] Stage V-v4 exact iter ${TAG}: job ${JOB_ID}"
  wait_for_job "${JOB_ID}" "exact iter ${TAG}" "${RUN_DIR}/iter${TAG}/exact_shard_rank000_of$(printf "%03d" "${WORLD_SIZE}").npz"

  "${PYTHON_BIN}" -m ml_phase.hpc     --merge     --run-dir "${RUN_DIR}"     --iteration "${ITER}"     --world-size "${WORLD_SIZE}"     --positive-delta-gap-tol "${POSITIVE_DELTA_GAP_TOL}"

  "${PYTHON_BIN}" -m ml_phase.append_trusted     --dataset "${RUN_DIR}/dataset_iter${TAG}.npz"     --trusted-exact "${RUN_DIR}/iter${TAG}/exact_trusted_iter${TAG}.npz"     --output-npz "${RUN_DIR}/dataset_iter${NEXT}.npz"     --output-csv "${RUN_DIR}/dataset_iter${NEXT}.csv"     --output-root "${OUTPUT_ROOT}"

  "${PYTHON_BIN}" scripts/stagev_v4_update_reward.py     --config "${CONFIG_JSON}"     --output-root "${OUTPUT_ROOT}"     --run-id "${RUN_ID}"     --iteration "${ITER}"

  echo "[done] Stage V-v4 iter ${TAG}; wrote dataset_iter${NEXT}"
done

echo "[done] Stage V-v4 production loop completed through exact iter $(printf "%03d" "${FINAL_EXACT_ITER}")"
