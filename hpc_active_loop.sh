#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}"
CONDA_ENV="${CONDA_ENV:-}"
CUDA_MODULE="${CUDA_MODULE:-compiler/cuda/cuda-12.8.1}"

RUN_MODE="${RUN_MODE:-discovery}"
RUN_ID_WAS_SET="${RUN_ID+x}"
if [ "${RUN_MODE}" = "discovery" ] && [ -z "${RUN_ID_WAS_SET}" ]; then
  echo "[error] RUN_ID must be explicitly set for discovery mode to avoid accidental new runs." >&2
  echo "[hint] RUN_ID=active_boundary_discovery_512seed_256x50 START_ITER=17 N_ITERS=83 bash hpc_active_loop.sh" >&2
  exit 1
fi
RUN_ID="${RUN_ID:-active_boundary_loop_512x50_acquisition_only}"

START_ITER="${START_ITER:-0}"
N_ITERS="${N_ITERS:-100}"
CANDIDATE_DOMAIN_MODE="${CANDIDATE_DOMAIN_MODE:-full}"
INITIALIZATION="${INITIALIZATION:-random_grid}"
INITIAL_SEED_SIZE="${INITIAL_SEED_SIZE:-512}"
BATCH_SIZE_MAX="${BATCH_SIZE_MAX:-256}"
BATCH_SIZE_MIN="${BATCH_SIZE_MIN:-0}"
BATCH_SIZE_MIN_BEFORE_MIN_ITER="${BATCH_SIZE_MIN_BEFORE_MIN_ITER:-64}"
BATCH_SIZE_MIN_AFTER_MIN_ITER="${BATCH_SIZE_MIN_AFTER_MIN_ITER:-0}"
SELECTION_MODE="${SELECTION_MODE:-stochastic}"
SAMPLING_POWER="${SAMPLING_POWER:-2.0}"
SAMPLING_POWER_START="${SAMPLING_POWER_START:-1.5}"
SAMPLING_POWER_MID="${SAMPLING_POWER_MID:-2.5}"
SAMPLING_POWER_END="${SAMPLING_POWER_END:-4.0}"
SAMPLING_POWER_MID_ITER="${SAMPLING_POWER_MID_ITER:-10}"
SAMPLING_POWER_END_ITER="${SAMPLING_POWER_END_ITER:-30}"
SAMPLING_POWER_SCHEDULE="${SAMPLING_POWER_SCHEDULE:-piecewise}"
SCORE_THRESHOLD_ABS="${SCORE_THRESHOLD_ABS:-0.0}"
SCORE_THRESHOLD_REL="${SCORE_THRESHOLD_REL:-0.0}"
ACTIVE_POOL_RULE="${ACTIVE_POOL_RULE:-max_threshold}"
ACTIVE_POOL_QUANTILE="${ACTIVE_POOL_QUANTILE:-0.90}"
ACTIVE_POOL_QUANTILE_SCHEDULE="${ACTIVE_POOL_QUANTILE_SCHEDULE:-piecewise}"
ACTIVE_POOL_QUANTILE_START="${ACTIVE_POOL_QUANTILE_START:-${ACTIVE_POOL_QUANTILE}}"
ACTIVE_POOL_QUANTILE_MID="${ACTIVE_POOL_QUANTILE_MID:-0.95}"
ACTIVE_POOL_QUANTILE_END="${ACTIVE_POOL_QUANTILE_END:-0.98}"
ACTIVE_POOL_QUANTILE_MID_ITER="${ACTIVE_POOL_QUANTILE_MID_ITER:-10}"
ACTIVE_POOL_QUANTILE_END_ITER="${ACTIVE_POOL_QUANTILE_END_ITER:-30}"
ACTIVE_POOL_REL_TO_P95="${ACTIVE_POOL_REL_TO_P95:-0.7}"
ACTIVE_POOL_MIN_QUANTILE="${ACTIVE_POOL_MIN_QUANTILE:-0.70}"
ACTIVE_POOL_MAX_FRACTION_START="${ACTIVE_POOL_MAX_FRACTION_START:-0.20}"
ACTIVE_POOL_MAX_FRACTION_END="${ACTIVE_POOL_MAX_FRACTION_END:-0.05}"
ACTIVE_POOL_MAX_FRACTION_END_ITER="${ACTIVE_POOL_MAX_FRACTION_END_ITER:-30}"
ACTIVE_SELECTION_MIN_ITERATIONS="${ACTIVE_SELECTION_MIN_ITERATIONS:-5}"
NO_UNDERFILLED_BATCH_AFTER_MIN_ITER="${NO_UNDERFILLED_BATCH_AFTER_MIN_ITER:-0}"
B_DELTA_GATE_MODE="${B_DELTA_GATE_MODE:-normal_sc_competition}"
Q_BOUNDARY_GATE_MODE="${Q_BOUNDARY_GATE_MODE:-psc}"
INTERIOR_FILTER_MODE="${INTERIOR_FILTER_MODE:-soft_penalty}"
INTERIOR_PENALTY_START_ITER="${INTERIOR_PENALTY_START_ITER:-10}"
INTERIOR_PENALTY_EARLY="${INTERIOR_PENALTY_EARLY:-0.5}"
INTERIOR_PENALTY_LATE="${INTERIOR_PENALTY_LATE:-0.1}"
P_CONF_THRESHOLD="${P_CONF_THRESHOLD:-0.98}"
U_NS_LOW="${U_NS_LOW:-0.05}"
U_UF_LOW="${U_UF_LOW:-0.05}"
G_PHASE_LOW="${G_PHASE_LOW:-0.05}"
E_Q_LOW="${E_Q_LOW:-0.05}"
E_EXT_LOW="${E_EXT_LOW:-0.05}"
W_EXT_SCHEDULE="${W_EXT_SCHEDULE:-piecewise}"
W_EXT_START="${W_EXT_START:-0.15}"
W_EXT_MID="${W_EXT_MID:-0.08}"
W_EXT_END="${W_EXT_END:-0.03}"
W_EXT_MID_ITER="${W_EXT_MID_ITER:-10}"
W_EXT_END_ITER="${W_EXT_END_ITER:-30}"
RANDOM_SEED="${RANDOM_SEED:-42}"
FINITE_T_BAND_WIDTH="${FINITE_T_BAND_WIDTH:-}"
HIDDEN_GROUND_TRUTH="${HIDDEN_GROUND_TRUTH:-}"
POINTS_PER_ITER="${POINTS_PER_ITER:-512}"
WORLD_SIZE="${WORLD_SIZE:-8}"
N_ENSEMBLE="${N_ENSEMBLE:-5}"
REG_EPOCHS="${REG_EPOCHS:-240}"
CLS_EPOCHS="${CLS_EPOCHS:-240}"
BATCH_SIZE="${BATCH_SIZE:-512}"
PARTITION_STRATEGY="${PARTITION_STRATEGY:-round_robin}"
POLL_SEC="${POLL_SEC:-30}"
MAX_WAIT_SEC="${MAX_WAIT_SEC:-604800}"
ACCOUNTING_DELAY_SEC="${ACCOUNTING_DELAY_SEC:-180}"
POSITIVE_DELTA_GAP_TOL="${POSITIVE_DELTA_GAP_TOL:-1e-8}"
EXCLUDE_NODES="${EXCLUDE_NODES:-gpuh01}"
ENABLE_EARLY_STOP="${ENABLE_EARLY_STOP:-1}"
MIN_NEW_POINTS_PER_ITER="${MIN_NEW_POINTS_PER_ITER:-8}"
MAX_LOW_APPEND_ITERS="${MAX_LOW_APPEND_ITERS:-2}"
ENABLE_STOP_CONTROLLER="${ENABLE_STOP_CONTROLLER:-1}"
STOP_MIN_ITERATIONS="${STOP_MIN_ITERATIONS:-5}"
STOP_PATIENCE="${STOP_PATIENCE:-4}"
STOP_MAX_ITERATIONS="${STOP_MAX_ITERATIONS:-}"
STOP_MAX_EXACT_CALLS="${STOP_MAX_EXACT_CALLS:-}"
STOP_MAP_TOL="${STOP_MAP_TOL:-0.002}"
STOP_SURPRISE_TOL="${STOP_SURPRISE_TOL:-0.05}"
STOP_SELECTED_A0_RATIO_TOL="${STOP_SELECTED_A0_RATIO_TOL:-0.15}"
STOP_QEDGE_RATE_TOL="${STOP_QEDGE_RATE_TOL:-0.01}"
STOP_RERUN_RATE_TOL="${STOP_RERUN_RATE_TOL:-0.01}"
STOP_ALLOW_MISSING_BOUNDARY="${STOP_ALLOW_MISSING_BOUNDARY:-0}"
BOUNDARY_REFINEMENT_MODE="${BOUNDARY_REFINEMENT_MODE:-diagnostic}"
BOUNDARY_KT_BIN_WIDTH="${BOUNDARY_KT_BIN_WIDTH:-0.005}"
BOUNDARY_MAX_LOCAL_SPACING="${BOUNDARY_MAX_LOCAL_SPACING:-0.035}"
BOUNDARY_POSITION_TOL="${BOUNDARY_POSITION_TOL:-0.00375}"
DRY_RUN="${DRY_RUN:-0}"

Q_EXPANSION_FLAG="${Q_EXPANSION_FLAG:---enable-q-expansion}"
DELTA_REFINEMENT_FLAG="${DELTA_REFINEMENT_FLAG:---enable-delta-refinement}"
MAX_Q_REFINEMENTS="${MAX_Q_REFINEMENTS:-3}"
MAX_DELTA_REFINEMENTS="${MAX_DELTA_REFINEMENTS:-2}"
N_DELTA_REFINED="${N_DELTA_REFINED:-300}"

WARM_START="${WARM_START:-}"

cd "${PROJECT_DIR}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "[error] PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

if [ "${RUN_MODE}" = "refinement" ] && [ ! -f "${WARM_START}" ]; then
  echo "[error] refinement mode requires an existing warm-start npz: ${WARM_START}" >&2
  exit 1
fi

if [ "${START_ITER}" -lt 0 ]; then
  echo "[error] START_ITER must be non-negative: ${START_ITER}" >&2
  exit 1
fi

END_ITER=$((START_ITER + N_ITERS))
if [ -z "${STOP_MAX_ITERATIONS}" ]; then
  STOP_MAX_ITERATIONS="${END_ITER}"
fi

run_dir="ML_Phase/active_runs/${RUN_ID}"
resume_dataset=""
if [ "${START_ITER}" -gt 0 ]; then
  resume_dataset="${run_dir}/dataset_iter$(printf '%03d' "${START_ITER}").npz"
fi

echo "[preflight] PROJECT_DIR=${PROJECT_DIR}"
echo "[preflight] RUN_ID=${RUN_ID}"
echo "[preflight] RUN_MODE=${RUN_MODE}"
echo "[preflight] CANDIDATE_DOMAIN_MODE=${CANDIDATE_DOMAIN_MODE}"
echo "[preflight] START_ITER=${START_ITER}"
echo "[preflight] N_ITERS_THIS_RUN=${N_ITERS}"
echo "[preflight] END_ITER_EXCLUSIVE=${END_ITER}"
echo "[preflight] TARGET_TOTAL_ITERATIONS=${END_ITER}"
echo "[preflight] POINTS_PER_ITER=${POINTS_PER_ITER}"
echo "[preflight] BATCH_SIZE_MAX=${BATCH_SIZE_MAX}"
echo "[preflight] ACTIVE_POOL_QUANTILE=${ACTIVE_POOL_QUANTILE}"
echo "[preflight] ACTIVE_POOL_QUANTILE_SCHEDULE=${ACTIVE_POOL_QUANTILE_SCHEDULE}"
echo "[preflight] ACTIVE_POOL_REL_TO_P95=${ACTIVE_POOL_REL_TO_P95}"
echo "[preflight] ACTIVE_POOL_MAX_FRACTION_START=${ACTIVE_POOL_MAX_FRACTION_START}"
echo "[preflight] ACTIVE_POOL_MAX_FRACTION_END=${ACTIVE_POOL_MAX_FRACTION_END}"
echo "[preflight] SAMPLING_POWER=${SAMPLING_POWER}"
echo "[preflight] SAMPLING_POWER_SCHEDULE=${SAMPLING_POWER_SCHEDULE}"
echo "[preflight] B_DELTA_GATE_MODE=${B_DELTA_GATE_MODE}"
echo "[preflight] INTERIOR_FILTER_MODE=${INTERIOR_FILTER_MODE}"
echo "[preflight] W_EXT_SCHEDULE=${W_EXT_SCHEDULE}"
echo "[preflight] WORLD_SIZE=${WORLD_SIZE}"
echo "[preflight] STOP_MAX_ITERATIONS=${STOP_MAX_ITERATIONS}"
if [ -n "${resume_dataset}" ]; then
  echo "[preflight] expected resume dataset=${resume_dataset}"
else
  echo "[preflight] expected resume dataset=N/A for START_ITER=0"
fi

case "$(basename "${PROJECT_DIR}"):${RUN_ID}" in
  *discovery*:active_boundary_loop_512x50_acquisition_only)
    echo "[error] PROJECT_DIR looks like a discovery package but RUN_ID is the old default: ${RUN_ID}" >&2
    echo "[hint] set RUN_ID=active_boundary_discovery_512seed_256x50 explicitly." >&2
    exit 1
    ;;
esac

if [ "${START_ITER}" -gt 0 ] && [ ! -f "${resume_dataset}" ]; then
  echo "[error] START_ITER=${START_ITER}, but resume dataset is missing: ${resume_dataset}" >&2
  echo "[hint] check RUN_ID or use START_ITER=<next completed dataset index>." >&2
  exit 1
fi

if [ "${START_ITER}" -eq 0 ] && [ -d "${run_dir}" ]; then
  existing_dataset="$(find "${run_dir}" -maxdepth 1 -name 'dataset_iter*.npz' -print -quit 2>/dev/null || true)"
  if [ -n "${existing_dataset}" ]; then
    echo "[error] Existing run directory contains dataset files: ${run_dir}" >&2
    echo "[hint] Use START_ITER=<next> to resume, or choose a new RUN_ID." >&2
    exit 1
  fi
fi

if [ "${DRY_RUN}" = "1" ]; then
  echo "[dry-run] preflight passed; no sbatch jobs will be submitted."
  exit 0
fi

if ! command -v flock >/dev/null 2>&1; then
  echo "[error] flock is required for the PROJECT_DIR/RUN_ID active-loop lock, but it is not available." >&2
  exit 1
fi

lock_file="${PROJECT_DIR}/.active_loop.${RUN_ID}.lock"
exec 9>"${lock_file}"
if ! flock -n 9; then
  echo "[error] another active loop holds lock: ${lock_file}" >&2
  echo "[hint] lock content:" >&2
  cat "${lock_file}" >&2 || true
  exit 1
fi
printf 'pid=%s\ntime=%s\nproject_dir=%s\nrun_id=%s\n' "$$" "$(date -Is)" "${PROJECT_DIR}" "${RUN_ID}" > "${lock_file}"

now_iso() {
  date -Is
}

status_update() {
  local iter_dir="$1"
  shift
  mkdir -p "${iter_dir}"
  local status_path="${iter_dir}/status.json"
  "${PYTHON_BIN}" - "${status_path}" "$@" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
updates = sys.argv[2:]
if path.exists():
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = {}
else:
    data = {}
for item in updates:
    key, raw = item.split("=", 1)
    lower = raw.lower()
    if lower == "true":
        value = True
    elif lower == "false":
        value = False
    elif lower == "null":
        value = None
    else:
        try:
            value = int(raw)
        except ValueError:
            try:
                value = float(raw)
            except ValueError:
                value = raw
    data[key] = value
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
tmp.replace(path)
PY
}

status_get() {
  local iter_dir="$1"
  local key="$2"
  local default_value="$3"
  local status_path="${iter_dir}/status.json"
  "${PYTHON_BIN}" - "${status_path}" "${key}" "${default_value}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
default = sys.argv[3]
try:
    data = json.loads(path.read_text(encoding="utf-8"))
    value = data.get(key, default)
except Exception:
    value = default
if isinstance(value, bool):
    print("true" if value else "false")
elif value is None:
    print("")
else:
    print(value)
PY
}

status_true() {
  [ "$(status_get "$1" "$2" "false")" = "true" ]
}

json_get_int() {
  local json_path="$1"
  local key="$2"
  local default_value="$3"
  "${PYTHON_BIN}" -c 'import json, sys
path, key, default_value = sys.argv[1], sys.argv[2], int(sys.argv[3])
try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(int(data.get(key, default_value)))
except Exception:
    print(default_value)
' "${json_path}" "${key}" "${default_value}"
}

csv_data_rows() {
  local csv_path="$1"
  "${PYTHON_BIN}" -c 'import sys
path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as f:
        n = sum(1 for _ in f)
    print(max(0, n - 1))
except Exception:
    print(0)
' "${csv_path}"
}

shards_found() {
  local iter_dir="$1"
  local world_size="$2"
  local found=0
  local rank
  for (( rank=0; rank<world_size; rank++ )); do
    local tag
    tag="$(printf '%03d' "${rank}")"
    local npz="${iter_dir}/exact_shard_rank${tag}_of$(printf '%03d' "${world_size}").npz"
    local json="${iter_dir}/exact_shard_rank${tag}_of$(printf '%03d' "${world_size}").json"
    if [ -s "${npz}" ] && [ -s "${json}" ]; then
      found=$((found + 1))
    fi
  done
  echo "${found}"
}

exact_shards_complete() {
  local iter_dir="$1"
  local world_size="$2"
  [ "$(shards_found "${iter_dir}" "${world_size}")" -eq "${world_size}" ]
}

sacct_decision() {
  local job_id="$1"
  local expected_array_count="$2"
  local sacct_text="$3"
  SACCT_TEXT="${sacct_text}" "${PYTHON_BIN}" - "${job_id}" "${expected_array_count}" <<'PY'
import os
import re
import sys

job_id = sys.argv[1]
expected = int(sys.argv[2])
rows = []
for raw in os.environ.get("SACCT_TEXT", "").splitlines():
    parts = raw.strip().split("|")
    if len(parts) < 3:
        continue
    jid, state, exit_code = parts[0], parts[1].split()[0], parts[2]
    if "." in jid:
        continue
    rows.append((jid, state, exit_code))

failure_states = {"FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}
running_states = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED", "RESIZING"}

if expected > 0:
    pattern = re.compile(rf"^{re.escape(job_id)}_(\d+)$")
    elems = {}
    for jid, state, exit_code in rows:
        m = pattern.match(jid)
        if not m:
            continue
        idx = int(m.group(1))
        if 0 <= idx < expected:
            elems[idx] = (state, exit_code)
    if any(state in failure_states for state, _ in elems.values()):
        bad = [f"{job_id}_{idx}:{state}:{code}" for idx, (state, code) in sorted(elems.items()) if state in failure_states]
        print("failure|" + ",".join(bad))
    elif len(elems) == expected and all(state == "COMPLETED" and code == "0:0" for state, code in elems.values()):
        print(f"success|array completed {len(elems)}/{expected}")
    elif any(state in running_states for state, _ in elems.values()):
        print(f"running|array active records {len(elems)}/{expected}")
    elif elems:
        states = ",".join(f"{job_id}_{idx}:{state}:{code}" for idx, (state, code) in sorted(elems.items()))
        print(f"unknown|array records {len(elems)}/{expected}: {states}")
    else:
        print("unknown|no array element records yet")
else:
    records = [(state, code) for jid, state, code in rows if jid == job_id]
    if not records:
        print("unknown|no ordinary job record yet")
    elif any(state in failure_states for state, _ in records):
        print("failure|" + ",".join(f"{state}:{code}" for state, code in records if state in failure_states))
    elif any(state in running_states for state, _ in records):
        print("running|" + ",".join(f"{state}:{code}" for state, code in records))
    elif any(state == "COMPLETED" and code == "0:0" for state, code in records):
        print("success|ordinary job completed")
    else:
        print("unknown|" + ",".join(f"{state}:{code}" for state, code in records))
PY
}

wait_for_job() {
  local job_id="$1"
  local label="$2"
  local kind="${3:-ordinary}"
  local iter_dir="${4:-}"
  local expected_array_count="${5:-0}"
  local waited=0
  local accounting_waited=0
  echo "[wait] ${label}: job ${job_id}"
  while true; do
    if command -v squeue >/dev/null 2>&1; then
      local sq
      sq="$(squeue -j "${job_id}" -h -o "%i|%T" 2>/dev/null || true)"
      if [ -n "${sq}" ]; then
        echo "[wait] ${label}: squeue active: ${sq//$'\n'/, }"
        if [ "${waited}" -ge "${MAX_WAIT_SEC}" ]; then
          echo "[error] timeout waiting for ${label}: job ${job_id}" >&2
          exit 1
        fi
        sleep "${POLL_SEC}"
        waited=$((waited + POLL_SEC))
        continue
      fi
    fi

    local decision="unknown|sacct unavailable"
    if command -v sacct >/dev/null 2>&1; then
      local sacct_out
      sacct_out="$(sacct -j "${job_id}" --format=JobID,State,ExitCode -P -n 2>/dev/null || true)"
      decision="$(sacct_decision "${job_id}" "${expected_array_count}" "${sacct_out}")"
      echo "[wait] ${label}: sacct ${decision}"
    else
      echo "[wait] ${label}: sacct unavailable"
    fi

    case "${decision%%|*}" in
      success)
        echo "[done] ${label}: job ${job_id}; ${decision#*|}"
        return 0
        ;;
      failure)
        echo "[error] ${label}: job ${job_id} failed: ${decision#*|}" >&2
        exit 1
        ;;
      running)
        accounting_waited=0
        ;;
      unknown)
        accounting_waited=$((accounting_waited + POLL_SEC))
        ;;
    esac

    if [ "${kind}" = "exact" ] && [ -n "${iter_dir}" ] && exact_shards_complete "${iter_dir}" "${expected_array_count}"; then
      echo "[done] ${label}: all ${expected_array_count}/${expected_array_count} exact shard npz/json files are present and sacct has no failure."
      return 0
    fi

    if [ "${accounting_waited}" -ge "${ACCOUNTING_DELAY_SEC}" ]; then
      echo "[error] ${label}: job ${job_id} is absent from squeue and not resolved by sacct after ${ACCOUNTING_DELAY_SEC}s." >&2
      if [ "${kind}" = "exact" ] && [ -n "${iter_dir}" ]; then
        echo "[error] exact shards found: $(shards_found "${iter_dir}" "${expected_array_count}")/${expected_array_count}" >&2
      fi
      exit 1
    fi

    if [ "${waited}" -ge "${MAX_WAIT_SEC}" ]; then
      echo "[error] timeout waiting for ${label}: job ${job_id}" >&2
      exit 1
    fi
    sleep "${POLL_SEC}"
    waited=$((waited + POLL_SEC))
  done
}

latest_dataset_iter="${START_ITER}"
low_append_iters=0
early_stop_reason=""

for (( iter=START_ITER; iter<END_ITER; iter++ )); do
  iter_tag="$(printf '%03d' "${iter}")"
  next_tag="$(printf '%03d' "$((iter + 1))")"
  iter_dir="${run_dir}/iter${iter_tag}"
  mkdir -p "${iter_dir}"
  status_update "${iter_dir}" "iter=${iter}" "run_id=${RUN_ID}" "status_path=${iter_dir}/status.json"

  resume_dataset=""
  if [ "${iter}" -gt 0 ]; then
    resume_dataset="${run_dir}/dataset_iter${iter_tag}.npz"
    if [ ! -f "${resume_dataset}" ]; then
      echo "[error] resume dataset missing for iter ${iter_tag}: ${resume_dataset}" >&2
      exit 1
    fi
  fi

  if [ -f "${iter_dir}/selected_points.csv" ]; then
    echo "[iter ${iter_tag}] candidate generation already done; reusing ${iter_dir}/selected_points.csv"
    status_update "${iter_dir}" "candidate_done=true"
  else
    existing_candidate_job="$(status_get "${iter_dir}" "candidate_job_id" "")"
    if [ -n "${existing_candidate_job}" ] && status_true "${iter_dir}" "candidate_submitted"; then
      echo "[iter ${iter_tag}] waiting for existing candidate job ${existing_candidate_job}"
      wait_for_job "${existing_candidate_job}" "candidate generation iter ${iter_tag}"
    else
      echo "[iter ${iter_tag}] submit candidate generation"
      candidate_sbatch_args=(--parsable)
      if [ -n "${EXCLUDE_NODES}" ]; then
        candidate_sbatch_args+=(--exclude="${EXCLUDE_NODES}")
      fi
      candidate_job="$(
        sbatch "${candidate_sbatch_args[@]}" \
          --export=ALL,PROJECT_DIR="${PROJECT_DIR}",PYTHON_BIN="${PYTHON_BIN}",CONDA_ENV="${CONDA_ENV}",WARM_START="${WARM_START}",RUN_ID="${RUN_ID}",RUN_MODE="${RUN_MODE}",CANDIDATE_DOMAIN_MODE="${CANDIDATE_DOMAIN_MODE}",INITIALIZATION="${INITIALIZATION}",INITIAL_SEED_SIZE="${INITIAL_SEED_SIZE}",BATCH_SIZE_MAX="${BATCH_SIZE_MAX}",BATCH_SIZE_MIN="${BATCH_SIZE_MIN}",BATCH_SIZE_MIN_BEFORE_MIN_ITER="${BATCH_SIZE_MIN_BEFORE_MIN_ITER}",BATCH_SIZE_MIN_AFTER_MIN_ITER="${BATCH_SIZE_MIN_AFTER_MIN_ITER}",SELECTION_MODE="${SELECTION_MODE}",SAMPLING_POWER="${SAMPLING_POWER}",SAMPLING_POWER_START="${SAMPLING_POWER_START}",SAMPLING_POWER_MID="${SAMPLING_POWER_MID}",SAMPLING_POWER_END="${SAMPLING_POWER_END}",SAMPLING_POWER_MID_ITER="${SAMPLING_POWER_MID_ITER}",SAMPLING_POWER_END_ITER="${SAMPLING_POWER_END_ITER}",SAMPLING_POWER_SCHEDULE="${SAMPLING_POWER_SCHEDULE}",SCORE_THRESHOLD_ABS="${SCORE_THRESHOLD_ABS}",SCORE_THRESHOLD_REL="${SCORE_THRESHOLD_REL}",ACTIVE_POOL_RULE="${ACTIVE_POOL_RULE}",ACTIVE_POOL_QUANTILE="${ACTIVE_POOL_QUANTILE}",ACTIVE_POOL_QUANTILE_SCHEDULE="${ACTIVE_POOL_QUANTILE_SCHEDULE}",ACTIVE_POOL_QUANTILE_START="${ACTIVE_POOL_QUANTILE_START}",ACTIVE_POOL_QUANTILE_MID="${ACTIVE_POOL_QUANTILE_MID}",ACTIVE_POOL_QUANTILE_END="${ACTIVE_POOL_QUANTILE_END}",ACTIVE_POOL_QUANTILE_MID_ITER="${ACTIVE_POOL_QUANTILE_MID_ITER}",ACTIVE_POOL_QUANTILE_END_ITER="${ACTIVE_POOL_QUANTILE_END_ITER}",ACTIVE_POOL_REL_TO_P95="${ACTIVE_POOL_REL_TO_P95}",ACTIVE_POOL_MIN_QUANTILE="${ACTIVE_POOL_MIN_QUANTILE}",ACTIVE_POOL_MAX_FRACTION_START="${ACTIVE_POOL_MAX_FRACTION_START}",ACTIVE_POOL_MAX_FRACTION_END="${ACTIVE_POOL_MAX_FRACTION_END}",ACTIVE_POOL_MAX_FRACTION_END_ITER="${ACTIVE_POOL_MAX_FRACTION_END_ITER}",ACTIVE_SELECTION_MIN_ITERATIONS="${ACTIVE_SELECTION_MIN_ITERATIONS}",NO_UNDERFILLED_BATCH_AFTER_MIN_ITER="${NO_UNDERFILLED_BATCH_AFTER_MIN_ITER}",B_DELTA_GATE_MODE="${B_DELTA_GATE_MODE}",Q_BOUNDARY_GATE_MODE="${Q_BOUNDARY_GATE_MODE}",INTERIOR_FILTER_MODE="${INTERIOR_FILTER_MODE}",INTERIOR_PENALTY_START_ITER="${INTERIOR_PENALTY_START_ITER}",INTERIOR_PENALTY_EARLY="${INTERIOR_PENALTY_EARLY}",INTERIOR_PENALTY_LATE="${INTERIOR_PENALTY_LATE}",P_CONF_THRESHOLD="${P_CONF_THRESHOLD}",U_NS_LOW="${U_NS_LOW}",U_UF_LOW="${U_UF_LOW}",G_PHASE_LOW="${G_PHASE_LOW}",E_Q_LOW="${E_Q_LOW}",E_EXT_LOW="${E_EXT_LOW}",W_EXT_SCHEDULE="${W_EXT_SCHEDULE}",W_EXT_START="${W_EXT_START}",W_EXT_MID="${W_EXT_MID}",W_EXT_END="${W_EXT_END}",W_EXT_MID_ITER="${W_EXT_MID_ITER}",W_EXT_END_ITER="${W_EXT_END_ITER}",RANDOM_SEED="${RANDOM_SEED}",FINITE_T_BAND_WIDTH="${FINITE_T_BAND_WIDTH}",HIDDEN_GROUND_TRUTH="${HIDDEN_GROUND_TRUTH}",ITERATIONS=1,START_ITERATION="${iter}",RESUME_DATASET="${resume_dataset}",POINTS_PER_ITER="${POINTS_PER_ITER}",WORLD_SIZE="${WORLD_SIZE}",N_ENSEMBLE="${N_ENSEMBLE}",REG_EPOCHS="${REG_EPOCHS}",CLS_EPOCHS="${CLS_EPOCHS}",BATCH_SIZE="${BATCH_SIZE}",BOUNDARY_REFINEMENT_MODE="${BOUNDARY_REFINEMENT_MODE}",BOUNDARY_KT_BIN_WIDTH="${BOUNDARY_KT_BIN_WIDTH}",BOUNDARY_MAX_LOCAL_SPACING="${BOUNDARY_MAX_LOCAL_SPACING}",BOUNDARY_POSITION_TOL="${BOUNDARY_POSITION_TOL}" \
          scripts/slurm_active_refine.sh
      )"
      candidate_job="${candidate_job%%;*}"
      status_update "${iter_dir}" "candidate_job_id=${candidate_job}" "candidate_submitted=true" "candidate_submit_time=$(now_iso)"
      wait_for_job "${candidate_job}" "candidate generation iter ${iter_tag}"
    fi
    if [ ! -f "${iter_dir}/selected_points.csv" ]; then
      echo "[error] selected_points.csv missing after candidate generation: ${iter_dir}/selected_points.csv" >&2
      exit 1
    fi
    status_update "${iter_dir}" "candidate_done=true" "candidate_done_time=$(now_iso)"
  fi

  selected_count="$(csv_data_rows "${iter_dir}/selected_points.csv")"
  status_update "${iter_dir}" "selected_points=${selected_count}"
  echo "[iter ${iter_tag}] selected points: ${selected_count}"
  if [ "${selected_count}" -le 0 ]; then
    early_stop_reason="no selected candidate points in iter ${iter_tag}"
    echo "[stop] ${early_stop_reason}"
    cat > "${iter_dir}/stop_metrics_iter${iter_tag}.json" <<EOF
{
  "iteration": ${iter},
  "stop": true,
  "stop_reason": "no_available_candidates",
  "note": "No selected candidates is treated as an exceptional boundary case, not the main convergence criterion."
}
EOF
    status_update "${iter_dir}" "stop_checked=true" "stop_metrics=${iter_dir}/stop_metrics_iter${iter_tag}.json" "completed=true"
    break
  fi

  current_shards_found="$(shards_found "${iter_dir}" "${WORLD_SIZE}")"
  if status_true "${iter_dir}" "exact_done" || exact_shards_complete "${iter_dir}" "${WORLD_SIZE}"; then
    echo "[iter ${iter_tag}] exact oracle already done; shards=${current_shards_found}/${WORLD_SIZE}"
    status_update "${iter_dir}" "exact_done=true" "exact_done_time=$(now_iso)" "shards_expected=${WORLD_SIZE}" "shards_found=${current_shards_found}"
  else
    existing_exact_job="$(status_get "${iter_dir}" "exact_job_id" "")"
    if [ -n "${existing_exact_job}" ] && status_true "${iter_dir}" "exact_submitted"; then
      echo "[iter ${iter_tag}] waiting for existing exact oracle job ${existing_exact_job}"
      wait_for_job "${existing_exact_job}" "HPC exact oracle iter ${iter_tag}" "exact" "${iter_dir}" "${WORLD_SIZE}"
    else
      echo "[iter ${iter_tag}] submit HPC exact oracle"
      array_last=$((WORLD_SIZE - 1))
      exact_sbatch_args=(--parsable --array="0-${array_last}")
      if [ -n "${EXCLUDE_NODES}" ]; then
        exact_sbatch_args+=(--exclude="${EXCLUDE_NODES}")
      fi
      exact_job="$(
        sbatch "${exact_sbatch_args[@]}" \
          --export=ALL,PROJECT_DIR="${PROJECT_DIR}",PYTHON_BIN="${PYTHON_BIN}",CONDA_ENV="${CONDA_ENV}",CUDA_MODULE="${CUDA_MODULE}",RUN_ID="${RUN_ID}",ITER="${iter}",WORLD_SIZE="${WORLD_SIZE}",Q_EXPANSION_FLAG="${Q_EXPANSION_FLAG}",DELTA_REFINEMENT_FLAG="${DELTA_REFINEMENT_FLAG}",MAX_Q_REFINEMENTS="${MAX_Q_REFINEMENTS}",MAX_DELTA_REFINEMENTS="${MAX_DELTA_REFINEMENTS}",N_DELTA_REFINED="${N_DELTA_REFINED}",POSITIVE_DELTA_GAP_TOL="${POSITIVE_DELTA_GAP_TOL}" \
          scripts/slurm_exact_oracle_array.sh
      )"
      exact_job="${exact_job%%;*}"
      old_exact_job="$(status_get "${iter_dir}" "exact_job_id" "")"
      if [ -n "${old_exact_job}" ] && [ "${old_exact_job}" != "${exact_job}" ]; then
        echo "[error] iter ${iter_tag} would record multiple exact_job_id values: old=${old_exact_job}, new=${exact_job}" >&2
        echo "[hint] inspect ${iter_dir}/status.json before resubmitting exact oracle." >&2
        exit 1
      fi
      status_update "${iter_dir}" "exact_job_id=${exact_job}" "exact_submitted=true" "exact_submit_time=$(now_iso)" "shards_expected=${WORLD_SIZE}"
      wait_for_job "${exact_job}" "HPC exact oracle iter ${iter_tag}" "exact" "${iter_dir}" "${WORLD_SIZE}"
    fi
    current_shards_found="$(shards_found "${iter_dir}" "${WORLD_SIZE}")"
    if [ "${current_shards_found}" -ne "${WORLD_SIZE}" ]; then
      echo "[error] exact oracle wait finished but shards are incomplete: ${current_shards_found}/${WORLD_SIZE}" >&2
      exit 1
    fi
    status_update "${iter_dir}" "exact_done=true" "exact_done_time=$(now_iso)" "shards_expected=${WORLD_SIZE}" "shards_found=${current_shards_found}"
  fi

  merged_path="${iter_dir}/exact_merged_iter${iter_tag}.npz"
  trusted_path="${iter_dir}/exact_trusted_iter${iter_tag}.npz"
  if status_true "${iter_dir}" "merged" && [ -f "${merged_path}" ] && [ -f "${trusted_path}" ]; then
    echo "[iter ${iter_tag}] merge already done: ${merged_path}"
  else
    echo "[iter ${iter_tag}] merge shards"
    "${PYTHON_BIN}" -m ml_phase.hpc \
      --merge \
      --run-dir "${run_dir}" \
      --iteration "${iter}" \
      --world-size "${WORLD_SIZE}" \
      --positive-delta-gap-tol "${POSITIVE_DELTA_GAP_TOL}"
    if [ ! -f "${merged_path}" ] || [ ! -f "${trusted_path}" ]; then
      echo "[error] merge did not produce expected files: ${merged_path}, ${trusted_path}" >&2
      exit 1
    fi
    status_update "${iter_dir}" "merged=true" "merged_path=${merged_path}" "trusted_filtered=true" "trusted_path=${trusted_path}"
  fi

  dataset_in="${run_dir}/dataset_iter${iter_tag}.npz"
  dataset_out="${run_dir}/dataset_iter${next_tag}.npz"
  dataset_out_csv="${run_dir}/dataset_iter${next_tag}.csv"
  append_summary="${run_dir}/dataset_iter${next_tag}.append.json"
  if status_true "${iter_dir}" "appended" && [ -f "${dataset_out}" ]; then
    echo "[iter ${iter_tag}] append already done: ${dataset_out}"
  elif [ -f "${dataset_out}" ]; then
    echo "[iter ${iter_tag}] dataset_out already exists without appended=true; entering recovery check."
    if [ ! -f "${append_summary}" ]; then
      echo "[error] dataset exists but append summary is missing: ${append_summary}" >&2
      echo "[hint] run scripts/recover_active_iter.sh for this iteration or inspect manually; refusing to overwrite." >&2
      exit 1
    fi
    new_unique_samples="$(json_get_int "${append_summary}" "new_unique_samples_added" 0)"
    training_eligible_appended="$(json_get_int "${append_summary}" "training_eligible_points_appended" 0)"
    status_update "${iter_dir}" "appended=true" "append_summary=${append_summary}" "dataset_out=${dataset_out}" "new_unique_samples_added=${new_unique_samples}" "training_eligible_points_appended=${training_eligible_appended}"
  else
    echo "[iter ${iter_tag}] append trusted exact points"
    "${PYTHON_BIN}" -m ml_phase.append_trusted \
      --dataset "${dataset_in}" \
      --trusted-exact "${trusted_path}" \
      --output-npz "${dataset_out}" \
      --output-csv "${dataset_out_csv}" \
      --output-root "ML_Phase"
    if [ ! -f "${dataset_out}" ]; then
      echo "[error] append did not produce dataset: ${dataset_out}" >&2
      exit 1
    fi
    new_unique_samples="$(json_get_int "${append_summary}" "new_unique_samples_added" 0)"
    training_eligible_appended="$(json_get_int "${append_summary}" "training_eligible_points_appended" 0)"
    status_update "${iter_dir}" "appended=true" "append_summary=${append_summary}" "dataset_out=${dataset_out}" "new_unique_samples_added=${new_unique_samples}" "training_eligible_points_appended=${training_eligible_appended}"
  fi

  new_unique_samples="$(json_get_int "${append_summary}" "new_unique_samples_added" 0)"
  training_eligible_appended="$(json_get_int "${append_summary}" "training_eligible_points_appended" 0)"
  latest_dataset_iter="$((iter + 1))"
  echo "[iter ${iter_tag}] training eligible appended: ${training_eligible_appended}"
  echo "[iter ${iter_tag}] new unique samples added: ${new_unique_samples}"
  echo "[iter ${iter_tag}] rerun list: ${iter_dir}/rerun_points.csv"

  if [ "${ENABLE_EARLY_STOP}" = "1" ]; then
    if [ "${new_unique_samples}" -lt "${MIN_NEW_POINTS_PER_ITER}" ]; then
      low_append_iters="$((low_append_iters + 1))"
      echo "[warn] low new-sample append count ${new_unique_samples} < ${MIN_NEW_POINTS_PER_ITER}; streak=${low_append_iters}/${MAX_LOW_APPEND_ITERS}"
    else
      low_append_iters=0
    fi
  fi

  stop_metrics="${iter_dir}/stop_metrics_iter${iter_tag}.json"
  if [ "${ENABLE_STOP_CONTROLLER}" = "1" ]; then
    if status_true "${iter_dir}" "stop_checked" && [ -f "${stop_metrics}" ]; then
      echo "[iter ${iter_tag}] stop metrics already checked: ${stop_metrics}"
    else
      echo "[iter ${iter_tag}] evaluate convergence stop metrics"
      stop_args=(
        --run-dir "${run_dir}"
        --iteration "${iter}"
        --current-dataset "${dataset_out}"
        --output-root "ML_Phase"
        --min-iterations "${STOP_MIN_ITERATIONS}"
        --patience "${STOP_PATIENCE}"
        --max-iterations "${STOP_MAX_ITERATIONS}"
        --map-tol "${STOP_MAP_TOL}"
        --surprise-tol "${STOP_SURPRISE_TOL}"
        --selected-a0-ratio-tol "${STOP_SELECTED_A0_RATIO_TOL}"
        --qedge-rate-tol "${STOP_QEDGE_RATE_TOL}"
        --rerun-rate-tol "${STOP_RERUN_RATE_TOL}"
      )
      if [ -n "${STOP_MAX_EXACT_CALLS}" ]; then
        stop_args+=(--max-exact-calls "${STOP_MAX_EXACT_CALLS}")
      fi
      if [ "${STOP_ALLOW_MISSING_BOUNDARY}" = "1" ]; then
        stop_args+=(--allow-missing-boundary)
      fi
      "${PYTHON_BIN}" -m ml_phase.stop_controller "${stop_args[@]}"
      if [ ! -f "${stop_metrics}" ]; then
        echo "[error] stop controller did not produce metrics: ${stop_metrics}" >&2
        exit 1
      fi
      status_update "${iter_dir}" "stop_checked=true" "stop_metrics=${stop_metrics}"
    fi
    should_stop="$(json_get_int "${stop_metrics}" "stop" 0)"
    if [ "${should_stop}" -eq 1 ]; then
      early_stop_reason="$("${PYTHON_BIN}" -c 'import json, sys
path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(data.get("stop_reason") or "stop_controller")
except Exception:
    print("stop_controller")
' "${stop_metrics}")"
      status_update "${iter_dir}" "completed=true" "completed_time=$(now_iso)"
      echo "[stop] ${early_stop_reason}"
      break
    fi
  fi
  status_update "${iter_dir}" "completed=true" "completed_time=$(now_iso)"
done

echo "[final] build report"
"${PYTHON_BIN}" -m ml_phase.report_builder \
  --run-id "${RUN_ID}" \
  --run-root "ML_Phase/active_runs" \
  --output "ML_Phase/reports/active_learning_phase_boundary_report.tex"

if command -v pdflatex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode -halt-on-error \
    -output-directory ML_Phase/reports \
    ML_Phase/reports/active_learning_phase_boundary_report.tex
else
  echo "[warn] pdflatex not found; TeX report generated only."
fi

echo "[done] active-learning loop completed for RUN_ID=${RUN_ID}"
if [ -n "${early_stop_reason}" ]; then
  echo "[done] early stop reason: ${early_stop_reason}"
fi
echo "[result] run directory: ${run_dir}"
echo "[result] latest dataset: ${run_dir}/dataset_iter$(printf '%03d' "${latest_dataset_iter}").npz"
