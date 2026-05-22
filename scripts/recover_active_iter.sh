#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}"
RUN_ID=""
ITERATION=""
WORLD_SIZE="${WORLD_SIZE:-8}"
POSITIVE_DELTA_GAP_TOL="${POSITIVE_DELTA_GAP_TOL:-1e-8}"
STOP_MIN_ITERATIONS="${STOP_MIN_ITERATIONS:-5}"
STOP_PATIENCE="${STOP_PATIENCE:-4}"
STOP_MAX_ITERATIONS="${STOP_MAX_ITERATIONS:-}"
STOP_MAP_TOL="${STOP_MAP_TOL:-0.002}"
STOP_SURPRISE_TOL="${STOP_SURPRISE_TOL:-0.05}"
STOP_SELECTED_A0_RATIO_TOL="${STOP_SELECTED_A0_RATIO_TOL:-0.15}"
STOP_QEDGE_RATE_TOL="${STOP_QEDGE_RATE_TOL:-0.01}"
STOP_RERUN_RATE_TOL="${STOP_RERUN_RATE_TOL:-0.01}"
STOP_ALLOW_MISSING_BOUNDARY="${STOP_ALLOW_MISSING_BOUNDARY:-0}"
FORCE=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/recover_active_iter.sh --run-id RUN_ID --iteration ITER --world-size N [--force]

Recovers one active-learning iteration from existing exact shard files.
It does not resubmit exact oracle jobs and does not overwrite an existing
dataset_iter{ITER+1}.npz by default.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --iteration)
      ITERATION="$2"
      shift 2
      ;;
    --world-size)
      WORLD_SIZE="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[error] unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [ -z "${RUN_ID}" ] || [ -z "${ITERATION}" ]; then
  echo "[error] --run-id and --iteration are required." >&2
  usage >&2
  exit 1
fi

cd "${PROJECT_DIR}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "[error] PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

iter_tag="$(printf '%03d' "${ITERATION}")"
next_tag="$(printf '%03d' "$((ITERATION + 1))")"
run_dir="ML_Phase/active_runs/${RUN_ID}"
iter_dir="${run_dir}/iter${iter_tag}"
status_path="${iter_dir}/status.json"
merged_path="${iter_dir}/exact_merged_iter${iter_tag}.npz"
trusted_path="${iter_dir}/exact_trusted_iter${iter_tag}.npz"
dataset_in="${run_dir}/dataset_iter${iter_tag}.npz"
dataset_out="${run_dir}/dataset_iter${next_tag}.npz"
dataset_out_csv="${run_dir}/dataset_iter${next_tag}.csv"
append_summary="${run_dir}/dataset_iter${next_tag}.append.json"
stop_metrics="${iter_dir}/stop_metrics_iter${iter_tag}.json"

mkdir -p "${iter_dir}"

status_update() {
  "${PYTHON_BIN}" - "${status_path}" "$@" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if path.exists():
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = {}
else:
    data = {}
for item in sys.argv[2:]:
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

shards_found() {
  local found=0
  local rank
  for (( rank=0; rank<WORLD_SIZE; rank++ )); do
    local tag
    tag="$(printf '%03d' "${rank}")"
    local npz="${iter_dir}/exact_shard_rank${tag}_of$(printf '%03d' "${WORLD_SIZE}").npz"
    local json="${iter_dir}/exact_shard_rank${tag}_of$(printf '%03d' "${WORLD_SIZE}").json"
    if [ -s "${npz}" ] && [ -s "${json}" ]; then
      found=$((found + 1))
    fi
  done
  echo "${found}"
}

found="$(shards_found)"
echo "[recover] RUN_ID=${RUN_ID}"
echo "[recover] iteration=${ITERATION}"
echo "[recover] iter_dir=${iter_dir}"
echo "[recover] shards=${found}/${WORLD_SIZE}"
status_update "iter=${ITERATION}" "run_id=${RUN_ID}" "shards_expected=${WORLD_SIZE}" "shards_found=${found}"

if [ "${found}" -ne "${WORLD_SIZE}" ]; then
  echo "[error] exact shard files are incomplete: ${found}/${WORLD_SIZE}" >&2
  exit 1
fi
status_update "exact_done=true" "exact_done_time=$(date -Is)"

if [ ! -f "${dataset_in}" ]; then
  echo "[error] input dataset is missing: ${dataset_in}" >&2
  exit 1
fi

if [ -f "${merged_path}" ] && [ -f "${trusted_path}" ] && [ "${FORCE}" -ne 1 ]; then
  echo "[recover] merge already exists: ${merged_path}"
else
  echo "[recover] merge shards"
  "${PYTHON_BIN}" -m ml_phase.hpc \
    --merge \
    --run-dir "${run_dir}" \
    --iteration "${ITERATION}" \
    --world-size "${WORLD_SIZE}" \
    --positive-delta-gap-tol "${POSITIVE_DELTA_GAP_TOL}"
fi

if [ ! -f "${merged_path}" ] || [ ! -f "${trusted_path}" ]; then
  echo "[error] merge/trusted output missing after merge step." >&2
  echo "[error] expected: ${merged_path}" >&2
  echo "[error] expected: ${trusted_path}" >&2
  exit 1
fi
status_update "merged=true" "merged_path=${merged_path}" "trusted_filtered=true" "trusted_path=${trusted_path}"

if [ -f "${dataset_out}" ]; then
  echo "[recover] dataset already exists; refusing to overwrite: ${dataset_out}"
  if [ ! -f "${append_summary}" ]; then
    echo "[error] dataset exists but append summary is missing: ${append_summary}" >&2
    exit 1
  fi
else
  echo "[recover] append trusted exact points"
  "${PYTHON_BIN}" -m ml_phase.append_trusted \
    --dataset "${dataset_in}" \
    --trusted-exact "${trusted_path}" \
    --output-npz "${dataset_out}" \
    --output-csv "${dataset_out_csv}" \
    --output-root "ML_Phase"
fi

if [ ! -f "${dataset_out}" ]; then
  echo "[error] recovered dataset missing: ${dataset_out}" >&2
  exit 1
fi
new_unique_samples="$(json_get_int "${append_summary}" "new_unique_samples_added" 0)"
training_eligible_appended="$(json_get_int "${append_summary}" "training_eligible_points_appended" 0)"
status_update "appended=true" "append_summary=${append_summary}" "dataset_out=${dataset_out}" "new_unique_samples_added=${new_unique_samples}" "training_eligible_points_appended=${training_eligible_appended}"

if [ -f "${stop_metrics}" ] && [ "${FORCE}" -ne 1 ]; then
  echo "[recover] stop metrics already exist: ${stop_metrics}"
else
  if [ -z "${STOP_MAX_ITERATIONS}" ]; then
    STOP_MAX_ITERATIONS="$((ITERATION + 1))"
  fi
  echo "[recover] run stop controller"
  stop_args=(
    --run-dir "${run_dir}"
    --iteration "${ITERATION}"
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
  if [ "${STOP_ALLOW_MISSING_BOUNDARY}" = "1" ]; then
    stop_args+=(--allow-missing-boundary)
  fi
  "${PYTHON_BIN}" -m ml_phase.stop_controller "${stop_args[@]}"
fi

if [ ! -f "${stop_metrics}" ]; then
  echo "[error] stop metrics missing: ${stop_metrics}" >&2
  exit 1
fi
status_update "stop_checked=true" "stop_metrics=${stop_metrics}" "completed=true" "completed_time=$(date -Is)"

echo "[recover] completed iteration ${iter_tag}"
echo "[recover] next dataset: ${dataset_out}"
echo "[recover] suggested resume command:"
cat <<EOF
nohup env \\
RUN_ID="${RUN_ID}" \\
START_ITER=$((ITERATION + 1)) \\
N_ITERS=<target_total_minus_$((ITERATION + 1))> \\
BATCH_SIZE_MAX=256 \\
POINTS_PER_ITER=256 \\
STOP_MAX_ITERATIONS=<target_total_iterations> \\
bash hpc_active_loop.sh \\
> discovery_active_loop_resume_from_iter${next_tag}.nohup.log 2>&1 &

echo \$! > active_loop_resume.pid
tail -n 40 discovery_active_loop_resume_from_iter${next_tag}.nohup.log
ps -p \$(cat active_loop_resume.pid) -f
squeue -u \$USER
EOF
