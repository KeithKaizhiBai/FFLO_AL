#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}"
CONDA_ENV="${CONDA_ENV:-}"
CUDA_MODULE="${CUDA_MODULE:-compiler/cuda/cuda-12.8.1}"

RUN_ID="${RUN_ID:-active_boundary_qdelta_smoke}"
ITER="${ITER:-0}"
POINTS_PER_ITER="${POINTS_PER_ITER:-32}"
WORLD_SIZE="${WORLD_SIZE:-4}"
N_ENSEMBLE="${N_ENSEMBLE:-2}"
REG_EPOCHS="${REG_EPOCHS:-20}"
CLS_EPOCHS="${CLS_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
PARTITION_STRATEGY="${PARTITION_STRATEGY:-round_robin}"
POLL_SEC="${POLL_SEC:-30}"
MAX_WAIT_SEC="${MAX_WAIT_SEC:-604800}"
EXCLUDE_NODES="${EXCLUDE_NODES:-gpuh01}"
POSITIVE_DELTA_GAP_TOL="${POSITIVE_DELTA_GAP_TOL:-1e-8}"

WARM_START="${WARM_START:-eta_phase_diagram_nkt138_nja156_nd400_nq400_nk800_kc8_jc8_dc4_qc100_kk200_eb10000_fp64_libcusolver_cfg422bd68ce6/eta_phase_diagram_nkt138_nja156_nd400_nq400_nk800_kc8_jc8_dc4_qc100_kk200_eb10000_fp64_libcusolver_cfg422bd68ce6.npz}"

cd "${PROJECT_DIR}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "[error] PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

if [ ! -f "${WARM_START}" ]; then
  echo "[error] warm-start npz not found: ${WARM_START}" >&2
  exit 1
fi

if [ ! -f "scripts/slurm_active_refine.sh" ]; then
  echo "[error] missing scripts/slurm_active_refine.sh" >&2
  exit 1
fi

if [ ! -f "scripts/slurm_exact_oracle_array.sh" ]; then
  echo "[error] missing scripts/slurm_exact_oracle_array.sh" >&2
  exit 1
fi

wait_for_job() {
  local job_id="$1"
  local label="$2"
  local waited=0
  echo "[wait] ${label}: job ${job_id}"
  while squeue -j "${job_id}" -h | grep -q .; do
    if [ "${waited}" -ge "${MAX_WAIT_SEC}" ]; then
      echo "[error] timeout waiting for ${label}: job ${job_id}" >&2
      exit 1
    fi
    sleep "${POLL_SEC}"
    waited=$((waited + POLL_SEC))
  done
  echo "[done] ${label}: job ${job_id}"
  if command -v sacct >/dev/null 2>&1; then
    local state
    state="$(sacct -j "${job_id}" --format=State --noheader 2>/dev/null | head -n 1 | awk '{print $1}' || true)"
    if echo "${state}" | grep -Eq 'FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL'; then
      echo "[error] ${label} ended with state ${state}" >&2
      exit 1
    fi
  fi
}

echo "[config] PROJECT_DIR=${PROJECT_DIR}"
echo "[config] PYTHON_BIN=${PYTHON_BIN}"
echo "[config] RUN_ID=${RUN_ID}"
echo "[config] ITER=${ITER}"
echo "[config] POINTS_PER_ITER=${POINTS_PER_ITER}"
echo "[config] WORLD_SIZE=${WORLD_SIZE}"
echo "[config] EXCLUDE_NODES=${EXCLUDE_NODES}"

echo "[step 1] submit candidate-shard generation"
candidate_sbatch_args=(--parsable)
if [ -n "${EXCLUDE_NODES}" ]; then
  candidate_sbatch_args+=(--exclude="${EXCLUDE_NODES}")
fi
candidate_job="$(
  sbatch "${candidate_sbatch_args[@]}" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",PYTHON_BIN="${PYTHON_BIN}",CONDA_ENV="${CONDA_ENV}",WARM_START="${WARM_START}",RUN_ID="${RUN_ID}",ITERATIONS=1,POINTS_PER_ITER="${POINTS_PER_ITER}",WORLD_SIZE="${WORLD_SIZE}",N_ENSEMBLE="${N_ENSEMBLE}",REG_EPOCHS="${REG_EPOCHS}",CLS_EPOCHS="${CLS_EPOCHS}",BATCH_SIZE="${BATCH_SIZE}" \
    scripts/slurm_active_refine.sh
)"
wait_for_job "${candidate_job}" "candidate generation"

iter_dir="ML_Phase/active_runs/${RUN_ID}/iter$(printf '%03d' "${ITER}")"
if [ ! -d "${iter_dir}" ]; then
  echo "[error] iteration directory not found after candidate job: ${iter_dir}" >&2
  exit 1
fi

if [ ! -f "${iter_dir}/selected_points.csv" ]; then
  echo "[error] selected_points.csv not found: ${iter_dir}/selected_points.csv" >&2
  exit 1
fi

echo "[step 2] submit H100 exact-oracle array"
array_last=$((WORLD_SIZE - 1))
exact_sbatch_args=(--parsable --array="0-${array_last}")
if [ -n "${EXCLUDE_NODES}" ]; then
  exact_sbatch_args+=(--exclude="${EXCLUDE_NODES}")
fi
exact_job="$(
  sbatch "${exact_sbatch_args[@]}" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",PYTHON_BIN="${PYTHON_BIN}",CONDA_ENV="${CONDA_ENV}",CUDA_MODULE="${CUDA_MODULE}",RUN_ID="${RUN_ID}",ITER="${ITER}",WORLD_SIZE="${WORLD_SIZE}" \
    scripts/slurm_exact_oracle_array.sh
)"
wait_for_job "${exact_job}" "H100 exact oracle"

echo "[step 3] merge exact shards"
"${PYTHON_BIN}" -m ml_phase.hpc \
  --merge \
  --run-dir "ML_Phase/active_runs/${RUN_ID}" \
  --iteration "${ITER}" \
  --world-size "${WORLD_SIZE}" \
  --positive-delta-gap-tol "${POSITIVE_DELTA_GAP_TOL}"

echo "[step 4] build report"
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

echo "[result] selected points: ${iter_dir}/selected_points.csv"
echo "[result] merged exact: ${iter_dir}/exact_merged_iter$(printf '%03d' "${ITER}").npz"
echo "[result] rerun list: ${iter_dir}/rerun_points.csv"
echo "[result] report tex: ML_Phase/reports/active_learning_phase_boundary_report.tex"
echo "[done] one-click workflow completed for RUN_ID=${RUN_ID}"
