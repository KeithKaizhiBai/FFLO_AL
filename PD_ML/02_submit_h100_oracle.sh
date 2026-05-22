#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
RUN_ID="${RUN_ID:-active_boundary_h100_smoke}"
ITER="${ITER:-0}"
WORLD_SIZE="${WORLD_SIZE:-4}"
CONDA_ENV="${CONDA_ENV:-}"
CUDA_MODULE="${CUDA_MODULE:-compiler/cuda/cuda-12.8.1}"

cd "${PROJECT_DIR}"

ITER_DIR="ML_Phase/active_runs/${RUN_ID}/iter$(printf "%03d" "${ITER}")"
if [ ! -d "${ITER_DIR}" ]; then
  echo "[error] iteration directory not found: ${ITER_DIR}" >&2
  echo "[hint] run 01_prepare_candidate_shards.sh first and wait for it to finish." >&2
  exit 1
fi

if [ ! -f "${ITER_DIR}/selected_points_rank000_of$(printf "%03d" "${WORLD_SIZE}").csv" ]; then
  echo "[error] selected point shards for WORLD_SIZE=${WORLD_SIZE} not found in ${ITER_DIR}" >&2
  exit 1
fi

export PROJECT_DIR RUN_ID ITER WORLD_SIZE CONDA_ENV CUDA_MODULE

echo "[submit] H100 exact oracle array"
echo "[submit] RUN_ID=${RUN_ID}, ITER=${ITER}, WORLD_SIZE=${WORLD_SIZE}"
sbatch -J "al_exact_${RUN_ID}_${ITER}" -D "${PROJECT_DIR}" --array="0-$((WORLD_SIZE - 1))" scripts/slurm_exact_oracle_array.sh
