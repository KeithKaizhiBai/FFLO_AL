#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
RUN_ID="${RUN_ID:-active_boundary_h100_smoke}"
ITER="${ITER:-0}"
WORLD_SIZE="${WORLD_SIZE:-4}"
CONDA_ENV="${CONDA_ENV:-}"

cd "${PROJECT_DIR}"

if [ -n "${CONDA_ENV}" ]; then
  if [ -f "/public/software/apps/anaconda/etc/profile.d/conda.sh" ]; then
    source /public/software/apps/anaconda/etc/profile.d/conda.sh
  fi
  source activate "${CONDA_ENV}"
fi

python -m ml_phase.hpc \
  --run-dir "ML_Phase/active_runs/${RUN_ID}" \
  --iteration "${ITER}" \
  --world-size "${WORLD_SIZE}" \
  --merge

