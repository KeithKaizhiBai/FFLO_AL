#!/bin/bash
#SBATCH --job-name=al_refine
#SBATCH --partition=Intel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
CONDA_ENV="${CONDA_ENV:-}"

cd "${PROJECT_DIR}"

if command -v module >/dev/null 2>&1; then
  module purge || true
fi

if [ -n "${CONDA_ENV}" ]; then
  if [ -f "/public/software/apps/anaconda/etc/profile.d/conda.sh" ]; then
    source /public/software/apps/anaconda/etc/profile.d/conda.sh
  fi
  source activate "${CONDA_ENV}"
fi

WARM_START="${WARM_START:?set WARM_START}"
RUN_ID="${RUN_ID:?set RUN_ID}"
ITERATIONS="${ITERATIONS:-1}"
POINTS_PER_ITER="${POINTS_PER_ITER:-64}"
WORLD_SIZE="${WORLD_SIZE:-8}"
N_ENSEMBLE="${N_ENSEMBLE:-5}"
REG_EPOCHS="${REG_EPOCHS:-240}"
CLS_EPOCHS="${CLS_EPOCHS:-240}"
BATCH_SIZE="${BATCH_SIZE:-512}"
DRY_RUN_FLAG="${DRY_RUN_FLAG:-}"

python -m ml_phase.active_refine \
  --warm-start "${WARM_START}" \
  --run-id "${RUN_ID}" \
  --mode hpc \
  --iterations "${ITERATIONS}" \
  --points-per-iter "${POINTS_PER_ITER}" \
  --world-size "${WORLD_SIZE}" \
  --partition-strategy round_robin \
  --n-ensemble "${N_ENSEMBLE}" \
  --reg-epochs "${REG_EPOCHS}" \
  --cls-epochs "${CLS_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  ${DRY_RUN_FLAG}
