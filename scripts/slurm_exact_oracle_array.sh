#!/bin/bash
#SBATCH --job-name=al_exact
#SBATCH --partition=NV_H100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-7

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
CONDA_ENV="${CONDA_ENV:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_MODULE="${CUDA_MODULE:-compiler/cuda/cuda-12.8.1}"

cd "${PROJECT_DIR}"

if command -v module >/dev/null 2>&1; then
  module purge || true
  if [ -n "${CUDA_MODULE}" ]; then
    module load "${CUDA_MODULE}" || true
  fi
fi

if [ -n "${CONDA_ENV}" ] && [ "${PYTHON_BIN}" = "python" ]; then
  PYTHON_BIN="${HOME}/.conda/envs/${CONDA_ENV}/bin/python"
fi

RUN_ID="${RUN_ID:?set RUN_ID}"
ITER="${ITER:?set ITER}"
WORLD_SIZE="${WORLD_SIZE:-${SLURM_ARRAY_TASK_COUNT}}"
RANK="${SLURM_ARRAY_TASK_ID}"
Q_EXPANSION_FLAG="${Q_EXPANSION_FLAG:---enable-q-expansion}"
DELTA_REFINEMENT_FLAG="${DELTA_REFINEMENT_FLAG:---enable-delta-refinement}"
Q_EXPAND_FACTOR="${Q_EXPAND_FACTOR:-1.5}"
Q_EXPAND_PAD_STEPS="${Q_EXPAND_PAD_STEPS:-50}"
Q_MAX_ABS="${Q_MAX_ABS:-3.141592653589793}"
MAX_Q_REFINEMENTS="${MAX_Q_REFINEMENTS:-3}"
DELTA_REFINE_HALF_WIDTH="${DELTA_REFINE_HALF_WIDTH:-0.03}"
N_DELTA_REFINED="${N_DELTA_REFINED:-300}"
MAX_DELTA_REFINEMENTS="${MAX_DELTA_REFINEMENTS:-2}"
POSITIVE_DELTA_GAP_TOL="${POSITIVE_DELTA_GAP_TOL:-1e-8}"

"${PYTHON_BIN}" -m ml_phase.exact_oracle \
  --run-id "${RUN_ID}" \
  --iteration "${ITER}" \
  --rank "${RANK}" \
  --world-size "${WORLD_SIZE}" \
  --device cuda:0 \
  ${Q_EXPANSION_FLAG} \
  --q-expand-factor "${Q_EXPAND_FACTOR}" \
  --q-expand-pad-steps "${Q_EXPAND_PAD_STEPS}" \
  --q-max-abs "${Q_MAX_ABS}" \
  --max-q-refinements "${MAX_Q_REFINEMENTS}" \
  ${DELTA_REFINEMENT_FLAG} \
  --delta-refine-half-width "${DELTA_REFINE_HALF_WIDTH}" \
  --n-delta-refined "${N_DELTA_REFINED}" \
  --max-delta-refinements "${MAX_DELTA_REFINEMENTS}" \
  --positive-delta-gap-tol "${POSITIVE_DELTA_GAP_TOL}"
