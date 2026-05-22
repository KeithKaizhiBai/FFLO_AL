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
CUDA_MODULE="${CUDA_MODULE:-compiler/cuda/cuda-12.8.1}"

cd "${PROJECT_DIR}"

if command -v module >/dev/null 2>&1; then
  module purge || true
  if [ -n "${CUDA_MODULE}" ]; then
    module load "${CUDA_MODULE}" || true
  fi
fi

if [ -n "${CONDA_ENV}" ]; then
  if [ -f "/public/software/apps/anaconda/etc/profile.d/conda.sh" ]; then
    source /public/software/apps/anaconda/etc/profile.d/conda.sh
  fi
  source activate "${CONDA_ENV}"
fi

RUN_ID="${RUN_ID:?set RUN_ID}"
ITER="${ITER:?set ITER}"
WORLD_SIZE="${WORLD_SIZE:-${SLURM_ARRAY_TASK_COUNT}}"
RANK="${SLURM_ARRAY_TASK_ID}"

python -m ml_phase.exact_oracle \
  --run-id "${RUN_ID}" \
  --iteration "${ITER}" \
  --rank "${RANK}" \
  --world-size "${WORLD_SIZE}" \
  --device cuda:0
