#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
RUN_ID="${RUN_ID:-active_boundary_h100_smoke}"
ITERATIONS="${ITERATIONS:-1}"
POINTS_PER_ITER="${POINTS_PER_ITER:-32}"
WORLD_SIZE="${WORLD_SIZE:-4}"
CONDA_ENV="${CONDA_ENV:-}"
DRY_RUN_FLAG="${DRY_RUN_FLAG:---dry-run}"
N_ENSEMBLE="${N_ENSEMBLE:-2}"
REG_EPOCHS="${REG_EPOCHS:-20}"
CLS_EPOCHS="${CLS_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
WARM_START="${WARM_START:-eta_phase_diagram_nkt138_nja156_nd400_nq400_nk800_kc8_jc8_dc4_qc100_kk200_eb10000_fp64_libcusolver_cfg422bd68ce6/eta_phase_diagram_nkt138_nja156_nd400_nq400_nk800_kc8_jc8_dc4_qc100_kk200_eb10000_fp64_libcusolver_cfg422bd68ce6.npz}"

cd "${PROJECT_DIR}"

export PROJECT_DIR RUN_ID ITERATIONS POINTS_PER_ITER WORLD_SIZE CONDA_ENV
export WARM_START DRY_RUN_FLAG N_ENSEMBLE REG_EPOCHS CLS_EPOCHS BATCH_SIZE

echo "[submit] CPU-side active refine job"
echo "[submit] RUN_ID=${RUN_ID}"
echo "[submit] POINTS_PER_ITER=${POINTS_PER_ITER}, WORLD_SIZE=${WORLD_SIZE}"
echo "[submit] DRY_RUN_FLAG=${DRY_RUN_FLAG}"
sbatch -J "al_refine_${RUN_ID}" -D "${PROJECT_DIR}" scripts/slurm_active_refine.sh

