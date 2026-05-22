#!/bin/bash
set -euo pipefail

export PROJECT_DIR="${PROJECT_DIR:-$PWD}"
export PYTHON_BIN="${PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}"
export EXCLUDE_NODES="${EXCLUDE_NODES:-gpuh01}"
export WARM_START="${WARM_START:-eta_phase_diagram_nkt138_nja156_nd400_nq400_nk800_kc8_jc8_dc4_qc100_kk200_eb10000_fp64_libcusolver_cfg422bd68ce6/eta_phase_diagram_nkt138_nja156_nd400_nq400_nk800_kc8_jc8_dc4_qc100_kk200_eb10000_fp64_libcusolver_cfg422bd68ce6.npz}"

export RUN_ID="${RUN_ID:-active_boundary_warmup_512x50_acquisition_only}"
export RUN_MODE="${RUN_MODE:-refinement}"
export CANDIDATE_DOMAIN_MODE="${CANDIDATE_DOMAIN_MODE:-prior_band}"
export FINITE_T_BAND_WIDTH="${FINITE_T_BAND_WIDTH:-0.08}"
export SELECTION_MODE="${SELECTION_MODE:-topk}"
export START_ITER="${START_ITER:-0}"
export N_ITERS="${N_ITERS:-50}"
export POINTS_PER_ITER="${POINTS_PER_ITER:-512}"
export WORLD_SIZE="${WORLD_SIZE:-8}"

export N_ENSEMBLE="${N_ENSEMBLE:-5}"
export REG_EPOCHS="${REG_EPOCHS:-240}"
export CLS_EPOCHS="${CLS_EPOCHS:-240}"
export BATCH_SIZE="${BATCH_SIZE:-512}"

export ENABLE_EARLY_STOP="${ENABLE_EARLY_STOP:-1}"
export MIN_NEW_POINTS_PER_ITER="${MIN_NEW_POINTS_PER_ITER:-8}"
export MAX_LOW_APPEND_ITERS="${MAX_LOW_APPEND_ITERS:-2}"
export BOUNDARY_REFINEMENT_MODE="${BOUNDARY_REFINEMENT_MODE:-diagnostic}"

unset RESUME_DATASET

echo "[run] starting from warm-start exact dataset"
echo "[run] PROJECT_DIR=${PROJECT_DIR}"
echo "[run] RUN_ID=${RUN_ID}"
echo "[run] RUN_MODE=${RUN_MODE}"
echo "[run] CANDIDATE_DOMAIN_MODE=${CANDIDATE_DOMAIN_MODE}"
echo "[run] START_ITER=${START_ITER}"
echo "[run] N_ITERS=${N_ITERS}"
echo "[run] POINTS_PER_ITER=${POINTS_PER_ITER}"
echo "[run] WORLD_SIZE=${WORLD_SIZE}"
echo "[run] PYTHON_BIN=${PYTHON_BIN}"
echo "[run] EXCLUDE_NODES=${EXCLUDE_NODES}"
echo "[run] BOUNDARY_REFINEMENT_MODE=${BOUNDARY_REFINEMENT_MODE}"

bash hpc_active_loop.sh
