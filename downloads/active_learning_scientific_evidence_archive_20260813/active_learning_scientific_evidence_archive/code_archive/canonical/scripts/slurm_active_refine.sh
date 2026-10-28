#!/bin/bash
#SBATCH --job-name=al_refine
#SBATCH --partition=NV_H100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
CONDA_ENV="${CONDA_ENV:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-ML_Phase}"

cd "${PROJECT_DIR}"

if command -v module >/dev/null 2>&1; then
  module purge || true
fi

if [ -n "${CONDA_ENV}" ] && [ "${PYTHON_BIN}" = "python" ]; then
  PYTHON_BIN="${HOME}/.conda/envs/${CONDA_ENV}/bin/python"
fi

WARM_START="${WARM_START:-}"
RUN_ID="${RUN_ID:?set RUN_ID}"
ITERATIONS="${ITERATIONS:-1}"
START_ITERATION="${START_ITERATION:-0}"
RESUME_DATASET="${RESUME_DATASET:-}"
RUN_MODE="${RUN_MODE:-discovery}"
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
ACQUISITION_PROFILE="${ACQUISITION_PROFILE:-full}"
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
W_CLS_SIMPLE="${W_CLS_SIMPLE:-1.0}"
W_NS_SIMPLE="${W_NS_SIMPLE:-1.0}"
W_UF_SIMPLE="${W_UF_SIMPLE:-0.5}"
W_GRAD_SIMPLE="${W_GRAD_SIMPLE:-0.2}"
W_REG_SIMPLE="${W_REG_SIMPLE:-0.1}"
W_EXT_SIMPLE_SCHEDULE="${W_EXT_SIMPLE_SCHEDULE:-piecewise}"
W_EXT_SIMPLE_START="${W_EXT_SIMPLE_START:-0.02}"
W_EXT_SIMPLE_MID="${W_EXT_SIMPLE_MID:-0.01}"
W_EXT_SIMPLE_END="${W_EXT_SIMPLE_END:-0.0}"
W_EXT_SIMPLE_MID_ITER="${W_EXT_SIMPLE_MID_ITER:-10}"
W_EXT_SIMPLE_END_ITER="${W_EXT_SIMPLE_END_ITER:-30}"
SURPRISE_CLEANUP_QEDGE_PENALTY="${SURPRISE_CLEANUP_QEDGE_PENALTY:-0.85}"
SURPRISE_CLEANUP_QEDGE_FLOOR="${SURPRISE_CLEANUP_QEDGE_FLOOR:-0.05}"
SURPRISE_CLEANUP_RESPONSE_WEIGHT="${SURPRISE_CLEANUP_RESPONSE_WEIGHT:-0.25}"
SURPRISE_CLEANUP_EXPLORE_SCALE="${SURPRISE_CLEANUP_EXPLORE_SCALE:-0.5}"
RANDOM_SEED="${RANDOM_SEED:-42}"
FINITE_T_BAND_WIDTH="${FINITE_T_BAND_WIDTH:-}"
HIDDEN_GROUND_TRUTH="${HIDDEN_GROUND_TRUTH:-}"
POINTS_PER_ITER="${POINTS_PER_ITER:-256}"
ORACLE_MODE="${ORACLE_MODE:-robust_al}"
WORLD_SIZE="${WORLD_SIZE:-8}"
N_ENSEMBLE="${N_ENSEMBLE:-5}"
REG_EPOCHS="${REG_EPOCHS:-240}"
CLS_EPOCHS="${CLS_EPOCHS:-240}"
BATCH_SIZE="${BATCH_SIZE:-512}"
BOUNDARY_REFINEMENT_MODE="${BOUNDARY_REFINEMENT_MODE:-diagnostic}"
BOUNDARY_KT_BIN_WIDTH="${BOUNDARY_KT_BIN_WIDTH:-0.005}"
BOUNDARY_MAX_LOCAL_SPACING="${BOUNDARY_MAX_LOCAL_SPACING:-0.035}"
BOUNDARY_POSITION_TOL="${BOUNDARY_POSITION_TOL:-0.00375}"
DRY_RUN_FLAG="${DRY_RUN_FLAG:-}"
RESUME_ARGS=()
if [ -n "${RESUME_DATASET}" ]; then
  RESUME_ARGS=(--resume-dataset "${RESUME_DATASET}")
fi
WARM_ARGS=()
if [ -n "${WARM_START}" ]; then
  WARM_ARGS=(--warm-start "${WARM_START}")
fi
BAND_ARGS=()
if [ -n "${FINITE_T_BAND_WIDTH}" ]; then
  BAND_ARGS=(--finite-t-band-width "${FINITE_T_BAND_WIDTH}")
fi
HIDDEN_ARGS=()
if [ -n "${HIDDEN_GROUND_TRUTH}" ]; then
  HIDDEN_ARGS=(--hidden-ground-truth "${HIDDEN_GROUND_TRUTH}")
fi
UNDERFILL_ARGS=()
if [ "${NO_UNDERFILLED_BATCH_AFTER_MIN_ITER}" = "1" ]; then
  UNDERFILL_ARGS=(--no-underfilled-batch-after-min-iter)
fi

"${PYTHON_BIN}" -m ml_phase.active_refine \
  "${WARM_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  "${BAND_ARGS[@]}" \
  "${HIDDEN_ARGS[@]}" \
  --run-mode "${RUN_MODE}" \
  --candidate-domain-mode "${CANDIDATE_DOMAIN_MODE}" \
  --initialization "${INITIALIZATION}" \
  --initial-seed-size "${INITIAL_SEED_SIZE}" \
  --batch-size-max "${BATCH_SIZE_MAX}" \
  --batch-size-min "${BATCH_SIZE_MIN}" \
  --batch-size-min-before-min-iter "${BATCH_SIZE_MIN_BEFORE_MIN_ITER}" \
  --batch-size-min-after-min-iter "${BATCH_SIZE_MIN_AFTER_MIN_ITER}" \
  --selection-mode "${SELECTION_MODE}" \
  --sampling-power "${SAMPLING_POWER}" \
  --sampling-power-start "${SAMPLING_POWER_START}" \
  --sampling-power-mid "${SAMPLING_POWER_MID}" \
  --sampling-power-end "${SAMPLING_POWER_END}" \
  --sampling-power-mid-iter "${SAMPLING_POWER_MID_ITER}" \
  --sampling-power-end-iter "${SAMPLING_POWER_END_ITER}" \
  --sampling-power-schedule "${SAMPLING_POWER_SCHEDULE}" \
  --score-threshold-abs "${SCORE_THRESHOLD_ABS}" \
  --score-threshold-rel "${SCORE_THRESHOLD_REL}" \
  --acquisition-profile "${ACQUISITION_PROFILE}" \
  --active-pool-rule "${ACTIVE_POOL_RULE}" \
  --active-pool-quantile "${ACTIVE_POOL_QUANTILE}" \
  --active-pool-quantile-schedule "${ACTIVE_POOL_QUANTILE_SCHEDULE}" \
  --active-pool-quantile-start "${ACTIVE_POOL_QUANTILE_START}" \
  --active-pool-quantile-mid "${ACTIVE_POOL_QUANTILE_MID}" \
  --active-pool-quantile-end "${ACTIVE_POOL_QUANTILE_END}" \
  --active-pool-quantile-mid-iter "${ACTIVE_POOL_QUANTILE_MID_ITER}" \
  --active-pool-quantile-end-iter "${ACTIVE_POOL_QUANTILE_END_ITER}" \
  --active-pool-rel-to-p95 "${ACTIVE_POOL_REL_TO_P95}" \
  --active-pool-min-quantile "${ACTIVE_POOL_MIN_QUANTILE}" \
  --active-pool-max-fraction-start "${ACTIVE_POOL_MAX_FRACTION_START}" \
  --active-pool-max-fraction-end "${ACTIVE_POOL_MAX_FRACTION_END}" \
  --active-pool-max-fraction-end-iter "${ACTIVE_POOL_MAX_FRACTION_END_ITER}" \
  --active-selection-min-iterations "${ACTIVE_SELECTION_MIN_ITERATIONS}" \
  "${UNDERFILL_ARGS[@]}" \
  --b-delta-gate-mode "${B_DELTA_GATE_MODE}" \
  --q-boundary-gate-mode "${Q_BOUNDARY_GATE_MODE}" \
  --interior-filter-mode "${INTERIOR_FILTER_MODE}" \
  --interior-penalty-start-iter "${INTERIOR_PENALTY_START_ITER}" \
  --interior-penalty-early "${INTERIOR_PENALTY_EARLY}" \
  --interior-penalty-late "${INTERIOR_PENALTY_LATE}" \
  --p-conf-threshold "${P_CONF_THRESHOLD}" \
  --u-ns-low "${U_NS_LOW}" \
  --u-uf-low "${U_UF_LOW}" \
  --g-phase-low "${G_PHASE_LOW}" \
  --e-q-low "${E_Q_LOW}" \
  --e-ext-low "${E_EXT_LOW}" \
  --w-ext-schedule "${W_EXT_SCHEDULE}" \
  --w-ext-start "${W_EXT_START}" \
  --w-ext-mid "${W_EXT_MID}" \
  --w-ext-end "${W_EXT_END}" \
  --w-ext-mid-iter "${W_EXT_MID_ITER}" \
  --w-ext-end-iter "${W_EXT_END_ITER}" \
  --w-cls-simple "${W_CLS_SIMPLE}" \
  --w-ns-simple "${W_NS_SIMPLE}" \
  --w-uf-simple "${W_UF_SIMPLE}" \
  --w-grad-simple "${W_GRAD_SIMPLE}" \
  --w-reg-simple "${W_REG_SIMPLE}" \
  --w-ext-simple-schedule "${W_EXT_SIMPLE_SCHEDULE}" \
  --w-ext-simple-start "${W_EXT_SIMPLE_START}" \
  --w-ext-simple-mid "${W_EXT_SIMPLE_MID}" \
  --w-ext-simple-end "${W_EXT_SIMPLE_END}" \
  --w-ext-simple-mid-iter "${W_EXT_SIMPLE_MID_ITER}" \
  --w-ext-simple-end-iter "${W_EXT_SIMPLE_END_ITER}" \
  --surprise-cleanup-qedge-penalty "${SURPRISE_CLEANUP_QEDGE_PENALTY}" \
  --surprise-cleanup-qedge-floor "${SURPRISE_CLEANUP_QEDGE_FLOOR}" \
  --surprise-cleanup-response-weight "${SURPRISE_CLEANUP_RESPONSE_WEIGHT}" \
  --surprise-cleanup-explore-scale "${SURPRISE_CLEANUP_EXPLORE_SCALE}" \
  --random-seed "${RANDOM_SEED}" \
  --start-iteration "${START_ITERATION}" \
  --run-id "${RUN_ID}" \
  --mode hpc \
  --iterations "${ITERATIONS}" \
  --points-per-iter "${POINTS_PER_ITER}" \
  --oracle-mode "${ORACLE_MODE}" \
  --output-root "${OUTPUT_ROOT}" \
  --world-size "${WORLD_SIZE}" \
  --partition-strategy round_robin \
  --n-ensemble "${N_ENSEMBLE}" \
  --reg-epochs "${REG_EPOCHS}" \
  --cls-epochs "${CLS_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --boundary-refinement-mode "${BOUNDARY_REFINEMENT_MODE}" \
  --boundary-kt-bin-width "${BOUNDARY_KT_BIN_WIDTH}" \
  --boundary-max-local-spacing "${BOUNDARY_MAX_LOCAL_SPACING}" \
  --boundary-position-tol "${BOUNDARY_POSITION_TOL}" \
  ${DRY_RUN_FLAG}
