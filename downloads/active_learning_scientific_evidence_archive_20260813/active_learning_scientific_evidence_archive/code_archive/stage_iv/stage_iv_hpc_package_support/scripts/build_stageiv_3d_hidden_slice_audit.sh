#!/bin/bash
set -euo pipefail

STAGEIV_EXPECTED_OUTPUT_ROOT="ML_Phase_StageIV_Topology3D"
STAGEIV_EXPECTED_RUN_ID="active_phase_topology_3d_t_ja_mu_from_scratch_v1"
STAGEIV_EXPECTED_CONFIG_JSON="configs/stageiv_3d_production.json"

if [ -n "${OUTPUT_ROOT:-}" ] && [ "${OUTPUT_ROOT}" != "${STAGEIV_EXPECTED_OUTPUT_ROOT}" ]; then
  echo "[error] refusing stale OUTPUT_ROOT=${OUTPUT_ROOT}; expected ${STAGEIV_EXPECTED_OUTPUT_ROOT}" >&2
  exit 31
fi
if [ -n "${RUN_ID:-}" ] && [ "${RUN_ID}" != "${STAGEIV_EXPECTED_RUN_ID}" ]; then
  echo "[error] refusing stale RUN_ID=${RUN_ID}; expected ${STAGEIV_EXPECTED_RUN_ID}" >&2
  exit 32
fi
if [ -n "${CONFIG_JSON:-}" ] && [ "${CONFIG_JSON}" != "${STAGEIV_EXPECTED_CONFIG_JSON}" ]; then
  echo "[error] refusing stale CONFIG_JSON=${CONFIG_JSON}; expected ${STAGEIV_EXPECTED_CONFIG_JSON}" >&2
  exit 33
fi

OUTPUT_ROOT="${STAGEIV_EXPECTED_OUTPUT_ROOT}"
RUN_ID="${STAGEIV_EXPECTED_RUN_ID}"
CONFIG_JSON="${STAGEIV_EXPECTED_CONFIG_JSON}"
export OUTPUT_ROOT RUN_ID CONFIG_JSON

PYTHON_BIN="${PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}"
RUN_DIR="${OUTPUT_ROOT}/active_runs/${RUN_ID}"
OUT_DIR="${OUTPUT_ROOT}/reports/stageiv_3d_hidden_slice_audit"

if [[ -z "${REFERENCE_DATASET:-}" ]]; then
  echo "[warn] REFERENCE_DATASET is not set; hidden-slice audit will return inconclusive."
  "${PYTHON_BIN}" scripts/stageiv_3d_hidden_slice_audit.py     --run-dir "${RUN_DIR}"     --output-dir "${OUT_DIR}"     --config "${CONFIG_JSON}"
else
  "${PYTHON_BIN}" scripts/stageiv_3d_hidden_slice_audit.py     --run-dir "${RUN_DIR}"     --output-dir "${OUT_DIR}"     --config "${CONFIG_JSON}"     --reference-dataset "${REFERENCE_DATASET}"
fi
