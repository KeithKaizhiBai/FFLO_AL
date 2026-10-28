#!/bin/bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-ML_Phase_512_TopoTrivial_FullLoop}"
RUN_ID="${RUN_ID:-active_phase_topology_from_scratch_full_loop_v1}"
RUN_DIR="${OUTPUT_ROOT}/active_runs/${RUN_ID}"

echo "[status] run_dir=${RUN_DIR}"
if command -v squeue >/dev/null 2>&1; then
  squeue -u "${USER}" || true
fi
if [ -d "${RUN_DIR}" ]; then
  find "${RUN_DIR}" -maxdepth 2 -type f \( -name 'dataset_iter*.npz' -o -name 'stop_metrics_iter*.json' -o -name 'exact_trusted_iter*.npz' \) | sort
else
  echo "[status] run directory not found"
fi
