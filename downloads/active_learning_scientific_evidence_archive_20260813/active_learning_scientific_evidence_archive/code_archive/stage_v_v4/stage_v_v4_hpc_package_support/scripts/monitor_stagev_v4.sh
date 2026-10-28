#!/bin/bash
set -euo pipefail
OUTPUT_ROOT="ML_Phase_StageV_V4_GraphAcq"
RUN_ID="stagev_v4_phase_structure_graph_acquisition_3d_v1"
SQUEUE_BIN="${SQUEUE_BIN:-/opt/gridview/slurm-2.3.3/bin/squeue}"
RUN_DIR="${OUTPUT_ROOT}/active_runs/${RUN_ID}"
echo "run_dir=${RUN_DIR}"
if [ ! -d "${RUN_DIR}" ]; then echo status=run_dir_missing; exit 0; fi
find "${RUN_DIR}" -maxdepth 1 -name 'dataset_iter*.npz' | sed -E 's/.*dataset_iter([0-9]+)\.npz/\1/' | sort -n | tail -1 | xargs -r echo latest_dataset_iteration=
find "${RUN_DIR}" -maxdepth 1 -type d -name 'iter[0-9][0-9][0-9]' | sed -E 's/.*iter([0-9]+)/\1/' | sort -n | tail -1 | xargs -r echo latest_exact_iteration=
"${SQUEUE_BIN}" -u "$USER" || true
