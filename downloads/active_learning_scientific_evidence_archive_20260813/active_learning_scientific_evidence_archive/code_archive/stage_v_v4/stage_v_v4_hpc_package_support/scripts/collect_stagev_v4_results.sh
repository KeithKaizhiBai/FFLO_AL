#!/bin/bash
set -euo pipefail
OUTPUT_ROOT="ML_Phase_StageV_V4_GraphAcq"
RUN_ID="stagev_v4_phase_structure_graph_acquisition_3d_v1"
ARCHIVE="${OUTPUT_ROOT}/stagev_v4_graph_acq_results.tar.gz"
mkdir -p "${OUTPUT_ROOT}"
tar -czf "${ARCHIVE}" "${OUTPUT_ROOT}/active_runs/${RUN_ID}" "${OUTPUT_ROOT}/preflight" 2>/dev/null || true
echo "[done] archive=${ARCHIVE}"
