#!/bin/bash
set -euo pipefail
OUTPUT_ROOT="ML_Phase_StageV_V3_SparseArc"
RUN_ID="stagev_v3_sparse_arc_boundary_support_3d_v1"
ARCHIVE="${OUTPUT_ROOT}/stagev_v3_sparse_arc_results.tar.gz"
mkdir -p "${OUTPUT_ROOT}"
tar -czf "${ARCHIVE}" "${OUTPUT_ROOT}/active_runs/${RUN_ID}" "${OUTPUT_ROOT}/preflight" 2>/dev/null || true
echo "[done] archive=${ARCHIVE}"
