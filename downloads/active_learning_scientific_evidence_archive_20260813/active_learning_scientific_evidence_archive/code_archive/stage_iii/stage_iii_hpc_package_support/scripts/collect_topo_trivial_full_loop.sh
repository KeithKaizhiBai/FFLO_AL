#!/bin/bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-ML_Phase_512_TopoTrivial_FullLoop}"
RUN_ID="${RUN_ID:-active_phase_topology_from_scratch_full_loop_v1}"
ARCHIVE="${OUTPUT_ROOT}/active_phase_topology_from_scratch_full_loop_v1_results.tar.gz"

mkdir -p "${OUTPUT_ROOT}"
tar -czf "${ARCHIVE}" \
  "${OUTPUT_ROOT}/active_runs/${RUN_ID}" \
  "${OUTPUT_ROOT}/figures" \
  "${OUTPUT_ROOT}/reports" \
  2>/dev/null || tar -czf "${ARCHIVE}" "${OUTPUT_ROOT}/active_runs/${RUN_ID}"

echo "[done] archive=${ARCHIVE}"
