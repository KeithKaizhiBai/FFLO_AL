#!/bin/bash
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
REPORT_ROOT="${REPORT_ROOT:-${SUBMIT_DIR}}"
if [ "$(basename "${REPORT_ROOT}")" = "scripts" ]; then
  REPORT_ROOT="$(cd "${REPORT_ROOT}/.." && pwd)"
else
  REPORT_ROOT="$(cd "${REPORT_ROOT}" && pwd)"
fi
PROJECT_DIR="${PROJECT_DIR:-$(cd "${REPORT_ROOT}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "${PROJECT_DIR}"

"${PYTHON_BIN}" "${REPORT_ROOT}/scripts/phase_qwindow_delta_refinement_audit.py" collect --report-root "${REPORT_ROOT}"
