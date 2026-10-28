#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="${PACKAGE_ROOT:-${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}}"

if [ -z "${RUN_ROOT:-}" ]; then
  if [ -f "${PACKAGE_ROOT}/RUN_MANIFEST.json" ]; then
    RUN_ROOT="${PACKAGE_ROOT}/local_refinement_refactor_variant_suite_run"
  elif [ -d "${PACKAGE_ROOT}/hpc_packages/local_refinement_refactor_variant_suite" ]; then
    RUN_ROOT="${PACKAGE_ROOT}/hpc_packages/local_refinement_refactor_variant_suite/local_refinement_refactor_variant_suite_run"
    PACKAGE_ROOT="${PACKAGE_ROOT}/hpc_packages/local_refinement_refactor_variant_suite"
  elif [ -n "${SCRATCH:-}" ]; then
    RUN_ROOT="${SCRATCH}/local_refinement_refactor_variant_suite_run"
  elif [ -n "${TMPDIR:-}" ]; then
    RUN_ROOT="${TMPDIR}/local_refinement_refactor_variant_suite_run"
  else
    RUN_ROOT="${HOME}/local_refinement_refactor_variant_suite_run"
  fi
fi

mkdir -p "${RUN_ROOT}"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
PACKAGE_ROOT="$(cd "${PACKAGE_ROOT}" && pwd)"
PROJECT_DIR="${PACKAGE_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PACKAGE_ROOT RUN_ROOT PROJECT_DIR

cd "${PACKAGE_ROOT}"
"${PYTHON_BIN}" scripts/build_local_refinement_performance_report.py \
  --result-root "${RUN_ROOT}" \
  --output-dir "${RUN_ROOT}/reports/local_refinement_refactor/variant_regression/performance_report" \
  --run-root "${RUN_ROOT}"

echo "variant performance report ready: ${RUN_ROOT}/reports/local_refinement_refactor/variant_regression/performance_report"
