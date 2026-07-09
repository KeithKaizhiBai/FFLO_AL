#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PACKAGE_ROOT="${PACKAGE_ROOT:-${REPO_ROOT}/hpc_packages/local_refinement_rankcap_acceptance}"
exec bash "${PACKAGE_ROOT}/scripts/submit_rankcap_acceptance_regression.sh" "$@"
