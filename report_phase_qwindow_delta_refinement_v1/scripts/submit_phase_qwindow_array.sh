#!/bin/bash
#SBATCH --job-name=phase_qwin
#SBATCH --partition=NV_H100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --exclude=gpuh01

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

"${PYTHON_BIN}" "${REPORT_ROOT}/scripts/phase_qwindow_delta_refinement_audit.py" run-qwindow   --report-root "${REPORT_ROOT}"   --rank "${SLURM_ARRAY_TASK_ID}"   --world-size "${SLURM_ARRAY_TASK_COUNT}"   --device cuda:0
