#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
RUN_ID="${RUN_ID:-active_boundary_h100_smoke}"
CONDA_ENV="${CONDA_ENV:-}"

cd "${PROJECT_DIR}"

if [ -n "${CONDA_ENV}" ]; then
  if [ -f "/public/software/apps/anaconda/etc/profile.d/conda.sh" ]; then
    source /public/software/apps/anaconda/etc/profile.d/conda.sh
  fi
  source activate "${CONDA_ENV}"
fi

python -m ml_phase.report_builder --run-id "${RUN_ID}"

if command -v pdflatex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode -output-directory ML_Phase/reports ML_Phase/reports/active_learning_phase_boundary_report.tex
else
  echo "[warn] pdflatex not found; generated TeX only."
fi

