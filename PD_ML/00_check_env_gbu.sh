#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
CONDA_ENV="${CONDA_ENV:-}"
CUDA_MODULE="${CUDA_MODULE:-compiler/cuda/cuda-12.8.1}"

cd "${PROJECT_DIR}"

echo "[check] project: ${PROJECT_DIR}"
echo "[check] python: $(command -v python || true)"

if command -v module >/dev/null 2>&1; then
  echo "[check] module is available"
  module av 2>&1 | head -n 40 || true
  if [ -n "${CUDA_MODULE}" ]; then
    module load "${CUDA_MODULE}" || true
  fi
else
  echo "[check] module command not found"
fi

if [ -n "${CONDA_ENV}" ]; then
  if [ -f "/public/software/apps/anaconda/etc/profile.d/conda.sh" ]; then
    source /public/software/apps/anaconda/etc/profile.d/conda.sh
  fi
  source activate "${CONDA_ENV}"
fi

python - <<'PY'
import importlib.util
mods = ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "torch"]
print("[check] python modules:", {m: bool(importlib.util.find_spec(m)) for m in mods})
import torch
print("[check] torch:", torch.__version__)
print("[check] cuda available:", torch.cuda.is_available())
print("[check] note: cuda can be False on login/CPU nodes; verify it inside an NV_H100 job.")
if torch.cuda.is_available():
    print("[check] gpu count:", torch.cuda.device_count())
    print("[check] gpu 0:", torch.cuda.get_device_name(0))
PY

if command -v sinfo >/dev/null 2>&1; then
  echo "[check] sinfo:"
  sinfo -p NV_H100,Intel,AMD,debug || true
fi
