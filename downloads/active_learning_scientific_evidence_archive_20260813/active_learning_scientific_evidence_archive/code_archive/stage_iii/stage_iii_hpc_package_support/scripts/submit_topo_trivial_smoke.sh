#!/bin/bash
set -euo pipefail

export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export PYTHONUTF8=1
export PYTHONIOENCODING="utf-8"

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" scripts/preflight_topo_trivial_full_loop.py
"${PYTHON_BIN}" -m py_compile ml_phase/topology_oracle.py ml_phase/exact_oracle.py ml_phase/acquisition.py ml_phase/active_refine.py ml_phase/config.py scripts/run_topology_pass_dataset_iter035.py
