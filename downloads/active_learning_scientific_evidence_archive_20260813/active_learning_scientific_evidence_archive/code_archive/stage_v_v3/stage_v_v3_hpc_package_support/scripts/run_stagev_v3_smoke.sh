#!/bin/bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"
"${PYTHON_BIN}" -m py_compile ml_phase/stagev_v3.py scripts/stagev_v3_select.py scripts/stagev_v3_update_reward.py scripts/stagev_v3_smoke.py
"${PYTHON_BIN}" scripts/stagev_v3_smoke.py --output-dir reports/stagev_v3_sparse_arc_smoke
