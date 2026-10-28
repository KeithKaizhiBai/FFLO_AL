#!/bin/bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"
"${PYTHON_BIN}" -m py_compile ml_phase/stagev_v4.py scripts/stagev_v4_select.py scripts/stagev_v4_update_reward.py scripts/stagev_v4_smoke.py
"${PYTHON_BIN}" scripts/stagev_v4_smoke.py --output-dir reports/stagev_v4_graph_acq_smoke
