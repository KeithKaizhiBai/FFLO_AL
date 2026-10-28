#!/bin/bash
set -euo pipefail

if [ -z "${START_ITER:-}" ]; then
  echo "[error] set START_ITER to the next exact iteration to run." >&2
  exit 2
fi

export CONFIRM_STAGEIV_FULL_LOOP="${CONFIRM_STAGEIV_FULL_LOOP:-1}"
bash scripts/submit_stageiv_3d_full_loop.sh
