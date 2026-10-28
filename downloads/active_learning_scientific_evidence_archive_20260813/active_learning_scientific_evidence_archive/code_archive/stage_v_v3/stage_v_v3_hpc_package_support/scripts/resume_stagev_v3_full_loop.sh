#!/bin/bash
set -euo pipefail
START_ITER="${START_ITER:?Set START_ITER first.}"
export START_ITER
exec bash scripts/submit_stagev_v3_full_loop.sh
