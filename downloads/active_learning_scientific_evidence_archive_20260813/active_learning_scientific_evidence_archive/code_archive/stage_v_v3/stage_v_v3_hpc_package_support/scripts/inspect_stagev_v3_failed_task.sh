#!/bin/bash
set -euo pipefail
JOB="${1:?Usage: bash scripts/inspect_stagev_v3_failed_task.sh JOBID[_TASKID]}"
SACCT_BIN="${SACCT_BIN:-/opt/gridview/slurm-2.3.3/bin/sacct}"
"${SACCT_BIN}" -j "${JOB%%_*}" --format=JobID,JobName,State,ExitCode,Elapsed,Start,End,NodeList,Reason --parsable2 || true
for f in slurm-${JOB}.out slurm-${JOB%%_*}_${JOB#*_}.out slurm-select-${JOB}.out; do
  if [ -f "$f" ]; then
    echo "==== $f ===="
    tail -n 220 "$f"
  fi
done
