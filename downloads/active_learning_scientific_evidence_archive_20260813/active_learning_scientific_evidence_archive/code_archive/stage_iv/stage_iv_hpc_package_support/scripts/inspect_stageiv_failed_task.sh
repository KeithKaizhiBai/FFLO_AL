#!/bin/bash
set -euo pipefail

JOB_ID="${1:-${JOB_ID:-}}"
if [ -z "${JOB_ID}" ]; then
  echo "[error] usage: bash scripts/inspect_stageiv_failed_task.sh <job_id>" >&2
  exit 2
fi

sacct -j "${JOB_ID}" --format=JobID,JobName,State,ExitCode,Elapsed,Start,End,NodeList,Reason --parsable2 || true
for f in slurm-"${JOB_ID}"*.out; do
  [ -e "$f" ] || continue
  echo "===== $f ====="
  tail -n 160 "$f" || true
done
