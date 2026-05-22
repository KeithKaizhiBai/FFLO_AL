#!/bin/bash
set -euo pipefail

LOG_FILE="${LOG_FILE:-active_loop_$(date +%Y%m%d_%H%M%S).log}"
PID_FILE="${PID_FILE:-active_loop.pid}"

if [ -f "${PID_FILE}" ]; then
  old_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [ -n "${old_pid}" ] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "[error] existing active loop appears to be running: pid=${old_pid}" >&2
    echo "[hint] use: tail -f ${LOG_FILE}" >&2
    exit 1
  fi
fi

nohup bash run_from_warmup_512.sh > "${LOG_FILE}" 2>&1 &
pid="$!"
echo "${pid}" > "${PID_FILE}"

echo "[background] started active-learning loop"
echo "[background] pid=${pid}"
echo "[background] pid file=${PID_FILE}"
echo "[background] log file=${LOG_FILE}"
echo "[background] follow log with: tail -f ${LOG_FILE}"
