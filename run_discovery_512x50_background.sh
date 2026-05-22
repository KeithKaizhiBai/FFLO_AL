#!/bin/bash
set -euo pipefail

LOG_FILE="${LOG_FILE:-discovery_active_loop_$(date +%Y%m%d_%H%M%S).log}"
PID_FILE="${PID_FILE:-discovery_active_loop.pid}"

if [ -f "${PID_FILE}" ]; then
  old_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [ -n "${old_pid}" ] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "[error] existing discovery loop appears to be running: pid=${old_pid}" >&2
    echo "[hint] use: tail -f ${LOG_FILE}" >&2
    exit 1
  fi
fi

nohup bash run_discovery_512x50.sh > "${LOG_FILE}" 2>&1 &
pid="$!"
echo "${pid}" > "${PID_FILE}"

echo "[background] started discovery active-learning loop"
echo "[background] pid=${pid}"
echo "[background] pid file=${PID_FILE}"
echo "[background] log file=${LOG_FILE}"
echo "[background] follow log with: tail -f ${LOG_FILE}"
