#!/bin/bash
set -euo pipefail

# Recover the Stage IV 3D exact-oracle iteration 000 rank 0 shard, then resume
# the full-loop controller from iteration 001.
#
# Expected use on GBU:
#   cd ~/bkz/Fu_FFLO/active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624
#   bash scripts/recover_stageiv_iter000_rank0_and_resume.sh
#
# If this file is uploaded to the package root instead of scripts/, run:
#   bash recover_stageiv_iter000_rank0_and_resume.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/recover_stageiv_failed_exact_iter.sh" ]; then
  PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
  PACKAGE_ROOT="$(pwd)"
fi
cd "${PACKAGE_ROOT}"

unset OUTPUT_ROOT RUN_ID CONFIG_JSON || true

export PYTHON_BIN="${PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}"
export EXCLUDE_NODES="${EXCLUDE_NODES:-gpuh01,gpuh14}"
export WORLD_SIZE="${WORLD_SIZE:-8}"

RUN_ID_EXPECTED="active_phase_topology_3d_t_ja_mu_from_scratch_v1"
OUTPUT_ROOT_EXPECTED="ML_Phase_StageIV_Topology3D"
RUN_DIR="${OUTPUT_ROOT_EXPECTED}/active_runs/${RUN_ID_EXPECTED}"

echo "[info] package_root=${PACKAGE_ROOT}"
echo "[info] run_dir=${RUN_DIR}"
echo "[info] PYTHON_BIN=${PYTHON_BIN}"
echo "[info] EXCLUDE_NODES=${EXCLUDE_NODES}"
echo "[info] recovering iter000 rank000 only"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "[error] PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

if [ ! -f "scripts/recover_stageiv_failed_exact_iter.sh" ]; then
  echo "[error] missing scripts/recover_stageiv_failed_exact_iter.sh" >&2
  echo "[hint] run this script from the extracted Stage IV package root." >&2
  exit 3
fi

if [ ! -d "${RUN_DIR}/iter000" ]; then
  echo "[error] missing iter000 directory: ${RUN_DIR}/iter000" >&2
  exit 4
fi

ITER=0 FAILED_RANKS=0 bash scripts/recover_stageiv_failed_exact_iter.sh

missing=""
for rank in $(seq 0 7); do
  shard="${RUN_DIR}/iter000/exact_shard_rank$(printf "%03d" "${rank}")_of008.npz"
  if [ ! -s "${shard}" ]; then
    missing="${missing} ${rank}"
  fi
done

if [ -n "${missing}" ]; then
  echo "[error] missing shards after recovery:${missing}" >&2
  exit 10
fi

if [ ! -s "${RUN_DIR}/iter000/exact_merged_iter000.npz" ]; then
  echo "[error] missing merged exact file: ${RUN_DIR}/iter000/exact_merged_iter000.npz" >&2
  exit 11
fi

if [ ! -s "${RUN_DIR}/iter000/exact_trusted_iter000.npz" ]; then
  echo "[error] missing trusted exact file: ${RUN_DIR}/iter000/exact_trusted_iter000.npz" >&2
  exit 12
fi

if [ ! -s "${RUN_DIR}/dataset_iter001.npz" ] || [ ! -s "${RUN_DIR}/dataset_iter001.csv" ]; then
  echo "[error] recovery did not produce dataset_iter001 outputs" >&2
  exit 13
fi

echo "[done] iter000 recovered; dataset_iter001 exists"

if [ ! -f "scripts/resume_stageiv_3d_full_loop.sh" ]; then
  echo "[error] missing scripts/resume_stageiv_3d_full_loop.sh" >&2
  exit 14
fi

export CONFIRM_STAGEIV_FULL_LOOP=1
resume_log="active_phase_topology_3d_t_ja_mu_from_scratch_v1_resume_iter001.nohup.log"

echo "[submit] resuming Stage IV controller from START_ITER=1"
START_ITER=1 nohup bash scripts/resume_stageiv_3d_full_loop.sh > "${resume_log}" 2>&1 &

echo "[submitted] resume_controller_pid=$!"
echo "[log] ${resume_log}"
echo "[monitor] tail -f ${resume_log}"
