from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_phase.stageiv_3d import STAGEIV_OUTPUT_ROOT, STAGEIV_RUN_ID, StageIV3DConfig


PACKAGE_NAME = f"{STAGEIV_RUN_ID}_identity_guard_hpc_20260624"
PACKAGE_ROOT = ROOT / "hpc_packages" / PACKAGE_NAME
ARCHIVE_PATH = ROOT / "hpc_packages" / f"{PACKAGE_NAME}.tar.gz"


def run(cmd: list[str], cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.replace("\r\n", "\n").replace("\r", "\n"), encoding="utf-8", newline="\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_snapshot() -> dict[str, Any]:
    commit = run(["git", "rev-parse", "HEAD"])
    status = run(["git", "status", "--short"])
    diff = run(["git", "diff", "--stat"])
    return {
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else "unknown",
        "git_status_short": status.stdout,
        "git_diff_stat": diff.stdout,
        "working_tree_has_changes": bool(status.stdout.strip()),
    }


def ignore_common(_dir: str, names: list[str]) -> set[str]:
    blocked = {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "hpc_packages",
        "output",
        "outputs",
        "active_runs",
        "datasets",
        "figures",
        "reports",
    }
    out: set[str] = set()
    for name in names:
        lower = name.lower()
        if name in blocked or lower.endswith((".pyc", ".pyo", ".tmp", ".bak")):
            out.add(name)
        if lower.startswith("slurm-") and lower.endswith(".out"):
            out.add(name)
    return out


def copy_tree(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copytree(src, dst, ignore=ignore_common)


def copy_file(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def normalize_shell(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(root.rglob("*.sh")):
        data = path.read_bytes()
        notes: list[str] = []
        status = "pass"
        if data.startswith(b"\xef\xbb\xbf"):
            data = data[3:]
            notes.append("removed_bom")
        if b"\r\n" in data or b"\r" in data:
            data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            notes.append("normalized_lf")
        try:
            data.decode("ascii")
        except UnicodeDecodeError:
            status = "fail"
            notes.append("non_ascii")
        path.write_bytes(data)
        rows.append({"path": path.relative_to(root).as_posix(), "status": status, "notes": ";".join(notes)})
    return rows


def stageiv_identity_guard_shell() -> str:
    return f"""STAGEIV_EXPECTED_OUTPUT_ROOT="{STAGEIV_OUTPUT_ROOT}"
STAGEIV_EXPECTED_RUN_ID="{STAGEIV_RUN_ID}"
STAGEIV_EXPECTED_CONFIG_JSON="configs/stageiv_3d_production.json"

if [ -n "${{OUTPUT_ROOT:-}}" ] && [ "${{OUTPUT_ROOT}}" != "${{STAGEIV_EXPECTED_OUTPUT_ROOT}}" ]; then
  echo "[error] refusing stale OUTPUT_ROOT=${{OUTPUT_ROOT}}; expected ${{STAGEIV_EXPECTED_OUTPUT_ROOT}}" >&2
  exit 31
fi
if [ -n "${{RUN_ID:-}}" ] && [ "${{RUN_ID}}" != "${{STAGEIV_EXPECTED_RUN_ID}}" ]; then
  echo "[error] refusing stale RUN_ID=${{RUN_ID}}; expected ${{STAGEIV_EXPECTED_RUN_ID}}" >&2
  exit 32
fi
if [ -n "${{CONFIG_JSON:-}}" ] && [ "${{CONFIG_JSON}}" != "${{STAGEIV_EXPECTED_CONFIG_JSON}}" ]; then
  echo "[error] refusing stale CONFIG_JSON=${{CONFIG_JSON}}; expected ${{STAGEIV_EXPECTED_CONFIG_JSON}}" >&2
  exit 33
fi

OUTPUT_ROOT="${{STAGEIV_EXPECTED_OUTPUT_ROOT}}"
RUN_ID="${{STAGEIV_EXPECTED_RUN_ID}}"
CONFIG_JSON="${{STAGEIV_EXPECTED_CONFIG_JSON}}"
export OUTPUT_ROOT RUN_ID CONFIG_JSON
"""


def freeze_stageiv_shell_identity(root: Path) -> list[dict[str, Any]]:
    """Freeze generated shell scripts to the Stage IV 3D run identity.

    The previous package allowed ambient OUTPUT_ROOT/RUN_ID variables to override
    the 3D defaults.  That made it possible to accidentally resume an older 2D
    topology run from a contaminated login-shell environment.  Production
    identity is now fixed by the package; a conflicting external value is a hard
    error instead of a silent fallback.
    """

    assignment_lines = [
        f'OUTPUT_ROOT="${{OUTPUT_ROOT:-{STAGEIV_OUTPUT_ROOT}}}"',
        f'RUN_ID="${{RUN_ID:-{STAGEIV_RUN_ID}}}"',
        'CONFIG_JSON="${CONFIG_JSON:-configs/stageiv_3d_production.json}"',
    ]
    guard = stageiv_identity_guard_shell().rstrip() + "\n\n"
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.sh")):
        text = path.read_text(encoding="utf-8")
        original_text = text
        removed: list[str] = []
        for line in assignment_lines:
            if line in text:
                text = text.replace(line + "\n", "")
                removed.append(line.split("=", 1)[0])
        if removed and "STAGEIV_EXPECTED_OUTPUT_ROOT" not in text:
            marker = "set -euo pipefail\n\n"
            if marker in text:
                text = text.replace(marker, marker + guard, 1)
            else:
                text = guard + text
        if text != original_text:
            write_text(path, text)
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "identity_guard_inserted": bool(removed),
                "removed_assignments": ",".join(removed),
            }
        )
    return rows


def scan_stale_2d_markers(root: Path) -> list[dict[str, str]]:
    markers = [
        "ML_Phase_512_TopoTrivial_FullLoop",
        "active_phase_topology_from_scratch_full_loop_v2",
    ]
    text_suffixes = {
        ".cfg",
        ".csv",
        ".json",
        ".md",
        ".py",
        ".sh",
        ".tex",
        ".txt",
        ".yaml",
        ".yml",
    }
    findings: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for marker in markers:
            if marker in text:
                findings.append({"path": path.relative_to(root).as_posix(), "marker": marker})
    return findings


def slurm_script() -> str:
    return """#!/bin/bash
#SBATCH --job-name=stageiv3d_exact
#SBATCH --partition=NV_H100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --exclude=gpuh01,gpuh14
#SBATCH --output=slurm-%A_%a.out

set -euo pipefail

export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export PYTHONUTF8=1
export PYTHONIOENCODING="utf-8"

host_name="$(hostname 2>/dev/null || true)"
if [ "${host_name%%.*}" = "gpuh01" ] || [ "${host_name%%.*}" = "gpuh14" ]; then
  echo "[error] refusing to run on excluded node ${host_name%%.*}" >&2
  exit 42
fi

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-ML_Phase_StageIV_Topology3D}"
RUN_ID="${RUN_ID:-active_phase_topology_3d_t_ja_mu_from_scratch_v1}"
ITER="${ITER:?ITER must be set}"
WORLD_SIZE="${WORLD_SIZE:-8}"
RANK="${SLURM_ARRAY_TASK_ID:-0}"
RUN_DIR="${OUTPUT_ROOT}/active_runs/${RUN_ID}"

cd "${PROJECT_DIR}"

echo "run_id=${RUN_ID}"
echo "iteration=${ITER}"
echo "rank=${RANK}"
echo "world_size=${WORLD_SIZE}"
echo "hostname=${host_name}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true
"${PYTHON_BIN}" - <<'PY'
import torch
print("torch.__version__=", torch.__version__)
print("torch.version.cuda=", torch.version.cuda)
print("torch.cuda.is_available()=", torch.cuda.is_available())
PY

"${PYTHON_BIN}" -m ml_phase.exact_oracle \
  --active-root "${OUTPUT_ROOT}/active_runs" \
  --run-id "${RUN_ID}" \
  --iteration "${ITER}" \
  --rank "${RANK}" \
  --world-size "${WORLD_SIZE}" \
  --device cuda:0 \
  --save-every 1 \
  --oracle-mode robust_incremental \
  --enable-q-expansion \
  --enable-incremental-q-expansion \
  --enable-delta-refinement \
  --allow-ambiguous-output \
  --enable-local-box-instrumentation \
  --enable-basin-clustering \
  --enable-selective-refinement \
  --max-refined-minima 3 \
  --max-optional-refined-basins 3 \
  --no-mandatory-basins-can-exceed-cap \
  --high-risk-overflow-policy rank_and_cap \
  --max-edge-risk-basins 1 \
  --max-delta-near-eps-basins 2 \
  --max-near-degenerate-basins 2 \
  --enable-topology-classification \
  --topology-gap-nk "${TOPOLOGY_GAP_NK:-2048}" \
  --topology-gap-backend "${TOPOLOGY_GAP_BACKEND:-gpu}" \
  --topology-gap-tol-rel "${TOPOLOGY_GAP_TOL_REL:-1e-8}" \
  --topology-gap-tol-abs "${TOPOLOGY_GAP_TOL_ABS:-0.0}" \
  --topology-gap-k-chunk "${TOPOLOGY_GAP_K_CHUNK:-512}"
"""


def submit_script() -> str:
    return """#!/bin/bash
set -euo pipefail

export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export PYTHONUTF8=1
export PYTHONIOENCODING="utf-8"

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-ML_Phase_StageIV_Topology3D}"
RUN_ID="${RUN_ID:-active_phase_topology_3d_t_ja_mu_from_scratch_v1}"
CONFIG_JSON="${CONFIG_JSON:-configs/stageiv_3d_production.json}"
RUN_DIR="${OUTPUT_ROOT}/active_runs/${RUN_ID}"
WORLD_SIZE="${WORLD_SIZE:-8}"
START_ITER="${START_ITER:-0}"
MAX_ACQUISITION_BATCHES="${MAX_ACQUISITION_BATCHES:-24}"
FINAL_EXACT_ITER="${FINAL_EXACT_ITER:-$MAX_ACQUISITION_BATCHES}"
PARTITION_STRATEGY="${PARTITION_STRATEGY:-cost_aware}"
SELECT_DEVICE="${SELECT_DEVICE:-cpu}"
EXCLUDE_NODES="${EXCLUDE_NODES:-gpuh01,gpuh14}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"
POSITIVE_DELTA_GAP_TOL="${POSITIVE_DELTA_GAP_TOL:-1e-8}"

if [ "${CONFIRM_STAGEIV_FULL_LOOP:-0}" != "1" ]; then
  echo "[error] set CONFIRM_STAGEIV_FULL_LOOP=1 to submit the production Stage IV full loop." >&2
  exit 2
fi

cd "${PROJECT_DIR}"

if [ "${START_ITER}" = "0" ] && [ -e "${RUN_DIR}" ]; then
  echo "[error] run directory already exists: ${RUN_DIR}" >&2
  echo "Set START_ITER>0 only if you are intentionally resuming a completed prefix." >&2
  exit 4
fi

mkdir -p "${OUTPUT_ROOT}/logs"
"${PYTHON_BIN}" scripts/stageiv_3d_preflight.py \
  --config "${CONFIG_JSON}" \
  --output-dir "${OUTPUT_ROOT}/preflight" \
  --n-ja "${PREFLIGHT_N_JA:-48}" \
  --n-mu "${PREFLIGHT_N_MU:-64}" \
  --n-k "${PREFLIGHT_N_K:-1024}"

wait_for_job() {
  local job_id="$1"
  local iter_tag="$2"
  while true; do
    active="$(squeue -h -j "${job_id}" -o "%i|%T" 2>/dev/null | paste -sd, - || true)"
    if [ -z "${active}" ]; then
      break
    fi
    echo "[wait] Stage IV exact iter ${iter_tag}: squeue active: ${active}"
    sleep "${SLEEP_SECONDS}"
  done
  failures="$(sacct -j "${job_id}" --format=JobID,State,ExitCode --parsable2 | awk -F'|' 'NR>1 && $1 !~ /\\.batch|\\.extern/ && $2 !~ /COMPLETED/ {print $1\":\"$2\":\"$3}' | paste -sd, -)"
  if [ -n "${failures}" ]; then
    echo "[error] Stage IV exact iter ${iter_tag}: job ${job_id} failed: ${failures}" >&2
    exit 5
  fi
}

for ITER in $(seq "${START_ITER}" "${FINAL_EXACT_ITER}"); do
  TAG="$(printf "%03d" "${ITER}")"
  NEXT="$(printf "%03d" "$((ITER + 1))")"
  if [ "${ITER}" = "0" ]; then
    "${PYTHON_BIN}" scripts/stageiv_3d_select.py \
      --config "${CONFIG_JSON}" \
      --mode seed \
      --iteration "${ITER}" \
      --output-root "${OUTPUT_ROOT}" \
      --run-id "${RUN_ID}" \
      --world-size "${WORLD_SIZE}" \
      --partition-strategy "${PARTITION_STRATEGY}"
  else
    "${PYTHON_BIN}" scripts/stageiv_3d_select.py \
      --config "${CONFIG_JSON}" \
      --mode acquisition \
      --iteration "${ITER}" \
      --output-root "${OUTPUT_ROOT}" \
      --run-id "${RUN_ID}" \
      --dataset "${RUN_DIR}/dataset_iter${TAG}.npz" \
      --world-size "${WORLD_SIZE}" \
      --partition-strategy "${PARTITION_STRATEGY}" \
      --device "${SELECT_DEVICE}"
  fi

  export PROJECT_DIR PYTHON_BIN OUTPUT_ROOT RUN_ID ITER WORLD_SIZE
  JOB_ID="$(sbatch --parsable --exclude="${EXCLUDE_NODES}" --array=0-$((WORLD_SIZE - 1)) scripts/slurm_stageiv_exact_array.sh)"
  echo "[submit] Stage IV exact iter ${TAG}: job ${JOB_ID}"
  wait_for_job "${JOB_ID}" "${TAG}"

  "${PYTHON_BIN}" -m ml_phase.hpc \
    --merge \
    --run-dir "${RUN_DIR}" \
    --iteration "${ITER}" \
    --world-size "${WORLD_SIZE}" \
    --positive-delta-gap-tol "${POSITIVE_DELTA_GAP_TOL}"

  "${PYTHON_BIN}" -m ml_phase.append_trusted \
    --dataset "${RUN_DIR}/dataset_iter${TAG}.npz" \
    --trusted-exact "${RUN_DIR}/iter${TAG}/exact_trusted_iter${TAG}.npz" \
    --output-npz "${RUN_DIR}/dataset_iter${NEXT}.npz" \
    --output-csv "${RUN_DIR}/dataset_iter${NEXT}.csv" \
    --output-root "${OUTPUT_ROOT}"

  echo "[done] Stage IV iter ${TAG}; wrote dataset_iter${NEXT}"
done

echo "[done] Stage IV full loop completed through exact iter $(printf "%03d" "${FINAL_EXACT_ITER}")"
"""


def collect_script() -> str:
    return """#!/bin/bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-ML_Phase_StageIV_Topology3D}"
RUN_ID="${RUN_ID:-active_phase_topology_3d_t_ja_mu_from_scratch_v1}"
ARCHIVE="${OUTPUT_ROOT}/stageiv_3d_topology_full_loop_results.tar.gz"

tar -czf "${ARCHIVE}" \
  "${OUTPUT_ROOT}/active_runs/${RUN_ID}" \
  "${OUTPUT_ROOT}/preflight" 2>/dev/null || tar -czf "${ARCHIVE}" "${OUTPUT_ROOT}"

echo "[done] archive=${ARCHIVE}"
"""


def preflight_script() -> str:
    return """#!/bin/bash
set -euo pipefail

export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export PYTHONUTF8=1
export PYTHONIOENCODING="utf-8"

PYTHON_BIN="${PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}"
CONFIG_JSON="${CONFIG_JSON:-configs/stageiv_3d_production.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-ML_Phase_StageIV_Topology3D}"

"${PYTHON_BIN}" scripts/stageiv_3d_preflight.py \
  --config "${CONFIG_JSON}" \
  --output-dir "${OUTPUT_ROOT}/preflight" \
  --n-ja "${PREFLIGHT_N_JA:-96}" \
  --n-mu "${PREFLIGHT_N_MU:-128}" \
  --n-k "${PREFLIGHT_N_K:-2048}"
"""


def submit_ready_script() -> str:
    return """#!/bin/bash
set -euo pipefail

export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export PYTHONUTF8=1
export PYTHONIOENCODING="utf-8"

PYTHON_BIN="${PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-ML_Phase_StageIV_Topology3D}"
CONFIG_JSON="${CONFIG_JSON:-configs/stageiv_3d_production.json}"
OUT_DIR="${OUTPUT_ROOT}/reports/stageiv_3d_submit_check"

"${PYTHON_BIN}" scripts/stageiv_3d_submit_check.py \
  --root . \
  --config "${CONFIG_JSON}" \
  --output-dir "${OUT_DIR}" \
  --python-bin "${PYTHON_BIN}"
"""


def resume_script() -> str:
    return """#!/bin/bash
set -euo pipefail

if [ -z "${START_ITER:-}" ]; then
  echo "[error] set START_ITER to the next exact iteration to run." >&2
  exit 2
fi

export CONFIRM_STAGEIV_FULL_LOOP="${CONFIRM_STAGEIV_FULL_LOOP:-1}"
bash scripts/submit_stageiv_3d_full_loop.sh
"""


def failed_rank_recovery_script() -> str:
    return """#!/bin/bash
set -euo pipefail

export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export PYTHONUTF8=1
export PYTHONIOENCODING="utf-8"

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-ML_Phase_StageIV_Topology3D}"
RUN_ID="${RUN_ID:-active_phase_topology_3d_t_ja_mu_from_scratch_v1}"
RUN_DIR="${OUTPUT_ROOT}/active_runs/${RUN_ID}"
WORLD_SIZE="${WORLD_SIZE:-8}"
EXCLUDE_NODES="${EXCLUDE_NODES:-gpuh01,gpuh14}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"
POSITIVE_DELTA_GAP_TOL="${POSITIVE_DELTA_GAP_TOL:-1e-8}"

if [ -z "${ITER:-}" ]; then
  echo "[error] set ITER to the failed exact iteration, for example ITER=0." >&2
  exit 2
fi
if [ -z "${FAILED_RANKS:-}" ]; then
  echo "[error] set FAILED_RANKS to a Slurm array list, for example FAILED_RANKS=3 or FAILED_RANKS=1,4." >&2
  exit 2
fi

cd "${PROJECT_DIR}"

TAG="$(printf "%03d" "${ITER}")"
NEXT="$(printf "%03d" "$((ITER + 1))")"

echo "[recover] iteration=${TAG}"
echo "[recover] failed_ranks=${FAILED_RANKS}"
echo "[recover] run_dir=${RUN_DIR}"

if [ ! -d "${RUN_DIR}/iter${TAG}" ]; then
  echo "[error] missing iteration directory: ${RUN_DIR}/iter${TAG}" >&2
  exit 3
fi

export PROJECT_DIR PYTHON_BIN OUTPUT_ROOT RUN_ID ITER WORLD_SIZE
JOB_ID="$(sbatch --parsable --exclude="${EXCLUDE_NODES}" --array="${FAILED_RANKS}" scripts/slurm_stageiv_exact_array.sh)"
echo "[submit] Stage IV recovery exact iter ${TAG}: job ${JOB_ID}"

while true; do
  active="$(squeue -h -j "${JOB_ID}" -o "%i|%T" 2>/dev/null | paste -sd, - || true)"
  if [ -z "${active}" ]; then
    break
  fi
  echo "[wait] Stage IV recovery iter ${TAG}: squeue active: ${active}"
  sleep "${SLEEP_SECONDS}"
done

failures="$(sacct -j "${JOB_ID}" --format=JobID,State,ExitCode --parsable2 | awk -F'|' 'NR>1 && $1 !~ /\\.batch|\\.extern/ && $2 !~ /COMPLETED/ {print $1\":\"$2\":\"$3}' | paste -sd, -)"
if [ -n "${failures}" ]; then
  echo "[error] Stage IV recovery iter ${TAG}: job ${JOB_ID} failed: ${failures}" >&2
  exit 5
fi

missing=""
for rank in $(seq 0 "$((WORLD_SIZE - 1))"); do
  shard="${RUN_DIR}/iter${TAG}/exact_shard_rank$(printf "%03d" "${rank}")_of$(printf "%03d" "${WORLD_SIZE}").npz"
  if [ ! -s "${shard}" ]; then
    missing="${missing} ${rank}"
  fi
done
if [ -n "${missing}" ]; then
  echo "[error] still missing exact shard ranks:${missing}" >&2
  exit 6
fi

"${PYTHON_BIN}" -m ml_phase.hpc \
  --merge \
  --run-dir "${RUN_DIR}" \
  --iteration "${ITER}" \
  --world-size "${WORLD_SIZE}" \
  --positive-delta-gap-tol "${POSITIVE_DELTA_GAP_TOL}"

"${PYTHON_BIN}" -m ml_phase.append_trusted \
  --dataset "${RUN_DIR}/dataset_iter${TAG}.npz" \
  --trusted-exact "${RUN_DIR}/iter${TAG}/exact_trusted_iter${TAG}.npz" \
  --output-npz "${RUN_DIR}/dataset_iter${NEXT}.npz" \
  --output-csv "${RUN_DIR}/dataset_iter${NEXT}.csv" \
  --output-root "${OUTPUT_ROOT}"

echo "[done] recovered Stage IV iter ${TAG}; wrote dataset_iter${NEXT}"
echo "[next] continue with: START_ITER=$((ITER + 1)) bash scripts/resume_stageiv_3d_full_loop.sh"
"""


def monitor_script() -> str:
    return """#!/bin/bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-ML_Phase_StageIV_Topology3D}"
RUN_ID="${RUN_ID:-active_phase_topology_3d_t_ja_mu_from_scratch_v1}"
RUN_DIR="${OUTPUT_ROOT}/active_runs/${RUN_ID}"

echo "[monitor] run_dir=${RUN_DIR}"
squeue -u "${USER}" || true
find "${RUN_DIR}" -maxdepth 2 -type f \\( -name 'dataset_iter*.npz' -o -name 'stageiv_selection_summary.json' -o -name 'exact_merged_iter*.npz' \\) | sort | tail -n 40 || true
"""


def hpc_status_script() -> str:
    return """#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-ML_Phase_StageIV_Topology3D}"
RUN_ID="${RUN_ID:-active_phase_topology_3d_t_ja_mu_from_scratch_v1}"
CONFIG_JSON="${CONFIG_JSON:-configs/stageiv_3d_production.json}"
OUT_DIR="${OUTPUT_ROOT}/reports/stageiv_3d_hpc_status"

CMD=(
  "${PYTHON_BIN}" scripts/stageiv_3d_hpc_status.py
  --output-root "${OUTPUT_ROOT}"
  --run-id "${RUN_ID}"
  --config "${CONFIG_JSON}"
  --output-dir "${OUT_DIR}"
  --world-size "${WORLD_SIZE:-8}"
)

if [[ -n "${JOB_ID:-}" ]]; then
  CMD+=(--job-id "${JOB_ID}")
fi

"${CMD[@]}"
"""


def return_check_script() -> str:
    return """#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-ML_Phase_StageIV_Topology3D}"
RUN_ID="${RUN_ID:-active_phase_topology_3d_t_ja_mu_from_scratch_v1}"
CONFIG_JSON="${CONFIG_JSON:-configs/stageiv_3d_production.json}"
RETURN_PATH="${RETURN_PATH:-${OUTPUT_ROOT}}"
OUT_DIR="${OUT_DIR:-${OUTPUT_ROOT}/reports/stageiv_3d_return_check}"

"${PYTHON_BIN}" scripts/stageiv_3d_return_check.py \
  --return-path "${RETURN_PATH}" \
  --run-id "${RUN_ID}" \
  --config "${CONFIG_JSON}" \
  --output-dir "${OUT_DIR}" \
  --world-size "${WORLD_SIZE:-8}"
"""


def checkpoint_script() -> str:
    return """#!/bin/bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-ML_Phase_StageIV_Topology3D}"
RUN_ID="${RUN_ID:-active_phase_topology_3d_t_ja_mu_from_scratch_v1}"
RUN_DIR="${OUTPUT_ROOT}/active_runs/${RUN_ID}"

echo "[checkpoint] datasets:"
ls -lh "${RUN_DIR}"/dataset_iter*.npz 2>/dev/null || true
echo "[checkpoint] iteration directories:"
find "${RUN_DIR}" -maxdepth 1 -type d -name 'iter*' | sort || true
"""


def failed_inspect_script() -> str:
    return """#!/bin/bash
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
"""


def postrun_report_script() -> str:
    return """#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-ML_Phase_StageIV_Topology3D}"
RUN_ID="${RUN_ID:-active_phase_topology_3d_t_ja_mu_from_scratch_v1}"
RUN_DIR="${OUTPUT_ROOT}/active_runs/${RUN_ID}"
OUT_DIR="${OUTPUT_ROOT}/reports/stageiv_3d_postrun_summary"

"${PYTHON_BIN}" scripts/stageiv_3d_postrun_report.py \
  --run-dir "${RUN_DIR}" \
  --output-dir "${OUT_DIR}"
"""


def convergence_audit_script() -> str:
    return """#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-ML_Phase_StageIV_Topology3D}"
RUN_ID="${RUN_ID:-active_phase_topology_3d_t_ja_mu_from_scratch_v1}"
RUN_DIR="${OUTPUT_ROOT}/active_runs/${RUN_ID}"
CONFIG_JSON="${CONFIG_JSON:-configs/stageiv_3d_production.json}"
OUT_DIR="${OUTPUT_ROOT}/reports/stageiv_3d_convergence_audit"

"${PYTHON_BIN}" scripts/stageiv_3d_convergence_audit.py \
  --run-dir "${RUN_DIR}" \
  --output-dir "${OUT_DIR}" \
  --config "${CONFIG_JSON}" \
  --audit-cloud-size "${AUDIT_CLOUD_SIZE:-20000}" \
  --support-radius "${AUDIT_SUPPORT_RADIUS:-0.075}" \
  --neighbor-k "${AUDIT_NEIGHBOR_K:-8}" \
  --surface-max-distance "${AUDIT_SURFACE_MAX_DISTANCE:-0.18}" \
  --component-radius "${AUDIT_COMPONENT_RADIUS:-0.035}" \
  --component-min-size "${AUDIT_COMPONENT_MIN_SIZE:-12}"
"""


def hidden_slice_audit_script() -> str:
    return """#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-ML_Phase_StageIV_Topology3D}"
RUN_ID="${RUN_ID:-active_phase_topology_3d_t_ja_mu_from_scratch_v1}"
RUN_DIR="${OUTPUT_ROOT}/active_runs/${RUN_ID}"
CONFIG_JSON="${CONFIG_JSON:-configs/stageiv_3d_production.json}"
OUT_DIR="${OUTPUT_ROOT}/reports/stageiv_3d_hidden_slice_audit"

if [[ -z "${REFERENCE_DATASET:-}" ]]; then
  echo "[warn] REFERENCE_DATASET is not set; hidden-slice audit will return inconclusive."
  "${PYTHON_BIN}" scripts/stageiv_3d_hidden_slice_audit.py \
    --run-dir "${RUN_DIR}" \
    --output-dir "${OUT_DIR}" \
    --config "${CONFIG_JSON}"
else
  "${PYTHON_BIN}" scripts/stageiv_3d_hidden_slice_audit.py \
    --run-dir "${RUN_DIR}" \
    --output-dir "${OUT_DIR}" \
    --config "${CONFIG_JSON}" \
    --reference-dataset "${REFERENCE_DATASET}"
fi
"""


def all_postrun_reports_script() -> str:
    return """#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-ML_Phase_StageIV_Topology3D}"
RUN_ID="${RUN_ID:-active_phase_topology_3d_t_ja_mu_from_scratch_v1}"
RUN_DIR="${OUTPUT_ROOT}/active_runs/${RUN_ID}"
CONFIG_JSON="${CONFIG_JSON:-configs/stageiv_3d_production.json}"
OUT_DIR="${OUTPUT_ROOT}/reports/stageiv_3d_postrun_bundle"

CMD=(
  "${PYTHON_BIN}" scripts/stageiv_3d_postrun_bundle.py
  --run-dir "${RUN_DIR}"
  --output-dir "${OUT_DIR}"
  --config "${CONFIG_JSON}"
  --audit-cloud-size "${AUDIT_CLOUD_SIZE:-20000}"
  --support-radius "${AUDIT_SUPPORT_RADIUS:-0.075}"
  --neighbor-k "${AUDIT_NEIGHBOR_K:-8}"
  --surface-max-distance "${AUDIT_SURFACE_MAX_DISTANCE:-0.18}"
  --component-radius "${AUDIT_COMPONENT_RADIUS:-0.035}"
  --component-min-size "${AUDIT_COMPONENT_MIN_SIZE:-12}"
  --hidden-grid-n "${HIDDEN_GRID_N:-201}"
  --hidden-knn-k "${HIDDEN_KNN_K:-8}"
  --hidden-support-radius "${HIDDEN_SUPPORT_RADIUS:-0.075}"
  --hidden-last-n "${HIDDEN_LAST_N:-5}"
  --hidden-component-min-size "${HIDDEN_COMPONENT_MIN_SIZE:-24}"
)

if [[ -n "${REFERENCE_DATASET:-}" ]]; then
  CMD+=(--reference-dataset "${REFERENCE_DATASET}")
else
  echo "[warn] REFERENCE_DATASET is not set; bundle decision will remain hidden-slice inconclusive."
fi

"${CMD[@]}"
"""


def environment_text() -> str:
    return """# Stage IV 3D HPC Environment

This package is intended for the GBU Slurm cluster H100 partition.

## Required Runtime

```text
partition: NV_H100
excluded nodes: gpuh01,gpuh14
python: /public_hw/home/sci_bfu/.conda/envs/my_env/bin/python
cuda: use the CUDA runtime visible inside the allocated Slurm job
```

The login node may report `torch.cuda.is_available() = False`; CUDA must be
checked inside an allocated `NV_H100` job.  The Slurm exact-array script prints
`hostname`, `CUDA_VISIBLE_DEVICES`, `nvidia-smi`, `torch.__version__`,
`torch.version.cuda`, and `torch.cuda.is_available()` before running exact
points.

## Required Environment Variables

```bash
export CONFIRM_STAGEIV_FULL_LOOP=1
export PYTHON_BIN=/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python
export EXCLUDE_NODES=gpuh01,gpuh14
```

Optional post-run validation variable:

```bash
export REFERENCE_DATASET=/path/to/stageiii_frozen_reference.npz
```

`REFERENCE_DATASET` is used only by the hidden fixed-mu validation audit.  It
must not be copied into the Stage IV training data or used to seed the cold
start run.

## Encoding Policy

Shell scripts are ASCII-safe, LF-normalized, and have no UTF-8 BOM.  Python,
JSON, YAML, and Markdown files are UTF-8.  The local Windows build cannot run
`bash -n` because WSL has no installed distribution; run `bash -n scripts/*.sh`
on the Linux login node after extraction if an additional shell syntax check is
desired.
"""


def package_project_summary_text() -> str:
    return f"""# Project Summary

## 1. Project Goal

This self-contained HPC package runs the Stage IV cold-start active-learning
calculation for the three-dimensional parameter space `(kBT/t, J_A/t, mu/t)`.
It is intended to generate a fresh 3D thermodynamic and topology-aware dataset,
not to continue any previous 2D active-learning run.

## 2. Current Architecture

- `ml_phase/` contains the copied production code needed by the exact oracle,
  active-learning selection, topology diagnostics, merge, append, and report
  utilities.
- `scripts/stageiv_3d_select.py` creates the seed and acquisition batches.
- `scripts/slurm_stageiv_exact_array.sh` evaluates exact oracle shards on the
  GBU `NV_H100` partition.
- `scripts/submit_stageiv_3d_full_loop.sh` orchestrates selection, Slurm exact
  shards, merge, and trusted append for the full Stage IV run.
- `configs/stageiv_3d_production.json` is the frozen production configuration.

## 3. Important Files

- `RUN_MANIFEST.json`: package provenance, validation results, and hashes.
- `HPC_COMMANDS_STAGEIV_3D.md`: upload and Slurm command checklist.
- `ENVIRONMENT_STAGEIV_3D_HPC.md`: cluster runtime and encoding notes.
- `configs/stageiv_3d_production.json`: expected `run_id`, `output_root`, and
  Stage IV acquisition settings.

## 4. Major Decisions

- Date: 2026-06-24
- Decision: Freeze the package run identity to `{STAGEIV_RUN_ID}` under
  `{STAGEIV_OUTPUT_ROOT}`.
- Reason: A previous HPC shell environment allowed stale 2D `OUTPUT_ROOT` or
  `RUN_ID` variables to redirect a Stage IV submission into an older run.
- Consequences: Generated shell scripts now reject conflicting `OUTPUT_ROOT`,
  `RUN_ID`, or `CONFIG_JSON` values instead of silently using them.

## 5. Completed Milestones

- Date: 2026-06-24
- Milestone: Built a self-contained Stage IV 3D HPC package with identity
  guards, LF-normalized shell scripts, `gpuh01,gpuh14` exclusion, preflight smoke
  checks, status checks, return checks, failed-rank recovery, and post-run
  report-only audit scripts.
- Evidence: `RUN_MANIFEST.json` and package validation fields.

## 6. Recent Major Changes

- Date: 2026-06-24
- Files changed: generated package shell scripts and package-local docs.
- Summary: The package refuses stale 2D run identity variables and avoids
  copying local historical project-memory text into the upload bundle.
- Why it matters: The uploaded package can be run from a contaminated login
  shell without accidentally writing to a prior 2D output namespace.

## 7. Validation and Tests

- Python package compile check: `python -m py_compile`.
- Package preflight smoke check: `scripts/stageiv_3d_preflight.py` with a small
  grid.
- Submit-ready smoke check: `scripts/stageiv_3d_submit_check.py`.
- Status and return-bundle smoke checks are report-only and do not launch exact
  calculations.
- Shell scripts are LF-normalized and ASCII-safe.

## 8. Open Problems

- Scientific convergence cannot be assessed until the external Stage IV HPC
  outputs return.
- Hidden fixed-mu validation requires an explicit `REFERENCE_DATASET` path at
  post-run report time.

## 9. Next Steps

- Upload this package to the GBU cluster.
- Extract into a fresh directory.
- Do not export stale `OUTPUT_ROOT`, `RUN_ID`, or `CONFIG_JSON`; the scripts
  will reject conflicting values.
- Run `bash scripts/check_stageiv_3d_submit_ready.sh`.
- Submit with `CONFIRM_STAGEIV_FULL_LOOP=1`.
"""


def readme_text() -> str:
    return f"""# Stage IV 3D Topology-Aware Cold-Start Full Loop

Run id:

```text
{STAGEIV_RUN_ID}
```

This package is self-contained for the Stage IV cold-start 3D active-learning run over
`(kBT/t, J_A/t, mu/t)`. It does not import Stage III datasets or checkpoints.

For the exact command checklist, see:

```text
HPC_COMMANDS_STAGEIV_3D.md
```

## Submit

```bash
cd ~/bkz/Fu_FFLO/{PACKAGE_NAME}
unset OUTPUT_ROOT RUN_ID CONFIG_JSON
export CONFIRM_STAGEIV_FULL_LOOP=1
export PYTHON_BIN=/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python
nohup bash scripts/submit_stageiv_3d_full_loop.sh > {STAGEIV_RUN_ID}.nohup.log 2>&1 &
```

The submit script uses `--exclude=gpuh01,gpuh14` and the Slurm script also
refuses to run if the runtime hostname resolves to `gpuh01` or `gpuh14`.

## Collect

```bash
bash scripts/collect_stageiv_3d_results.sh
```

Run collection after the Slurm loop finishes.  The post-run reports should then
be built from the returned/extracted output directory.

## Operations

```bash
bash scripts/check_stageiv_3d_submit_ready.sh
bash scripts/run_stageiv_3d_preflight.sh
bash scripts/monitor_stageiv_3d_run.sh
bash scripts/check_stageiv_3d_hpc_status.sh
RETURN_PATH=ML_Phase_StageIV_Topology3D bash scripts/check_stageiv_3d_return_bundle.sh
START_ITER=7 bash scripts/resume_stageiv_3d_full_loop.sh
ITER=0 FAILED_RANKS=3 bash scripts/recover_stageiv_failed_exact_iter.sh
bash scripts/inspect_stageiv_3d_checkpoint.sh
bash scripts/inspect_stageiv_failed_task.sh <job_id>
bash scripts/collect_stageiv_3d_results.sh
bash scripts/build_stageiv_3d_postrun_summary.sh
bash scripts/build_stageiv_3d_convergence_audit.sh
REFERENCE_DATASET=/path/to/stageiii_frozen_reference.npz bash scripts/build_stageiv_3d_hidden_slice_audit.sh
REFERENCE_DATASET=/path/to/stageiii_frozen_reference.npz bash scripts/build_stageiv_3d_all_postrun_reports.sh
```

## Notes

- Production seed size: 1024 scrambled Sobol points.
- Acquisition batch size: 256.
- Acquisition batches: 24 after seed.
- Exact oracle: robust incremental q-window with basin-level rank-and-cap K3.
- Topology diagnostics: Pfaffian Z2 plus full-BZ bulk gap.
"""


def hpc_commands_text() -> str:
    return f"""# Stage IV 3D HPC Command Checklist

This file is the operational checklist for the self-contained Stage IV 3D
topology-aware cold-start full-loop package.  It is intentionally command-only
and does not define physics, tolerances, acquisition rules, or convergence
criteria.

Run id:

```text
{STAGEIV_RUN_ID}
```

Output root:

```text
{STAGEIV_OUTPUT_ROOT}
```

Expected final cumulative dataset after the seed plus 24 acquisition batches:

```text
{STAGEIV_OUTPUT_ROOT}/active_runs/{STAGEIV_RUN_ID}/dataset_iter025.npz
```

## 1. After Extracting On HPC

```bash
cd ~/bkz/Fu_FFLO/{PACKAGE_NAME}
unset OUTPUT_ROOT RUN_ID CONFIG_JSON
export PYTHON_BIN=/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python
bash scripts/check_stageiv_3d_submit_ready.sh
```

The submit-ready checker is report-only.  It does not submit Slurm jobs, merge
shards, append datasets, run exact calculations, or continue active learning.

## 2. Submit The Production Full Loop

```bash
unset OUTPUT_ROOT RUN_ID CONFIG_JSON
export CONFIRM_STAGEIV_FULL_LOOP=1
nohup bash scripts/submit_stageiv_3d_full_loop.sh > {STAGEIV_RUN_ID}.nohup.log 2>&1 &
```

The Slurm script excludes `gpuh01,gpuh14` and also refuses to run if the runtime
host name resolves to `gpuh01` or `gpuh14`.

The package identity is frozen.  If `OUTPUT_ROOT`, `RUN_ID`, or `CONFIG_JSON`
is already set to a value other than the Stage IV 3D expected value, generated
scripts exit with an error instead of silently writing into another run.

## 3. Monitor The Run

```bash
tail -n 120 {STAGEIV_RUN_ID}.nohup.log
bash scripts/check_stageiv_3d_hpc_status.sh
```

The status checker is read-only.  It is useful when the Slurm queue is empty
but the checkpoint, archive, or returned-output state is unclear.

## 4. Recover One Failed Exact-Array Rank

Use this path only when one or a few Slurm array ranks fail while the other
expected exact shards for the same iteration are present or can be preserved.

```bash
bash scripts/inspect_stageiv_failed_task.sh <job_id>
ITER=<failed_iteration> FAILED_RANKS=<rank_list> bash scripts/recover_stageiv_failed_exact_iter.sh
START_ITER=<next_iteration> bash scripts/resume_stageiv_3d_full_loop.sh
```

Example:

```bash
bash scripts/inspect_stageiv_failed_task.sh 81110
ITER=0 FAILED_RANKS=3 bash scripts/recover_stageiv_failed_exact_iter.sh
START_ITER=1 bash scripts/resume_stageiv_3d_full_loop.sh
```

Do not rerun a whole full loop just because one exact-array rank hit a
transient CUDA device busy/unavailable error.

## 5. Resume From A Completed Prefix

If the loop stopped after a completed merge/append boundary, resume from the
next exact iteration:

```bash
START_ITER=<next_iteration> bash scripts/resume_stageiv_3d_full_loop.sh
```

Do not set `START_ITER` to an iteration whose selected-points partition or
trusted append state is incomplete.

## 6. Collect Results After Completion

```bash
bash scripts/collect_stageiv_3d_results.sh
```

Expected archive:

```text
{STAGEIV_OUTPUT_ROOT}/stageiv_3d_topology_full_loop_results.tar.gz
```

## 7. Check Returned Results Before Analysis

After downloading or extracting the returned directory:

```bash
RETURN_PATH={STAGEIV_OUTPUT_ROOT} bash scripts/check_stageiv_3d_return_bundle.sh
```

The returned-result checker accepts either a returned directory or tar archive.
It does not extract archives and does not run exact calculations.

## 8. Build Post-Run Report-Only Audits

```bash
REFERENCE_DATASET=/path/to/stageiii_frozen_reference.npz bash scripts/build_stageiv_3d_all_postrun_reports.sh
```

The reference dataset is used only for hidden fixed-mu validation.  It must not
be merged into Stage IV training.

Primary post-run decision file:

```text
{STAGEIV_OUTPUT_ROOT}/reports/stageiv_3d_postrun_bundle/stageiv_3d_postrun_bundle_decision.json
```

## Do Not Do

- Do not merge Stage III `dataset_iter018`, Phase-II `dataset_iter035`, or any
  previous topology-derived dataset into this cold-start Stage IV training run.
- Do not interpret package preflight or submit-ready success as scientific
  convergence.
- Do not mark a missing boundary or surface as zero shift.
- Do not modify production thermodynamic phase criteria, topology formulas,
  acquisition rules, StopController thresholds, or exact-oracle tolerances from
  this operational package.
- Do not use `gpuh01` or `gpuh14`.
- Do not restart from scratch for a single-rank CUDA device-busy failure before
  trying the failed-rank recovery command.
"""


def build_package() -> dict[str, Any]:
    if PACKAGE_ROOT.exists():
        shutil.rmtree(PACKAGE_ROOT)
    PACKAGE_ROOT.mkdir(parents=True, exist_ok=True)
    copy_tree(ROOT / "ml_phase", PACKAGE_ROOT / "ml_phase")
    for file_name in [
        "eta_phase_diagram_cuda.py",
        "tfflo_1d_cuda.py",
        "AGENTS.md",
        "MODEL_SPEC.md",
        "README.md",
    ]:
        copy_file(ROOT / file_name, PACKAGE_ROOT / file_name)
    write_text(PACKAGE_ROOT / "docs" / "PROJECT_SUMMARY.md", package_project_summary_text())
    for doc in [
        "MODEL_SPEC.md",
        "NUMERICS_SPEC.md",
        "DECISIONS.md",
        "StageIV_MultiDim_Phase_Diagram.md",
    ]:
        copy_file(ROOT / "docs" / doc, PACKAGE_ROOT / "docs" / doc)
    for script in [
        "stageiv_3d_select.py",
        "stageiv_3d_preflight.py",
        "stageiv_3d_submit_check.py",
        "stageiv_3d_postrun_report.py",
        "stageiv_3d_convergence_audit.py",
        "stageiv_3d_hidden_slice_audit.py",
        "stageiv_3d_postrun_bundle.py",
        "stageiv_3d_hpc_status.py",
        "stageiv_3d_return_check.py",
    ]:
        copy_file(ROOT / "scripts" / script, PACKAGE_ROOT / "scripts" / script)

    prod_cfg = StageIV3DConfig()
    smoke_cfg = StageIV3DConfig(initial_seed_size=32, batch_size=16, max_acquisition_batches=1, candidate_pool_size=512, model_ensemble=2, reg_epochs=8, cls_epochs=8)
    write_json(PACKAGE_ROOT / "configs" / "stageiv_3d_production.json", prod_cfg.to_dict())
    write_json(PACKAGE_ROOT / "configs" / "stageiv_3d_smoke.json", smoke_cfg.to_dict())
    write_text(PACKAGE_ROOT / "scripts" / "slurm_stageiv_exact_array.sh", slurm_script())
    write_text(PACKAGE_ROOT / "scripts" / "submit_stageiv_3d_full_loop.sh", submit_script())
    write_text(PACKAGE_ROOT / "scripts" / "collect_stageiv_3d_results.sh", collect_script())
    write_text(PACKAGE_ROOT / "scripts" / "run_stageiv_3d_preflight.sh", preflight_script())
    write_text(PACKAGE_ROOT / "scripts" / "check_stageiv_3d_submit_ready.sh", submit_ready_script())
    write_text(PACKAGE_ROOT / "scripts" / "resume_stageiv_3d_full_loop.sh", resume_script())
    write_text(PACKAGE_ROOT / "scripts" / "recover_stageiv_failed_exact_iter.sh", failed_rank_recovery_script())
    write_text(PACKAGE_ROOT / "scripts" / "monitor_stageiv_3d_run.sh", monitor_script())
    write_text(PACKAGE_ROOT / "scripts" / "check_stageiv_3d_hpc_status.sh", hpc_status_script())
    write_text(PACKAGE_ROOT / "scripts" / "check_stageiv_3d_return_bundle.sh", return_check_script())
    write_text(PACKAGE_ROOT / "scripts" / "inspect_stageiv_3d_checkpoint.sh", checkpoint_script())
    write_text(PACKAGE_ROOT / "scripts" / "inspect_stageiv_failed_task.sh", failed_inspect_script())
    write_text(PACKAGE_ROOT / "scripts" / "build_stageiv_3d_postrun_summary.sh", postrun_report_script())
    write_text(PACKAGE_ROOT / "scripts" / "build_stageiv_3d_convergence_audit.sh", convergence_audit_script())
    write_text(PACKAGE_ROOT / "scripts" / "build_stageiv_3d_hidden_slice_audit.sh", hidden_slice_audit_script())
    write_text(PACKAGE_ROOT / "scripts" / "build_stageiv_3d_all_postrun_reports.sh", all_postrun_reports_script())
    package_readme = readme_text()
    write_text(PACKAGE_ROOT / "README.md", package_readme)
    write_text(PACKAGE_ROOT / "README_STAGEIV_3D_HPC.md", package_readme)
    write_text(PACKAGE_ROOT / "ENVIRONMENT_STAGEIV_3D_HPC.md", environment_text())
    write_text(PACKAGE_ROOT / "HPC_COMMANDS_STAGEIV_3D.md", hpc_commands_text())

    identity_guard_rows = freeze_stageiv_shell_identity(PACKAGE_ROOT)
    shell_rows = normalize_shell(PACKAGE_ROOT)
    py_files = [str(p.relative_to(PACKAGE_ROOT)) for p in sorted(PACKAGE_ROOT.rglob("*.py"))]
    compile_cmd = ["python", "-m", "py_compile", *py_files]
    compile_result = run(compile_cmd, cwd=PACKAGE_ROOT)
    bash_results: list[dict[str, Any]] = []
    for shell_path in sorted(PACKAGE_ROOT.rglob("*.sh")):
        bash_check = run(["bash", "-n", str(shell_path.relative_to(PACKAGE_ROOT))], cwd=PACKAGE_ROOT)
        bash_output_text = (bash_check.stdout + bash_check.stderr).replace("\x00", "")
        bash_unavailable = (
            bash_check.returncode != 0
            and "Windows Subsystem for Linux has no installed distributions" in bash_output_text
        )
        bash_results.append(
            {
                "path": shell_path.relative_to(PACKAGE_ROOT).as_posix(),
                "returncode": bash_check.returncode,
                "status": "skipped_local_bash_unavailable" if bash_unavailable else ("pass" if bash_check.returncode == 0 else "fail"),
                "stdout": "" if bash_unavailable else bash_check.stdout,
                "stderr": "" if bash_unavailable else bash_check.stderr,
                "notes": "local Windows bash maps to WSL without an installed distro" if bash_unavailable else "",
            }
        )
    preflight = run(
        [
            "python",
            "scripts/stageiv_3d_preflight.py",
            "--config",
            "configs/stageiv_3d_production.json",
            "--output-dir",
            "reports/stageiv_3d_preflight_package",
            "--n-ja",
            "8",
            "--n-mu",
            "8",
            "--n-k",
            "128",
        ],
        cwd=PACKAGE_ROOT,
    )
    submit_check = run(
        [
            "python",
            "scripts/stageiv_3d_submit_check.py",
            "--root",
            ".",
            "--config",
            "configs/stageiv_3d_production.json",
            "--output-dir",
            "reports/stageiv_3d_submit_check_package",
            "--python-bin",
            "python",
            "--allow-missing-manifest",
        ],
        cwd=PACKAGE_ROOT,
    )
    status_smoke = run(
        [
            "python",
            "scripts/stageiv_3d_hpc_status.py",
            "--output-root",
            "reports/stageiv_3d_status_smoke_output",
            "--output-dir",
            "reports/stageiv_3d_status_smoke",
            "--config",
            "configs/stageiv_3d_production.json",
        ],
        cwd=PACKAGE_ROOT,
    )
    return_check_smoke = run(
        [
            "python",
            "scripts/stageiv_3d_return_check.py",
            "--return-path",
            "ML_Phase_StageIV_Topology3D",
            "--config",
            "configs/stageiv_3d_production.json",
            "--output-dir",
            "reports/stageiv_3d_return_check_package",
        ],
        cwd=PACKAGE_ROOT,
    )
    stale_2d_marker_scan = scan_stale_2d_markers(PACKAGE_ROOT)

    manifest = {
        "package_name": PACKAGE_NAME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": STAGEIV_RUN_ID,
        "output_root": STAGEIV_OUTPUT_ROOT,
        "git": git_snapshot(),
        "stageiv_config": prod_cfg.to_dict(),
        "identity_guard": {
            "expected_output_root": STAGEIV_OUTPUT_ROOT,
            "expected_run_id": STAGEIV_RUN_ID,
            "expected_config_json": "configs/stageiv_3d_production.json",
            "script_rows": identity_guard_rows,
        },
        "stale_2d_marker_scan": stale_2d_marker_scan,
        "shell_normalization": shell_rows,
        "bash_n_results": bash_results,
        "generated_data_dir_check": {
            "ml_phase_active_runs_exists": (PACKAGE_ROOT / "ml_phase" / "active_runs").exists(),
            "ml_phase_datasets_exists": (PACKAGE_ROOT / "ml_phase" / "datasets").exists(),
            "ml_phase_figures_exists": (PACKAGE_ROOT / "ml_phase" / "figures").exists(),
            "ml_phase_reports_exists": (PACKAGE_ROOT / "ml_phase" / "reports").exists(),
        },
        "py_compile_returncode": compile_result.returncode,
        "py_compile_stdout": compile_result.stdout,
        "py_compile_stderr": compile_result.stderr,
        "package_preflight_returncode": preflight.returncode,
        "package_preflight_stdout": preflight.stdout,
        "package_preflight_stderr": preflight.stderr,
        "package_submit_check_returncode": submit_check.returncode,
        "package_submit_check_stdout": submit_check.stdout,
        "package_submit_check_stderr": submit_check.stderr,
        "package_status_smoke_returncode": status_smoke.returncode,
        "package_status_smoke_stdout": status_smoke.stdout,
        "package_status_smoke_stderr": status_smoke.stderr,
        "package_return_check_smoke_returncode": return_check_smoke.returncode,
        "package_return_check_smoke_stdout": return_check_smoke.stdout,
        "package_return_check_smoke_stderr": return_check_smoke.stderr,
    }
    generated_clean = not any(manifest["generated_data_dir_check"].values())
    bash_clean = all(str(r["status"]) in {"pass", "skipped_local_bash_unavailable"} for r in bash_results)
    manifest["package_validation_status"] = (
        "pass"
        if compile_result.returncode == 0
        and preflight.returncode == 0
        and submit_check.returncode == 0
        and status_smoke.returncode == 0
        and return_check_smoke.returncode == 0
        and all(r["status"] == "pass" for r in shell_rows)
        and not stale_2d_marker_scan
        and bash_clean
        and generated_clean
        else "fail"
    )
    write_json(PACKAGE_ROOT / "RUN_MANIFEST.json", manifest)
    for pycache in sorted(PACKAGE_ROOT.rglob("__pycache__")):
        shutil.rmtree(pycache, ignore_errors=True)

    if ARCHIVE_PATH.exists():
        ARCHIVE_PATH.unlink()
    with tarfile.open(ARCHIVE_PATH, "w:gz") as tar:
        tar.add(PACKAGE_ROOT, arcname=PACKAGE_ROOT.name)
    archive_sha = sha256_file(ARCHIVE_PATH)
    write_text(ARCHIVE_PATH.with_suffix(ARCHIVE_PATH.suffix + ".sha256"), f"{archive_sha}  {ARCHIVE_PATH.name}\n")
    manifest["archive_path"] = str(ARCHIVE_PATH)
    manifest["archive_sha256"] = archive_sha
    with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
        manifest["tar_member_count"] = len(tar.getmembers())
        manifest["tar_listing_sample"] = [m.name for m in tar.getmembers()[:40]]
    write_json(PACKAGE_ROOT / "RUN_MANIFEST.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the self-contained Stage IV 3D topology-aware HPC package.")
    p.add_argument("--print-json", action="store_true", help="Print package manifest.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_package()
    if args.print_json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(f"Wrote package: {PACKAGE_ROOT}")
        print(f"Wrote archive: {ARCHIVE_PATH}")
        print(f"Validation: {manifest['package_validation_status']}")


if __name__ == "__main__":
    main()
