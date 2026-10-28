#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SUBMIT_REL = Path("scripts/submit_stagev_acqv2_full_loop.sh")
SELECT_REL = Path("scripts/slurm_stagev_acqv2_select.sh")

OLD_SELECTION_BLOCK = '''  if [ "${ITER}" = "0" ]; then
    "${PYTHON_BIN}" scripts/stagev_acqv2_select.py \\
      --config "${CONFIG_JSON}" \\
      --mode seed \\
      --iteration "${ITER}" \\
      --output-root "${OUTPUT_ROOT}" \\
      --run-id "${RUN_ID}" \\
      --world-size "${WORLD_SIZE}" \\
      --partition-strategy "${PARTITION_STRATEGY}"
  else
    "${PYTHON_BIN}" scripts/stagev_acqv2_select.py \\
      --config "${CONFIG_JSON}" \\
      --mode acquisition \\
      --iteration "${ITER}" \\
      --output-root "${OUTPUT_ROOT}" \\
      --run-id "${RUN_ID}" \\
      --dataset "${RUN_DIR}/dataset_iter${TAG}.npz" \\
      --world-size "${WORLD_SIZE}" \\
      --partition-strategy "${PARTITION_STRATEGY}" \\
      --device "${SELECT_DEVICE}"
  fi

  export PROJECT_DIR PYTHON_BIN OUTPUT_ROOT RUN_ID ITER WORLD_SIZE
'''

NEW_SELECTION_BLOCK = '''  if [ "${ITER}" = "0" ]; then
    SELECT_MODE="seed"
  else
    SELECT_MODE="acquisition"
  fi
  export PROJECT_DIR PYTHON_BIN OUTPUT_ROOT RUN_ID ITER WORLD_SIZE CONFIG_JSON PARTITION_STRATEGY SELECT_DEVICE SELECT_MODE
  SELECT_JOB_ID="$(sbatch --parsable --exclude="${EXCLUDE_NODES}" scripts/slurm_stagev_acqv2_select.sh)"
  echo "[submit] Stage V selection iter ${TAG}: job ${SELECT_JOB_ID}"
  wait_for_job "${SELECT_JOB_ID}" "${TAG}-selection"

  export PROJECT_DIR PYTHON_BIN OUTPUT_ROOT RUN_ID ITER WORLD_SIZE
'''

SLURM_SELECT_SCRIPT = '''#!/bin/bash
#SBATCH --job-name=stagev_select
#SBATCH --partition=NV_H100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --exclude=gpuh01,gpuh14
#SBATCH --output=slurm-select-%j.out

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
OUTPUT_ROOT="${OUTPUT_ROOT:-ML_Phase_StageV_AcqV2}"
RUN_ID="${RUN_ID:-stagev_acqv2_boundary_support_learned_residual_3d_v1}"
CONFIG_JSON="${CONFIG_JSON:-configs/stagev_acqv2_production.json}"
ITER="${ITER:?ITER must be set}"
WORLD_SIZE="${WORLD_SIZE:-8}"
SELECT_MODE="${SELECT_MODE:?SELECT_MODE must be seed or acquisition}"
PARTITION_STRATEGY="${PARTITION_STRATEGY:-cost_aware}"
SELECT_DEVICE="${SELECT_DEVICE:-cpu}"
RUN_DIR="${OUTPUT_ROOT}/active_runs/${RUN_ID}"
TAG="$(printf "%03d" "${ITER}")"

cd "${PROJECT_DIR}"

echo "stage=selection"
echo "run_id=${RUN_ID}"
echo "iteration=${ITER}"
echo "select_mode=${SELECT_MODE}"
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

if [ "${SELECT_MODE}" = "seed" ]; then
  "${PYTHON_BIN}" scripts/stagev_acqv2_select.py \\
    --config "${CONFIG_JSON}" \\
    --mode seed \\
    --iteration "${ITER}" \\
    --output-root "${OUTPUT_ROOT}" \\
    --run-id "${RUN_ID}" \\
    --world-size "${WORLD_SIZE}" \\
    --partition-strategy "${PARTITION_STRATEGY}"
elif [ "${SELECT_MODE}" = "acquisition" ]; then
  "${PYTHON_BIN}" scripts/stagev_acqv2_select.py \\
    --config "${CONFIG_JSON}" \\
    --mode acquisition \\
    --iteration "${ITER}" \\
    --output-root "${OUTPUT_ROOT}" \\
    --run-id "${RUN_ID}" \\
    --dataset "${RUN_DIR}/dataset_iter${TAG}.npz" \\
    --world-size "${WORLD_SIZE}" \\
    --partition-strategy "${PARTITION_STRATEGY}" \\
    --device "${SELECT_DEVICE}"
else
  echo "[error] invalid SELECT_MODE=${SELECT_MODE}" >&2
  exit 43
fi
'''


def find_package_root(explicit_root: str | None) -> Path:
    if explicit_root:
        root = Path(explicit_root).expanduser().resolve()
        if (root / SUBMIT_REL).exists():
            return root
        raise FileNotFoundError(f"cannot find {SUBMIT_REL} under explicit root: {root}")

    starts = [Path.cwd().resolve(), Path(__file__).resolve().parent]
    seen: set[Path] = set()
    for start in starts:
        for candidate in [start, *start.parents]:
            if candidate in seen:
                continue
            seen.add(candidate)
            if (candidate / SUBMIT_REL).exists():
                return candidate
    raise FileNotFoundError("cannot locate Stage V package root. Run from package root or pass --root.")


def write_select_script(root: Path) -> None:
    path = root / SELECT_REL
    path.write_text(SLURM_SELECT_SCRIPT, encoding="utf-8", newline="\n")
    path.chmod(0o755)


def find_selection_block_by_markers(text: str) -> tuple[int, int]:
    lines = text.splitlines(keepends=True)
    selector_index = None
    for idx, line in enumerate(lines):
        if "scripts/stagev_acqv2_select.py" in line:
            selector_index = idx
            break
    if selector_index is None:
        raise RuntimeError("submit script does not contain scripts/stagev_acqv2_select.py")

    start_index = None
    for idx in range(selector_index, -1, -1):
        line = lines[idx].strip()
        if line.startswith("if ") and "ITER" in line and ' = "0"' in line and line.endswith("then"):
            start_index = idx
            break
    if start_index is None:
        context = "".join(lines[max(0, selector_index - 8):selector_index + 8])
        raise RuntimeError("could not locate selection if/then block before selector call:\n" + context)

    fi_index = None
    for idx in range(selector_index + 1, len(lines)):
        if lines[idx].strip() == "fi":
            fi_index = idx
            break
    if fi_index is None:
        context = "".join(lines[start_index:start_index + 80])
        raise RuntimeError("could not locate closing fi for selection block:\n" + context)

    end_index = fi_index + 1
    while end_index < len(lines) and lines[end_index].strip() == "":
        end_index += 1
    if end_index < len(lines):
        stripped = lines[end_index].strip()
        if stripped.startswith("export ") and "PROJECT_DIR" in stripped and "ITER" in stripped and "WORLD_SIZE" in stripped:
            end_index += 1
    return start_index, end_index


def patch_submit(root: Path) -> str:
    path = root / SUBMIT_REL
    text = path.read_text(encoding="utf-8")
    if "slurm_stagev_acqv2_select.sh" in text:
        return "already_patched"
    if OLD_SELECTION_BLOCK in text:
        new_text = text.replace(OLD_SELECTION_BLOCK, NEW_SELECTION_BLOCK)
        status = "patched_exact_block"
    else:
        lines = text.splitlines(keepends=True)
        start_index, end_index = find_selection_block_by_markers(text)
        new_text = "".join(lines[:start_index]) + NEW_SELECTION_BLOCK + "".join(lines[end_index:])
        status = "patched_marker_block"
    path.write_text(new_text, encoding="utf-8", newline="\n")
    path.chmod(0o755)
    return status


def verify(root: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", "py_compile", str(root / "scripts/stagev_acqv2_select.py")],
        check=True,
    )
    bash = subprocess.run(["bash", "-n", str(root / SUBMIT_REL)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if bash.returncode != 0:
        raise RuntimeError(f"bash -n failed for {SUBMIT_REL}:\n{bash.stderr}")
    bash = subprocess.run(["bash", "-n", str(root / SELECT_REL)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if bash.returncode != 0:
        raise RuntimeError(f"bash -n failed for {SELECT_REL}:\n{bash.stderr}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch Stage V package so acquisition selection runs under Slurm.")
    parser.add_argument("--root", default=None, help="Stage V package root. Defaults to auto-detect.")
    parser.add_argument("--no-verify", action="store_true", help="Skip py_compile and bash -n checks.")
    args = parser.parse_args()

    root = find_package_root(args.root)
    write_select_script(root)
    status = patch_submit(root)
    if not args.no_verify:
        verify(root)

    payload = {
        "package_root": str(root),
        "submit_patch_status": status,
        "selection_script": str(root / SELECT_REL),
        "selection_runs_under_slurm": True,
        "excluded_nodes": ["gpuh01", "gpuh14"],
        "resume_recommendation": "START_ITER=14 after existing dataset_iter014",
    }
    out = root / "stagev_slurm_selection_hotfix_applied.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("")
    print("Resume example:")
    print("export CONFIRM_STAGEV_PRODUCTION=1")
    print(
        "START_ITER=14 FINAL_EXACT_ITER=17 nohup bash scripts/resume_stagev_acqv2_full_loop.sh "
        "> stagev_resume_iter014_to017_slurmselect.nohup.log 2>&1 &"
    )


if __name__ == "__main__":
    main()
