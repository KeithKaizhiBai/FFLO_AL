from __future__ import annotations

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

from ml_phase.stagev_v2 import STAGEV2_OUTPUT_ROOT, STAGEV2_RUN_ID, StageV2Config


PACKAGE_NAME = f"{STAGEV2_RUN_ID}_hpc"
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
        "active_runs",
        "reports",
        "output",
        "outputs",
        "datasets",
        "figures",
    }
    out: set[str] = set()
    for name in names:
        lower = name.lower()
        if name in blocked or lower.endswith((".pyc", ".pyo", ".tmp", ".bak", ".tar.gz", ".zip")):
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


def normalize_text_files(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    suffixes = {".py", ".sh", ".md", ".json", ".yaml", ".yml", ".txt"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
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
            data.decode("utf-8")
        except UnicodeDecodeError:
            status = "fail"
            notes.append("invalid_utf8")
        path.write_bytes(data)
        rows.append({"path": path.relative_to(root).as_posix(), "status": status, "notes": ";".join(notes)})
    return rows


def slurm_exact_script() -> str:
    return f"""#!/bin/bash
#SBATCH --job-name=stagev2_exact
#SBATCH --partition=NV_H100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --exclude=gpuh01,gpuh14
#SBATCH --output=slurm-%A_%a.out

set -euo pipefail

export LANG="${{LANG:-C.UTF-8}}"
export LC_ALL="${{LC_ALL:-C.UTF-8}}"
export PYTHONUTF8=1
export PYTHONIOENCODING="utf-8"

host_name="$(hostname 2>/dev/null || true)"
if [ "${{host_name%%.*}}" = "gpuh01" ] || [ "${{host_name%%.*}}" = "gpuh14" ]; then
  echo "[error] refusing to run on excluded node ${{host_name%%.*}}" >&2
  exit 42
fi

PROJECT_DIR="${{PROJECT_DIR:-$PWD}}"
PYTHON_BIN="${{PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}}"
OUTPUT_ROOT="{STAGEV2_OUTPUT_ROOT}"
RUN_ID="{STAGEV2_RUN_ID}"
ITER="${{ITER:?ITER must be set}}"
WORLD_SIZE="${{WORLD_SIZE:-8}}"
RANK="${{SLURM_ARRAY_TASK_ID:-0}}"
export OUTPUT_ROOT RUN_ID

cd "${{PROJECT_DIR}}"

echo "run_id=${{RUN_ID}}"
echo "iteration=${{ITER}}"
echo "rank=${{RANK}}"
echo "world_size=${{WORLD_SIZE}}"
echo "hostname=${{host_name}}"
echo "CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-}}"
nvidia-smi || true
"${{PYTHON_BIN}}" - <<'PY'
import torch
print("torch.__version__=", torch.__version__)
print("torch.version.cuda=", torch.version.cuda)
print("torch.cuda.is_available()=", torch.cuda.is_available())
PY

"${{PYTHON_BIN}}" -m ml_phase.exact_oracle \
  --active-root "${{OUTPUT_ROOT}}/active_runs" \
  --run-id "${{RUN_ID}}" \
  --iteration "${{ITER}}" \
  --rank "${{RANK}}" \
  --world-size "${{WORLD_SIZE}}" \
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
  --topology-gap-nk "${{TOPOLOGY_GAP_NK:-2048}}" \
  --topology-gap-backend "${{TOPOLOGY_GAP_BACKEND:-gpu}}" \
  --topology-gap-tol-rel "${{TOPOLOGY_GAP_TOL_REL:-1e-8}}" \
  --topology-gap-tol-abs "${{TOPOLOGY_GAP_TOL_ABS:-0.0}}" \
  --topology-gap-k-chunk "${{TOPOLOGY_GAP_K_CHUNK:-512}}"
"""


def slurm_select_script() -> str:
    return f"""#!/bin/bash
#SBATCH --job-name=stagev2_select
#SBATCH --partition=NV_H100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --exclude=gpuh01,gpuh14
#SBATCH --output=slurm-select-%j.out

set -euo pipefail

export LANG="${{LANG:-C.UTF-8}}"
export LC_ALL="${{LC_ALL:-C.UTF-8}}"
export PYTHONUTF8=1
export PYTHONIOENCODING="utf-8"

host_name="$(hostname 2>/dev/null || true)"
if [ "${{host_name%%.*}}" = "gpuh01" ] || [ "${{host_name%%.*}}" = "gpuh14" ]; then
  echo "[error] refusing to run on excluded node ${{host_name%%.*}}" >&2
  exit 42
fi

PROJECT_DIR="${{PROJECT_DIR:-$PWD}}"
PYTHON_BIN="${{PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}}"
OUTPUT_ROOT="{STAGEV2_OUTPUT_ROOT}"
RUN_ID="{STAGEV2_RUN_ID}"
CONFIG_JSON="${{CONFIG_JSON:-configs/stagev_v2_production.json}}"
ITER="${{ITER:?ITER must be set}}"
WORLD_SIZE="${{WORLD_SIZE:-8}}"
SELECT_MODE="${{SELECT_MODE:?SELECT_MODE must be seed or acquisition}}"
PARTITION_STRATEGY="${{PARTITION_STRATEGY:-cost_aware}}"
SELECT_DEVICE="${{SELECT_DEVICE:-cpu}}"
RUN_DIR="${{OUTPUT_ROOT}}/active_runs/${{RUN_ID}}"
TAG="$(printf "%03d" "${{ITER}}")"
export OUTPUT_ROOT RUN_ID

cd "${{PROJECT_DIR}}"

echo "stage=selection"
echo "run_id=${{RUN_ID}}"
echo "iteration=${{ITER}}"
echo "select_mode=${{SELECT_MODE}}"
echo "world_size=${{WORLD_SIZE}}"
echo "hostname=${{host_name}}"
echo "CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-}}"
nvidia-smi || true

if [ "${{SELECT_MODE}}" = "seed" ]; then
  "${{PYTHON_BIN}}" scripts/stagev_v2_select.py \
    --config "${{CONFIG_JSON}}" \
    --mode seed \
    --iteration "${{ITER}}" \
    --output-root "${{OUTPUT_ROOT}}" \
    --run-id "${{RUN_ID}}" \
    --world-size "${{WORLD_SIZE}}" \
    --partition-strategy "${{PARTITION_STRATEGY}}"
elif [ "${{SELECT_MODE}}" = "acquisition" ]; then
  "${{PYTHON_BIN}}" scripts/stagev_v2_select.py \
    --config "${{CONFIG_JSON}}" \
    --mode acquisition \
    --iteration "${{ITER}}" \
    --output-root "${{OUTPUT_ROOT}}" \
    --run-id "${{RUN_ID}}" \
    --dataset "${{RUN_DIR}}/dataset_iter${{TAG}}.npz" \
    --world-size "${{WORLD_SIZE}}" \
    --partition-strategy "${{PARTITION_STRATEGY}}" \
    --device "${{SELECT_DEVICE}}"
else
  echo "[error] invalid SELECT_MODE=${{SELECT_MODE}}" >&2
  exit 43
fi
"""


def submit_script() -> str:
    return f"""#!/bin/bash
set -euo pipefail

export LANG="${{LANG:-C.UTF-8}}"
export LC_ALL="${{LC_ALL:-C.UTF-8}}"
export PYTHONUTF8=1
export PYTHONIOENCODING="utf-8"

PROJECT_DIR="${{PROJECT_DIR:-$PWD}}"
PYTHON_BIN="${{PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}}"
OUTPUT_ROOT="{STAGEV2_OUTPUT_ROOT}"
RUN_ID="{STAGEV2_RUN_ID}"
CONFIG_JSON="${{CONFIG_JSON:-configs/stagev_v2_production.json}}"
RUN_DIR="${{OUTPUT_ROOT}}/active_runs/${{RUN_ID}}"
WORLD_SIZE="${{WORLD_SIZE:-8}}"
START_ITER="${{START_ITER:-0}}"
FINAL_EXACT_ITER="${{FINAL_EXACT_ITER:-96}}"
PARTITION_STRATEGY="${{PARTITION_STRATEGY:-cost_aware}}"
SELECT_DEVICE="${{SELECT_DEVICE:-cpu}}"
EXCLUDE_NODES="${{EXCLUDE_NODES:-gpuh01,gpuh14}}"
SLEEP_SECONDS="${{SLEEP_SECONDS:-60}}"
POSITIVE_DELTA_GAP_TOL="${{POSITIVE_DELTA_GAP_TOL:-1e-8}}"

if [ "${{CONFIRM_STAGEV2_PRODUCTION:-0}}" != "1" ]; then
  echo "[error] set CONFIRM_STAGEV2_PRODUCTION=1 to submit Stage V-v2." >&2
  exit 2
fi

cd "${{PROJECT_DIR}}"

if [ "${{START_ITER}}" = "0" ] && [ -e "${{RUN_DIR}}" ]; then
  echo "[error] run directory already exists: ${{RUN_DIR}}" >&2
  echo "Set START_ITER>0 only for intentional resume." >&2
  exit 4
fi

mkdir -p "${{OUTPUT_ROOT}}/logs"
"${{PYTHON_BIN}}" scripts/stagev_v2_preflight.py \
  --config "${{CONFIG_JSON}}" \
  --output-dir "${{OUTPUT_ROOT}}/preflight"

wait_for_job() {{
  local job_id="$1"
  local iter_tag="$2"
  while true; do
    active="$(squeue -h -j "${{job_id}}" -o "%i|%T" 2>/dev/null | paste -sd, - || true)"
    if [ -z "${{active}}" ]; then
      break
    fi
    echo "[wait] Stage V-v2 ${{iter_tag}}: squeue active: ${{active}}"
    sleep "${{SLEEP_SECONDS}}"
  done
  failures="$(sacct -j "${{job_id}}" --format=JobID,State,ExitCode --parsable2 | awk -F'|' 'NR>1 && $1 !~ /\\.batch|\\.extern/ && $2 !~ /COMPLETED/ {{print $1\":\"$2\":\"$3}}' | paste -sd, -)"
  if [ -n "${{failures}}" ]; then
    echo "[error] Stage V-v2 ${{iter_tag}}: job ${{job_id}} failed: ${{failures}}" >&2
    exit 5
  fi
}}

for ITER in $(seq "${{START_ITER}}" "${{FINAL_EXACT_ITER}}"); do
  TAG="$(printf "%03d" "${{ITER}}")"
  NEXT="$(printf "%03d" "$((ITER + 1))")"
  if [ "${{ITER}}" = "0" ]; then
    SELECT_MODE="seed"
  else
    SELECT_MODE="acquisition"
  fi
  export PROJECT_DIR PYTHON_BIN OUTPUT_ROOT RUN_ID ITER WORLD_SIZE CONFIG_JSON PARTITION_STRATEGY SELECT_DEVICE SELECT_MODE
  SELECT_JOB_ID="$(sbatch --parsable --exclude="${{EXCLUDE_NODES}}" scripts/slurm_stagev_v2_select.sh)"
  echo "[submit] Stage V-v2 selection iter ${{TAG}}: job ${{SELECT_JOB_ID}}"
  wait_for_job "${{SELECT_JOB_ID}}" "selection iter ${{TAG}}"

  export PROJECT_DIR PYTHON_BIN OUTPUT_ROOT RUN_ID ITER WORLD_SIZE
  JOB_ID="$(sbatch --parsable --exclude="${{EXCLUDE_NODES}}" --array=0-$((WORLD_SIZE - 1)) scripts/slurm_stagev_v2_exact_array.sh)"
  echo "[submit] Stage V-v2 exact iter ${{TAG}}: job ${{JOB_ID}}"
  wait_for_job "${{JOB_ID}}" "exact iter ${{TAG}}"

  "${{PYTHON_BIN}}" -m ml_phase.hpc \
    --merge \
    --run-dir "${{RUN_DIR}}" \
    --iteration "${{ITER}}" \
    --world-size "${{WORLD_SIZE}}" \
    --positive-delta-gap-tol "${{POSITIVE_DELTA_GAP_TOL}}"

  "${{PYTHON_BIN}}" -m ml_phase.append_trusted \
    --dataset "${{RUN_DIR}}/dataset_iter${{TAG}}.npz" \
    --trusted-exact "${{RUN_DIR}}/iter${{TAG}}/exact_trusted_iter${{TAG}}.npz" \
    --output-npz "${{RUN_DIR}}/dataset_iter${{NEXT}}.npz" \
    --output-csv "${{RUN_DIR}}/dataset_iter${{NEXT}}.csv" \
    --output-root "${{OUTPUT_ROOT}}"

  "${{PYTHON_BIN}}" scripts/stagev_v2_update_reward.py \
    --config "${{CONFIG_JSON}}" \
    --output-root "${{OUTPUT_ROOT}}" \
    --run-id "${{RUN_ID}}" \
    --iteration "${{ITER}}"

  echo "[done] Stage V-v2 iter ${{TAG}}; wrote dataset_iter${{NEXT}}"
done

echo "[done] Stage V-v2 production loop completed through exact iter $(printf "%03d" "${{FINAL_EXACT_ITER}}")"
"""


def preflight_script() -> str:
    return """from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_phase.stagev_v2 import STAGEV2_RUN_ID, StageV2Config


def run(cmd):
    return subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def main():
    p = argparse.ArgumentParser(description="Stage V-v2 package preflight.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    cfg = StageV2Config.from_json(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pyc = run([sys.executable, "-m", "py_compile", "ml_phase/stagev_v2.py", "scripts/stagev_v2_select.py", "scripts/stagev_v2_update_reward.py"])
    status = "pass" if pyc.returncode == 0 and cfg.run_id == STAGEV2_RUN_ID else "fail"
    checks = {
        "status": status,
        "run_id": cfg.run_id,
        "output_root": cfg.output_root,
        "initial_seed_size": cfg.initial_seed_size,
        "micro_batch_size": cfg.micro_batch_size,
        "max_micro_batches": cfg.max_micro_batches,
        "cold_start": True,
        "stageiv_data_used_for_training": False,
        "stagev1_data_used_for_training": False,
        "excluded_nodes": ["gpuh01", "gpuh14"],
        "python": sys.executable,
        "confirm_stagev2_production": os.environ.get("CONFIRM_STAGEV2_PRODUCTION", "0"),
        "py_compile_returncode": pyc.returncode,
        "py_compile_stderr": pyc.stderr,
    }
    (args.output_dir / "stagev_v2_preflight.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    (args.output_dir / "stagev_v2_preflight.md").write_text(
        "# Stage V-v2 Preflight\\n\\n"
        f"- status: `{status}`\\n"
        f"- run_id: `{cfg.run_id}`\\n"
        "- excluded_nodes: `gpuh01,gpuh14`\\n"
        "- cold_start: `True`\\n",
        encoding="utf-8",
    )
    print(json.dumps(checks, indent=2, sort_keys=True))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
"""


def write_scripts(root: Path) -> None:
    write_text(root / "scripts" / "slurm_stagev_v2_exact_array.sh", slurm_exact_script())
    write_text(root / "scripts" / "slurm_stagev_v2_select.sh", slurm_select_script())
    write_text(root / "scripts" / "submit_stagev_v2_full_loop.sh", submit_script())
    write_text(
        root / "scripts" / "resume_stagev_v2_full_loop.sh",
        "#!/bin/bash\nset -euo pipefail\nSTART_ITER=\"${START_ITER:?Set START_ITER first.}\"\nexport START_ITER\nexec bash scripts/submit_stagev_v2_full_loop.sh\n",
    )
    write_text(
        root / "scripts" / "monitor_stagev_v2.sh",
        f"#!/bin/bash\nset -euo pipefail\nOUTPUT_ROOT=\"{STAGEV2_OUTPUT_ROOT}\"\nRUN_ID=\"{STAGEV2_RUN_ID}\"\nRUN_DIR=\"${{OUTPUT_ROOT}}/active_runs/${{RUN_ID}}\"\necho \"run_dir=${{RUN_DIR}}\"\nif [ ! -d \"${{RUN_DIR}}\" ]; then echo status=run_dir_missing; exit 0; fi\nfind \"${{RUN_DIR}}\" -maxdepth 1 -name 'dataset_iter*.npz' | sed -E 's/.*dataset_iter([0-9]+)\\.npz/\\1/' | sort -n | tail -1 | xargs -r echo latest_dataset_iteration=\nfind \"${{RUN_DIR}}\" -maxdepth 1 -type d -name 'iter[0-9][0-9][0-9]' | sed -E 's/.*iter([0-9]+)/\\1/' | sort -n | tail -1 | xargs -r echo latest_exact_iteration=\nsqueue -u \"$USER\" || true\n",
    )
    write_text(
        root / "scripts" / "inspect_stagev_v2_failed_task.sh",
        "#!/bin/bash\nset -euo pipefail\nJOB=\"${1:?Usage: bash scripts/inspect_stagev_v2_failed_task.sh JOBID[_TASKID]}\"\nsacct -j \"${JOB%%_*}\" --format=JobID,JobName,State,ExitCode,Elapsed,Start,End,NodeList,Reason --parsable2 || true\nfor f in slurm-${JOB}.out slurm-${JOB%%_*}_${JOB#*_}.out slurm-select-${JOB}.out; do\n  if [ -f \"$f\" ]; then\n    echo \"==== $f ====\"\n    tail -n 220 \"$f\"\n  fi\ndone\n",
    )
    write_text(
        root / "scripts" / "collect_stagev_v2_results.sh",
        f"#!/bin/bash\nset -euo pipefail\nOUTPUT_ROOT=\"{STAGEV2_OUTPUT_ROOT}\"\nRUN_ID=\"{STAGEV2_RUN_ID}\"\nARCHIVE=\"${{OUTPUT_ROOT}}/stagev_v2_multihead_results.tar.gz\"\nmkdir -p \"${{OUTPUT_ROOT}}\"\ntar -czf \"${{ARCHIVE}}\" \"${{OUTPUT_ROOT}}/active_runs/${{RUN_ID}}\" \"${{OUTPUT_ROOT}}/preflight\" 2>/dev/null || true\necho \"[done] archive=${{ARCHIVE}}\"\n",
    )
    write_text(
        root / "scripts" / "run_stagev_v2_smoke.sh",
        "#!/bin/bash\nset -euo pipefail\nPYTHON_BIN=\"${PYTHON_BIN:-python}\"\n\"${PYTHON_BIN}\" -m py_compile ml_phase/stagev_v2.py scripts/stagev_v2_select.py scripts/stagev_v2_update_reward.py scripts/stagev_v2_smoke.py\n\"${PYTHON_BIN}\" scripts/stagev_v2_smoke.py --output-dir reports/stagev_v2_multihead_smoke\n",
    )
    write_text(root / "scripts" / "stagev_v2_preflight.py", preflight_script())
    for path in sorted((root / "scripts").glob("*.sh")):
        path.chmod(0o755)


def config_variants() -> dict[str, StageV2Config]:
    base = StageV2Config()
    return {
        "stagev_v2_production": base,
        "stagev_v2_microbatch128": StageV2Config(micro_batch_size=128, max_micro_batches=48),
        "stagev_v2_a0_only": StageV2Config(learned_lambda_max=0.0, learned_min_reward_samples=10**9),
        "stagev_v2_lower_mu_extension": StageV2Config(mu_min=-1.0, mu_max=1.5, guard_mu_min=-1.25, guard_mu_max=2.0),
        "stagev_v2_smoke": StageV2Config(initial_seed_size=128, micro_batch_size=16, max_micro_batches=2, candidate_pool_size=2048, model_ensemble=2, reg_epochs=4, cls_epochs=4),
    }


def write_configs(root: Path) -> None:
    for name, cfg in config_variants().items():
        write_json(root / "configs" / f"{name}.json", cfg.to_dict())
    write_json(
        root / "schemas" / "stagev_v2_exact_output_schema.json",
        {
            "schema_note": "Stage V-v2 uses existing exact oracle fields. free_energy_gap_to_normal is the stored F_SC_minus_F_normal margin.",
            "required_or_alias_fields": {
                "F_SC_minus_F_normal": "free_energy_gap_to_normal",
                "P0": "topology_p0",
                "Ppi": "topology_ppi",
                "bulk_gap": "topology_bulk_gap",
                "topology_label": "topology_label_code",
                "spectral_status": "topology_spectral_status_code",
            },
        },
    )


def write_readme(root: Path, snapshot: dict[str, Any]) -> None:
    write_text(
        root / "README.md",
        f"""# Stage V-v2 Multi-head Boundary-learning HPC Package

Run id:

```text
{STAGEV2_RUN_ID}
```

This is a new cold-start Stage V-v2 package. It does not use Stage III, Stage
IV, or Stage V-v1 data/checkpoints for training initialization or seed points.

## Smoke

```bash
cd ~/bkz/Fu_FFLO/{PACKAGE_NAME}
export PYTHON_BIN=/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python
bash scripts/run_stagev_v2_smoke.sh
```

## Production

```bash
export CONFIRM_STAGEV2_PRODUCTION=1
nohup bash scripts/submit_stagev_v2_full_loop.sh > {STAGEV2_RUN_ID}.nohup.log 2>&1 &
```

Resume:

```bash
export CONFIRM_STAGEV2_PRODUCTION=1
START_ITER=24 nohup bash scripts/resume_stagev_v2_full_loop.sh > {STAGEV2_RUN_ID}_resume_iter024.nohup.log 2>&1 &
```

Monitor:

```bash
bash scripts/monitor_stagev_v2.sh
squeue -u "$USER"
```

Collect:

```bash
bash scripts/collect_stagev_v2_results.sh
```

## Safety

- Slurm scripts and `sbatch` commands exclude `gpuh01,gpuh14`.
- Production refuses to run unless `CONFIRM_STAGEV2_PRODUCTION=1`.
- `START_ITER=0` refuses to overwrite an existing run directory.
- Selection runs as a Slurm job, not as a CPU-heavy login-node task.

## Git Snapshot

- commit: `{snapshot.get("git_commit", "unknown")}`
- dirty worktree: `{snapshot.get("working_tree_has_changes", False)}`
""",
    )


def validate_package(root: Path) -> dict[str, Any]:
    py_files = [
        "ml_phase/stagev_v2.py",
        "scripts/stagev_v2_select.py",
        "scripts/stagev_v2_update_reward.py",
        "scripts/stagev_v2_smoke.py",
        "scripts/stagev_v2_preflight.py",
    ]
    pyc = run([sys.executable, "-m", "py_compile", *py_files], cwd=root)
    text_rows = normalize_text_files(root)
    bash_rows: list[dict[str, Any]] = []
    bash = shutil.which("bash")
    for path in sorted((root / "scripts").glob("*.sh")):
        rel = path.relative_to(root).as_posix()
        if bash is None:
            bash_rows.append({"path": rel, "status": "skipped", "reason": "bash_not_found"})
            continue
        res = run([bash, "-n", rel], cwd=root)
        combined = (res.stdout + res.stderr).replace("\x00", "").lower()
        if res.returncode != 0 and "windows subsystem for linux has no installed distributions" in combined:
            bash_rows.append({"path": rel, "status": "skipped", "reason": "wsl_bash_unavailable"})
        else:
            bash_rows.append({"path": rel, "status": "pass" if res.returncode == 0 else "fail", "stderr": res.stderr})
    broken_symlinks = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_symlink() and not p.exists()]
    exact_script = (root / "scripts" / "slurm_stagev_v2_exact_array.sh").read_text(encoding="utf-8")
    select_script = (root / "scripts" / "slurm_stagev_v2_select.sh").read_text(encoding="utf-8")
    submit = (root / "scripts" / "submit_stagev_v2_full_loop.sh").read_text(encoding="utf-8")
    return {
        "py_compile_status": "pass" if pyc.returncode == 0 else "fail",
        "py_compile_stderr": pyc.stderr,
        "text_encoding": text_rows,
        "bash_n": bash_rows,
        "broken_symlinks": broken_symlinks,
        "excluded_nodes_present": all("gpuh01,gpuh14" in text for text in [exact_script, select_script, submit]),
        "selection_runs_under_slurm": "slurm_stagev_v2_select.sh" in submit,
    }


def write_checksums(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            rows.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    write_text(root / "SHA256SUMS.txt", "\n".join(rows) + "\n")


def remove_pycache(root: Path) -> None:
    for path in sorted(root.rglob("__pycache__"), reverse=True):
        if path.is_dir():
            shutil.rmtree(path)


def create_archive(root: Path, archive: Path) -> list[str]:
    if archive.exists():
        archive.unlink()
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(root, arcname=root.name)
    with tarfile.open(archive, "r:gz") as tar:
        return tar.getnames()


def main() -> None:
    snapshot = git_snapshot()
    if PACKAGE_ROOT.exists():
        shutil.rmtree(PACKAGE_ROOT)
    PACKAGE_ROOT.mkdir(parents=True, exist_ok=True)
    copy_tree(ROOT / "ml_phase", PACKAGE_ROOT / "ml_phase")
    copy_tree(ROOT / "tests", PACKAGE_ROOT / "tests")
    copy_file(ROOT / "eta_phase_diagram_cuda.py", PACKAGE_ROOT / "eta_phase_diagram_cuda.py")
    copy_file(ROOT / "tfflo_1d_cuda.py", PACKAGE_ROOT / "tfflo_1d_cuda.py")
    for doc in [
        "AGENTS.md",
        "README.md",
        "MODEL_SPEC.md",
        "docs/PROJECT_SUMMARY.md",
        "docs/MODEL_SPEC.md",
        "docs/NUMERICS_SPEC.md",
        "docs/DECISIONS.md",
        "docs/StageV_v1_AcqFun_Learn_prompt.md",
    ]:
        copy_file(ROOT / doc, PACKAGE_ROOT / doc)
    for script in [
        "scripts/stagev_v2_select.py",
        "scripts/stagev_v2_update_reward.py",
        "scripts/stagev_v2_smoke.py",
        "scripts/package_stagev_v2_hpc.py",
    ]:
        copy_file(ROOT / script, PACKAGE_ROOT / script)
    write_configs(PACKAGE_ROOT)
    write_scripts(PACKAGE_ROOT)
    write_readme(PACKAGE_ROOT, snapshot)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_name": PACKAGE_NAME,
        "run_id": STAGEV2_RUN_ID,
        "output_root": STAGEV2_OUTPUT_ROOT,
        "stage": "Stage V-v2",
        "cold_start": True,
        "stageiv_data_used_for_training": False,
        "stagev1_data_used_for_training": False,
        "initial_seed_size": StageV2Config().initial_seed_size,
        "micro_batch_size": StageV2Config().micro_batch_size,
        "max_micro_batches": StageV2Config().max_micro_batches,
        "excluded_nodes": ["gpuh01", "gpuh14"],
        "selection_runs_under_slurm": True,
        **snapshot,
    }
    write_json(PACKAGE_ROOT / "run_manifest.json", manifest)
    validation = validate_package(PACKAGE_ROOT)
    remove_pycache(PACKAGE_ROOT)
    write_json(PACKAGE_ROOT / "package_validation.json", validation)
    write_text(
        PACKAGE_ROOT / "package_validation.md",
        "# Package Validation\n\n"
        f"- py_compile_status: `{validation['py_compile_status']}`\n"
        f"- excluded_nodes_present: `{validation['excluded_nodes_present']}`\n"
        f"- selection_runs_under_slurm: `{validation['selection_runs_under_slurm']}`\n"
        "- bash_n: see `package_validation.json`\n",
    )
    write_checksums(PACKAGE_ROOT)
    tar_listing = create_archive(PACKAGE_ROOT, ARCHIVE_PATH)
    summary = {
        "package_root": str(PACKAGE_ROOT),
        "archive_path": str(ARCHIVE_PATH),
        "archive_size_bytes": ARCHIVE_PATH.stat().st_size,
        "archive_sha256": sha256_file(ARCHIVE_PATH),
        "tar_member_count": len(tar_listing),
        "validation": validation,
    }
    out = ROOT / "reports" / "stagev_v2_hpc_package"
    write_json(out / "stagev_v2_package_summary.json", summary)
    write_text(
        out / "stagev_v2_implementation_report.md",
        "# Stage V-v2 Multi-head Acquisition Implementation Report\n\n"
        "## Repository Audit\n\n"
        "- Stage V-v1 scalar residual selected too much normal/SC and global Sobol support.\n"
        "- Stage V-v2 is a new cold-start package with no Stage IV or Stage V-v1 training initialization.\n\n"
        "## Implemented Architecture\n\n"
        "- Per-boundary heads: `ns`, `uf`, `p0`, `ppi`, `gap`.\n"
        "- Per-boundary reward normalization and independent `lambda_s`.\n"
        "- Automatic `alpha_s` update from boundary-support deficits.\n"
        "- Per-boundary rank normalization before log-sum-exp score combination.\n"
        "- Proposal-source density correction to control global Sobol volume bias.\n"
        "- Slurm-based selection to avoid CPU-heavy login-node work.\n\n"
        "## Exact Output Schema\n\n"
        "- Exact oracle was not modified in this package.\n"
        "- Existing `free_energy_gap_to_normal` is used as the stored `F_SC_minus_F_normal` margin.\n"
        "- Existing topology fields `topology_p0`, `topology_ppi`, and `topology_bulk_gap` are used.\n\n"
        "## Validation\n\n"
        f"- archive: `{summary['archive_path']}`\n"
        f"- sha256: `{summary['archive_sha256']}`\n"
        f"- py_compile_status: `{validation['py_compile_status']}`\n"
        f"- excluded_nodes_present: `{validation['excluded_nodes_present']}`\n"
        f"- selection_runs_under_slurm: `{validation['selection_runs_under_slurm']}`\n\n"
        "## Do Not Claim\n\n"
        "- Do not claim Stage V-v2 has converged before the production run.\n"
        "- Do not claim learned acquisition outperforms v1 before matched-budget comparison.\n"
        "- Do not claim Stage IV or Stage V-v1 data was used for training.\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
