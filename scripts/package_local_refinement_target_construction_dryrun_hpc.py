from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "hpc_packages"
PACKAGE_NAME = "local_refinement_target_construction_dryrun"
RESULT_ARCHIVE = "local_refinement_target_construction_dryrun_results.tar.gz"
FIXED_POINTS = ROOT / "reports" / "local_refinement_refactor" / "stage_00_baseline" / "fixed_point_regression_points.csv"
RUNNABLE_VARIANTS = [
    "baseline",
    "cluster_only",
    "rank_and_cap_k3",
    "rank_and_cap_k2",
    "rank_and_cap_energy_window",
]
DEFAULT_MAX_CONCURRENT = 16
DEFAULT_POINT_TIME = "02:00:00"


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree(src: Path, dst: Path, ignore: Iterable[str] = ()) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    ignore_set = set(ignore)

    def ignore_func(_dir: str, names: list[str]) -> set[str]:
        return {
            name
            for name in names
            if name in ignore_set
            or name == "__pycache__"
            or name.endswith(".pyc")
            or name in {"active_runs", "datasets", "figures", "hpc_jobs", "models", "reports"}
        }

    shutil.copytree(src, dst, ignore=ignore_func)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _normalize_shell_scripts(root: Path) -> None:
    for path in root.rglob("*.sh"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        path.write_text(text.replace("\r\n", "\n").replace("\r", "\n"), encoding="utf-8", newline="\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8", newline="\n")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_validation_config(package_root: Path) -> dict[str, Any]:
    source_rows = _read_csv_rows(FIXED_POINTS)
    validation_rows: list[dict[str, Any]] = []
    for point_id, row in enumerate(source_rows):
        validation_rows.append({"point_id": point_id, **row})
    _write_csv(package_root / "config" / "validation_points.csv", validation_rows)
    _write_csv(
        package_root / "config" / "validation_variants.csv",
        [{"variant": variant, "role": "baseline" if variant == "baseline" else "candidate"} for variant in RUNNABLE_VARIANTS],
    )
    return {
        "expected_fixed_points": len(validation_rows),
        "expected_variants": len(RUNNABLE_VARIANTS),
        "expected_point_tasks": len(validation_rows),
        "validation_points": "config/validation_points.csv",
        "validation_variants": "config/validation_variants.csv",
    }


def _shell_preamble() -> str:
    return """SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="${PACKAGE_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-}"
if [ -z "${RUN_ROOT}" ]; then
  if [ -w "${PACKAGE_ROOT}" ]; then
    RUN_ROOT="${PACKAGE_ROOT}/local_refinement_target_construction_dryrun_run"
  elif [ -n "${SCRATCH:-}" ]; then
    RUN_ROOT="${SCRATCH}/local_refinement_target_construction_dryrun_run"
  elif [ -n "${TMPDIR:-}" ]; then
    RUN_ROOT="${TMPDIR}/local_refinement_target_construction_dryrun_run"
  else
    RUN_ROOT="${HOME}/local_refinement_target_construction_dryrun_run"
  fi
fi
mkdir -p "${RUN_ROOT}"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
export PACKAGE_ROOT RUN_ROOT
"""


def _slurm_point_script() -> str:
    return f"""#!/bin/bash
#SBATCH --job-name=lr_target_dry
#SBATCH --partition=NV_H100
#SBATCH --exclude=gpuh01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time={DEFAULT_POINT_TIME}
#SBATCH --output=%x-%A_%a.out

set -euo pipefail
{_shell_preamble()}
PYTHON_BIN="${{PYTHON_BIN:-python}}"
TASK_ID="${{SLURM_ARRAY_TASK_ID:-0}}"
mkdir -p "${{RUN_ROOT}}/logs"
env_file="${{RUN_ROOT}}/logs/point_${{TASK_ID}}_env_snapshot.txt"
{{
  echo "timestamp=$(date -Is)"
  echo "hostname=$(hostname)"
  echo "package_root=${{PACKAGE_ROOT}}"
  echo "run_root=${{RUN_ROOT}}"
  echo "point_id=${{TASK_ID}}"
  echo "cuda_visible_devices=${{CUDA_VISIBLE_DEVICES:-}}"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
  "${{PYTHON_BIN}}" - <<'PY'
import sys, torch
print("python=" + sys.version.replace("\\n", " "))
print("torch_version=" + getattr(torch, "__version__", "unknown"))
print("torch_cuda_available=" + str(torch.cuda.is_available()))
print("torch_cuda_version=" + str(getattr(torch.version, "cuda", "")))
PY
}} > "${{env_file}}" 2>&1
cd "${{PACKAGE_ROOT}}"
"${{PYTHON_BIN}}" scripts/run_local_refinement_target_construction_point.py \\
  --package-root "${{PACKAGE_ROOT}}" \\
  --run-root "${{RUN_ROOT}}" \\
  --point-id "${{TASK_ID}}" \\
  --device cuda
"""


def _postprocess_script() -> str:
    return """#!/bin/bash
#SBATCH --job-name=lr_target_post
#SBATCH --exclude=gpuh01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
""" + _shell_preamble() + """
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "${PACKAGE_ROOT}"
"${PYTHON_BIN}" scripts/aggregate_local_refinement_target_construction_dryrun.py \\
  --package-root "${PACKAGE_ROOT}" \\
  --run-root "${RUN_ROOT}"
echo "return_archive=${RUN_ROOT}/local_refinement_target_construction_dryrun_results.tar.gz"
"""


def _submit_script(expected_points: int) -> str:
    last_task = max(0, int(expected_points) - 1)
    return f"""#!/bin/bash
set -euo pipefail
{_shell_preamble()}
PYTHON_BIN="${{PYTHON_BIN:-python}}"
PARTITION="${{PARTITION:-NV_H100}}"
EXCLUDE_NODES="${{EXCLUDE_NODES:-gpuh01}}"
MAX_CONCURRENT="${{MAX_CONCURRENT:-{DEFAULT_MAX_CONCURRENT}}}"
POINT_TIME="${{POINT_TIME:-{DEFAULT_POINT_TIME}}}"
mkdir -p "${{RUN_ROOT}}/logs"
cd "${{PACKAGE_ROOT}}"
"${{PYTHON_BIN}}" scripts/preflight_target_construction_dryrun_hpc.py \\
  --package-root "${{PACKAGE_ROOT}}" \\
  --run-root "${{RUN_ROOT}}" \\
  --output-json "${{RUN_ROOT}}/logs/preflight.json"
array_id=$(sbatch --parsable \\
  --partition="${{PARTITION}}" \\
  --exclude="${{EXCLUDE_NODES}}" \\
  --time="${{POINT_TIME}}" \\
  --array="0-{last_task}%${{MAX_CONCURRENT}}" \\
  scripts/slurm_target_construction_point_array.sh)
echo "${{array_id}}" > "${{RUN_ROOT}}/logs/target_construction_point_array.jobid"
post_id=$(sbatch --parsable \\
  --exclude="${{EXCLUDE_NODES}}" \\
  --dependency=afterany:${{array_id}} \\
  scripts/slurm_target_construction_postprocess.sh)
echo "${{post_id}}" > "${{RUN_ROOT}}/logs/target_construction_postprocess.jobid"
echo "submitted point array: ${{array_id}}"
echo "submitted postprocess afterany:${{array_id}}: ${{post_id}}"
echo "run root: ${{RUN_ROOT}}"
echo "return archive: ${{RUN_ROOT}}/{RESULT_ARCHIVE}"
echo "status: ${{PYTHON_BIN}} scripts/check_target_construction_dryrun_hpc_status.py --package-root ${{PACKAGE_ROOT}} --run-root ${{RUN_ROOT}} --query-scheduler"
"""


def _alias_script(target: str) -> str:
    return f"""#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
exec bash "${{SCRIPT_DIR}}/{target}" "$@"
"""


def _preflight_script() -> str:
    return """from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from pathlib import Path


REQUIRED_PATHS = [
    "README.md",
    "RUN_MANIFEST.json",
    "config/validation_points.csv",
    "config/validation_variants.csv",
    "eta_phase_diagram_cuda.py",
    "tfflo_1d_cuda.py",
    "ml_phase/exact_oracle.py",
    "scripts/run_local_refinement_target_construction_point.py",
    "scripts/aggregate_local_refinement_target_construction_dryrun.py",
    "scripts/check_target_construction_dryrun_hpc_status.py",
    "scripts/slurm_target_construction_point_array.sh",
    "scripts/slurm_target_construction_postprocess.sh",
    "scripts/submit_target_construction_dryrun_workflow.sh",
]


def count_rows(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight target-construction dry-run HPC package.")
    parser.add_argument("--package-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-json", type=Path, default=Path("preflight.json"))
    parser.add_argument("--expected-points", type=int, default=32)
    args = parser.parse_args()
    package_root = args.package_root.resolve()
    run_root = args.run_root.resolve()
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    failures: list[str] = []

    for rel in REQUIRED_PATHS:
        if not (package_root / rel).exists():
            failures.append(f"missing required path: {rel}")

    points = package_root / "config" / "validation_points.csv"
    if points.exists():
        n = count_rows(points)
        if n != int(args.expected_points):
            failures.append(f"validation point count mismatch: {n} != {args.expected_points}")

    for path in package_root.rglob("*.sh"):
        raw = path.read_bytes()
        rel = path.relative_to(package_root).as_posix()
        if b"\\r\\n" in raw or b"\\r" in raw:
            failures.append(f"shell script is not LF-only: {rel}")
        if path.name.startswith("slurm_") and b"gpuh01" not in raw:
            failures.append(f"slurm script does not mention gpuh01 exclusion: {rel}")

    for module_name in ["eta_phase_diagram_cuda", "ml_phase.exact_oracle"]:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures.append(f"import failed for {module_name}: {exc!r}")

    summary = {
        "status": "pass" if not failures else "fail",
        "package_root": str(package_root),
        "run_root": str(run_root),
        "failures": failures,
    }
    output_json = args.output_json if args.output_json.is_absolute() else run_root / args.output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
"""


def _readme(expected_points: int) -> str:
    variants = "\n".join(f"- `{variant}`" for variant in RUNNABLE_VARIANTS)
    return f"""# Local Refinement Target-Construction Dry-Run Package

This is a complete HPC upload package for the 32 fixed-point
target-construction-only gate.

It runs:

```text
coarse scan -> q expansion -> candidate detection -> optional clustering ->
risk annotation -> energy-window marking -> final target selection
```

It does not run local refinement boxes and does not run active learning.

## Variants

{variants}

Each Slurm array task computes one fixed point and applies all variants to the
same shared coarse/q-expansion candidate set. This avoids 32 x 5 repeated
coarse scans.

## Defaults

```text
points: {expected_points}
partition: NV_H100
max_concurrent: {DEFAULT_MAX_CONCURRENT}
point_time: {DEFAULT_POINT_TIME}
excluded nodes: gpuh01
run root: $PACKAGE_ROOT/local_refinement_target_construction_dryrun_run
```

## Run

```bash
tar -xzf {PACKAGE_NAME}.tar.gz
cd {PACKAGE_NAME}
bash scripts/submit_target_construction_dryrun_workflow.sh
```

Status:

```bash
python scripts/check_target_construction_dryrun_hpc_status.py --package-root . --run-root "$RUN_ROOT" --query-scheduler
```

Return:

```text
$RUN_ROOT/{RESULT_ARCHIVE}
```

Important outputs inside the return archive:

```text
reports/local_refinement_refactor/target_construction_dryrun/tables/target_construction_by_point.csv
reports/local_refinement_refactor/target_construction_dryrun/tables/target_construction_candidates.csv
reports/local_refinement_refactor/target_construction_dryrun/tables/target_construction_summary.csv
reports/local_refinement_refactor/target_construction_dryrun/summary/target_construction_gate_status.json
reports/local_refinement_refactor/target_construction_dryrun/decision_log.md
```
"""


def _write_archive_sidecars(archive: Path, package_root: Path) -> None:
    sha = _sha256_file(archive)
    archive.with_suffix(archive.suffix + ".sha256").write_text(f"{sha}  {archive.name}\n", encoding="utf-8", newline="\n")
    metadata = {
        "package_name": PACKAGE_NAME,
        "archive_name": archive.name,
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": sha,
        "package_file_count": sum(1 for path in package_root.rglob("*") if path.is_file()),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "return_archive": RESULT_ARCHIVE,
        "excluded_nodes_default": "gpuh01",
    }
    archive.with_suffix(archive.suffix + ".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8", newline="\n")


def build_package() -> Path:
    if not FIXED_POINTS.exists():
        raise FileNotFoundError(f"missing fixed-point CSV: {FIXED_POINTS}")
    package_root = OUT_ROOT / PACKAGE_NAME
    if package_root.exists():
        shutil.rmtree(package_root)
    package_root.mkdir(parents=True)

    _copy_tree(ROOT / "ml_phase", package_root / "ml_phase")
    _copy_file(ROOT / "eta_phase_diagram_cuda.py", package_root / "eta_phase_diagram_cuda.py")
    _copy_file(ROOT / "tfflo_1d_cuda.py", package_root / "tfflo_1d_cuda.py")
    for script in [
        "run_local_refinement_target_construction_point.py",
        "aggregate_local_refinement_target_construction_dryrun.py",
        "check_target_construction_dryrun_hpc_status.py",
        "run_local_refinement_fixed_point_regression.py",
    ]:
        _copy_file(ROOT / "scripts" / script, package_root / "scripts" / script)
    for doc in [
        "PROJECT_SUMMARY.md",
        "MODEL_SPEC.md",
        "NUMERICS_SPEC.md",
        "DECISIONS.md",
        "LOCAL_REFINEMENT_REFACTOR_STATUS.md",
        "LOCAL_REFINEMENT_REFACTOR_DECISION_LOG.md",
    ]:
        src = ROOT / "docs" / doc
        if src.exists():
            _copy_file(src, package_root / "docs" / doc)
    if (ROOT / "MODEL_SPEC.md").exists():
        _copy_file(ROOT / "MODEL_SPEC.md", package_root / "MODEL_SPEC.md")
    config = _write_validation_config(package_root)

    manifest = {
        "package_name": PACKAGE_NAME,
        "purpose": "32 fixed-point local-refinement target-construction-only dry-run gate",
        "active_learning": "not_run",
        "local_box_scan": "not_run",
        "fixed_points": config["validation_points"],
        "expected_fixed_points": config["expected_fixed_points"],
        "expected_point_tasks": config["expected_point_tasks"],
        "variants": RUNNABLE_VARIANTS,
        "scheduler": "slurm_array",
        "array_dimension": "fixed_point_id",
        "default_max_concurrent": DEFAULT_MAX_CONCURRENT,
        "default_point_time": DEFAULT_POINT_TIME,
        "excluded_nodes_default": "gpuh01",
        "return_archive": RESULT_ARCHIVE,
        "workflow_submit": "scripts/submit_target_construction_dryrun_workflow.sh",
        "worker": "scripts/run_local_refinement_target_construction_point.py",
        "aggregator": "scripts/aggregate_local_refinement_target_construction_dryrun.py",
        "status_checker": "scripts/check_target_construction_dryrun_hpc_status.py",
    }
    _write_text(package_root / "RUN_MANIFEST.json", json.dumps(manifest, indent=2))
    _write_text(package_root / "README.md", _readme(config["expected_fixed_points"]))
    _write_text(package_root / "scripts" / "preflight_target_construction_dryrun_hpc.py", _preflight_script())
    _write_text(package_root / "scripts" / "slurm_target_construction_point_array.sh", _slurm_point_script())
    _write_text(package_root / "scripts" / "slurm_target_construction_postprocess.sh", _postprocess_script())
    _write_text(package_root / "scripts" / "submit_target_construction_dryrun_workflow.sh", _submit_script(config["expected_fixed_points"]))
    _write_text(package_root / "run_target_construction_dryrun.sh", _alias_script("scripts/submit_target_construction_dryrun_workflow.sh"))
    _normalize_shell_scripts(package_root)

    archive = OUT_ROOT / f"{PACKAGE_NAME}.tar.gz"
    if archive.exists():
        archive.unlink()
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(package_root, arcname=PACKAGE_NAME)
    _write_archive_sidecars(archive, package_root)
    return archive


def main() -> None:
    archive = build_package()
    print(f"Wrote package: {archive}")


if __name__ == "__main__":
    main()
