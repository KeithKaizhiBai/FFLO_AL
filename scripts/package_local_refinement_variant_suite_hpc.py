from __future__ import annotations

import hashlib
import csv
import json
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_local_refinement_fixed_point_regression import resolve_variant_config

OUT_ROOT = ROOT / "hpc_packages"
PACKAGE_NAME = "local_refinement_refactor_variant_suite"
RESULT_ARCHIVE = "local_refinement_refactor_variant_suite_results.tar.gz"
FIXED_POINTS = ROOT / "reports" / "local_refinement_refactor" / "stage_00_baseline" / "fixed_point_regression_points.csv"
RUNNABLE_VARIANTS = [
    "baseline",
    "cluster_only",
    "cluster_optional_k3",
    "cluster_optional_k2",
    "cluster_energy_window",
]
COMPARISON_VARIANTS = [variant for variant in RUNNABLE_VARIANTS if variant != "baseline"]
DEFAULT_MAX_CONCURRENT = 32
DEFAULT_POINT_TIME = "02:00:00"
ML_PHASE_OUTPUT_DIRS = (
    "active_runs",
    "datasets",
    "figures",
    "hpc_jobs",
    "models",
    "reports",
)
REPORT_IGNORE_DIRS = {
    "regression_dry_run",
    "regression_gpu_baseline",
    "regression_gpu_instrumented",
    "baseline_vs_instrumented",
    "return_bundle_metadata",
    "imported_results",
}


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree(src: Path, dst: Path, ignore: Iterable[str] = ()) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    ignore_set = set(ignore)

    def ignore_func(_dir: str, names: list[str]) -> set[str]:
        return {name for name in names if name in ignore_set or name == "__pycache__"}

    shutil.copytree(src, dst, ignore=ignore_func)


def _copy_stage_reports(dst: Path) -> None:
    src = ROOT / "reports" / "local_refinement_refactor"
    if not src.exists():
        return
    _copy_tree(src, dst, ignore=REPORT_IGNORE_DIRS)


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


def _fixed_point_count(package_root: Path) -> int:
    fixed_points = package_root / "fixed_points" / "fixed_point_regression_points.csv"
    if not fixed_points.exists():
        return 0
    with fixed_points.open("r", newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def _write_archive_sidecars(archive: Path, package_root: Path) -> None:
    sha256 = _sha256_file(archive)
    archive.with_suffix(archive.suffix + ".sha256").write_text(
        f"{sha256}  {archive.name}\n",
        encoding="utf-8",
        newline="\n",
    )
    metadata = {
        "package_name": PACKAGE_NAME,
        "purpose": "Stage 2-4 local-refinement variant point-wise GPU array regression suite",
        "active_learning": "not_run",
        "archive_name": archive.name,
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": sha256,
        "package_file_count": sum(1 for path in package_root.rglob("*") if path.is_file()),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "return_archive": RESULT_ARCHIVE,
        "scheduler": "slurm_array",
        "expected_tasks": len(RUNNABLE_VARIANTS) * _fixed_point_count(package_root),
    }
    archive.with_suffix(archive.suffix + ".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _shell_roots_preamble() -> str:
    return """SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="${PACKAGE_ROOT:-${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}}"
RUN_ROOT="${RUN_ROOT:-}"
if [ -z "${RUN_ROOT}" ]; then
  if [ -w "${PACKAGE_ROOT}" ]; then
    RUN_ROOT="${PACKAGE_ROOT}/local_refinement_refactor_variant_suite_run"
  elif [ -n "${SCRATCH:-}" ]; then
    RUN_ROOT="${SCRATCH}/local_refinement_refactor_variant_suite_run"
  elif [ -n "${TMPDIR:-}" ]; then
    RUN_ROOT="${TMPDIR}/local_refinement_refactor_variant_suite_run"
  else
    RUN_ROOT="${HOME}/local_refinement_refactor_variant_suite_run"
  fi
fi
mkdir -p "${RUN_ROOT}"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
PROJECT_DIR="${PACKAGE_ROOT}"
export PACKAGE_ROOT RUN_ROOT PROJECT_DIR
"""


def _variant_output_dir(variant: str) -> str:
    return f"reports/local_refinement_refactor/variant_regression/{variant}"


def _read_fixed_point_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
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


def _write_task_config(package_root: Path, fixed_points_path: Path) -> dict[str, Any]:
    point_rows = _read_fixed_point_rows(fixed_points_path)
    validation_points: list[dict[str, Any]] = []
    for point_id, row in enumerate(point_rows):
        validation_points.append({"point_id": point_id, **row})

    variant_rows = [
        {
            "variant": variant,
            "comparison_role": "baseline" if variant == "baseline" else "candidate",
            "config_json": json.dumps(resolve_variant_config(variant), sort_keys=True),
        }
        for variant in RUNNABLE_VARIANTS
    ]
    task_rows: list[dict[str, Any]] = []
    task_id = 0
    for variant in RUNNABLE_VARIANTS:
        for point in validation_points:
            task_rows.append(
                {
                    "task_id": task_id,
                    "variant": variant,
                    **point,
                }
            )
            task_id += 1

    _write_csv(package_root / "config" / "validation_points.csv", validation_points)
    _write_csv(package_root / "config" / "validation_variants.csv", variant_rows)
    _write_csv(package_root / "config" / "task_matrix.csv", task_rows)
    _write_text(
        package_root / "config" / "equivalence_tolerances.json",
        json.dumps(
            {
                "max_q_opt_abs_diff": 1.0e-10,
                "max_delta_opt_abs_diff": 1.0e-10,
                "max_deltaf_abs_diff": 1.0e-8,
            },
            indent=2,
        ),
    )
    return {
        "expected_fixed_points": len(validation_points),
        "expected_tasks": len(task_rows),
        "task_matrix": "config/task_matrix.csv",
        "validation_points": "config/validation_points.csv",
        "validation_variants": "config/validation_variants.csv",
    }


def _slurm_variant_script(variant: str) -> str:
    out_dir = _variant_output_dir(variant)
    return f"""#!/bin/bash
#SBATCH --job-name=lr_{variant[:16]}
#SBATCH --partition=NV_H100
#SBATCH --exclude=gpuh01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

{_shell_roots_preamble()}
PYTHON_BIN="${{PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}}"
CUDA_MODULE="${{CUDA_MODULE:-compiler/cuda/cuda-12.8.1}}"

cd "${{PACKAGE_ROOT}}"

if command -v module >/dev/null 2>&1; then
  module purge || true
  if [ -n "${{CUDA_MODULE}}" ]; then
    module load "${{CUDA_MODULE}}" || true
  fi
fi

mkdir -p "${{RUN_ROOT}}/logs" "${{RUN_ROOT}}/{out_dir}"
ENV_LOG="${{RUN_ROOT}}/logs/{variant}_env_snapshot.txt"

{{
  echo "timestamp=$(date -Is)"
  echo "hostname=$(hostname)"
  echo "package_root=${{PACKAGE_ROOT}}"
  echo "run_root=${{RUN_ROOT}}"
  echo "python_bin=${{PYTHON_BIN}}"
  echo "variant={variant}"
  echo "cuda_visible_devices=${{CUDA_VISIBLE_DEVICES:-}}"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
  "${{PYTHON_BIN}}" - <<'PY'
import sys
import torch
print("python=" + sys.version.replace("\\n", " "))
print("torch_version=" + str(torch.__version__))
print("torch_cuda_available=" + str(torch.cuda.is_available()))
print("torch_cuda_version=" + str(torch.version.cuda))
PY
}} > "${{ENV_LOG}}" 2>&1

"${{PYTHON_BIN}}" - <<'PY' >> "${{ENV_LOG}}" 2>&1
import sys
import torch
try:
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    probe = torch.empty(1, device="cuda")
    probe += 1
    torch.cuda.synchronize()
    print("cuda_runtime_probe=pass")
except Exception as exc:
    print("cuda_runtime_probe=fail")
    print("cuda_runtime_error=" + repr(exc))
    sys.exit(42)
PY

"${{PYTHON_BIN}}" -m py_compile \\
  ml_phase/exact_oracle.py \\
  scripts/run_local_refinement_fixed_point_regression.py \\
  scripts/compare_local_refinement_variants.py \\
  scripts/build_local_refinement_performance_report.py

"${{PYTHON_BIN}}" scripts/run_local_refinement_fixed_point_regression.py \\
  --points-file "${{PACKAGE_ROOT}}/fixed_points/fixed_point_regression_points.csv" \\
  --output-dir "${{RUN_ROOT}}/{out_dir}" \\
  --run-root "${{RUN_ROOT}}" \\
  --device cuda \\
  --variant-name {variant} \\
  --enable-local-box-instrumentation
"""


def _submit_variant_script(variant: str) -> str:
    return f"""#!/bin/bash
set -euo pipefail
{_shell_roots_preamble()}
cd "${{PACKAGE_ROOT}}"
mkdir -p "${{RUN_ROOT}}/logs"
job_id=$(sbatch --parsable scripts/slurm_variant_{variant}.sh)
echo "${{job_id}}" > "${{RUN_ROOT}}/logs/variant_{variant}.jobid"
echo "submitted {variant}: ${{job_id}}"
"""


def _compare_script() -> str:
    commands = []
    for variant in COMPARISON_VARIANTS:
        commands.append(
            f'''"${{PYTHON_BIN}}" scripts/compare_local_refinement_variants.py \\
  --baseline-csv "${{RUN_ROOT}}/{_variant_output_dir("baseline")}/baseline_pointwise.csv" \\
  --candidate-csv "${{RUN_ROOT}}/{_variant_output_dir(variant)}/{variant}_pointwise.csv" \\
  --output-dir "${{RUN_ROOT}}/reports/local_refinement_refactor/variant_regression/comparisons/baseline_vs_{variant}" \\
  --run-root "${{RUN_ROOT}}"
'''
        )
    return """#!/bin/bash
set -euo pipefail
""" + _shell_roots_preamble() + """
PYTHON_BIN="${PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}"
export PYTHON_BIN
cd "${PACKAGE_ROOT}"
""" + "\n".join(commands) + """
echo "variant comparisons ready: ${RUN_ROOT}/reports/local_refinement_refactor/variant_regression/comparisons"
"""


def _collect_script() -> str:
    return """#!/bin/bash
set -euo pipefail
""" + _shell_roots_preamble() + """
RESULT_ARCHIVE="${RESULT_ARCHIVE:-local_refinement_refactor_variant_suite_results.tar.gz}"
RETURN_METADATA_DIR="${RUN_ROOT}/reports/local_refinement_refactor/variant_regression/return_bundle_metadata"
mkdir -p "${RETURN_METADATA_DIR}"
{
  echo "archive=${RUN_ROOT}/${RESULT_ARCHIVE}"
  echo "timestamp=$(date -Is)"
  echo "package_root=${PACKAGE_ROOT}"
  echo "run_root=${RUN_ROOT}"
} > "${RETURN_METADATA_DIR}/return_manifest.txt"
tar -czf "${RUN_ROOT}/${RESULT_ARCHIVE}" \
  -C "${PACKAGE_ROOT}" README.md RUN_MANIFEST.json config fixed_points/fixed_point_regression_points.csv \
  -C "${RUN_ROOT}" logs reports/local_refinement_refactor/variant_regression
echo "return archive: ${RUN_ROOT}/${RESULT_ARCHIVE}"
"""


def _slurm_point_array_script() -> str:
    return """#!/bin/bash
#SBATCH --job-name=lr_var_point
#SBATCH --partition=NV_H100
#SBATCH --exclude=gpuh01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

""" + _shell_roots_preamble() + """
PYTHON_BIN="${PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}"
CUDA_MODULE="${CUDA_MODULE:-compiler/cuda/cuda-12.8.1}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

cd "${PACKAGE_ROOT}"

if command -v module >/dev/null 2>&1; then
  module purge || true
  if [ -n "${CUDA_MODULE}" ]; then
    module load "${CUDA_MODULE}" || true
  fi
fi

mkdir -p "${RUN_ROOT}/logs"
ENV_LOG="${RUN_ROOT}/logs/variant_point_${SLURM_ARRAY_JOB_ID:-manual}_${TASK_ID}_env_snapshot.txt"

{
  echo "timestamp=$(date -Is)"
  echo "hostname=$(hostname)"
  echo "package_root=${PACKAGE_ROOT}"
  echo "run_root=${RUN_ROOT}"
  echo "python_bin=${PYTHON_BIN}"
  echo "slurm_array_job_id=${SLURM_ARRAY_JOB_ID:-}"
  echo "slurm_array_task_id=${TASK_ID}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
  "${PYTHON_BIN}" - <<'PY'
import sys
import torch
print("python=" + sys.version.replace("\\n", " "))
print("torch_version=" + str(torch.__version__))
print("torch_cuda_available=" + str(torch.cuda.is_available()))
print("torch_cuda_version=" + str(torch.version.cuda))
PY
} > "${ENV_LOG}" 2>&1

"${PYTHON_BIN}" - <<'PY' >> "${ENV_LOG}" 2>&1
import sys
import torch
try:
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    probe = torch.empty(1, device="cuda")
    probe += 1
    torch.cuda.synchronize()
    print("cuda_runtime_probe=pass")
except Exception as exc:
    print("cuda_runtime_probe=fail")
    print("cuda_runtime_error=" + repr(exc))
    sys.exit(42)
PY

"${PYTHON_BIN}" -m py_compile \
  ml_phase/exact_oracle.py \
  scripts/run_local_refinement_variant_point.py \
  scripts/aggregate_local_refinement_variant_array_suite.py \
  scripts/compare_local_refinement_variants.py

"${PYTHON_BIN}" scripts/run_local_refinement_variant_point.py \
  --package-root "${PACKAGE_ROOT}" \
  --run-root "${RUN_ROOT}" \
  --task-matrix "${PACKAGE_ROOT}/config/task_matrix.csv" \
  --task-id "${TASK_ID}" \
  --device cuda \
  --enable-local-box-instrumentation
"""


def _postprocess_slurm_script() -> str:
    return """#!/bin/bash
#SBATCH --job-name=lr_variant_post
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail

""" + _shell_roots_preamble() + """
PYTHON_BIN="${PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}"
export PYTHON_BIN
cd "${PACKAGE_ROOT}"

set +e
"${PYTHON_BIN}" scripts/aggregate_local_refinement_variant_array_suite.py \
  --package-root "${PACKAGE_ROOT}" \
  --run-root "${RUN_ROOT}" \
  --task-matrix "${PACKAGE_ROOT}/config/task_matrix.csv" \
  --fail-if-incomplete
aggregate_status=$?
set -e

bash scripts/collect_variant_suite_outputs.sh
exit "${aggregate_status}"
"""


def _workflow_submit_script() -> str:
    return """#!/bin/bash
set -euo pipefail
""" + _shell_roots_preamble() + """
PYTHON_BIN="${PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}"
cd "${PACKAGE_ROOT}"
mkdir -p "${RUN_ROOT}/logs"

"${PYTHON_BIN}" scripts/preflight_local_refinement_variant_suite_hpc.py \
  --package-root "${PACKAGE_ROOT}" \
  --run-root "${RUN_ROOT}" \
  --output-json "${RUN_ROOT}/reports/local_refinement_refactor/variant_regression/preflight.json"

task_count=$("${PYTHON_BIN}" - <<'PY'
import csv
from pathlib import Path
path = Path("config/task_matrix.csv")
with path.open(newline="", encoding="utf-8") as f:
    print(sum(1 for _ in csv.DictReader(f)))
PY
)
last_task=$((task_count - 1))
if [ "${last_task}" -lt 0 ]; then
  echo "empty task matrix" >&2
  exit 2
fi

MAX_CONCURRENT="${MAX_CONCURRENT:-32}"
POINT_TIME="${POINT_TIME:-02:00:00}"
EXCLUDE_NODES="${EXCLUDE_NODES:-gpuh01}"
PARTITION="${PARTITION:-NV_H100}"

array_args=(
  --parsable
  --partition="${PARTITION}"
  --exclude="${EXCLUDE_NODES}"
  --time="${POINT_TIME}"
  --array="0-${last_task}%${MAX_CONCURRENT}"
  --output="${RUN_ROOT}/logs/variant_point_%A_%a.out"
  --error="${RUN_ROOT}/logs/variant_point_%A_%a.err"
)
array_id=$(sbatch "${array_args[@]}" scripts/slurm_variant_point_array.sh)
echo "${array_id}" > "${RUN_ROOT}/logs/variant_point_array.jobid"
echo "submitted point-wise variant array: ${array_id}"
echo "task_count=${task_count}"
echo "max_concurrent=${MAX_CONCURRENT}"
echo "point_time=${POINT_TIME}"

POSTPROCESS_PARTITION="${POSTPROCESS_PARTITION:-${PARTITION}}"
post_args=(--partition="${POSTPROCESS_PARTITION}")
if [ -n "${POSTPROCESS_SBATCH_ARGS:-}" ]; then
  read -r -a extra_post_args <<< "${POSTPROCESS_SBATCH_ARGS}"
  post_args+=("${extra_post_args[@]}")
fi
postprocess_id=$(sbatch --parsable --dependency=afterany:${array_id} "${post_args[@]}" scripts/slurm_variant_suite_postprocess.sh)
echo "${postprocess_id}" > "${RUN_ROOT}/logs/variant_suite_postprocess.jobid"
echo "submitted postprocess afterany:${array_id}: ${postprocess_id}"
echo "run root: ${RUN_ROOT}"
echo "after completion, return ${RUN_ROOT}/local_refinement_refactor_variant_suite_results.tar.gz"
echo "status check: ${PYTHON_BIN} scripts/check_variant_suite_hpc_status.py --package-root ${PACKAGE_ROOT} --run-root ${RUN_ROOT} --query-scheduler"
"""


def _workflow_alias_script(target: str) -> str:
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
from pathlib import Path
import sys


REQUIRED_PATHS = [
    "README.md",
    "RUN_MANIFEST.json",
    "config/variants.json",
    "config/validation_points.csv",
    "config/validation_variants.csv",
    "config/task_matrix.csv",
    "config/equivalence_tolerances.json",
    "code_snapshot/ml_phase/exact_oracle.py",
    "code_snapshot/scripts/run_local_refinement_fixed_point_regression.py",
    "fixed_points/fixed_point_regression_points.csv",
    "eta_phase_diagram_cuda.py",
    "tfflo_1d_cuda.py",
    "ml_phase/exact_oracle.py",
    "reports/local_refinement_refactor",
    "scripts/run_local_refinement_variant_point.py",
    "scripts/aggregate_local_refinement_variant_array_suite.py",
    "scripts/run_local_refinement_fixed_point_regression.py",
    "scripts/compare_local_refinement_variants.py",
    "scripts/build_local_refinement_performance_report.py",
    "scripts/collect_local_refinement_performance_report.sh",
    "scripts/submit_local_refinement_fixed_point_regression.sh",
    "scripts/submit_local_refinement_instrumented_benchmark.sh",
    "scripts/slurm_variant_point_array.sh",
    "scripts/submit_variant_suite_regression_workflow.sh",
    "scripts/collect_variant_suite_outputs.sh",
    "scripts/import_local_refinement_variant_suite_results.py",
    "scripts/check_variant_suite_hpc_status.py",
    "tests/test_local_refinement_variant_suite_package.py",
]
EXPECTED_VARIANTS = [
    "baseline",
    "cluster_only",
    "cluster_optional_k3",
    "cluster_optional_k2",
    "cluster_energy_window",
]


def count_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight the local-refinement variant-suite package.")
    parser.add_argument("--package-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-json", type=Path, default=Path("reports/local_refinement_refactor/variant_regression/preflight.json"))
    parser.add_argument("--expected-fixed-points", type=int, default=32)
    args = parser.parse_args()
    package_root = args.package_root.resolve()
    run_root = args.run_root.resolve()
    failures: list[str] = []
    checked: dict[str, object] = {"package_root": str(package_root), "run_root": str(run_root)}

    for rel_path in REQUIRED_PATHS:
        if not (package_root / rel_path).exists():
            failures.append(f"missing required path: {rel_path}")

    manifest_path = package_root / "RUN_MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checked["package_name"] = manifest.get("package_name")
        checked["variants"] = manifest.get("variants")
        if manifest.get("package_name") != "local_refinement_refactor_variant_suite":
            failures.append(f"manifest package_name mismatch: {manifest.get('package_name')}")
        if manifest.get("active_learning") != "not_run":
            failures.append(f"manifest active_learning should be not_run, got {manifest.get('active_learning')}")
        if manifest.get("variants") != EXPECTED_VARIANTS:
            failures.append(f"manifest variants mismatch: {manifest.get('variants')}")

    fixed_points = package_root / "fixed_points" / "fixed_point_regression_points.csv"
    if fixed_points.exists():
        row_count = count_rows(fixed_points)
        checked["fixed_point_count"] = row_count
        if row_count != args.expected_fixed_points:
            failures.append(f"fixed-point row count mismatch: {row_count} != {args.expected_fixed_points}")

    task_matrix = package_root / "config" / "task_matrix.csv"
    if task_matrix.exists():
        task_count = count_rows(task_matrix)
        checked["task_count"] = task_count
        expected_tasks = int(args.expected_fixed_points) * len(EXPECTED_VARIANTS)
        if task_count != expected_tasks:
            failures.append(f"task-matrix row count mismatch: {task_count} != {expected_tasks}")

    package_root_text = str(package_root)
    if package_root_text not in sys.path:
        sys.path.insert(0, package_root_text)
    for module_name in ["eta_phase_diagram_cuda", "ml_phase.exact_oracle"]:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures.append(f"import check failed for {module_name}: {exc!r}")

    summary = {"status": "pass" if not failures else "fail", "checked": checked, "failures": failures}
    output_json = args.output_json if args.output_json.is_absolute() else run_root / args.output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
"""


def _readme() -> str:
    variants = "\n".join(f"- `{variant}`: `{resolve_variant_config(variant)}`" for variant in RUNNABLE_VARIANTS)
    return f"""# Local Refinement Refactor Variant Suite

This package runs a complete fixed-point GPU regression for the
local-refinement optimization variants. It does not run active learning and
does not append training data.

The suite is submitted as a point-wise Slurm array over:

```text
variant x fixed_point_id
```

This package recomputes `baseline` and all candidate variants from scratch. It
does not merge or depend on any previous partially completed HPC result.

## Variants

{variants}

Stage 5 branch reuse and Stage 6 adaptive boxes are included only as code and
documentation prototypes. They are not submitted as runnable production
variants by this package.

## Output Location

Runtime outputs are written under `RUN_ROOT`.  If `RUN_ROOT` is unset and the
extracted package is writable, scripts default to:

```text
$PACKAGE_ROOT/local_refinement_refactor_variant_suite_run
```

This keeps `logs/`, `reports/`, and the return archive inside the extracted
upload package rather than the repository root or the login directory.

## Run on HPC

```bash
tar -xzf local_refinement_refactor_variant_suite.tar.gz
cd local_refinement_refactor_variant_suite
bash scripts/submit_variant_suite_regression_workflow.sh
```

Runbook-compatible aliases are also included:

```bash
bash scripts/submit_local_refinement_fixed_point_regression.sh
bash scripts/submit_local_refinement_instrumented_benchmark.sh
```

The default submission uses:

```text
MAX_CONCURRENT=32
POINT_TIME=02:00:00
PARTITION=NV_H100
EXCLUDE_NODES=gpuh01
```

Override them if needed:

```bash
MAX_CONCURRENT=48 POINT_TIME=03:00:00 bash scripts/submit_variant_suite_regression_workflow.sh
```

The postprocess job uses `afterany`, not `afterok`, so a return archive is
created even when individual point tasks fail or time out. The archive then
contains `summary/missing_or_failed_tasks.csv` and the Slurm logs needed for
diagnosis.

Manual aggregation is also possible after the array job finishes:

```bash
python scripts/aggregate_local_refinement_variant_array_suite.py --package-root . --run-root "$RUN_ROOT"
bash scripts/collect_variant_suite_outputs.sh
```

If `squeue` no longer shows jobs but the return archive is unclear, inspect
the run root:

```bash
python scripts/check_variant_suite_hpc_status.py --package-root . --run-root "$RUN_ROOT" --query-scheduler
```

Return:

```text
$RUN_ROOT/local_refinement_refactor_variant_suite_results.tar.gz
```

## Local Return Check

After downloading the returned archive, run from the project checkout:

```bash
python scripts/import_local_refinement_variant_suite_results.py local_refinement_refactor_variant_suite_results.tar.gz
```

By default this extracts into `imported_results/` next to the returned archive
and writes `latest_variant_suite_import_manifest.json` there.

The returned archive also includes:

```text
reports/local_refinement_refactor/variant_regression/summary/
reports/local_refinement_refactor/variant_regression/performance_report/
reports/local_refinement_refactor/variant_regression/point_tasks/
```

Key files:

```text
summary/array_suite_status.json
summary/task_status.csv
summary/missing_or_failed_tasks.csv
summary/equivalence_matrix.csv
performance_report/runtime_summary.csv
performance_report/local_box_summary.csv
performance_report/performance_summary.json
performance_report/performance_report.md
decision_log.md
```
"""


def build_package() -> Path:
    if not FIXED_POINTS.exists():
        raise FileNotFoundError(f"Missing fixed-point CSV: {FIXED_POINTS}")
    package_root = OUT_ROOT / PACKAGE_NAME
    if package_root.exists():
        shutil.rmtree(package_root)
    package_root.mkdir(parents=True)

    _copy_tree(ROOT / "ml_phase", package_root / "ml_phase", ignore=ML_PHASE_OUTPUT_DIRS)
    for script in [
        "run_local_refinement_variant_point.py",
        "aggregate_local_refinement_variant_array_suite.py",
        "run_local_refinement_fixed_point_regression.py",
        "compare_local_refinement_variants.py",
        "build_local_refinement_performance_report.py",
        "import_local_refinement_variant_suite_results.py",
        "check_variant_suite_hpc_status.py",
        "audit_local_refinement_stage_reports.py",
        "audit_local_refinement_runbook_tests.py",
    ]:
        _copy_file(ROOT / "scripts" / script, package_root / "scripts" / script)
    _copy_file(
        ROOT / "scripts" / "collect_local_refinement_performance_report.sh",
        package_root / "scripts" / "collect_local_refinement_performance_report.sh",
    )
    _copy_file(ROOT / "eta_phase_diagram_cuda.py", package_root / "eta_phase_diagram_cuda.py")
    _copy_file(ROOT / "tfflo_1d_cuda.py", package_root / "tfflo_1d_cuda.py")
    _copy_file(FIXED_POINTS, package_root / "fixed_points" / "fixed_point_regression_points.csv")
    task_config = _write_task_config(package_root, package_root / "fixed_points" / "fixed_point_regression_points.csv")

    _copy_tree(ROOT / "tests", package_root / "tests")
    _copy_stage_reports(package_root / "reports" / "local_refinement_refactor")
    _copy_tree(ROOT / "ml_phase", package_root / "code_snapshot" / "ml_phase", ignore=ML_PHASE_OUTPUT_DIRS)
    _copy_tree(ROOT / "scripts", package_root / "code_snapshot" / "scripts")
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

    variant_configs = {variant: resolve_variant_config(variant) for variant in RUNNABLE_VARIANTS}
    manifest = {
        "package_name": PACKAGE_NAME,
        "purpose": "Stage 2-4 local-refinement variant point-wise GPU array regression suite",
        "active_learning": "not_run",
        "fixed_points": "fixed_points/fixed_point_regression_points.csv",
        "expected_fixed_points": task_config["expected_fixed_points"],
        "expected_tasks": task_config["expected_tasks"],
        "task_matrix": task_config["task_matrix"],
        "validation_points": task_config["validation_points"],
        "validation_variants": task_config["validation_variants"],
        "variants": RUNNABLE_VARIANTS,
        "comparison_variants": COMPARISON_VARIANTS,
        "variant_configs": variant_configs,
        "scheduler": "slurm_array",
        "array_dimension": "variant x fixed_point_id",
        "default_max_concurrent": DEFAULT_MAX_CONCURRENT,
        "default_point_time": DEFAULT_POINT_TIME,
        "package_layout": [
            "README.md",
            "RUN_MANIFEST.json",
            "config/",
            "code_snapshot/",
            "scripts/",
            "tests/",
            "reports/",
        ],
        "output_root": "reports/local_refinement_refactor/variant_regression",
        "workflow_submit": "scripts/submit_variant_suite_regression_workflow.sh",
        "array_worker": "scripts/run_local_refinement_variant_point.py",
        "array_aggregator": "scripts/aggregate_local_refinement_variant_array_suite.py",
        "hpc_status_check": "scripts/check_variant_suite_hpc_status.py",
        "workflow_aliases": {
            "scripts/submit_local_refinement_fixed_point_regression.sh": "scripts/submit_variant_suite_regression_workflow.sh",
            "scripts/submit_local_refinement_instrumented_benchmark.sh": "scripts/submit_variant_suite_regression_workflow.sh",
        },
        "return_archive": RESULT_ARCHIVE,
        "writable_run_root_env": "RUN_ROOT",
        "excluded_nodes_default": "gpuh01",
        "notes": [
            "This array package recomputes baseline and all candidate variants from scratch.",
            "Postprocess uses afterany so a diagnostic return archive is produced even when point tasks fail.",
            "Stage 5 branch reuse is not integrated into production loop.",
            "Stage 6 adaptive boxes are diagnostics-only and not a runnable variant.",
        ],
    }
    _write_text(package_root / "RUN_MANIFEST.json", json.dumps(manifest, indent=2))
    _write_text(package_root / "config" / "variants.json", json.dumps(variant_configs, indent=2))
    _write_text(package_root / "README.md", _readme())
    _write_text(package_root / "scripts" / "preflight_local_refinement_variant_suite_hpc.py", _preflight_script())
    _write_text(package_root / "scripts" / "slurm_variant_point_array.sh", _slurm_point_array_script())
    _write_text(package_root / "scripts" / "collect_variant_suite_outputs.sh", _collect_script())
    _write_text(package_root / "scripts" / "slurm_variant_suite_postprocess.sh", _postprocess_slurm_script())
    _write_text(package_root / "scripts" / "submit_variant_suite_regression_workflow.sh", _workflow_submit_script())
    _write_text(
        package_root / "scripts" / "submit_local_refinement_fixed_point_regression.sh",
        _workflow_alias_script("submit_variant_suite_regression_workflow.sh"),
    )
    _write_text(
        package_root / "scripts" / "submit_local_refinement_instrumented_benchmark.sh",
        _workflow_alias_script("submit_variant_suite_regression_workflow.sh"),
    )
    _write_text(
        package_root / "run_full_variant_array_suite.sh",
        _workflow_alias_script("scripts/submit_variant_suite_regression_workflow.sh"),
    )
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
