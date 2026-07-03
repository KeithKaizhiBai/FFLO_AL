from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "hpc_packages"
PACKAGE_NAME = "local_refinement_refactor_stage01_instrumentation"
FIXED_POINTS = ROOT / "reports" / "local_refinement_refactor" / "stage_00_baseline" / "fixed_point_regression_points.csv"
ML_PHASE_OUTPUT_DIRS = (
    "active_runs",
    "datasets",
    "figures",
    "hpc_jobs",
    "models",
    "reports",
)


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


def _write_archive_sidecars(archive: Path, package_root: Path) -> None:
    sha256 = _sha256_file(archive)
    sidecar = archive.with_suffix(archive.suffix + ".sha256")
    sidecar.write_text(f"{sha256}  {archive.name}\n", encoding="utf-8", newline="\n")
    metadata = {
        "package_name": PACKAGE_NAME,
        "purpose": "Stage 1 local-box instrumentation fixed-point GPU regression",
        "active_learning": "not_run",
        "archive": archive.name,
        "archive_name": archive.name,
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": sha256,
        "package_file_count": sum(1 for path in package_root.rglob("*") if path.is_file()),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "validation_hint": "python scripts/validate_local_refinement_hpc_package.py",
    }
    archive.with_suffix(archive.suffix + ".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _shell_roots_preamble() -> str:
    return """SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="${PACKAGE_ROOT:-${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}}"
RUN_ROOT="${RUN_ROOT:-}"
if [ -z "${RUN_ROOT}" ]; then
  if [ -w "${PACKAGE_ROOT}" ]; then
    RUN_ROOT="${PACKAGE_ROOT}/local_refinement_refactor_stage1_run"
  elif [ -n "${SCRATCH:-}" ]; then
    RUN_ROOT="${SCRATCH}/local_refinement_refactor_stage1_run"
  elif [ -n "${TMPDIR:-}" ]; then
    RUN_ROOT="${TMPDIR}/local_refinement_refactor_stage1_run"
  else
    RUN_ROOT="${HOME}/local_refinement_refactor_stage1_run"
  fi
fi
mkdir -p "${RUN_ROOT}"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
PROJECT_DIR="${PACKAGE_ROOT}"
export PACKAGE_ROOT RUN_ROOT PROJECT_DIR
"""


def _slurm_script(instrumented: bool) -> str:
    name = "instrumented" if instrumented else "baseline"
    flag = "--enable-local-box-instrumentation" if instrumented else ""
    out_dir = f"reports/local_refinement_refactor/stage_01_instrumentation/regression_gpu_{name}"
    if not instrumented:
        out_dir = "reports/local_refinement_refactor/stage_00_baseline/regression_gpu_baseline"
    return f"""#!/bin/bash
#SBATCH --job-name=lr_{name}
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

mkdir -p "${{RUN_ROOT}}/logs" "${{RUN_ROOT}}/reports/local_refinement_refactor/stage_00_baseline" "${{RUN_ROOT}}/reports/local_refinement_refactor/stage_01_instrumentation"
ENV_LOG="${{RUN_ROOT}}/logs/{name}_regression_env_snapshot.txt"

{{
  echo "timestamp=$(date -Is)"
  echo "hostname=$(hostname)"
  echo "package_root=${{PACKAGE_ROOT}}"
  echo "run_root=${{RUN_ROOT}}"
  echo "python_bin=${{PYTHON_BIN}}"
  echo "variant={name}"
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
  scripts/verify_local_refinement_stage1_gate.py \\
  scripts/collect_local_refinement_stage1_outputs.py \\
  scripts/validate_local_refinement_hpc_package.py \\
  scripts/preflight_local_refinement_stage1_hpc.py

"${{PYTHON_BIN}}" scripts/run_local_refinement_fixed_point_regression.py \\
  --points-file "${{PACKAGE_ROOT}}/fixed_points/fixed_point_regression_points.csv" \\
  --output-dir "${{RUN_ROOT}}/{out_dir}" \\
  --run-root "${{RUN_ROOT}}" \\
  --device cuda \\
  --variant-name baseline \\
  {flag}
"""


def _submit_script(target: str) -> str:
    return f"""#!/bin/bash
set -euo pipefail
{_shell_roots_preamble()}
cd "${{PACKAGE_ROOT}}"
mkdir -p "${{RUN_ROOT}}/logs"
job_id=$(sbatch --parsable scripts/slurm_{target}.sh)
echo "${{job_id}}" > "${{RUN_ROOT}}/logs/{target}.jobid"
echo "submitted {target}: ${{job_id}}"
"""


def _compare_script() -> str:
    return """#!/bin/bash
set -euo pipefail
""" + _shell_roots_preamble() + """
PYTHON_BIN="${PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}"
cd "${PACKAGE_ROOT}"

"${PYTHON_BIN}" scripts/compare_local_refinement_variants.py \
  --baseline-csv "${RUN_ROOT}/reports/local_refinement_refactor/stage_00_baseline/regression_gpu_baseline/baseline_pointwise.csv" \
  --candidate-csv "${RUN_ROOT}/reports/local_refinement_refactor/stage_01_instrumentation/regression_gpu_instrumented/baseline_pointwise.csv" \
  --output-dir "${RUN_ROOT}/reports/local_refinement_refactor/stage_01_instrumentation/baseline_vs_instrumented" \
  --run-root "${RUN_ROOT}"

echo "comparison ready: ${RUN_ROOT}/reports/local_refinement_refactor/stage_01_instrumentation/baseline_vs_instrumented"
"""


def _verify_script() -> str:
    return """#!/bin/bash
set -euo pipefail
""" + _shell_roots_preamble() + """
PYTHON_BIN="${PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}"
cd "${PACKAGE_ROOT}"

"${PYTHON_BIN}" scripts/verify_local_refinement_stage1_gate.py --run-root "${RUN_ROOT}"

echo "stage1 gate status: ${RUN_ROOT}/reports/local_refinement_refactor/stage_01_instrumentation/baseline_vs_instrumented/stage1_gate_status.json"
"""


def _collect_script() -> str:
    return """#!/bin/bash
set -euo pipefail
""" + _shell_roots_preamble() + """
PYTHON_BIN="${PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}"
RESULT_ARCHIVE="${RESULT_ARCHIVE:-local_refinement_refactor_stage1_regression_results.tar.gz}"

cd "${PACKAGE_ROOT}"

"${PYTHON_BIN}" scripts/collect_local_refinement_stage1_outputs.py \
  --package-root "${PACKAGE_ROOT}" \
  --run-root "${RUN_ROOT}" \
  --archive "${RESULT_ARCHIVE}"
"""


def _postprocess_slurm_script() -> str:
    return """#!/bin/bash
#SBATCH --job-name=lr_stage1_gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail

""" + _shell_roots_preamble() + """
PYTHON_BIN="${PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}"

cd "${PACKAGE_ROOT}"

mkdir -p "${RUN_ROOT}/logs"
{
  echo "timestamp=$(date -Is)"
  echo "hostname=$(hostname)"
  echo "package_root=${PACKAGE_ROOT}"
  echo "run_root=${RUN_ROOT}"
  echo "python_bin=${PYTHON_BIN}"
  echo "job=stage1_postprocess"
} > "${RUN_ROOT}/logs/stage1_postprocess_env_snapshot.txt"

bash scripts/compare_stage1_regression.sh
bash scripts/verify_stage1_gate.sh
bash scripts/collect_stage1_regression_outputs.sh
"""


def _workflow_submit_script() -> str:
    return """#!/bin/bash
set -euo pipefail
""" + _shell_roots_preamble() + """
PYTHON_BIN="${PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}"
cd "${PACKAGE_ROOT}"
mkdir -p "${RUN_ROOT}/logs"

"${PYTHON_BIN}" scripts/preflight_local_refinement_stage1_hpc.py \
  --package-root "${PACKAGE_ROOT}" \
  --run-root "${RUN_ROOT}" \
  --output-json "${RUN_ROOT}/reports/local_refinement_refactor/stage_01_instrumentation/stage1_runtime_preflight.json"

baseline_id=$(sbatch --parsable scripts/slurm_stage0_baseline_regression.sh)
echo "${baseline_id}" > "${RUN_ROOT}/logs/stage0_baseline_regression.jobid"
echo "submitted stage0 baseline regression: ${baseline_id}"

instrumented_id=$(sbatch --parsable scripts/slurm_stage1_instrumented_regression.sh)
echo "${instrumented_id}" > "${RUN_ROOT}/logs/stage1_instrumented_regression.jobid"
echo "submitted stage1 instrumented regression: ${instrumented_id}"

post_args=()
if [ -n "${POSTPROCESS_SBATCH_ARGS:-}" ]; then
  read -r -a post_args <<< "${POSTPROCESS_SBATCH_ARGS}"
fi
postprocess_id=$(sbatch --parsable --dependency=afterok:${baseline_id}:${instrumented_id} "${post_args[@]}" scripts/slurm_stage1_postprocess.sh)
echo "${postprocess_id}" > "${RUN_ROOT}/logs/stage1_postprocess.jobid"
echo "submitted stage1 postprocess afterok:${baseline_id}:${instrumented_id}: ${postprocess_id}"
echo "run root: ${RUN_ROOT}"
echo "after completion, return ${RUN_ROOT}/local_refinement_refactor_stage1_regression_results.tar.gz"
"""


def _workflow_alias_script(target: str) -> str:
    return f"""#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
exec bash "${{SCRIPT_DIR}}/{target}" "$@"
"""


def _readme() -> str:
    return """# Local Refinement Refactor Stage 1 Instrumentation Package

This package validates logging-only local-box instrumentation on the fixed-point
regression set. It does not run active learning and does not append training
data.

## Run on HPC

By default, runtime outputs are written under the extracted package directory:

```bash
tar -xzf local_refinement_refactor_stage01_instrumentation.tar.gz
cd local_refinement_refactor_stage01_instrumentation
export RUN_ROOT="${RUN_ROOT:-$PWD/local_refinement_refactor_stage1_run}"
mkdir -p "$RUN_ROOT"
```

```bash
python scripts/preflight_local_refinement_stage1_hpc.py --package-root . --run-root "$RUN_ROOT"
bash scripts/submit_stage1_regression_workflow.sh
```

Runbook-compatible aliases are also included:

```bash
bash scripts/submit_local_refinement_fixed_point_regression.sh
bash scripts/submit_local_refinement_instrumented_benchmark.sh
```

If `RUN_ROOT` is unset, scripts write under
`$PACKAGE_ROOT/local_refinement_refactor_stage1_run` when the extracted package
root is writable.  If the package root is not writable, scripts fall back to
`$SCRATCH/local_refinement_refactor_stage1_run`,
`$TMPDIR/local_refinement_refactor_stage1_run`, or
`$HOME/local_refinement_refactor_stage1_run`.

The GPU exact-regression Slurm scripts submit to `NV_H100` and exclude
`gpuh01` by default because that node has shown an incompatible NVIDIA driver
for the current PyTorch/CUDA environment.  Each exact job also performs a CUDA
runtime tensor-allocation probe before running the fixed-point solver.

If the cluster requires an explicit partition for the postprocess job:

```bash
POSTPROCESS_SBATCH_ARGS="--partition=NV_H100" bash scripts/submit_stage1_regression_workflow.sh
```

Alternatively, submit each step manually:

```bash
bash scripts/submit_stage0_baseline_regression.sh
bash scripts/submit_stage1_instrumented_regression.sh
```

After both jobs complete:

```bash
bash scripts/compare_stage1_regression.sh
bash scripts/verify_stage1_gate.sh
bash scripts/collect_stage1_regression_outputs.sh
```

## Pass Criteria

- `phase_candidate` mismatch count = 0
- `trusted_exact` mismatch count = 0
- `training_eligible_exact` mismatch count = 0
- `q_unresolved` mismatch count = 0
- `delta_unresolved` mismatch count = 0
- `rerun_required` mismatch count = 0

The instrumented job should additionally write:

```text
$RUN_ROOT/reports/local_refinement_refactor/stage_01_instrumentation/regression_gpu_instrumented/baseline_local_box_timing.csv
```
"""


def build_package() -> Path:
    if not FIXED_POINTS.exists():
        raise FileNotFoundError(f"Missing fixed-point CSV: {FIXED_POINTS}")
    package_root = OUT_ROOT / PACKAGE_NAME
    if package_root.exists():
        shutil.rmtree(package_root)
    package_root.mkdir(parents=True)

    # Runnable package content.
    _copy_tree(ROOT / "ml_phase", package_root / "ml_phase", ignore=ML_PHASE_OUTPUT_DIRS)
    scripts_to_copy = [
        "run_local_refinement_fixed_point_regression.py",
        "compare_local_refinement_variants.py",
        "verify_local_refinement_stage1_gate.py",
        "collect_local_refinement_stage1_outputs.py",
        "validate_local_refinement_hpc_package.py",
        "preflight_local_refinement_stage1_hpc.py",
    ]
    for script in scripts_to_copy:
        _copy_file(ROOT / "scripts" / script, package_root / "scripts" / script)
    _copy_file(ROOT / "eta_phase_diagram_cuda.py", package_root / "eta_phase_diagram_cuda.py")
    _copy_file(ROOT / "tfflo_1d_cuda.py", package_root / "tfflo_1d_cuda.py")
    _copy_file(FIXED_POINTS, package_root / "fixed_points" / "fixed_point_regression_points.csv")

    # Documentation and reproducibility snapshot.
    for doc in [
        "LOCAL_REFINEMENT_REFACTOR_MASTER_PLAN.md",
        "LOCAL_REFINEMENT_REFACTOR_DECISION_LOG.md",
        "LOCAL_REFINEMENT_REFACTOR_STATUS.md",
        "LOCAL_REFINEMENT_REFACTOR_CALL_GRAPH.md",
        "PROJECT_SUMMARY.md",
    ]:
        src = ROOT / "docs" / doc
        if src.exists():
            _copy_file(src, package_root / "docs" / doc)
    _copy_tree(ROOT / "ml_phase", package_root / "code_snapshot" / "ml_phase", ignore=ML_PHASE_OUTPUT_DIRS)
    _copy_tree(ROOT / "scripts", package_root / "code_snapshot" / "scripts")

    _write_text(package_root / "README.md", _readme())
    _write_text(package_root / "scripts" / "slurm_stage0_baseline_regression.sh", _slurm_script(instrumented=False))
    _write_text(package_root / "scripts" / "slurm_stage1_instrumented_regression.sh", _slurm_script(instrumented=True))
    _write_text(package_root / "scripts" / "submit_stage0_baseline_regression.sh", _submit_script("stage0_baseline_regression"))
    _write_text(package_root / "scripts" / "submit_stage1_instrumented_regression.sh", _submit_script("stage1_instrumented_regression"))
    _write_text(package_root / "scripts" / "compare_stage1_regression.sh", _compare_script())
    _write_text(package_root / "scripts" / "verify_stage1_gate.sh", _verify_script())
    _write_text(package_root / "scripts" / "collect_stage1_regression_outputs.sh", _collect_script())
    _write_text(package_root / "scripts" / "slurm_stage1_postprocess.sh", _postprocess_slurm_script())
    _write_text(package_root / "scripts" / "submit_stage1_regression_workflow.sh", _workflow_submit_script())
    _write_text(
        package_root / "scripts" / "submit_local_refinement_fixed_point_regression.sh",
        _workflow_alias_script("submit_stage1_regression_workflow.sh"),
    )
    _write_text(
        package_root / "scripts" / "submit_local_refinement_instrumented_benchmark.sh",
        _workflow_alias_script("submit_stage1_regression_workflow.sh"),
    )
    manifest = {
        "package_name": PACKAGE_NAME,
        "purpose": "Stage 1 local-box instrumentation fixed-point GPU regression",
        "active_learning": "not_run",
        "fixed_points": "fixed_points/fixed_point_regression_points.csv",
        "baseline_output": "reports/local_refinement_refactor/stage_00_baseline/regression_gpu_baseline",
        "instrumented_output": "reports/local_refinement_refactor/stage_01_instrumentation/regression_gpu_instrumented",
        "comparison_output": "reports/local_refinement_refactor/stage_01_instrumentation/baseline_vs_instrumented",
        "instrumentation_flag": "--enable-local-box-instrumentation",
        "expected_local_box_csv": "reports/local_refinement_refactor/stage_01_instrumentation/regression_gpu_instrumented/baseline_local_box_timing.csv",
        "workflow_submit": "scripts/submit_stage1_regression_workflow.sh",
        "workflow_aliases": {
            "scripts/submit_local_refinement_fixed_point_regression.sh": "scripts/submit_stage1_regression_workflow.sh",
            "scripts/submit_local_refinement_instrumented_benchmark.sh": "scripts/submit_stage1_regression_workflow.sh",
        },
        "preflight": "scripts/preflight_local_refinement_stage1_hpc.py",
        "postprocess_job": "scripts/slurm_stage1_postprocess.sh",
        "return_archive": "local_refinement_refactor_stage1_regression_results.tar.gz",
        "writable_run_root_env": "RUN_ROOT",
        "expected_fixed_points": 32,
    }
    _write_text(package_root / "RUN_MANIFEST.json", json.dumps(manifest, indent=2))
    _write_text(
        package_root / "reports" / "local_refinement_refactor" / "stage_01_instrumentation" / "decision_log.md",
        "# Stage 1 HPC Regression Decision Log\n\nThis package is for fixed-point regression only. It must not be used as an active-learning run package.\n",
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
