from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "active_phase_topology_from_scratch_full_loop_v1"
PACKAGE_NAME = "active_phase_topology_from_scratch_full_loop_v1_hpc"
OUTPUT_ROOT = "ML_Phase_512_TopoTrivial_FullLoop"
RESULT_ARCHIVE = "active_phase_topology_from_scratch_full_loop_v1_results.tar.gz"
REPORT_ROOT = ROOT / "reports" / "topo_trivial_full_loop_package"
REFERENCE_REPORT = ROOT / "reports" / "topology_pass_dataset_iter035_v1"
PACKAGE_ROOT = ROOT / "hpc_packages" / PACKAGE_NAME
ARCHIVE_PATH = ROOT / "hpc_packages" / f"{PACKAGE_NAME}.tar.gz"


def run(cmd: list[str], cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.replace("\r\n", "\n").replace("\r", "\n"), encoding="utf-8", newline="\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_hash(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted(paths):
        if not path.is_file():
            continue
        h.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        h.update(b"\0")
        h.update(sha256_file(path).encode("ascii"))
        h.update(b"\0")
    return h.hexdigest()


def git_snapshot() -> dict[str, str]:
    status = run(["git", "status", "--short"])
    diff = run(["git", "diff", "--stat"])
    commit = run(["git", "rev-parse", "HEAD"])
    return {
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else "unknown",
        "git_status_short": status.stdout.strip(),
        "git_diff_stat": diff.stdout.strip(),
        "working_tree_has_changes": "yes" if status.stdout.strip() else "no",
    }


def ignore_common(_dir: str, names: list[str]) -> set[str]:
    blocked = {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "hpc_packages",
        "output",
        "outputs",
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
    if not src.exists():
        return
    shutil.copytree(src, dst, ignore=ignore_common)


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def normalize_shell_scripts(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(root.rglob("*.sh")):
        data = path.read_bytes()
        status = "pass"
        notes: list[str] = []
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


def add_gpuh01_runtime_guard(path: Path) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    if "hostname_guard_gpuh01" in text:
        return
    marker = "set -euo pipefail\n"
    guard = (
        "\n"
        "# hostname_guard_gpuh01\n"
        "host_name=\"$(hostname 2>/dev/null || true)\"\n"
        "if [ \"${host_name%%.*}\" = \"gpuh01\" ]; then\n"
        "  echo \"[error] refusing to run on excluded node gpuh01\" >&2\n"
        "  exit 42\n"
        "fi\n"
    )
    if marker in text:
        text = text.replace(marker, marker + guard, 1)
    else:
        text = guard + "\n" + text
    write_text(path, text)


def submit_script() -> str:
    return """#!/bin/bash
set -euo pipefail

export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export PYTHONUTF8=1
export PYTHONIOENCODING="utf-8"

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-ML_Phase_512_TopoTrivial_FullLoop}"
RUN_ID="${RUN_ID:-active_phase_topology_from_scratch_full_loop_v1}"
RUN_DIR="${OUTPUT_ROOT}/active_runs/${RUN_ID}"
N_ITERS="${N_ITERS:-51}"
WORLD_SIZE="${WORLD_SIZE:-8}"
EXCLUDE_NODES="${EXCLUDE_NODES:-gpuh01}"

if [ "${CONFIRM_TOPO_FULL_LOOP:-0}" != "1" ]; then
  echo "[error] set CONFIRM_TOPO_FULL_LOOP=1 to submit the production run." >&2
  exit 2
fi

if [ -e "${RUN_DIR}" ]; then
  echo "[error] run directory already exists: ${RUN_DIR}" >&2
  exit 4
fi

echo "[submit] project_dir=${PROJECT_DIR}"
echo "[submit] run_id=${RUN_ID}"
echo "[submit] output_root=${OUTPUT_ROOT}"
echo "[submit] n_iters=${N_ITERS}"
echo "[submit] world_size=${WORLD_SIZE}"
echo "[submit] exclude_nodes=${EXCLUDE_NODES}"

cd "${PROJECT_DIR}"

RUN_MODE="discovery" \\
CANDIDATE_DOMAIN_MODE="full" \\
INITIALIZATION="sobol_scrambled" \\
INITIAL_SEED_SIZE="512" \\
BATCH_SIZE_MAX="256" \\
BATCH_SIZE_MIN="0" \\
POINTS_PER_ITER="256" \\
N_ITERS="${N_ITERS}" \\
WORLD_SIZE="${WORLD_SIZE}" \\
OUTPUT_ROOT="${OUTPUT_ROOT}" \\
RUN_ID="${RUN_ID}" \\
PYTHON_BIN="${PYTHON_BIN}" \\
ORACLE_MODE="robust_incremental" \\
INCREMENTAL_Q_EXPANSION_FLAG="--enable-incremental-q-expansion" \\
LOCAL_BOX_INSTRUMENTATION_FLAG="--enable-local-box-instrumentation" \\
ENABLE_BASIN_CLUSTERING_FLAG="--enable-basin-clustering" \\
ENABLE_SELECTIVE_REFINEMENT_FLAG="--enable-selective-refinement" \\
MAX_REFINED_MINIMA="3" \\
MAX_OPTIONAL_REFINED_BASINS="3" \\
MANDATORY_BASINS_CAN_EXCEED_CAP_FLAG="" \\
HIGH_RISK_OVERFLOW_POLICY="rank_and_cap" \\
MAX_EDGE_RISK_BASINS="1" \\
MAX_DELTA_NEAR_EPS_BASINS="2" \\
MAX_NEAR_DEGENERATE_BASINS="2" \\
ACQUISITION_PROFILE="topo_trivial" \\
TOPOLOGY_CLASSIFICATION_FLAG="--enable-topology-classification" \\
TOPOLOGY_GAP_NK="${TOPOLOGY_GAP_NK:-2048}" \\
TOPOLOGY_GAP_BACKEND="${TOPOLOGY_GAP_BACKEND:-gpu}" \\
TOPOLOGY_GAP_TOL_REL="${TOPOLOGY_GAP_TOL_REL:-1e-8}" \\
TOPOLOGY_GAP_TOL_ABS="${TOPOLOGY_GAP_TOL_ABS:-0.0}" \\
TOPOLOGY_GAP_K_CHUNK="${TOPOLOGY_GAP_K_CHUNK:-512}" \\
EXCLUDE_NODES="${EXCLUDE_NODES}" \\
STOP_SURPRISE_MODE="trusted" \\
TRUSTED_SURPRISE_MIN_DENOMINATOR="64" \\
TRUSTED_SURPRISE_MIN_FRACTION="0.25" \\
STOP_MAX_ITERATIONS="${N_ITERS}" \\
bash hpc_active_loop.sh
"""


def smoke_script() -> str:
    return """#!/bin/bash
set -euo pipefail

export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export PYTHONUTF8=1
export PYTHONIOENCODING="utf-8"

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" scripts/preflight_topo_trivial_full_loop.py
"${PYTHON_BIN}" -m py_compile ml_phase/topology_oracle.py ml_phase/exact_oracle.py ml_phase/acquisition.py ml_phase/active_refine.py ml_phase/config.py scripts/run_topology_pass_dataset_iter035.py
"""


def status_script() -> str:
    return """#!/bin/bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-ML_Phase_512_TopoTrivial_FullLoop}"
RUN_ID="${RUN_ID:-active_phase_topology_from_scratch_full_loop_v1}"
RUN_DIR="${OUTPUT_ROOT}/active_runs/${RUN_ID}"

echo "[status] run_dir=${RUN_DIR}"
if command -v squeue >/dev/null 2>&1; then
  squeue -u "${USER}" || true
fi
if [ -d "${RUN_DIR}" ]; then
  find "${RUN_DIR}" -maxdepth 2 -type f \\( -name 'dataset_iter*.npz' -o -name 'stop_metrics_iter*.json' -o -name 'exact_trusted_iter*.npz' \\) | sort
else
  echo "[status] run directory not found"
fi
"""


def collect_script() -> str:
    return """#!/bin/bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-ML_Phase_512_TopoTrivial_FullLoop}"
RUN_ID="${RUN_ID:-active_phase_topology_from_scratch_full_loop_v1}"
ARCHIVE="${OUTPUT_ROOT}/active_phase_topology_from_scratch_full_loop_v1_results.tar.gz"

mkdir -p "${OUTPUT_ROOT}"
tar -czf "${ARCHIVE}" \\
  "${OUTPUT_ROOT}/active_runs/${RUN_ID}" \\
  "${OUTPUT_ROOT}/figures" \\
  "${OUTPUT_ROOT}/reports" \\
  2>/dev/null || tar -czf "${ARCHIVE}" "${OUTPUT_ROOT}/active_runs/${RUN_ID}"

echo "[done] archive=${ARCHIVE}"
"""


def preflight_script() -> str:
    return r'''from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "topo_trivial_full_loop_package"


def add(rows, check, status, detail):
    rows.append({"check": check, "status": status, "detail": detail})


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["check", "status", "detail"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = []
    add(rows, "topology_oracle_exists", "pass" if (ROOT / "ml_phase" / "topology_oracle.py").exists() else "fail", "ml_phase/topology_oracle.py")
    add(rows, "topology_pass_script_exists", "pass" if (ROOT / "scripts" / "run_topology_pass_dataset_iter035.py").exists() else "fail", "scripts/run_topology_pass_dataset_iter035.py")
    active_text = (ROOT / "ml_phase" / "active_refine.py").read_text(encoding="utf-8", errors="replace")
    add(rows, "sobol_initialization_supported", "pass" if "sobol_scrambled" in active_text else "fail", "active_refine initialization choices")
    has_topo_acq = ("A_spectral" in active_text) and ("topo_trivial" in active_text)
    add(rows, "topology_aware_acquisition_supported", "fail" if not has_topo_acq else "pass", "production active_refine must expose topology-aware acquisition before full-loop submission")
    submit_text = (ROOT / "scripts" / "submit_topo_trivial_full_loop.sh").read_text(encoding="utf-8", errors="replace")
    add(rows, "submit_script_excludes_gpuh01", "pass" if "gpuh01" in submit_text else "fail", "EXCLUDE_NODES=gpuh01")
    add(rows, "submit_script_uses_topo_trivial_profile", "pass" if 'ACQUISITION_PROFILE="topo_trivial"' in submit_text else "fail", "no silent phase-only fallback")
    add(rows, "submit_script_enables_topology_exact", "pass" if "--enable-topology-classification" in submit_text else "fail", "exact shards must write topology fields")

    summary_path = ROOT / "topology_reference" / "topology_pass_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        counts = {(str(r.get("thermo_phase")), str(r.get("topology_label"))): int(r.get("count", 0)) for r in summary.get("counts", [])}
        expected = {
            ("uniform_SC", "trivial"): 715,
            ("FFLO", "trivial"): 3127,
            ("FFLO", "topological"): 1515,
            ("FFLO", "gapless_SC"): 195,
            ("FFLO", "topology_unresolved"): 15,
        }
        for key, value in expected.items():
            got = counts.get(key, 0)
            add(rows, f"reference_count_{key[0]}_{key[1]}", "pass" if got == value else "fail", f"got={got}; expected={value}")
    else:
        add(rows, "reference_summary_exists", "fail", str(summary_path))

    shell_failures = []
    for path in sorted(ROOT.rglob("*.sh")):
        data = path.read_bytes()
        if data.startswith(b"\xef\xbb\xbf") or b"\r\n" in data or b"\r" in data:
            shell_failures.append(path.relative_to(ROOT).as_posix())
        try:
            data.decode("ascii")
        except UnicodeDecodeError:
            shell_failures.append(path.relative_to(ROOT).as_posix() + ":non_ascii")
    add(rows, "shell_scripts_ascii_lf_no_bom", "pass" if not shell_failures else "fail", ";".join(shell_failures[:10]))

    status = "pass" if all(r["status"] == "pass" for r in rows) else "blocked"
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "tables" / "package_preflight_checks.csv", rows)
    (OUT / "package_preflight.json").write_text(json.dumps({"status": status, "checks": rows}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "checks": rows}, indent=2))


if __name__ == "__main__":
    main()
'''


def package_config() -> dict[str, Any]:
    return {
        "run_id": RUN_ID,
        "output_root": OUTPUT_ROOT,
        "result_archive": RESULT_ARCHIVE,
        "run_mode": "discovery",
        "initialization": "sobol_scrambled",
        "initial_seed_size": 512,
        "batch_size_max": 256,
        "default_n_iters": 51,
        "default_acquisition_batches": 50,
        "oracle_mode": "robust_incremental",
        "rankcap": {
            "enabled": True,
            "variant": "k3",
            "max_total_refined_basins": 3,
            "max_optional_refined_basins": 3,
            "mandatory_basins_can_exceed_cap": False,
            "high_risk_overflow_policy": "rank_and_cap",
        },
        "stop_controller": {
            "surprise_mode": "trusted",
            "trusted_surprise_min_denominator": 64,
            "trusted_surprise_min_fraction": 0.25,
        },
        "topology_reference": {
            "source": "reports/topology_pass_dataset_iter035_v1",
            "training_warm_start": False,
            "purpose": "unit/regression/reference/post-run comparison only",
        },
        "production_gate": {
            "topology_aware_acquisition_required": True,
            "acquisition_profile": "topo_trivial",
            "exact_topology_classification_flag": "--enable-topology-classification",
        },
        "excluded_nodes_default": "gpuh01",
        "disabled_features": ["k2", "energy_window", "branch_reuse", "Powell", "adaptive_box", "GPU_batching", "Hamiltonian_cache"],
    }


def readme_text(config: dict[str, Any]) -> str:
    return f"""# {PACKAGE_NAME}

Independent Stage III topology/trivial full-loop HPC package.

## Run Identity

- run_id: `{RUN_ID}`
- output_root: `{OUTPUT_ROOT}`
- default iterations: `{config['default_n_iters']}` (512 Sobol seed plus 50 acquisition batches)
- default batch size: `256`
- default excluded node: `gpuh01`

## Important Gate

This package intentionally does not silently fall back to the old thermodynamic-only
active-learning loop. The production submit script uses:

```text
ACQUISITION_PROFILE=topo_trivial
TOPOLOGY_CLASSIFICATION_FLAG=--enable-topology-classification
```

It still exits unless `CONFIRM_TOPO_FULL_LOOP=1` is set, so uploading the package
does not accidentally submit the production run.

## Preflight

```bash
python scripts/preflight_topo_trivial_full_loop.py
bash scripts/submit_topo_trivial_smoke.sh
```

## Production Submission

```bash
export PYTHON_BIN=/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python
export CONFIRM_TOPO_FULL_LOOP=1
nohup bash scripts/submit_topo_trivial_full_loop.sh > {RUN_ID}.nohup.log 2>&1 &
```

## Status and Collection

```bash
bash scripts/status_topo_trivial_full_loop.sh
bash scripts/collect_topo_trivial_full_loop.sh
```

## Reference Data

`topology_reference/` contains the frozen `dataset_iter035` offline topology
pass outputs for regression and post-run comparison. It must not be used as the
warm-start training dataset for this cold-start topology-aware loop.
"""


def build_package() -> dict[str, Any]:
    if PACKAGE_ROOT.exists():
        shutil.rmtree(PACKAGE_ROOT)
    PACKAGE_ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "hpc_packages").mkdir(parents=True, exist_ok=True)

    copy_tree(ROOT / "ml_phase", PACKAGE_ROOT / "ml_phase")
    copy_tree(ROOT / "tests", PACKAGE_ROOT / "tests")
    copy_tree(REFERENCE_REPORT, PACKAGE_ROOT / "topology_reference")
    for name in [
        "AGENTS.md",
        "MODEL_SPEC.md",
        "README.md",
        "eta_phase_diagram_cuda.py",
        "tfflo_1d_cuda.py",
        "hpc_active_loop.sh",
    ]:
        copy_file(ROOT / name, PACKAGE_ROOT / name)
    docs_dst = PACKAGE_ROOT / "docs"
    docs_dst.mkdir(parents=True, exist_ok=True)
    for name in [
        "PROJECT_SUMMARY.md",
        "MODEL_SPEC.md",
        "NUMERICS_SPEC.md",
        "DECISIONS.md",
        "StageIII_TopoTrivial_Classification.md",
        "Topo_Trivial_FullLoop_Build.md",
    ]:
        copy_file(ROOT / "docs" / name, docs_dst / name)

    scripts_dst = PACKAGE_ROOT / "scripts"
    scripts_dst.mkdir(parents=True, exist_ok=True)
    for name in [
        "run_topology_pass_dataset_iter035.py",
        "slurm_active_refine.sh",
        "slurm_exact_oracle_array.sh",
        "recover_active_iter.sh",
    ]:
        copy_file(ROOT / "scripts" / name, scripts_dst / name)

    write_text(scripts_dst / "submit_topo_trivial_full_loop.sh", submit_script())
    write_text(scripts_dst / "submit_topo_trivial_smoke.sh", smoke_script())
    write_text(scripts_dst / "status_topo_trivial_full_loop.sh", status_script())
    write_text(scripts_dst / "collect_topo_trivial_full_loop.sh", collect_script())
    write_text(scripts_dst / "preflight_topo_trivial_full_loop.py", preflight_script())
    write_json(PACKAGE_ROOT / "configs" / "topo_trivial_full_loop_config.json", package_config())
    write_text(PACKAGE_ROOT / "README_TOPO_TRIVIAL_FULL_LOOP.md", readme_text(package_config()))

    for rel in [
        "scripts/slurm_active_refine.sh",
        "scripts/slurm_exact_oracle_array.sh",
    ]:
        add_gpuh01_runtime_guard(PACKAGE_ROOT / rel)

    shell_rows = normalize_shell_scripts(PACKAGE_ROOT)

    initial_files = [p for p in PACKAGE_ROOT.rglob("*") if p.is_file()]
    manifest = {
        "package_name": PACKAGE_NAME,
        "run_id": RUN_ID,
        "output_root": OUTPUT_ROOT,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(initial_files),
        "package_tree_hash": tree_hash(initial_files),
        "config": package_config(),
        "git": git_snapshot(),
    }
    write_json(PACKAGE_ROOT / "RUN_MANIFEST.json", manifest)

    validation = validate_package(PACKAGE_ROOT, shell_rows)
    write_json(PACKAGE_ROOT / "PACKAGE_VALIDATION.json", validation)
    write_csv(PACKAGE_ROOT / "reports" / "topo_trivial_full_loop_package" / "tables" / "package_validation_checks.csv", validation["checks"])

    for cache_dir in sorted(PACKAGE_ROOT.rglob("__pycache__"), reverse=True):
        shutil.rmtree(cache_dir, ignore_errors=True)

    final_files = [p for p in PACKAGE_ROOT.rglob("*") if p.is_file()]
    manifest.update(
        {
            "file_count": len(final_files),
            "package_tree_hash": tree_hash(final_files),
            "validation_status": validation["status"],
            "production_submission_status": validation["production_submission_status"],
        }
    )
    write_json(PACKAGE_ROOT / "RUN_MANIFEST.json", manifest)

    if ARCHIVE_PATH.exists():
        ARCHIVE_PATH.unlink()
    with tarfile.open(ARCHIVE_PATH, "w:gz") as tar:
        tar.add(PACKAGE_ROOT, arcname=PACKAGE_NAME)
    archive_sha = sha256_file(ARCHIVE_PATH)
    write_text(ARCHIVE_PATH.with_suffix(ARCHIVE_PATH.suffix + ".sha256"), f"{archive_sha}  {ARCHIVE_PATH.name}\n")

    metadata = {
        **manifest,
        "archive": str(ARCHIVE_PATH),
        "archive_sha256": archive_sha,
        "validation_status": validation["status"],
        "production_submission_status": validation["production_submission_status"],
    }
    write_json(ARCHIVE_PATH.with_suffix(ARCHIVE_PATH.suffix + ".metadata.json"), metadata)
    write_package_report(metadata, validation)
    return metadata


def validate_package(root: Path, shell_rows: list[dict[str, str]]) -> dict[str, Any]:
    checks: list[dict[str, str]] = []

    def add(check: str, status: str, detail: str) -> None:
        checks.append({"check": check, "status": status, "detail": detail})

    required = [
        "ml_phase/topology_oracle.py",
        "ml_phase/active_refine.py",
        "scripts/run_topology_pass_dataset_iter035.py",
        "scripts/preflight_topo_trivial_full_loop.py",
        "scripts/submit_topo_trivial_full_loop.sh",
        "scripts/submit_topo_trivial_smoke.sh",
        "scripts/status_topo_trivial_full_loop.sh",
        "scripts/collect_topo_trivial_full_loop.sh",
        "topology_reference/topology_pass_summary.json",
        "configs/topo_trivial_full_loop_config.json",
        "RUN_MANIFEST.json",
    ]
    for rel in required:
        add(f"exists_{rel}", "pass" if (root / rel).exists() else "fail", rel)

    active_text = (root / "ml_phase" / "active_refine.py").read_text(encoding="utf-8", errors="replace")
    add("sobol_scrambled_supported", "pass" if "sobol_scrambled" in active_text else "fail", "active_refine.py")
    has_topology_acq = ("A_spectral" in active_text) and ("topo_trivial" in active_text)
    add("topology_aware_acquisition_supported", "fail" if not has_topology_acq else "pass", "required before production submission")

    submit_text = (root / "scripts" / "submit_topo_trivial_full_loop.sh").read_text(encoding="utf-8", errors="replace")
    add("submit_requires_confirm", "pass" if "CONFIRM_TOPO_FULL_LOOP" in submit_text else "fail", "submit guard")
    add("submit_uses_topo_trivial_profile", "pass" if 'ACQUISITION_PROFILE="topo_trivial"' in submit_text else "fail", "topology profile")
    add("submit_enables_topology_exact", "pass" if "--enable-topology-classification" in submit_text else "fail", "topology exact fields")
    add("submit_excludes_gpuh01", "pass" if "EXCLUDE_NODES=\"${EXCLUDE_NODES:-gpuh01}\"" in submit_text else "fail", "gpuh01")

    shell_fail = [r for r in shell_rows if r["status"] != "pass"]
    add("shell_scripts_ascii_lf_no_bom", "pass" if not shell_fail else "fail", ";".join(r["path"] for r in shell_fail[:10]))

    compile_cmd = [
        "python",
        "-m",
        "py_compile",
        "ml_phase/topology_oracle.py",
        "ml_phase/exact_oracle.py",
        "ml_phase/acquisition.py",
        "ml_phase/active_refine.py",
        "ml_phase/config.py",
        "scripts/run_topology_pass_dataset_iter035.py",
        "scripts/preflight_topo_trivial_full_loop.py",
    ]
    compiled = run(compile_cmd, cwd=root)
    add("py_compile_critical_files", "pass" if compiled.returncode == 0 else "fail", (compiled.stderr or compiled.stdout).strip())

    bash = run(["where.exe", "bash"])
    if bash.returncode == 0:
        bash_path = bash.stdout.splitlines()[0].strip()
        if bash_path.lower().endswith("\\system32\\bash.exe"):
            add("bash_syntax", "warn", "only Windows WSL launcher found locally; no installed Linux distribution for bash -n")
        else:
            failures = []
            for rel in [
                "scripts/submit_topo_trivial_full_loop.sh",
                "scripts/submit_topo_trivial_smoke.sh",
                "scripts/status_topo_trivial_full_loop.sh",
                "scripts/collect_topo_trivial_full_loop.sh",
            ]:
                out = run([bash_path, "-n", rel], cwd=root)
                if out.returncode != 0:
                    failures.append(f"{rel}:{out.stderr.strip()}")
            add("bash_syntax", "pass" if not failures else "fail", "; ".join(failures))
    else:
        add("bash_syntax", "warn", "bash not found on local Windows PATH")

    if any(row["check"] == "topology_aware_acquisition_supported" and row["status"] == "fail" for row in checks):
        production = "blocked_until_topology_aware_acquisition_is_implemented"
    else:
        production = "ready"
    package_status = "pass" if all(row["status"] in {"pass", "warn"} for row in checks) else "fail"
    return {
        "status": package_status,
        "production_submission_status": production,
        "checks": checks,
    }


def write_package_report(metadata: dict[str, Any], validation: dict[str, Any]) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(REPORT_ROOT / "package_manifest.json", metadata)
    write_csv(REPORT_ROOT / "tables" / "package_validation_checks.csv", validation["checks"])
    decision = [
        "# Topology/Trivial Full-Loop HPC Package Decision Log",
        "",
        f"- package_status: {validation['status']}",
        f"- production_submission_status: {validation['production_submission_status']}",
        f"- archive: `{metadata['archive']}`",
        f"- archive_sha256: `{metadata['archive_sha256']}`",
        "",
        "Decision: the package is built as a self-contained topology-aware full-loop package. Production submission requires `CONFIRM_TOPO_FULL_LOOP=1`, uses `ACQUISITION_PROFILE=topo_trivial`, and enables topology diagnostics in each exact shard.",
    ]
    write_text(REPORT_ROOT / "decision_log.md", "\n".join(decision) + "\n")
    lines = [
        "# Topology/Trivial Full-Loop HPC Package Build Report",
        "",
        "## Summary",
        "",
        f"- package: `{PACKAGE_NAME}`",
        f"- archive: `{metadata['archive']}`",
        f"- archive sha256: `{metadata['archive_sha256']}`",
        f"- package validation status: `{validation['status']}`",
        f"- production submission status: `{validation['production_submission_status']}`",
        "",
        "## What Is Included",
        "",
        "- `ml_phase/` source snapshot, including the topology oracle.",
        "- selected `scripts/` entry points and package-local submit/status/collect/preflight scripts.",
        "- selected docs and the Stage III build runbook.",
        "- `topology_reference/` copied from `reports/topology_pass_dataset_iter035_v1` for regression/reference use only.",
        "- package manifest, validation JSON/CSV, and archive checksum.",
        "",
        "## Production Gate",
        "",
        "The submit script intentionally exits unless `CONFIRM_TOPO_FULL_LOOP=1` is set. It uses `ACQUISITION_PROFILE=topo_trivial` and `--enable-topology-classification`, preventing a silent fallback to the thermodynamic-only acquisition path.",
        "",
        "## Validation Checks",
        "",
    ]
    for row in validation["checks"]:
        lines.append(f"- {row['check']}: {row['status']} ({row['detail']})")
    write_text(REPORT_ROOT / "package_build_report.md", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Stage III topology/trivial full-loop HPC package.")
    parser.add_argument("--no-build", action="store_true", help="Only report intended paths.")
    args = parser.parse_args()
    if args.no_build:
        print(json.dumps({"package_root": str(PACKAGE_ROOT), "archive": str(ARCHIVE_PATH)}, indent=2))
        return
    metadata = build_package()
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
