from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HPC_ROOT = ROOT / "hpc_packages"


COMMON_FILES = [
    "AGENTS.md",
    "MODEL_SPEC.md",
    "hpc_active_loop.sh",
    "eta_phase_diagram_cuda.py",
    "tfflo_1d_cuda.py",
]

COMMON_DIRS = [
    "ml_phase",
    "report",
    "tests",
]

DOC_FILES = [
    "docs/PROJECT_SUMMARY.md",
    "docs/MODEL_SPEC.md",
    "docs/NUMERICS_SPEC.md",
    "docs/DECISIONS.md",
    "docs/LOCAL_REFINEMENT_REFACTOR_DECISION_LOG.md",
    "docs/TRUSTED_SURPRISE_STOPCONTROLLER_PLAN.md",
    "docs/TRUSTED_SURPRISE_STOPCONTROLLER_DECISION_LOG.md",
]

REPORT_DIRS = [
    "reports/trusted_surprise_counterfactual",
    "reports/surprise_review_recheck",
    "reports/surprise_decomposition_audit",
    "reports/stopcontroller_surprise_replay",
]

SCRIPT_FILES = [
    "scripts/replay_stopcontroller_surprise_modes.py",
    "scripts/run_rankcap_k3_active_loop_package.py",
    "scripts/export_active_run_report_md.py",
    "scripts/build_trusted_surprise_counterfactual.py",
    "scripts/dev_check_stop_controller.py",
    "scripts/recover_active_iter.sh",
    "scripts/slurm_active_refine.sh",
    "scripts/slurm_exact_oracle_array.sh",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_text_lf(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = src.read_text(encoding="utf-8", errors="replace")
    dst.write_text(text.replace("\r\n", "\n").replace("\r", "\n"), encoding="utf-8", newline="\n")


def copy_path(src_rel: str, dst_root: Path) -> None:
    src = ROOT / src_rel
    dst = dst_root / src_rel
    if not src.exists():
        return
    if src.is_dir():
        ignore = shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            ".pytest_cache",
            "*.aux",
            "*.log",
            "*.out",
            "active_runs",
            "datasets",
            "figures",
            "hpc_jobs",
            "models",
            "reports",
        )
        shutil.copytree(src, dst, ignore=ignore, dirs_exist_ok=True)
        for sh in dst.rglob("*.sh"):
            copy_text_lf(sh, sh)
    else:
        if src.suffix == ".sh":
            copy_text_lf(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def write_executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8", newline="\n")


def write_common_scripts(pkg: Path, trusted: bool) -> None:
    mode = "trusted" if trusted else "all_selected"
    validation_run_id = "active_boundary_discovery_trusted_surprise_short_validation_v1"
    output_root = "ML_Phase_512_TrustedSurprise_Short"
    write_executable(
        pkg / "scripts" / "replay_existing_full_loop_stopcontroller.sh",
        f"""#!/bin/bash
set -euo pipefail
PROJECT_DIR="${{PROJECT_DIR:-$PWD}}"
PYTHON_BIN="${{PYTHON_BIN:-python}}"
RUN_DIR="${{RUN_DIR:-$PROJECT_DIR/rankcap_k3_full_loop/ML_Phase_512_RankCapK3_FullLoop/active_runs/active_boundary_discovery_rankcap_k3_full_loop_v1}}"
OUT_DIR="${{OUT_DIR:-$PROJECT_DIR/reports/stopcontroller_surprise_replay}}"
cd "$PROJECT_DIR"
"$PYTHON_BIN" scripts/replay_stopcontroller_surprise_modes.py \\
  --run-dir "$RUN_DIR" \\
  --output-dir "$OUT_DIR" \\
  --trusted-surprise-min-denominator "${{TRUSTED_SURPRISE_MIN_DENOMINATOR:-64}}" \\
  --trusted-surprise-min-fraction "${{TRUSTED_SURPRISE_MIN_FRACTION:-0.25}}"
""",
    )
    write_executable(
        pkg / "scripts" / "submit_trusted_surprise_short_validation.sh",
        f"""#!/bin/bash
set -euo pipefail
PROJECT_DIR="${{PROJECT_DIR:-$PWD}}"
PYTHON_BIN="${{PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}}"
RUN_ID="${{RUN_ID:-{validation_run_id}}}"
OUTPUT_ROOT="${{OUTPUT_ROOT:-{output_root}}}"
WORLD_SIZE="${{WORLD_SIZE:-8}}"
N_ITERS="${{N_ITERS:-4}}"
EXCLUDE_NODES="${{EXCLUDE_NODES:-gpuh01}}"
cd "$PROJECT_DIR"
if [ -e "${{OUTPUT_ROOT}}/active_runs/${{RUN_ID}}" ]; then
  echo "[error] run directory already exists: ${{OUTPUT_ROOT}}/active_runs/${{RUN_ID}}" >&2
  exit 1
fi
env \\
  PROJECT_DIR="$PROJECT_DIR" \\
  PYTHON_BIN="$PYTHON_BIN" \\
  OUTPUT_ROOT="$OUTPUT_ROOT" \\
  RUN_ID="$RUN_ID" \\
  RUN_MODE="discovery" \\
  START_ITER="0" \\
  N_ITERS="$N_ITERS" \\
  CANDIDATE_DOMAIN_MODE="full" \\
  ACQUISITION_PROFILE="full" \\
  ORACLE_MODE="robust_incremental" \\
  INCREMENTAL_Q_EXPANSION_FLAG="--enable-incremental-q-expansion" \\
  LOCAL_BOX_INSTRUMENTATION_FLAG="--enable-local-box-instrumentation" \\
  ENABLE_BASIN_CLUSTERING_FLAG="--enable-basin-clustering" \\
  ENABLE_SELECTIVE_REFINEMENT_FLAG="--enable-selective-refinement" \\
  MAX_REFINED_MINIMA="3" \\
  MAX_OPTIONAL_REFINED_BASINS="3" \\
  MANDATORY_BASINS_CAN_EXCEED_CAP_FLAG="--no-mandatory-basins-can-exceed-cap" \\
  HIGH_RISK_OVERFLOW_POLICY="rank_and_cap" \\
  MAX_EDGE_RISK_BASINS="1" \\
  MAX_DELTA_NEAR_EPS_BASINS="2" \\
  MAX_NEAR_DEGENERATE_BASINS="2" \\
  ENERGY_WINDOW_PRUNING_FLAG="" \\
  WORLD_SIZE="$WORLD_SIZE" \\
  PARTITION_STRATEGY="round_robin" \\
  EXCLUDE_NODES="$EXCLUDE_NODES" \\
  ENABLE_EARLY_STOP="1" \\
  ENABLE_STOP_CONTROLLER="1" \\
  STOP_MIN_ITERATIONS="${{STOP_MIN_ITERATIONS:-5}}" \\
  STOP_PATIENCE="${{STOP_PATIENCE:-4}}" \\
  STOP_MAX_ITERATIONS="${{STOP_MAX_ITERATIONS:-$N_ITERS}}" \\
  STOP_SURPRISE_MODE="{mode}" \\
  TRUSTED_SURPRISE_MIN_DENOMINATOR="${{TRUSTED_SURPRISE_MIN_DENOMINATOR:-64}}" \\
  TRUSTED_SURPRISE_MIN_FRACTION="${{TRUSTED_SURPRISE_MIN_FRACTION:-0.25}}" \\
  POINTS_PER_ITER="${{POINTS_PER_ITER:-256}}" \\
  INITIAL_SEED_SIZE="${{INITIAL_SEED_SIZE:-512}}" \\
  BATCH_SIZE_MAX="${{BATCH_SIZE_MAX:-256}}" \\
  BATCH_SIZE_MIN="${{BATCH_SIZE_MIN:-0}}" \\
  BATCH_SIZE_MIN_BEFORE_MIN_ITER="${{BATCH_SIZE_MIN_BEFORE_MIN_ITER:-64}}" \\
  BATCH_SIZE_MIN_AFTER_MIN_ITER="${{BATCH_SIZE_MIN_AFTER_MIN_ITER:-0}}" \\
  N_ENSEMBLE="${{N_ENSEMBLE:-5}}" \\
  REG_EPOCHS="${{REG_EPOCHS:-240}}" \\
  CLS_EPOCHS="${{CLS_EPOCHS:-240}}" \\
  BATCH_SIZE="${{BATCH_SIZE:-512}}" \\
  bash hpc_active_loop.sh
""",
    )
    write_executable(
        pkg / "scripts" / "collect_trusted_surprise_validation.sh",
        """#!/bin/bash
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
OUTPUT_ROOT="${OUTPUT_ROOT:-ML_Phase_512_TrustedSurprise_Short}"
RUN_ID="${RUN_ID:-active_boundary_discovery_trusted_surprise_short_validation_v1}"
ARCHIVE="${ARCHIVE:-stopcontroller_trusted_surprise_validation_results.tar.gz}"
cd "$PROJECT_DIR"
tar -czf "$ARCHIVE" \
  "$OUTPUT_ROOT/active_runs/$RUN_ID" \
  "$OUTPUT_ROOT/reports" \
  reports/stopcontroller_surprise_replay 2>/dev/null || \
tar -czf "$ARCHIVE" "$OUTPUT_ROOT/active_runs/$RUN_ID" "$OUTPUT_ROOT/reports"
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
echo "[done] archive=$ARCHIVE"
""",
    )
    write_executable(
        pkg / "scripts" / "resume_trusted_surprise_validation.sh",
        """#!/bin/bash
set -euo pipefail
if [ -z "${START_ITER:-}" ]; then
  echo "[error] set START_ITER to the next dataset index before resume." >&2
  exit 1
fi
PROJECT_DIR="${PROJECT_DIR:-$PWD}" \
RUN_ID="${RUN_ID:-active_boundary_discovery_trusted_surprise_short_validation_v1}" \
OUTPUT_ROOT="${OUTPUT_ROOT:-ML_Phase_512_TrustedSurprise_Short}" \
STOP_SURPRISE_MODE="${STOP_SURPRISE_MODE:-trusted}" \
bash hpc_active_loop.sh
""",
    )
    write_executable(
        pkg / "scripts" / "stop_trusted_surprise_validation.sh",
        """#!/bin/bash
set -euo pipefail
RUN_ID="${RUN_ID:-active_boundary_discovery_trusted_surprise_short_validation_v1}"
if ! command -v squeue >/dev/null 2>&1; then
  echo "[error] squeue not available; run scancel manually for jobs matching $RUN_ID." >&2
  exit 1
fi
squeue -u "${USER}" -h -o "%i %j" | awk -v pat="$RUN_ID" '$2 ~ pat {print $1}' | xargs -r scancel
""",
    )


def write_readme(pkg: Path, package_name: str, trusted: bool) -> None:
    mode = "trusted" if trusted else "all_selected"
    readme = f"""# {package_name}

This is an independent StopController surprise-mode package.

Mode:

```text
STOP_SURPRISE_MODE={mode}
```

The package does not change phase criteria, Delta tolerances, final ambiguity
tolerances, acquisition weights, candidate domain, rankcap_k3 local refinement,
required_pass_count, patience, or surprise_tol.

## Local Replay

```bash
bash scripts/replay_existing_full_loop_stopcontroller.sh
```

This requires the saved full-loop run directory to be present locally.

## Trusted Short Validation

```bash
bash scripts/submit_trusted_surprise_short_validation.sh
```

The submit script defaults to:

```text
EXCLUDE_NODES=gpuh01
N_ITERS=4
WORLD_SIZE=8
robust_incremental + rankcap_k3
```

## Collect

```bash
bash scripts/collect_trusted_surprise_validation.sh
```

## Resume

```bash
START_ITER=<next_dataset_index> bash scripts/resume_trusted_surprise_validation.sh
```

## Stop Running Jobs

```bash
bash scripts/stop_trusted_surprise_validation.sh
```
"""
    (pkg / "README.md").write_text(readme, encoding="utf-8", newline="\n")


def write_manifest(pkg: Path, package_name: str, trusted: bool) -> None:
    payload = {
        "package_name": package_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stop_surprise_mode": "trusted" if trusted else "all_selected",
        "trusted_surprise_min_denominator": 64,
        "trusted_surprise_min_fraction": 0.25,
        "exclude_nodes_default": "gpuh01",
        "forbidden_changes": [
            "phase criterion",
            "Delta tolerance",
            "final ambiguity tolerance",
            "acquisition weights",
            "candidate domain",
            "rankcap_k3 local refinement",
            "required_pass_count",
            "patience",
            "surprise_tol",
        ],
        "entrypoints": [
            "scripts/replay_existing_full_loop_stopcontroller.sh",
            "scripts/submit_trusted_surprise_short_validation.sh",
            "scripts/collect_trusted_surprise_validation.sh",
            "scripts/resume_trusted_surprise_validation.sh",
            "scripts/stop_trusted_surprise_validation.sh",
        ],
    }
    (pkg / "RUN_MANIFEST.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_config_snapshot(pkg: Path, trusted: bool) -> None:
    mode = "trusted" if trusted else "all_selected"
    config_dir = pkg / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stop_surprise_mode": mode,
        "trusted_surprise_min_denominator": 64,
        "trusted_surprise_min_fraction": 0.25,
        "surprise_tol_changed": False,
        "required_pass_count_changed": False,
        "patience_changed": False,
        "phase_criterion_changed": False,
        "acquisition_changed": False,
        "rankcap_k3_changed": False,
        "exclude_nodes_default": "gpuh01",
    }
    (config_dir / "stopcontroller_surprise_config.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    code_snapshot = pkg / "code_snapshot"
    code_snapshot.mkdir(parents=True, exist_ok=True)
    (code_snapshot / "README.md").write_text(
        "# Code Snapshot\n\n"
        "The runnable code snapshot is stored at the package root so the Slurm\n"
        "entry points can run without rewriting import paths.  This directory is\n"
        "kept as an explicit package marker for audit tooling.\n",
        encoding="utf-8",
        newline="\n",
    )


def checksums(pkg: Path) -> None:
    rows = []
    for path in sorted(pkg.rglob("*")):
        if path.is_file() and path.name != "CHECKSUMS.sha256":
            rows.append(f"{sha256_file(path)}  {path.relative_to(pkg).as_posix()}")
    (pkg / "CHECKSUMS.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")


def build_package(package_name: str, trusted: bool) -> Path:
    pkg = HPC_ROOT / package_name
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir(parents=True, exist_ok=True)

    for rel in COMMON_FILES + DOC_FILES + SCRIPT_FILES:
        copy_path(rel, pkg)
    for rel in COMMON_DIRS + REPORT_DIRS:
        copy_path(rel, pkg)
    write_common_scripts(pkg, trusted=trusted)
    write_readme(pkg, package_name, trusted=trusted)
    write_manifest(pkg, package_name, trusted=trusted)
    write_config_snapshot(pkg, trusted=trusted)
    checksums(pkg)

    archive = HPC_ROOT / f"{package_name}.tar.gz"
    if archive.exists():
        archive.unlink()
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(pkg, arcname=package_name)
    sha_path = HPC_ROOT / f"{package_name}.tar.gz.sha256"
    sha_path.write_text(f"{sha256_file(archive)}  {archive.name}\n", encoding="utf-8")
    meta = {
        "package_name": package_name,
        "path": str(pkg),
        "archive": str(archive),
        "archive_size_bytes": archive.stat().st_size,
        "sha256": sha256_file(archive),
    }
    (HPC_ROOT / f"{package_name}.tar.gz.metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return archive


def main() -> None:
    HPC_ROOT.mkdir(parents=True, exist_ok=True)
    archives = [
        build_package("stopcontroller_all_selected_baseline_snapshot", trusted=False),
        build_package("stopcontroller_trusted_surprise_v1", trusted=True),
    ]
    print(json.dumps({"archives": [str(p) for p in archives]}, indent=2))


if __name__ == "__main__":
    main()
