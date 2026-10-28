from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
HPC_ROOT = ROOT / "hpc_packages"

PACKAGE_NAME = "rankcap_k3_tail_surprise_continuation_v1"
RUN_ID = "active_boundary_discovery_rankcap_k3_tail_surprise_continuation_v1"
OUTPUT_ROOT = "ML_Phase_512_RankCapK3_TailContinuation"
RESULT_ARCHIVE = "rankcap_k3_tail_surprise_continuation_results.tar.gz"

SOURCE_OUTPUT_ROOT = ROOT / "rankcap_k3_full_loop" / "ML_Phase_512_RankCapK3_FullLoop"
SOURCE_RUN_ID = "active_boundary_discovery_rankcap_k3_full_loop_v1"
SOURCE_RUN_DIR = SOURCE_OUTPUT_ROOT / "active_runs" / SOURCE_RUN_ID
START_ITER = 31
TAIL_FIRST_ITER = 26

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
    "docs/report_qa/20260619_tail_surprise_continuation_package.md",
]

SCRIPT_FILES = [
    "scripts/export_active_run_report_md.py",
    "scripts/recover_active_iter.sh",
    "scripts/replay_stopcontroller_surprise_modes.py",
    "scripts/run_rankcap_k3_active_loop_package.py",
    "scripts/slurm_active_refine.sh",
    "scripts/slurm_exact_oracle_array.sh",
]

REPORT_DIRS = [
    "reports/rankcap_k3_full_loop_enhanced",
    "reports/last5_selection_decomposition",
    "reports/surprise_decomposition_audit",
    "reports/surprise_review_recheck",
    "reports/trusted_surprise_counterfactual",
]

TAIL_KEEP_NAMES = {
    "boundary_displacement_iter*.json",
    "candidate_scores_boundary.csv",
    "exact_merged_iter*.npz",
    "exact_training_iter*.npz",
    "exact_trusted_iter*.npz",
    "figures.json",
    "hpc_instructions.txt",
    "merge_summary_iter*.json",
    "metrics.json",
    "monitor_predictions_iter*.npz",
    "partition_metadata.json",
    "rerun_points.csv",
    "selected_points.csv",
    "selected_points_rank*_of*.csv",
    "selection_diagnostics.json",
    "selection_region_diagnostics_iter*.csv",
    "selection_region_diagnostics_iter*.json",
    "status.json",
    "stop_metrics_iter*.json",
}


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


def ignore_tree(_: str, names: list[str]) -> set[str]:
    blocked = {".git", "__pycache__", ".pytest_cache", "active_runs", "datasets", "figures", "hpc_jobs", "models"}
    return {name for name in names if name in blocked or name.endswith((".pyc", ".pyo"))}


def copy_path(src_rel: str, dst_root: Path) -> None:
    src = ROOT / src_rel
    if not src.exists():
        return
    dst = dst_root / src_rel
    if src.is_dir():
        shutil.copytree(src, dst, ignore=ignore_tree, dirs_exist_ok=True)
        for sh in dst.rglob("*.sh"):
            copy_text_lf(sh, sh)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix == ".sh":
        copy_text_lf(src, dst)
    else:
        shutil.copy2(src, dst)


def write_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8", newline="\n")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def require_source_artifacts() -> None:
    required = [
        SOURCE_RUN_DIR / f"dataset_iter{START_ITER:03d}.npz",
        SOURCE_RUN_DIR / f"dataset_iter{START_ITER:03d}.csv",
        SOURCE_RUN_DIR / "metrics_history.json",
        SOURCE_RUN_DIR / "stop_metrics_history.json",
        SOURCE_RUN_DIR / "run_config.json",
        SOURCE_RUN_DIR / f"iter{START_ITER - 1:03d}" / f"monitor_predictions_iter{START_ITER - 1:03d}.npz",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required restart artifacts:\n" + "\n".join(missing))


def copy_restart_artifacts(pkg: Path) -> dict[str, Any]:
    run_dir = pkg / OUTPUT_ROOT / "active_runs" / RUN_ID
    run_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for idx in range(TAIL_FIRST_ITER, START_ITER + 1):
        for suffix in [".npz", ".csv"]:
            src = SOURCE_RUN_DIR / f"dataset_iter{idx:03d}{suffix}"
            if src.exists():
                dst = run_dir / src.name
                shutil.copy2(src, dst)
                copied.append(dst.relative_to(pkg).as_posix())
        append = SOURCE_RUN_DIR / f"dataset_iter{idx:03d}.append.json"
        if append.exists():
            dst = run_dir / append.name
            shutil.copy2(append, dst)
            copied.append(dst.relative_to(pkg).as_posix())

    for name in ["metrics_history.json", "stop_metrics_history.json", "run_config.json", "stop_state.json"]:
        src = SOURCE_RUN_DIR / name
        if src.exists():
            dst = run_dir / name
            shutil.copy2(src, dst)
            copied.append(dst.relative_to(pkg).as_posix())

    for idx in range(TAIL_FIRST_ITER, START_ITER):
        src_dir = SOURCE_RUN_DIR / f"iter{idx:03d}"
        dst_dir = run_dir / src_dir.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for pattern in sorted(TAIL_KEEP_NAMES):
            for src in sorted(src_dir.glob(pattern)):
                if src.is_file():
                    dst = dst_dir / src.name
                    shutil.copy2(src, dst)
                    copied.append(dst.relative_to(pkg).as_posix())

    full_stop_history = read_json(SOURCE_RUN_DIR / "stop_metrics_history.json", [])
    final_stop = full_stop_history[-1] if full_stop_history else {}
    final_metrics = final_stop.get("metrics", {}) if isinstance(final_stop, dict) else {}
    restart_summary = {
        "source_output_root": str(SOURCE_OUTPUT_ROOT),
        "source_run_id": SOURCE_RUN_ID,
        "source_run_dir": str(SOURCE_RUN_DIR),
        "package_output_root": OUTPUT_ROOT,
        "package_run_id": RUN_ID,
        "start_iter": START_ITER,
        "tail_first_iter": TAIL_FIRST_ITER,
        "restart_dataset": f"{OUTPUT_ROOT}/active_runs/{RUN_ID}/dataset_iter{START_ITER:03d}.npz",
        "previous_monitor": f"{OUTPUT_ROOT}/active_runs/{RUN_ID}/iter{START_ITER - 1:03d}/monitor_predictions_iter{START_ITER - 1:03d}.npz",
        "final_source_stop_reason": final_stop.get("stop_reason"),
        "final_source_convergence_pass": final_stop.get("convergence_pass"),
        "final_source_label_surprise_all_selected": final_metrics.get(
            "label_surprise_all_selected",
            final_metrics.get("label_surprise_rate"),
        ),
        "final_source_label_surprise_trusted": final_metrics.get(
            "label_surprise_trusted",
            "not_available_in_original_all_selected_history",
        ),
        "copied_files_count": len(copied),
    }
    write_lf(run_dir / "tail_restart_manifest.json", json.dumps(restart_summary, indent=2, sort_keys=True))
    return restart_summary


def write_submit_script(pkg: Path, default_n_iters: int) -> None:
    write_lf(
        pkg / "scripts" / "submit_rankcap_k3_tail_surprise_continuation.sh",
        f"""#!/bin/bash
set -euo pipefail

PROJECT_DIR="${{PROJECT_DIR:-$PWD}}"
PYTHON_BIN="${{PYTHON_BIN:-/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python}}"
RUN_ID="${{RUN_ID:-{RUN_ID}}}"
OUTPUT_ROOT="${{OUTPUT_ROOT:-{OUTPUT_ROOT}}}"
WORLD_SIZE="${{WORLD_SIZE:-8}}"
EXCLUDE_NODES="${{EXCLUDE_NODES:-gpuh01}}"
START_ITER="${{START_ITER:-{START_ITER}}}"
N_ITERS="${{N_ITERS:-{default_n_iters}}}"
STOP_MAX_ITERATIONS="${{STOP_MAX_ITERATIONS:-$((START_ITER + N_ITERS))}}"

cd "${{PROJECT_DIR}}"

run_dir="${{OUTPUT_ROOT}}/active_runs/${{RUN_ID}}"
resume_dataset="${{run_dir}}/dataset_iter$(printf '%03d' "${{START_ITER}}").npz"
if [ ! -f "${{resume_dataset}}" ]; then
  echo "[error] missing package-local restart dataset: ${{resume_dataset}}" >&2
  exit 1
fi
previous_monitor="${{run_dir}}/iter$(printf '%03d' "$((START_ITER - 1))")/monitor_predictions_iter$(printf '%03d' "$((START_ITER - 1))").npz"
if [ ! -f "${{previous_monitor}}" ]; then
  echo "[error] missing previous monitor needed for boundary/phase-map shift: ${{previous_monitor}}" >&2
  exit 1
fi

echo "[submit] package={PACKAGE_NAME}"
echo "[submit] semantics=tail continuation from full-loop dataset_iter${{START_ITER}}"
echo "[submit] run_id=${{RUN_ID}}"
echo "[submit] output_root=${{OUTPUT_ROOT}}"
echo "[submit] start_iter=${{START_ITER}}"
echo "[submit] n_iters=${{N_ITERS}}"
echo "[submit] stop_max_iterations=${{STOP_MAX_ITERATIONS}}"
echo "[submit] world_size=${{WORLD_SIZE}}"
echo "[submit] exclude_nodes=${{EXCLUDE_NODES}}"
echo "[submit] stop_surprise_mode=trusted"

env \\
  PROJECT_DIR="${{PROJECT_DIR}}" \\
  PYTHON_BIN="${{PYTHON_BIN}}" \\
  OUTPUT_ROOT="${{OUTPUT_ROOT}}" \\
  RUN_ID="${{RUN_ID}}" \\
  RUN_MODE="discovery" \\
  START_ITER="${{START_ITER}}" \\
  N_ITERS="${{N_ITERS}}" \\
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
  WORLD_SIZE="${{WORLD_SIZE}}" \\
  PARTITION_STRATEGY="round_robin" \\
  EXCLUDE_NODES="${{EXCLUDE_NODES}}" \\
  ENABLE_EARLY_STOP="${{ENABLE_EARLY_STOP:-1}}" \\
  ENABLE_STOP_CONTROLLER="${{ENABLE_STOP_CONTROLLER:-1}}" \\
  STOP_MIN_ITERATIONS="${{STOP_MIN_ITERATIONS:-5}}" \\
  STOP_PATIENCE="${{STOP_PATIENCE:-4}}" \\
  STOP_MAX_ITERATIONS="${{STOP_MAX_ITERATIONS}}" \\
  STOP_SURPRISE_MODE="trusted" \\
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

echo "[collect] package continuation return archive"
bash scripts/collect_rankcap_k3_tail_surprise_continuation.sh
echo "[done] return archive: ${{OUTPUT_ROOT}}/{RESULT_ARCHIVE}"
""",
    )


def write_collect_script(pkg: Path) -> None:
    write_lf(
        pkg / "scripts" / "collect_rankcap_k3_tail_surprise_continuation.sh",
        f"""#!/bin/bash
set -euo pipefail

PROJECT_DIR="${{PROJECT_DIR:-$PWD}}"
OUTPUT_ROOT="${{OUTPUT_ROOT:-{OUTPUT_ROOT}}}"
RUN_ID="${{RUN_ID:-{RUN_ID}}}"
ARCHIVE="${{ARCHIVE:-{RESULT_ARCHIVE}}}"

cd "${{PROJECT_DIR}}"
mkdir -p "${{OUTPUT_ROOT}}/return"
tar -czf "${{OUTPUT_ROOT}}/${{ARCHIVE}}" \\
  "${{OUTPUT_ROOT}}/active_runs/${{RUN_ID}}" \\
  "${{OUTPUT_ROOT}}/reports" \\
  RUN_MANIFEST.json \\
  CHECKSUMS.sha256
sha256sum "${{OUTPUT_ROOT}}/${{ARCHIVE}}" > "${{OUTPUT_ROOT}}/${{ARCHIVE}}.sha256"
echo "[done] archive=${{OUTPUT_ROOT}}/${{ARCHIVE}}"
""",
    )


def write_readme(pkg: Path, default_n_iters: int, restart_summary: dict[str, Any]) -> None:
    write_lf(
        pkg / "README.md",
        f"""# {PACKAGE_NAME}

Independent tail-continuation package for testing whether additional
rankcap_k3 active-learning batches after the full-loop endpoint reduce
StopController surprise.

This package is self-contained for computation.  It includes:

- runnable code snapshot at the package root;
- package-local restart dataset `dataset_iter{START_ITER:03d}.npz`;
- tail datasets from `dataset_iter{TAIL_FIRST_ITER:03d}` through `dataset_iter{START_ITER:03d}`;
- selected tail iteration monitor/stop/exact artifacts from `iter{TAIL_FIRST_ITER:03d}` through `iter{START_ITER - 1:03d}`;
- `metrics_history.json`, `stop_metrics_history.json`, `run_config.json`, and `stop_state.json`;
- Slurm scripts with default `EXCLUDE_NODES=gpuh01`.

The package does not modify thermodynamic phase criteria, Delta refinement
tolerance, final ambiguity tolerance, acquisition formula, rankcap_k3 oracle
logic, or StopController thresholds.

## Run

```bash
nohup bash scripts/submit_rankcap_k3_tail_surprise_continuation.sh > {PACKAGE_NAME}.nohup.log 2>&1 &
```

Defaults:

```text
START_ITER={START_ITER}
N_ITERS={default_n_iters}
STOP_SURPRISE_MODE=trusted
WORLD_SIZE=8
EXCLUDE_NODES=gpuh01
```

## Collect Again

```bash
bash scripts/collect_rankcap_k3_tail_surprise_continuation.sh
```

## Restart Input Summary

```json
{json.dumps(restart_summary, indent=2, sort_keys=True)}
```
""",
    )


def write_manifest(pkg: Path, default_n_iters: int, restart_summary: dict[str, Any]) -> dict[str, Any]:
    manifest = {
        "package_name": PACKAGE_NAME,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": RUN_ID,
        "output_root": OUTPUT_ROOT,
        "source_output_root": str(SOURCE_OUTPUT_ROOT),
        "source_run_id": SOURCE_RUN_ID,
        "start_iter": START_ITER,
        "default_n_iters": default_n_iters,
        "result_archive": RESULT_ARCHIVE,
        "submit_script": "scripts/submit_rankcap_k3_tail_surprise_continuation.sh",
        "collect_script": "scripts/collect_rankcap_k3_tail_surprise_continuation.sh",
        "exclude_nodes_default": "gpuh01",
        "stop_surprise_mode": "trusted",
        "trusted_surprise_min_denominator": 64,
        "trusted_surprise_min_fraction": 0.25,
        "restart_summary": restart_summary,
        "forbidden_changes": [
            "thermodynamic phase criterion",
            "Delta refinement trigger tolerance",
            "final ambiguity tolerance",
            "acquisition formula",
            "candidate-domain strategy",
            "rankcap_k3 local-refinement policy",
            "StopController thresholds",
            "k2",
            "energy-window pruning",
            "branch reuse",
            "Powell",
            "adaptive box",
            "GPU batching",
            "Hamiltonian cache",
        ],
    }
    write_lf(pkg / "RUN_MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def write_checksums(pkg: Path) -> None:
    rows: list[str] = []
    for path in sorted(pkg.rglob("*")):
        if path.is_file() and path.name != "CHECKSUMS.sha256":
            rows.append(f"{sha256_file(path)}  {path.relative_to(pkg).as_posix()}")
    write_lf(pkg / "CHECKSUMS.sha256", "\n".join(rows))


def normalize_shell_scripts(pkg: Path) -> list[str]:
    normalized: list[str] = []
    for path in pkg.rglob("*.sh"):
        copy_text_lf(path, path)
        normalized.append(path.relative_to(pkg).as_posix())
    return normalized


def build_package(default_n_iters: int, force: bool) -> dict[str, Any]:
    require_source_artifacts()
    package_root = HPC_ROOT / PACKAGE_NAME
    archive_path = HPC_ROOT / f"{PACKAGE_NAME}.tar.gz"
    if package_root.exists():
        if not force:
            raise FileExistsError(f"package already exists; rerun with --force: {package_root}")
        shutil.rmtree(package_root)
    package_root.mkdir(parents=True, exist_ok=True)

    for rel in COMMON_FILES + DOC_FILES + SCRIPT_FILES:
        copy_path(rel, package_root)
    for rel in COMMON_DIRS + REPORT_DIRS:
        copy_path(rel, package_root)

    restart_summary = copy_restart_artifacts(package_root)
    write_submit_script(package_root, default_n_iters)
    write_collect_script(package_root)
    write_readme(package_root, default_n_iters, restart_summary)
    manifest = write_manifest(package_root, default_n_iters, restart_summary)
    normalized = normalize_shell_scripts(package_root)
    manifest["normalized_shell_scripts"] = normalized
    write_lf(package_root / "RUN_MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True))
    write_checksums(package_root)

    if archive_path.exists():
        if not force:
            raise FileExistsError(f"archive already exists; rerun with --force: {archive_path}")
        archive_path.unlink()
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(package_root, arcname=PACKAGE_NAME)
    archive_sha = sha256_file(archive_path)
    write_lf(HPC_ROOT / f"{PACKAGE_NAME}.tar.gz.sha256", f"{archive_sha}  {archive_path.name}")
    metadata = {
        "package_root": str(package_root),
        "archive": str(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "sha256": archive_sha,
        "manifest": manifest,
    }
    write_lf(HPC_ROOT / f"{PACKAGE_NAME}.tar.gz.metadata.json", json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Package rankcap_k3 tail surprise continuation HPC run.")
    parser.add_argument("--n-iters", type=int, default=5, help="Continuation acquisition iterations to run from dataset_iter031.")
    parser.add_argument("--force", action="store_true", help="Replace an existing package/archive with the same name.")
    args = parser.parse_args()
    HPC_ROOT.mkdir(parents=True, exist_ok=True)
    metadata = build_package(default_n_iters=args.n_iters, force=args.force)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
