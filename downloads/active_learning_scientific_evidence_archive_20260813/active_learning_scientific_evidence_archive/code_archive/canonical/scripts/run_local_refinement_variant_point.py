from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_local_refinement_fixed_point_regression import (
    _result_frame,
    resolve_variant_config,
)


DEFAULT_OUTPUT_ROOT = Path("reports/local_refinement_refactor/variant_regression")


def _default_run_root(package_root: Path) -> Path:
    env_run_root = os.environ.get("RUN_ROOT")
    if env_run_root:
        return Path(env_run_root)
    if os.access(package_root, os.W_OK):
        return package_root / "local_refinement_refactor_variant_suite_run"
    fallback = os.environ.get("SCRATCH") or os.environ.get("TMPDIR") or os.environ.get("HOME")
    if fallback:
        return Path(fallback) / "local_refinement_refactor_variant_suite_run"
    return package_root / "local_refinement_refactor_variant_suite_run"


def _run_path(path: Path, run_root: Path) -> Path:
    return path if path.is_absolute() else run_root / path


def _package_path(path: Path, package_root: Path) -> Path:
    return path if path.is_absolute() else package_root / path


def _read_task(task_matrix: Path, task_id: int) -> dict[str, str]:
    with task_matrix.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["task_id"]) == int(task_id):
                return row
    raise KeyError(f"task_id {task_id} not found in {task_matrix}")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        _atomic_write_text(path, "")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _rewrite_local_box_file(path: Path, *, task_id: int, point_id: int, variant_name: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["task_id"] = int(task_id)
        row["point_id"] = int(point_id)
        row["variant_name"] = variant_name
    _write_csv(path, rows)


def _task_output_paths(
    run_root: Path,
    output_root: Path,
    *,
    variant_name: str,
    point_id: int,
) -> dict[str, Path]:
    task_dir = run_root / output_root / "point_tasks" / variant_name
    stem = f"point_{int(point_id):03d}"
    return {
        "task_dir": task_dir,
        "csv": task_dir / f"{stem}.csv",
        "json": task_dir / f"{stem}.json",
        "npz": task_dir / f"{stem}.npz",
        "local_box": task_dir / f"{stem}_local_box_timing.csv",
    }


def run_task(
    *,
    package_root: Path,
    run_root: Path,
    task_matrix: Path,
    task_id: int,
    output_root: Path,
    device: str,
    enable_local_box_instrumentation: bool,
    force: bool,
) -> dict[str, Any]:
    task = _read_task(task_matrix, task_id)
    variant_name = str(task["variant"])
    point_id = int(task["point_id"])
    kT = float(task["kT"])
    JA = float(task["JA"])
    variant_config = resolve_variant_config(variant_name)
    paths = _task_output_paths(run_root, output_root, variant_name=variant_name, point_id=point_id)
    paths["task_dir"].mkdir(parents=True, exist_ok=True)

    if paths["json"].exists() and paths["csv"].exists() and not force:
        existing = json.loads(paths["json"].read_text(encoding="utf-8"))
        if existing.get("status") == "success":
            return existing

    status_base: dict[str, Any] = {
        "mode": "exact_point",
        "status": "running",
        "task_id": int(task_id),
        "variant_name": variant_name,
        "point_id": point_id,
        "kT": kT,
        "JA": JA,
        "source_category": task.get("category", "unknown"),
        "source_run": task.get("source_run", "unknown"),
        "source_iter": task.get("source_iter", "unknown"),
        "source_index": task.get("source_index", "unknown"),
        "device": device,
        "variant_config": variant_config,
        "output_csv": str(paths["csv"]),
        "output_npz": str(paths["npz"]),
        "local_box_output_file": str(paths["local_box"]) if enable_local_box_instrumentation else "N/A",
    }
    _atomic_write_text(paths["json"], json.dumps(status_base, indent=2))

    t0 = time.perf_counter()
    try:
        from eta_phase_diagram_cuda import EtaPhaseConfig
        from ml_phase.exact_oracle import evaluate_points

        result = evaluate_points(
            points=np.asarray([[kT, JA]], dtype=np.float64),
            cfg=EtaPhaseConfig(),
            device=device,
            save_every=1,
            output_file=paths["npz"],
            oracle_mode="robust_incremental",
            enable_q_expansion=True,
            enable_delta_refinement=True,
            enable_incremental_q_expansion=True,
            enable_local_box_instrumentation=bool(enable_local_box_instrumentation),
            local_box_output_file=paths["local_box"] if enable_local_box_instrumentation else None,
            **variant_config,
        )
        rows = _result_frame(result)
        if len(rows) != 1:
            raise RuntimeError(f"expected one result row for task {task_id}, got {len(rows)}")
        row = rows[0]
        row["task_id"] = int(task_id)
        row["point_id"] = point_id
        row["source_category"] = task.get("category", "unknown")
        row["source_run"] = task.get("source_run", "unknown")
        row["source_iter"] = task.get("source_iter", "unknown")
        row["source_index"] = task.get("source_index", "unknown")
        row["variant_name"] = variant_name
        _write_csv(paths["csv"], [row])
        if enable_local_box_instrumentation:
            _rewrite_local_box_file(paths["local_box"], task_id=task_id, point_id=point_id, variant_name=variant_name)

        status = {
            **status_base,
            "status": "success",
            "wall_runtime_sec": float(time.perf_counter() - t0),
            "point_total_runtime_sec": float(row.get("point_total_runtime_sec", 0.0)),
            "local_refinement_runtime_sec": float(row.get("local_refinement_runtime_sec", 0.0)),
        }
        _atomic_write_text(paths["json"], json.dumps(status, indent=2))
        return status
    except Exception as exc:
        status = {
            **status_base,
            "status": "failed",
            "wall_runtime_sec": float(time.perf_counter() - t0),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _atomic_write_text(paths["json"], json.dumps(status, indent=2))
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one local-refinement variant fixed-point task.")
    parser.add_argument("--package-root", type=Path, default=ROOT)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--task-matrix", type=Path, default=Path("config/task_matrix.csv"))
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--enable-local-box-instrumentation", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    package_root = args.package_root.resolve()
    run_root = (args.run_root if args.run_root is not None else _default_run_root(package_root)).resolve()
    task_matrix = _package_path(args.task_matrix, package_root)
    task_id = args.task_id
    if task_id is None:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    output_root = args.output_root

    status = run_task(
        package_root=package_root,
        run_root=run_root,
        task_matrix=task_matrix,
        task_id=int(task_id),
        output_root=output_root,
        device=args.device,
        enable_local_box_instrumentation=bool(args.enable_local_box_instrumentation),
        force=bool(args.force),
    )
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
