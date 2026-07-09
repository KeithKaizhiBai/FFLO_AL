from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import tarfile
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


ACCEPTANCE_VARIANTS = ["baseline", "rank_and_cap_k3"]
DEFAULT_REPORT_ROOT = Path("reports/local_refinement_rankcap_acceptance")
DEFAULT_RUN_ROOT_NAME = "local_refinement_rankcap_acceptance_run"
DEFAULT_FIXED_POINTS = ROOT / "reports" / "local_refinement_refactor" / "stage_00_baseline" / "fixed_point_regression_points.csv"
DEFAULT_TARGET_SOURCE = (
    ROOT
    / "local_refinement_target_construction_dryrun"
    / "local_refinement_target_construction_dryrun_run"
    / "reports"
    / "local_refinement_refactor"
    / "target_construction_dryrun"
    / "tables"
)
PACKAGE_ROOT = ROOT / "hpc_packages" / "local_refinement_rankcap_acceptance"
PACKAGE_ARCHIVE = ROOT / "hpc_packages" / "local_refinement_rankcap_acceptance.tar.gz"
RESULT_ARCHIVE = "local_refinement_rankcap_acceptance_results.tar.gz"

FLAG_COLUMNS = [
    "phase_candidate",
    "trusted_exact",
    "training_eligible_exact",
    "q_unresolved",
    "delta_unresolved",
    "rerun_required",
]
FLOAT_COLUMNS = ["DeltaF", "q_opt", "Delta_opt", "positive_delta_gap"]
TARGET_CAP = 3


def _default_run_root(package_root: Path) -> Path:
    env_run_root = os.environ.get("RUN_ROOT")
    if env_run_root:
        return Path(env_run_root)
    if os.access(package_root, os.W_OK):
        return package_root / DEFAULT_RUN_ROOT_NAME
    fallback = os.environ.get("SCRATCH") or os.environ.get("TMPDIR") or os.environ.get("HOME")
    if fallback:
        return Path(fallback) / DEFAULT_RUN_ROOT_NAME
    return package_root / DEFAULT_RUN_ROOT_NAME


def _package_path(path: Path, package_root: Path) -> Path:
    return path if path.is_absolute() else package_root / path


def _run_path(path: Path, run_root: Path) -> Path:
    return path if path.is_absolute() else run_root / path


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        if not fieldnames:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _fval(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _ival(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _mean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def _fmt(value: float) -> str:
    return "" if not math.isfinite(value) else f"{value:.6g}"


def _rel_report_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def normalize_fixed_points(source: Path, output_path: Path) -> list[dict[str, Any]]:
    rows = _read_csv(source)
    if not rows:
        raise FileNotFoundError(f"fixed-point source is empty or missing: {source}")
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        point_id = _ival(row.get("point_id", idx), idx)
        out.append({"point_id": point_id, **{k: v for k, v in row.items() if k != "point_id"}})
    _write_csv(output_path, out)
    return out


def write_task_matrix(fixed_points: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    task_id = 0
    for row in fixed_points:
        for variant in ACCEPTANCE_VARIANTS:
            tasks.append(
                {
                    "task_id": task_id,
                    "variant": variant,
                    "point_id": row["point_id"],
                    "category": row.get("category", "unknown"),
                    "source_run": row.get("source_run", "unknown"),
                    "source_iter": row.get("source_iter", "unknown"),
                    "source_index": row.get("source_index", "unknown"),
                    "kT": row["kT"],
                    "JA": row["JA"],
                }
            )
            task_id += 1
    _write_csv(path, tasks)
    return tasks


def _target_paths(run_root: Path, output_root: Path, point_id: int) -> dict[str, Path]:
    point_dir = run_root / output_root / "point_tasks" / "target"
    stem = f"point_{int(point_id):03d}"
    return {
        "json": point_dir / f"{stem}.json",
        "summary_csv": point_dir / f"{stem}_summary.csv",
        "candidate_csv": point_dir / f"{stem}_candidates.csv",
    }


def run_target_task(
    *,
    package_root: Path,
    run_root: Path,
    fixed_points: Path,
    point_id: int,
    output_root: Path,
    device: str,
    force: bool = False,
) -> dict[str, Any]:
    from eta_phase_diagram_cuda import EtaPhaseConfig
    from scripts.run_local_refinement_fixed_point_regression import resolve_variant_config
    from scripts.run_local_refinement_target_construction_point import (
        _construct_variant_targets,
        _read_point,
        _run_shared_coarse_scan,
    )
    from tfflo_1d_cuda import maybe_set_linalg_backend
    import torch

    paths = _target_paths(run_root, output_root, point_id)
    if paths["json"].exists() and paths["summary_csv"].exists() and not force:
        existing = json.loads(paths["json"].read_text(encoding="utf-8"))
        if existing.get("status") == "success":
            return existing

    point = _read_point(fixed_points, point_id)
    kT = float(point["kT"])
    JA = float(point["JA"])
    cfg = EtaPhaseConfig()
    cfg_scaled = cfg.scaled()
    maybe_set_linalg_backend(cfg_scaled)
    device_obj = torch.device(device)
    t0 = time.perf_counter()
    base_status = {
        "mode": "rankcap_acceptance_target_construction",
        "status": "running",
        "point_id": int(point_id),
        "kT": kT,
        "JA": JA,
        "source_category": point.get("category", "unknown"),
        "device": str(device_obj),
        "local_box_scan": "not_run",
    }
    _write_json(paths["json"], base_status)
    try:
        coarse = _run_shared_coarse_scan(kT, JA, cfg, device_obj)
        summary_rows: list[dict[str, Any]] = []
        candidate_rows: list[dict[str, Any]] = []
        for variant in ACCEPTANCE_VARIANTS:
            variant_config = resolve_variant_config(variant)
            summary, candidates = _construct_variant_targets(
                coarse["raw_minima"],
                cfg,
                variant,
                variant_config,
                coarse["final_scan"],
            )
            summary_row = {
                **base_status,
                **summary,
                "variant": variant,
                "status": "success",
                "q_expansion_count": int(len(coarse["expansion_directions"])),
                "total_q_points_evaluated": int(
                    coarse["counts"]["base_q_points_evaluated"]
                    + coarse["counts"]["added_q_points_evaluated"]
                    + coarse["counts"]["recomputed_q_points"]
                ),
                "total_estimated_grid_evaluations": int(
                    coarse["counts"]["base_grid_evaluations"]
                    + coarse["counts"]["incremental_q_grid_evaluations"]
                    + coarse["counts"]["fallback_full_rescan_grid_evaluations"]
                ),
            }
            summary_rows.append(summary_row)
            for candidate in candidates:
                candidate_rows.append(
                    {
                        "point_id": int(point_id),
                        "kT": kT,
                        "JA": JA,
                        "risk_category": point.get("category", "unknown"),
                        **candidate,
                    }
                )
        _write_csv(paths["summary_csv"], summary_rows)
        _write_csv(paths["candidate_csv"], candidate_rows)
        rank_rows = [r for r in summary_rows if r["variant"] == "rank_and_cap_k3"]
        gate_failures = [r for r in rank_rows if str(r.get("target_count_gate")) != "pass" or _ival(r.get("selected_refine_target_count")) > TARGET_CAP]
        status = {
            **base_status,
            "status": "success",
            "wall_runtime_sec": float(time.perf_counter() - t0),
            "variants": ACCEPTANCE_VARIANTS,
            "target_count_gate": "pass" if not gate_failures else "fail",
            "gate_failure_count": len(gate_failures),
            "summary_rows": len(summary_rows),
            "candidate_rows": len(candidate_rows),
        }
        _write_json(paths["json"], status)
        return status
    except Exception as exc:
        status = {
            **base_status,
            "status": "failed",
            "wall_runtime_sec": float(time.perf_counter() - t0),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(paths["json"], status)
        raise


def _read_task(task_matrix: Path, task_id: int) -> dict[str, str]:
    for row in _read_csv(task_matrix):
        if _ival(row.get("task_id")) == int(task_id):
            return row
    raise KeyError(f"task_id {task_id} not found in {task_matrix}")


def _regression_paths(run_root: Path, output_root: Path, variant: str, point_id: int) -> dict[str, Path]:
    point_dir = run_root / output_root / "point_tasks" / "regression" / variant
    stem = f"point_{int(point_id):03d}"
    return {
        "json": point_dir / f"{stem}.json",
        "csv": point_dir / f"{stem}.csv",
        "npz": point_dir / f"{stem}.npz",
        "local_box": point_dir / f"{stem}_local_box_timing.csv",
    }


def _oracle_result_row(result: Any, task: dict[str, str]) -> dict[str, Any]:
    def arr_value(name: str, default: float = float("nan")) -> Any:
        values = getattr(result, name, None)
        if values is None:
            return default
        return values[0]

    return {
        "task_id": _ival(task["task_id"]),
        "variant": task["variant"],
        "point_id": _ival(task["point_id"]),
        "kT": float(task["kT"]),
        "JA": float(task["JA"]),
        "source_category": task.get("category", "unknown"),
        "source_run": task.get("source_run", "unknown"),
        "phase_candidate": _ival(arr_value("phase_candidate", 0)),
        "phase_label": _ival(arr_value("phase_candidate", 0)),
        "q_opt": float(arr_value("q_opt")),
        "Delta_opt": float(arr_value("delta_opt")),
        "delta_opt": float(arr_value("delta_opt")),
        "DeltaF": float(arr_value("free_energy_gap_to_normal")),
        "positive_delta_gap": float(arr_value("positive_delta_gap")),
        "trusted_exact": _ival(arr_value("trusted_exact", 0)),
        "training_eligible_exact": _ival(arr_value("training_eligible_exact", 0)),
        "q_unresolved": _ival(arr_value("q_unresolved", 0)),
        "delta_unresolved": _ival(arr_value("delta_unresolved", 0)),
        "rerun_required": _ival(arr_value("rerun_required", 0)),
        "local_minima_detected_count": _ival(arr_value("local_minima_detected_count", 0)),
        "clustered_basin_count": _ival(arr_value("clustered_basin_count", 0)),
        "selected_refine_target_count": _ival(arr_value("selected_refine_target_count", 0)),
        "local_boxes_refined_count": _ival(arr_value("local_boxes_refined_count", 0)),
        "local_refinement_runtime_sec": float(arr_value("local_refinement_runtime_sec", 0.0)),
        "point_total_runtime_sec": float(arr_value("point_total_runtime_sec", 0.0)),
        "total_q_points_evaluated": _ival(arr_value("total_q_points_evaluated", 0)),
        "total_estimated_grid_evaluations": _ival(arr_value("total_estimated_grid_evaluations", 0)),
    }


def _rewrite_local_box_file(path: Path, task: dict[str, str]) -> None:
    rows = _read_csv(path)
    if not rows:
        return
    for row in rows:
        row["task_id"] = task["task_id"]
        row["variant"] = task["variant"]
        row["point_id"] = task["point_id"]
    _write_csv(path, rows)


def _gate_a_path(run_root: Path, output_root: Path) -> Path:
    return run_root / output_root / "summary" / "gate_a_status.json"


def _gate_a_passed(run_root: Path, output_root: Path) -> bool:
    path = _gate_a_path(run_root, output_root)
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("gate_a_status") == "pass"


def run_regression_task(
    *,
    package_root: Path,
    run_root: Path,
    task_matrix: Path,
    task_id: int,
    output_root: Path,
    device: str,
    force: bool = False,
) -> dict[str, Any]:
    task = _read_task(task_matrix, task_id)
    variant = task["variant"]
    point_id = _ival(task["point_id"])
    paths = _regression_paths(run_root, output_root, variant, point_id)
    if paths["json"].exists() and paths["csv"].exists() and not force:
        existing = json.loads(paths["json"].read_text(encoding="utf-8"))
        if existing.get("status") in {"success", "skipped_gate_a_failed"}:
            return existing

    base_status = {
        "mode": "rankcap_acceptance_regression_task",
        "status": "running",
        "task_id": int(task_id),
        "variant": variant,
        "point_id": point_id,
        "kT": float(task["kT"]),
        "JA": float(task["JA"]),
        "source_category": task.get("category", "unknown"),
        "device": device,
        "output_csv": str(paths["csv"]),
        "output_npz": str(paths["npz"]),
        "local_box_output_file": str(paths["local_box"]),
    }
    _write_json(paths["json"], base_status)

    if not _gate_a_passed(run_root, output_root):
        status = {**base_status, "status": "skipped_gate_a_failed", "wall_runtime_sec": 0.0}
        _write_json(paths["json"], status)
        return status

    t0 = time.perf_counter()
    try:
        from eta_phase_diagram_cuda import EtaPhaseConfig
        from ml_phase.exact_oracle import evaluate_points
        from scripts.run_local_refinement_fixed_point_regression import resolve_variant_config

        variant_config = resolve_variant_config(variant)
        result = evaluate_points(
            points=np.asarray([[float(task["kT"]), float(task["JA"])]], dtype=np.float64),
            cfg=EtaPhaseConfig(),
            device=device,
            save_every=1,
            output_file=paths["npz"],
            oracle_mode="robust_incremental",
            enable_q_expansion=True,
            enable_delta_refinement=True,
            enable_incremental_q_expansion=True,
            enable_local_box_instrumentation=True,
            local_box_output_file=paths["local_box"],
            **variant_config,
        )
        row = _oracle_result_row(result, task)
        _write_csv(paths["csv"], [row])
        _rewrite_local_box_file(paths["local_box"], task)
        status = {
            **base_status,
            "status": "success",
            "wall_runtime_sec": float(time.perf_counter() - t0),
            "point_total_runtime_sec": float(row["point_total_runtime_sec"]),
            "local_refinement_runtime_sec": float(row["local_refinement_runtime_sec"]),
            "local_boxes_refined_count": int(row["local_boxes_refined_count"]),
            "selected_refine_target_count": int(row["selected_refine_target_count"]),
        }
        _write_json(paths["json"], status)
        return status
    except Exception as exc:
        status = {
            **base_status,
            "status": "failed",
            "wall_runtime_sec": float(time.perf_counter() - t0),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(paths["json"], status)
        raise


def _load_target_rows_from_tasks(report_root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    point_dir = report_root / "point_tasks" / "target"
    summaries: list[dict[str, str]] = []
    candidates: list[dict[str, str]] = []
    statuses: list[dict[str, str]] = []
    for path in sorted(point_dir.glob("point_*_summary.csv")):
        summaries.extend(_read_csv(path))
    for path in sorted(point_dir.glob("point_*_candidates.csv")):
        candidates.extend(_read_csv(path))
    for path in sorted(point_dir.glob("point_*.json")):
        try:
            statuses.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            statuses.append({"status": "failed", "error": f"json decode failed: {exc}", "path": str(path)})
    return summaries, candidates, statuses


def _load_target_rows_from_source(target_source: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    summaries = _read_csv(target_source / "target_construction_by_point.csv")
    candidates = _read_csv(target_source / "target_construction_candidates.csv")
    filtered_summary = [r for r in summaries if r.get("variant") in ACCEPTANCE_VARIANTS]
    filtered_candidates = [r for r in candidates if r.get("variant") in ACCEPTANCE_VARIANTS]
    return filtered_summary, filtered_candidates, [{"status": "success", "source": str(target_source)}]


def _target_comparison_rows(target_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_point: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in target_rows:
        if row.get("variant") in ACCEPTANCE_VARIANTS:
            by_point[_ival(row.get("point_id"))][row["variant"]] = row
    out: list[dict[str, Any]] = []
    for point_id in sorted(by_point):
        base = by_point[point_id].get("baseline", {})
        rank = by_point[point_id].get("rank_and_cap_k3", {})
        out.append(
            {
                "point_id": point_id,
                "risk_category": rank.get("source_category", base.get("source_category", rank.get("risk_category", "unknown"))),
                "baseline_raw_candidate_count": base.get("raw_candidate_count", ""),
                "rank_and_cap_raw_candidate_count": rank.get("raw_candidate_count", ""),
                "baseline_clustered_basin_count": base.get("clustered_basin_count", ""),
                "rank_and_cap_clustered_basin_count": rank.get("clustered_basin_count", ""),
                "baseline_selected_refine_target_count": base.get("selected_refine_target_count", ""),
                "rank_and_cap_selected_refine_target_count": rank.get("selected_refine_target_count", ""),
                "rank_and_cap_mandatory_count": rank.get("mandatory_basin_count", ""),
                "rank_and_cap_optional_count": rank.get("selected_optional_count", rank.get("optional_basin_count", "")),
                "rank_and_cap_selected_mandatory_count": rank.get("selected_mandatory_count", ""),
                "rank_and_cap_mandatory_overflow": rank.get("mandatory_overflow", ""),
                "rank_and_cap_mandatory_overflow_count": rank.get("mandatory_overflow_count", ""),
                "rank_and_cap_target_count_gate": "pass" if _ival(rank.get("selected_refine_target_count"), 999) <= TARGET_CAP else "fail",
            }
        )
    return out


def _selection_reason_counts(candidate_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    for row in candidate_rows:
        if row.get("variant") != "rank_and_cap_k3":
            continue
        if _ival(row.get("selected_for_refinement")) != 1:
            continue
        counts[(row.get("variant", ""), row.get("rank_and_cap_selection_reason", "unknown"))] += 1
    return [
        {"variant": variant, "selection_reason": reason, "selected_target_count": count}
        for (variant, reason), count in sorted(counts.items())
    ]


def _load_regression_rows(report_root: Path) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    rows: list[dict[str, str]] = []
    statuses: list[dict[str, Any]] = []
    point_dir = report_root / "point_tasks" / "regression"
    for variant in ACCEPTANCE_VARIANTS:
        for path in sorted((point_dir / variant).glob("point_*.csv")):
            if not re.fullmatch(r"point_\d+\.csv", path.name):
                continue
            rows.extend(_read_csv(path))
        for path in sorted((point_dir / variant).glob("point_*.json")):
            try:
                statuses.append(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError as exc:
                statuses.append({"status": "failed", "error": f"json decode failed: {exc}", "path": str(path)})
    return rows, statuses


def _compare_regression(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_key[_ival(row.get("point_id"))][row.get("variant", "")] = row
    comparisons: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for point_id in sorted(by_key):
        base = by_key[point_id].get("baseline")
        rank = by_key[point_id].get("rank_and_cap_k3")
        if base is None or rank is None:
            continue
        row: dict[str, Any] = {
            "point_id": point_id,
            "risk_category": rank.get("source_category", base.get("source_category", "unknown")),
            "kT": base.get("kT", ""),
            "JA": base.get("JA", ""),
        }
        mismatch = False
        for col in FLAG_COLUMNS:
            b = base.get(col, "")
            r = rank.get(col, "")
            match = int(b == r)
            row[f"{col}_baseline"] = b
            row[f"{col}_rank_and_cap_k3"] = r
            row[f"{col}_match"] = match
            mismatch = mismatch or not bool(match)
        for col in FLOAT_COLUMNS:
            b = _fval(base.get(col))
            r = _fval(rank.get(col))
            diff = abs(b - r) if math.isfinite(b) and math.isfinite(r) else float("nan")
            row[f"{col}_baseline"] = _fmt(b)
            row[f"{col}_rank_and_cap_k3"] = _fmt(r)
            row[f"{col}_abs_diff"] = _fmt(diff)
        row["q_unresolved_increased"] = int(_ival(rank.get("q_unresolved")) > _ival(base.get("q_unresolved")))
        row["delta_unresolved_increased"] = int(_ival(rank.get("delta_unresolved")) > _ival(base.get("delta_unresolved")))
        row["any_gate_mismatch"] = int(mismatch or row["q_unresolved_increased"] or row["delta_unresolved_increased"])
        comparisons.append(row)
        if row["any_gate_mismatch"]:
            mismatches.append(row)
    return comparisons, mismatches


def _workload_rows(regression_rows: list[dict[str, str]], target_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target_by_point = {int(r["point_id"]): r for r in target_rows if str(r.get("point_id", "")).isdigit()}
    rows: list[dict[str, Any]] = []
    for row in regression_rows:
        point_id = _ival(row.get("point_id"))
        target = target_by_point.get(point_id, {})
        rows.append(
            {
                "variant": row.get("variant", ""),
                "point_id": point_id,
                "risk_category": row.get("source_category", target.get("risk_category", "unknown")),
                "local_boxes_refined_count": row.get("local_boxes_refined_count", ""),
                "selected_refine_target_count": row.get("selected_refine_target_count", ""),
                "rank_and_cap_selected_target_count_from_gate_a": target.get("rank_and_cap_selected_refine_target_count", ""),
                "local_refinement_runtime_sec": row.get("local_refinement_runtime_sec", ""),
                "point_total_runtime_sec": row.get("point_total_runtime_sec", ""),
                "total_q_points_evaluated": row.get("total_q_points_evaluated", ""),
                "total_estimated_grid_evaluations": row.get("total_estimated_grid_evaluations", ""),
            }
        )
    return rows


def _runtime_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for variant in ACCEPTANCE_VARIANTS:
        subset = [r for r in rows if r.get("variant") == variant]
        out.append(
            {
                "variant": variant,
                "point_count": len(subset),
                "mean_local_boxes_refined_count": _fmt(_mean([_fval(r.get("local_boxes_refined_count")) for r in subset])),
                "mean_local_refinement_runtime_sec": _fmt(_mean([_fval(r.get("local_refinement_runtime_sec")) for r in subset])),
                "mean_point_total_runtime_sec": _fmt(_mean([_fval(r.get("point_total_runtime_sec")) for r in subset])),
                "sum_local_refinement_runtime_sec": _fmt(sum(_fval(r.get("local_refinement_runtime_sec")) for r in subset if math.isfinite(_fval(r.get("local_refinement_runtime_sec"))))),
                "sum_point_total_runtime_sec": _fmt(sum(_fval(r.get("point_total_runtime_sec")) for r in subset if math.isfinite(_fval(r.get("point_total_runtime_sec"))))),
            }
        )
    return out


def _status_word(condition: bool | None) -> str:
    if condition is None:
        return "cannot determine"
    return "confirmed" if condition else "failed"


def _match_rate(comparisons: list[dict[str, Any]], col: str) -> float:
    if not comparisons:
        return float("nan")
    return sum(_ival(row.get(f"{col}_match")) for row in comparisons) / len(comparisons)


def _make_placeholder_figure(path: Path, title: str, message: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=13, weight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _make_figures(report_root: Path, target_rows: list[dict[str, Any]], regression_rows: list[dict[str, str]], comparisons: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = report_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    point_ids = [int(r["point_id"]) for r in target_rows]
    selected = [_fval(r.get("rank_and_cap_selected_refine_target_count")) for r in target_rows]
    fig, ax = plt.subplots(figsize=(8.0, 3.5))
    ax.bar(point_ids, selected, color="#4C78A8")
    ax.axhline(TARGET_CAP, color="#333333", linestyle="--", linewidth=1)
    ax.set_xlabel("point id")
    ax.set_ylabel("selected targets")
    ax.set_title("rank_and_cap_k3 selected target count by point")
    fig.tight_layout()
    fig.savefig(figures / "selected_target_count_by_point.png", dpi=180)
    plt.close(fig)

    if not regression_rows:
        for name in [
            "local_boxes_before_after.png",
            "point_runtime_before_after.png",
            "local_refinement_runtime_before_after.png",
            "deltaF_difference.png",
            "qopt_difference.png",
            "Deltaopt_difference.png",
        ]:
            _make_placeholder_figure(figures / name, "physics regression pending on HPC", "No local-box regression rows are available yet.")
        return

    def means_for(field: str) -> list[float]:
        out: list[float] = []
        for variant in ACCEPTANCE_VARIANTS:
            subset = [r for r in regression_rows if r.get("variant") == variant]
            out.append(_mean([_fval(r.get(field)) for r in subset]))
        return out

    for name, field, ylabel, title in [
        ("local_boxes_before_after.png", "local_boxes_refined_count", "mean boxes", "Mean local boxes refined"),
        ("point_runtime_before_after.png", "point_total_runtime_sec", "mean sec", "Mean point total runtime"),
        ("local_refinement_runtime_before_after.png", "local_refinement_runtime_sec", "mean sec", "Mean local-refinement runtime"),
    ]:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.bar(ACCEPTANCE_VARIANTS, means_for(field), color=["#4C78A8", "#54A24B"])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(figures / name, dpi=180)
        plt.close(fig)

    for name, field, ylabel in [
        ("deltaF_difference.png", "DeltaF_abs_diff", "|DeltaF diff|"),
        ("qopt_difference.png", "q_opt_abs_diff", "|q_opt diff|"),
        ("Deltaopt_difference.png", "Delta_opt_abs_diff", "|Delta_opt diff|"),
    ]:
        fig, ax = plt.subplots(figsize=(8.0, 3.5))
        ax.bar([_ival(r.get("point_id")) for r in comparisons], [_fval(r.get(field)) for r in comparisons], color="#F58518")
        ax.set_xlabel("point id")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " by point")
        fig.tight_layout()
        fig.savefig(figures / name, dpi=180)
        plt.close(fig)


def _write_markdown_report(
    report_root: Path,
    *,
    acceptance: dict[str, Any],
    runtime_rows: list[dict[str, Any]],
    mismatch_rows: list[dict[str, Any]],
    regression_pending: bool,
) -> None:
    def ans(key: str) -> str:
        return str(acceptance.get(key, "cannot determine"))

    md = f"""# Rank-and-Cap K3 Acceptance Report

Date: 2026-06-09

## Scope

This is a one-pipeline local-refinement acceptance check for:

- baseline robust_incremental
- rank_and_cap_k3

No k2, energy-window, branch reuse, Powell, adaptive box, GPU batching,
Hamiltonian cache, mini AL, or full AL is enabled.

## Acceptance Status

`acceptance_status = {acceptance['acceptance_status']}`

Physics regression status: `{acceptance['physics_regression_status']}`

## Required Answers

| Question | Answer | Evidence |
| --- | --- | --- |
| 1. 32 points all complete? | {ans('q01_32_points_complete')} | completed_points={acceptance.get('completed_points', '')} |
| 2. rank_and_cap_k3 selected targets <= 3? | {ans('q02_selected_targets_le_3')} | max_rankcap_selected_targets={acceptance.get('max_rankcap_selected_targets', '')} |
| 3. Is there mandatory overflow? | {ans('q03_mandatory_overflow_present')} | overflow_points={acceptance.get('mandatory_overflow_points', '')} |
| 4. Was overflow handled by rank_and_cap? | {ans('q04_overflow_handled')} | selected targets remain <= 3 |
| 5. phase_label 100% match? | {ans('q05_phase_label_100_match')} | match_rate={acceptance.get('phase_label_match_rate', '')} |
| 6. trusted_exact 100% match? | {ans('q06_trusted_exact_100_match')} | match_rate={acceptance.get('trusted_exact_match_rate', '')} |
| 7. training_eligible_exact 100% match? | {ans('q07_training_eligible_100_match')} | match_rate={acceptance.get('training_eligible_exact_match_rate', '')} |
| 8. q_unresolved increased? | {ans('q08_q_unresolved_not_increased')} | increased_count={acceptance.get('q_unresolved_increased_count', '')} |
| 9. delta_unresolved increased? | {ans('q09_delta_unresolved_not_increased')} | increased_count={acceptance.get('delta_unresolved_increased_count', '')} |
| 10. Any timeout? | {ans('q10_no_timeout')} | timeout_count={acceptance.get('timeout_count', '')} |
| 11. Average local boxes before / after? | {ans('q11_local_boxes_reduced')} | {acceptance.get('local_boxes_before_after', '')} |
| 12. Local-refinement runtime reduction? | {ans('q12_local_refinement_runtime_reduced')} | {acceptance.get('local_refinement_runtime_before_after', '')} |
| 13. Point total runtime reduction? | {ans('q13_point_total_runtime_not_higher')} | {acceptance.get('point_total_runtime_before_after', '')} |
| 14. Points closest to mismatch? | {ans('q14_closest_mismatch_points')} | see `tables/pointwise_regression_comparison.csv` |
| 15. Allow one-iteration AL validation? | {ans('q15_allow_one_iteration_al')} | requires acceptance_status=pass |

## Gate A: Target Construction

Gate A status: `{acceptance['gate_a_status']}`.

The rank_and_cap_k3 selected-target cap is evaluated before local-box scans.
Mandatory overflow is allowed only as recorded metadata; it must not increase
the final selected target count above 3.

## Gate B: Local-Box Physics Regression

Gate B status: `{acceptance['gate_b_status']}`.

{"The local-box physics regression has not been run locally. The report therefore records `physics regression pending on HPC` and does not claim physical equivalence." if regression_pending else "The local-box physics regression completed and was compared pointwise against baseline."}

## Gate C: Runtime and Workload

Gate C status: `{acceptance['gate_c_status']}`.

Runtime and workload statistics are reported only after Gate B has completed.

## Mismatch Points

`mismatch_points.csv` row count: {len(mismatch_rows)}

## Runtime Summary

| variant | point_count | mean local boxes | mean local-refinement runtime sec | mean point total runtime sec |
| --- | ---: | ---: | ---: | ---: |
"""
    for row in runtime_rows:
        md += (
            f"| {row.get('variant', '')} | {row.get('point_count', '')} | "
            f"{row.get('mean_local_boxes_refined_count', '')} | "
            f"{row.get('mean_local_refinement_runtime_sec', '')} | "
            f"{row.get('mean_point_total_runtime_sec', '')} |\n"
        )
    md += """
## Do-Not-Claim List

- Do not claim physics equivalence unless Gate B passes.
- Do not enter AL unless `acceptance_status = pass`.
- Do not rerun the old 72 timeout tasks.
- Do not interpret selected target count as phase count.

## Artifacts

- `tables/fixed_points.csv`
- `tables/target_construction_comparison.csv`
- `tables/pointwise_regression_comparison.csv`
- `tables/mismatch_points.csv`
- `tables/runtime_comparison.csv`
- `tables/workload_comparison.csv`
- `tables/selection_reason_counts.csv`
- `tables/acceptance_summary.csv`
- `figures/`
"""
    _write_text(report_root / "rankcap_acceptance_report.md", md)


def _write_decision_log(report_root: Path, acceptance: dict[str, Any]) -> None:
    text = f"""# Decision Log: Rank-and-Cap K3 Acceptance

Date: 2026-06-09

## Decision

`acceptance_status = {acceptance['acceptance_status']}`

## Evidence

- Gate A: {acceptance['gate_a_status']}
- Gate B: {acceptance['gate_b_status']}
- Gate C: {acceptance['gate_c_status']}
- completed_points: {acceptance.get('completed_points', '')}
- max_rankcap_selected_targets: {acceptance.get('max_rankcap_selected_targets', '')}
- phase_label_match_rate: {acceptance.get('phase_label_match_rate', '')}
- trusted_exact_match_rate: {acceptance.get('trusted_exact_match_rate', '')}
- training_eligible_exact_match_rate: {acceptance.get('training_eligible_exact_match_rate', '')}

## Next Action

{acceptance.get('next_action', '')}
"""
    _write_text(report_root / "decision_log.md", text)


def _write_pdf_report(report_root: Path) -> None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except Exception:
        return

    rows = _read_csv(report_root / "tables" / "acceptance_summary.csv")
    if not rows:
        return
    acceptance = rows[0]
    styles = getSampleStyleSheet()
    story: list[Any] = []
    story.append(Paragraph("Rank-and-Cap K3 Acceptance Report", styles["Title"]))
    story.append(Paragraph("Date: 2026-06-09", styles["BodyText"]))
    story.append(Spacer(1, 0.12 * inch))
    story.append(Paragraph("Scope", styles["Heading2"]))
    story.append(Paragraph("One-pipeline acceptance check: baseline robust_incremental vs rank_and_cap_k3 only. No k2, energy-window, branch reuse, Powell, adaptive box, GPU batching, Hamiltonian cache, mini AL, or full AL is enabled.", styles["BodyText"]))
    story.append(Paragraph("Acceptance Status", styles["Heading2"]))
    status_table = Table(
        [
            ["field", "value"],
            ["acceptance_status", acceptance.get("acceptance_status", "")],
            ["physics_regression_status", acceptance.get("physics_regression_status", "")],
            ["Gate A", acceptance.get("gate_a_status", "")],
            ["Gate B", acceptance.get("gate_b_status", "")],
            ["Gate C", acceptance.get("gate_c_status", "")],
            ["max rankcap selected targets", acceptance.get("max_rankcap_selected_targets", "")],
        ],
        colWidths=[2.3 * inch, 3.5 * inch],
    )
    status_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#B0B0B0")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(status_table)
    story.append(Spacer(1, 0.12 * inch))
    story.append(Paragraph("Required Answers", styles["Heading2"]))
    question_rows = [
        ["Question", "Answer"],
        ["32 points all complete?", acceptance.get("q01_32_points_complete", "")],
        ["selected targets <= 3?", acceptance.get("q02_selected_targets_le_3", "")],
        ["mandatory overflow present?", acceptance.get("q03_mandatory_overflow_present", "")],
        ["overflow handled?", acceptance.get("q04_overflow_handled", "")],
        ["phase_label 100% match?", acceptance.get("q05_phase_label_100_match", "")],
        ["trusted_exact 100% match?", acceptance.get("q06_trusted_exact_100_match", "")],
        ["training_eligible 100% match?", acceptance.get("q07_training_eligible_100_match", "")],
        ["q_unresolved not increased?", acceptance.get("q08_q_unresolved_not_increased", "")],
        ["delta_unresolved not increased?", acceptance.get("q09_delta_unresolved_not_increased", "")],
        ["no timeout?", acceptance.get("q10_no_timeout", "")],
        ["local boxes reduced?", acceptance.get("q11_local_boxes_reduced", "")],
        ["local-refinement runtime reduced?", acceptance.get("q12_local_refinement_runtime_reduced", "")],
        ["point total runtime not higher?", acceptance.get("q13_point_total_runtime_not_higher", "")],
        ["closest mismatch points available?", acceptance.get("q14_closest_mismatch_points", "")],
        ["allow one-iteration AL?", acceptance.get("q15_allow_one_iteration_al", "")],
    ]
    q_table = Table(question_rows, colWidths=[3.4 * inch, 2.4 * inch])
    q_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#B0B0B0")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.7),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F7F7")]),
            ]
        )
    )
    story.append(q_table)
    story.append(Spacer(1, 0.12 * inch))
    story.append(Paragraph("Interpretation", styles["Heading2"]))
    story.append(Paragraph(str(acceptance.get("next_action", "")), styles["BodyText"]))
    fig = report_root / "figures" / "selected_target_count_by_point.png"
    if fig.exists():
        story.append(Spacer(1, 0.15 * inch))
        story.append(Image(str(fig), width=6.4 * inch, height=2.8 * inch))
    SimpleDocTemplate(str(report_root / "rankcap_acceptance_report.pdf"), pagesize=letter).build(story)


def collect_acceptance(
    *,
    package_root: Path,
    run_root: Path,
    output_root: Path,
    fixed_points: Path,
    target_source: Path | None,
    create_archive: bool,
) -> dict[str, Any]:
    report_root = run_root / output_root
    tables = report_root / "tables"
    figures = report_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    fixed_rows = normalize_fixed_points(fixed_points, tables / "fixed_points.csv")

    target_rows, candidate_rows, target_statuses = _load_target_rows_from_tasks(report_root)
    if not target_rows and target_source is not None:
        target_rows, candidate_rows, target_statuses = _load_target_rows_from_source(target_source)
    target_comparison = _target_comparison_rows(target_rows)
    selection_counts = _selection_reason_counts(candidate_rows)
    _write_csv(tables / "target_construction_comparison.csv", target_comparison)
    _write_csv(tables / "selection_reason_counts.csv", selection_counts)

    regression_rows, regression_statuses = _load_regression_rows(report_root)
    comparisons, mismatches = _compare_regression(regression_rows)
    runtime_rows = _runtime_summary(regression_rows)
    workload_rows = _workload_rows(regression_rows, target_comparison)

    comparison_fields = [
        "point_id",
        "risk_category",
        "kT",
        "JA",
        "phase_candidate_baseline",
        "phase_candidate_rank_and_cap_k3",
        "phase_candidate_match",
        "trusted_exact_baseline",
        "trusted_exact_rank_and_cap_k3",
        "trusted_exact_match",
        "training_eligible_exact_baseline",
        "training_eligible_exact_rank_and_cap_k3",
        "training_eligible_exact_match",
        "q_unresolved_baseline",
        "q_unresolved_rank_and_cap_k3",
        "q_unresolved_match",
        "delta_unresolved_baseline",
        "delta_unresolved_rank_and_cap_k3",
        "delta_unresolved_match",
        "rerun_required_baseline",
        "rerun_required_rank_and_cap_k3",
        "rerun_required_match",
        "DeltaF_abs_diff",
        "DeltaF_baseline",
        "DeltaF_rank_and_cap_k3",
        "q_opt_abs_diff",
        "q_opt_baseline",
        "q_opt_rank_and_cap_k3",
        "Delta_opt_abs_diff",
        "Delta_opt_baseline",
        "Delta_opt_rank_and_cap_k3",
        "positive_delta_gap_abs_diff",
        "positive_delta_gap_baseline",
        "positive_delta_gap_rank_and_cap_k3",
        "q_unresolved_increased",
        "delta_unresolved_increased",
        "any_gate_mismatch",
    ]
    _write_csv(tables / "pointwise_regression_comparison.csv", comparisons, fieldnames=comparison_fields)
    _write_csv(tables / "mismatch_points.csv", mismatches, fieldnames=comparison_fields)
    _write_csv(tables / "runtime_comparison.csv", runtime_rows)
    _write_csv(
        tables / "workload_comparison.csv",
        workload_rows,
        fieldnames=[
            "variant",
            "point_id",
            "risk_category",
            "local_boxes_refined_count",
            "selected_refine_target_count",
            "rank_and_cap_selected_target_count_from_gate_a",
            "local_refinement_runtime_sec",
            "point_total_runtime_sec",
            "total_q_points_evaluated",
            "total_estimated_grid_evaluations",
        ],
    )

    expected_points = len(fixed_rows)
    rank_target_counts = [_ival(r.get("rank_and_cap_selected_refine_target_count"), 999) for r in target_comparison]
    completed_target_points = len({r["point_id"] for r in target_comparison if r.get("rank_and_cap_target_count_gate") == "pass"})
    max_rank_target = max(rank_target_counts, default=999)
    target_fail_statuses = [s for s in target_statuses if s.get("status") not in {"success"}]
    gate_a_ok = (
        completed_target_points == expected_points
        and max_rank_target <= TARGET_CAP
        and not target_fail_statuses
    )
    overflow_points = sum(1 for r in target_comparison if _ival(r.get("rank_and_cap_mandatory_overflow")) > 0)
    overflow_handled = gate_a_ok and all(_ival(r.get("rank_and_cap_selected_refine_target_count"), 999) <= TARGET_CAP for r in target_comparison)

    regression_point_ids = {(_ival(r.get("point_id")), r.get("variant")) for r in regression_rows}
    regression_complete = all((int(row["point_id"]), variant) in regression_point_ids for row in fixed_rows for variant in ACCEPTANCE_VARIANTS)
    failed_regression = [s for s in regression_statuses if s.get("status") not in {"success"}]
    timeout_count = sum(1 for s in regression_statuses if "TIMEOUT" in str(s.get("error", "")).upper() or "TIMEOUT" in str(s.get("status", "")).upper())
    phase_rate = _match_rate(comparisons, "phase_candidate")
    trusted_rate = _match_rate(comparisons, "trusted_exact")
    training_rate = _match_rate(comparisons, "training_eligible_exact")
    q_increased = sum(_ival(r.get("q_unresolved_increased")) for r in comparisons)
    delta_increased = sum(_ival(r.get("delta_unresolved_increased")) for r in comparisons)
    gate_b_ok = (
        regression_complete
        and not failed_regression
        and not mismatches
        and phase_rate == 1.0
        and trusted_rate == 1.0
        and training_rate == 1.0
        and q_increased == 0
        and delta_increased == 0
        and timeout_count == 0
    )

    by_variant_runtime = {r["variant"]: r for r in runtime_rows}
    base_runtime = by_variant_runtime.get("baseline", {})
    rank_runtime = by_variant_runtime.get("rank_and_cap_k3", {})
    base_boxes = _fval(base_runtime.get("mean_local_boxes_refined_count"))
    rank_boxes = _fval(rank_runtime.get("mean_local_boxes_refined_count"))
    base_local_runtime = _fval(base_runtime.get("mean_local_refinement_runtime_sec"))
    rank_local_runtime = _fval(rank_runtime.get("mean_local_refinement_runtime_sec"))
    base_total_runtime = _fval(base_runtime.get("mean_point_total_runtime_sec"))
    rank_total_runtime = _fval(rank_runtime.get("mean_point_total_runtime_sec"))

    gate_c_known = bool(gate_b_ok and runtime_rows)
    local_boxes_reduced = (rank_boxes <= TARGET_CAP and rank_boxes < base_boxes) if gate_c_known else None
    local_runtime_reduced = (rank_local_runtime < base_local_runtime) if gate_c_known else None
    total_runtime_not_higher = (rank_total_runtime <= base_total_runtime) if gate_c_known else None

    if gate_a_ok:
        gate_a_status = "pass"
    else:
        gate_a_status = "fail"
    if not regression_rows:
        gate_b_status = "pending_hpc_regression"
        gate_c_status = "pending_hpc_regression"
        acceptance_status = "pending_hpc_regression"
    elif gate_b_ok:
        gate_b_status = "pass"
        gate_c_status = "pass" if gate_c_known else "cannot determine"
        acceptance_status = "pass" if gate_a_ok and timeout_count == 0 else "fail"
    else:
        gate_b_status = "fail"
        gate_c_status = "not_evaluated_gate_b_failed"
        acceptance_status = "fail"
    if not gate_a_ok:
        acceptance_status = "fail"

    acceptance = {
        "acceptance_status": acceptance_status,
        "physics_regression_status": "pending on HPC" if not regression_rows else ("pass" if gate_b_ok else "fail"),
        "gate_a_status": gate_a_status,
        "gate_b_status": gate_b_status,
        "gate_c_status": gate_c_status,
        "expected_points": expected_points,
        "completed_points": expected_points if regression_complete else completed_target_points,
        "target_completed_points": completed_target_points,
        "regression_complete": int(regression_complete),
        "max_rankcap_selected_targets": max_rank_target if max_rank_target != 999 else "",
        "mandatory_overflow_points": overflow_points,
        "phase_label_match_rate": _fmt(phase_rate),
        "trusted_exact_match_rate": _fmt(trusted_rate),
        "training_eligible_exact_match_rate": _fmt(training_rate),
        "q_unresolved_increased_count": q_increased if comparisons else "",
        "delta_unresolved_increased_count": delta_increased if comparisons else "",
        "timeout_count": timeout_count,
        "mismatch_point_count": len(mismatches),
        "mismatch_points_empty": int(len(mismatches) == 0),
        "local_boxes_before_after": f"{_fmt(base_boxes)} / {_fmt(rank_boxes)}",
        "local_refinement_runtime_before_after": f"{_fmt(base_local_runtime)} / {_fmt(rank_local_runtime)}",
        "point_total_runtime_before_after": f"{_fmt(base_total_runtime)} / {_fmt(rank_total_runtime)}",
        "q01_32_points_complete": _status_word(regression_complete if regression_rows else None),
        "q02_selected_targets_le_3": _status_word(gate_a_ok),
        "q03_mandatory_overflow_present": "confirmed" if overflow_points > 0 else "failed",
        "q04_overflow_handled": _status_word(overflow_handled),
        "q05_phase_label_100_match": _status_word((phase_rate == 1.0) if comparisons else None),
        "q06_trusted_exact_100_match": _status_word((trusted_rate == 1.0) if comparisons else None),
        "q07_training_eligible_100_match": _status_word((training_rate == 1.0) if comparisons else None),
        "q08_q_unresolved_not_increased": _status_word((q_increased == 0) if comparisons else None),
        "q09_delta_unresolved_not_increased": _status_word((delta_increased == 0) if comparisons else None),
        "q10_no_timeout": _status_word((timeout_count == 0) if regression_statuses else None),
        "q11_local_boxes_reduced": _status_word(local_boxes_reduced),
        "q12_local_refinement_runtime_reduced": _status_word(local_runtime_reduced),
        "q13_point_total_runtime_not_higher": _status_word(total_runtime_not_higher),
        "q14_closest_mismatch_points": "cannot determine" if not comparisons else ("confirmed" if not mismatches else "failed"),
        "q15_allow_one_iteration_al": "confirmed" if acceptance_status == "pass" else "failed",
        "next_action": (
            "Run the generated HPC acceptance workflow and collect the returned report."
            if acceptance_status == "pending_hpc_regression"
            else "Do not enter AL; inspect mismatch_points.csv."
            if acceptance_status == "fail"
            else "User may decide whether to run one-iteration AL validation."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_csv(tables / "acceptance_summary.csv", [acceptance])
    _write_json(report_root / "summary" / "acceptance_summary.json", acceptance)
    _write_json(_gate_a_path(run_root, output_root), {"gate_a_status": gate_a_status, "max_rankcap_selected_targets": max_rank_target, "mandatory_overflow_points": overflow_points})

    _make_figures(report_root, target_comparison, regression_rows, comparisons)
    _write_markdown_report(
        report_root,
        acceptance=acceptance,
        runtime_rows=runtime_rows,
        mismatch_rows=mismatches,
        regression_pending=not bool(regression_rows),
    )
    _write_decision_log(report_root, acceptance)
    _write_pdf_report(report_root)

    if create_archive:
        archive = run_root / RESULT_ARCHIVE
        if archive.exists():
            archive.unlink()
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(report_root, arcname=str(output_root))
            for rel in ["README.md", "RUN_MANIFEST.json", "config"]:
                src = package_root / rel
                if src.exists():
                    tar.add(src, arcname=rel)
        acceptance["return_archive"] = str(archive)
        _write_json(report_root / "summary" / "acceptance_summary.json", acceptance)
    return acceptance


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)

    def ignore(_dir: str, names: list[str]) -> set[str]:
        blocked = {"__pycache__", ".git", "active_runs", "datasets", "figures", "hpc_jobs", "models", "reports"}
        return {name for name in names if name in blocked or name.endswith(".pyc")}

    shutil.copytree(src, dst, ignore=ignore)


def _normalize_shell_scripts(root: Path) -> None:
    for path in root.rglob("*.sh"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        path.write_text(text.replace("\r\n", "\n").replace("\r", "\n"), encoding="utf-8", newline="\n")


def _shell_preamble(run_root_name: str = DEFAULT_RUN_ROOT_NAME) -> str:
    return f"""SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
PACKAGE_ROOT="${{PACKAGE_ROOT:-$(cd "${{SCRIPT_DIR}}/.." && pwd)}}"
RUN_ROOT="${{RUN_ROOT:-}}"
if [ -z "${{RUN_ROOT}}" ]; then
  if [ -w "${{PACKAGE_ROOT}}" ]; then
    RUN_ROOT="${{PACKAGE_ROOT}}/{run_root_name}"
  elif [ -n "${{SCRATCH:-}}" ]; then
    RUN_ROOT="${{SCRATCH}}/{run_root_name}"
  elif [ -n "${{TMPDIR:-}}" ]; then
    RUN_ROOT="${{TMPDIR}}/{run_root_name}"
  else
    RUN_ROOT="${{HOME}}/{run_root_name}"
  fi
fi
mkdir -p "${{RUN_ROOT}}"
RUN_ROOT="$(cd "${{RUN_ROOT}}" && pwd)"
export PACKAGE_ROOT RUN_ROOT
"""


def _write_hpc_scripts(package_root: Path, fixed_count: int) -> None:
    scripts = package_root / "scripts"
    last_point = max(0, fixed_count - 1)
    last_task = max(0, fixed_count * 2 - 1)
    _write_text(
        scripts / "slurm_rankcap_acceptance_target_array.sh",
        f"""#!/bin/bash
#SBATCH --job-name=lr_rc_target
#SBATCH --partition=NV_H100
#SBATCH --exclude=gpuh01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%A_%a.out

set -euo pipefail
{_shell_preamble()}
PYTHON_BIN="${{PYTHON_BIN:-python}}"
cd "${{PACKAGE_ROOT}}"
"${{PYTHON_BIN}}" scripts/run_local_refinement_rankcap_acceptance.py \\
  --mode target-task \\
  --package-root "${{PACKAGE_ROOT}}" \\
  --run-root "${{RUN_ROOT}}" \\
  --fixed-points config/fixed_points.csv \\
  --point-id "${{SLURM_ARRAY_TASK_ID:-0}}" \\
  --device cuda \\
  --force
""",
    )
    _write_text(
        scripts / "slurm_rankcap_acceptance_target_collect.sh",
        f"""#!/bin/bash
#SBATCH --job-name=lr_rc_gatea
#SBATCH --partition=NV_H100
#SBATCH --exclude=gpuh01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

set -euo pipefail
{_shell_preamble()}
PYTHON_BIN="${{PYTHON_BIN:-python}}"
cd "${{PACKAGE_ROOT}}"
"${{PYTHON_BIN}}" scripts/run_local_refinement_rankcap_acceptance.py \\
  --mode collect \\
  --package-root "${{PACKAGE_ROOT}}" \\
  --run-root "${{RUN_ROOT}}" \\
  --fixed-points config/fixed_points.csv \\
  --no-archive
""",
    )
    _write_text(
        scripts / "slurm_rankcap_acceptance_regression_array.sh",
        f"""#!/bin/bash
#SBATCH --job-name=lr_rc_reg
#SBATCH --partition=NV_H100
#SBATCH --exclude=gpuh01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%A_%a.out

set -euo pipefail
{_shell_preamble()}
PYTHON_BIN="${{PYTHON_BIN:-python}}"
cd "${{PACKAGE_ROOT}}"
"${{PYTHON_BIN}}" scripts/run_local_refinement_rankcap_acceptance.py \\
  --mode regression-task \\
  --package-root "${{PACKAGE_ROOT}}" \\
  --run-root "${{RUN_ROOT}}" \\
  --task-matrix config/task_matrix.csv \\
  --task-id "${{SLURM_ARRAY_TASK_ID:-0}}" \\
  --device cuda \\
  --force
""",
    )
    _write_text(
        scripts / "slurm_rankcap_acceptance_collect.sh",
        f"""#!/bin/bash
#SBATCH --job-name=lr_rc_collect
#SBATCH --partition=NV_H100
#SBATCH --exclude=gpuh01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
{_shell_preamble()}
PYTHON_BIN="${{PYTHON_BIN:-python}}"
cd "${{PACKAGE_ROOT}}"
"${{PYTHON_BIN}}" scripts/run_local_refinement_rankcap_acceptance.py \\
  --mode collect \\
  --package-root "${{PACKAGE_ROOT}}" \\
  --run-root "${{RUN_ROOT}}" \\
  --fixed-points config/fixed_points.csv
echo "return_archive=${{RUN_ROOT}}/{RESULT_ARCHIVE}"
""",
    )
    _write_text(
        scripts / "submit_rankcap_acceptance_regression.sh",
        f"""#!/bin/bash
set -euo pipefail
{_shell_preamble()}
PYTHON_BIN="${{PYTHON_BIN:-python}}"
PARTITION="${{PARTITION:-NV_H100}}"
EXCLUDE_NODES="${{EXCLUDE_NODES:-gpuh01}}"
TARGET_MAX_CONCURRENT="${{TARGET_MAX_CONCURRENT:-16}}"
REGRESSION_MAX_CONCURRENT="${{REGRESSION_MAX_CONCURRENT:-32}}"
mkdir -p "${{RUN_ROOT}}/logs"
cd "${{PACKAGE_ROOT}}"

target_id=$(sbatch --parsable --partition="${{PARTITION}}" --exclude="${{EXCLUDE_NODES}}" --array="0-{last_point}%${{TARGET_MAX_CONCURRENT}}" scripts/slurm_rankcap_acceptance_target_array.sh)
echo "${{target_id}}" > "${{RUN_ROOT}}/logs/rankcap_target_array.jobid"
gate_id=$(sbatch --parsable --partition="${{PARTITION}}" --exclude="${{EXCLUDE_NODES}}" --dependency=afterany:${{target_id}} scripts/slurm_rankcap_acceptance_target_collect.sh)
echo "${{gate_id}}" > "${{RUN_ROOT}}/logs/rankcap_target_collect.jobid"
reg_id=$(sbatch --parsable --partition="${{PARTITION}}" --exclude="${{EXCLUDE_NODES}}" --dependency=afterok:${{gate_id}} --array="0-{last_task}%${{REGRESSION_MAX_CONCURRENT}}" scripts/slurm_rankcap_acceptance_regression_array.sh)
echo "${{reg_id}}" > "${{RUN_ROOT}}/logs/rankcap_regression_array.jobid"
collect_id=$(sbatch --parsable --partition="${{PARTITION}}" --exclude="${{EXCLUDE_NODES}}" --dependency=afterany:${{reg_id}} scripts/slurm_rankcap_acceptance_collect.sh)
echo "${{collect_id}}" > "${{RUN_ROOT}}/logs/rankcap_collect.jobid"

echo "target array: ${{target_id}}"
echo "target collect: ${{gate_id}}"
echo "regression array: ${{reg_id}}"
echo "final collect: ${{collect_id}}"
echo "run root: ${{RUN_ROOT}}"
echo "return archive: ${{RUN_ROOT}}/{RESULT_ARCHIVE}"
""",
    )
    _write_text(
        scripts / "collect_rankcap_acceptance_report.sh",
        f"""#!/bin/bash
set -euo pipefail
{_shell_preamble()}
PYTHON_BIN="${{PYTHON_BIN:-python}}"
cd "${{PACKAGE_ROOT}}"
"${{PYTHON_BIN}}" scripts/run_local_refinement_rankcap_acceptance.py \\
  --mode collect \\
  --package-root "${{PACKAGE_ROOT}}" \\
  --run-root "${{RUN_ROOT}}" \\
  --fixed-points config/fixed_points.csv
echo "return_archive=${{RUN_ROOT}}/{RESULT_ARCHIVE}"
""",
    )
    _normalize_shell_scripts(scripts)


def package_hpc(fixed_points: Path, package_root: Path = PACKAGE_ROOT, archive: Path = PACKAGE_ARCHIVE) -> dict[str, Any]:
    if package_root.exists():
        shutil.rmtree(package_root)
    package_root.mkdir(parents=True, exist_ok=True)
    for name in ["ml_phase"]:
        _copy_tree(ROOT / name, package_root / name)
    for name in ["eta_phase_diagram_cuda.py", "tfflo_1d_cuda.py", "MODEL_SPEC.md"]:
        shutil.copy2(ROOT / name, package_root / name)
    (package_root / "scripts").mkdir(parents=True, exist_ok=True)
    for name in [
        "run_local_refinement_rankcap_acceptance.py",
        "run_local_refinement_fixed_point_regression.py",
        "run_local_refinement_target_construction_point.py",
    ]:
        shutil.copy2(ROOT / "scripts" / name, package_root / "scripts" / name)
    fixed_rows = normalize_fixed_points(fixed_points, package_root / "config" / "fixed_points.csv")
    write_task_matrix(fixed_rows, package_root / "config" / "task_matrix.csv")
    _write_hpc_scripts(package_root, len(fixed_rows))
    manifest = {
        "package_name": "local_refinement_rankcap_acceptance",
        "fixed_points": len(fixed_rows),
        "variants": ACCEPTANCE_VARIANTS,
        "result_archive": RESULT_ARCHIVE,
        "submit_script": "scripts/submit_rankcap_acceptance_regression.sh",
        "collect_script": "scripts/collect_rankcap_acceptance_report.sh",
        "excludes": ["gpuh01"],
        "notes": "One-pipeline rank_and_cap_k3 acceptance: target gate, bounded local-box regression, then report collection.",
    }
    _write_json(package_root / "RUN_MANIFEST.json", manifest)
    _write_text(
        package_root / "README.md",
        """# Local-Refinement Rank-and-Cap Acceptance Package

Run from the extracted package root:

```bash
bash scripts/submit_rankcap_acceptance_regression.sh
```

The workflow runs Gate A target construction for baseline and rank_and_cap_k3,
then runs bounded local-box fixed-point regression only if Gate A passes.
It does not run k2, energy-window, branch reuse, Powell, adaptive box, GPU
batching, Hamiltonian cache, mini AL, or full AL.
""",
    )
    if archive.exists():
        archive.unlink()
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(package_root, arcname=package_root.name)
    return {"package_root": str(package_root), "archive": str(archive), "fixed_points": len(fixed_rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description="One-pipeline rank_and_cap_k3 acceptance workflow.")
    parser.add_argument("--mode", choices=["target-task", "regression-task", "collect", "package-hpc"], required=True)
    parser.add_argument("--package-root", type=Path, default=ROOT)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--fixed-points", type=Path, default=DEFAULT_FIXED_POINTS)
    parser.add_argument("--target-source", type=Path, default=None)
    parser.add_argument("--point-id", type=int, default=None)
    parser.add_argument("--task-matrix", type=Path, default=Path("config/task_matrix.csv"))
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-archive", action="store_true")
    args = parser.parse_args()

    package_root = args.package_root.resolve()
    run_root = (args.run_root or _default_run_root(package_root)).resolve()
    output_root = args.output_root
    fixed_points = _package_path(args.fixed_points, package_root)

    if args.mode == "target-task":
        point_id = args.point_id
        if point_id is None:
            point_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        status = run_target_task(
            package_root=package_root,
            run_root=run_root,
            fixed_points=fixed_points,
            point_id=int(point_id),
            output_root=output_root,
            device=args.device,
            force=bool(args.force),
        )
        print(json.dumps(status, indent=2))
        return 0 if status.get("status") == "success" else 1

    if args.mode == "regression-task":
        task_id = args.task_id
        if task_id is None:
            task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        status = run_regression_task(
            package_root=package_root,
            run_root=run_root,
            task_matrix=_package_path(args.task_matrix, package_root),
            task_id=int(task_id),
            output_root=output_root,
            device=args.device,
            force=bool(args.force),
        )
        print(json.dumps(status, indent=2))
        return 0 if status.get("status") in {"success", "skipped_gate_a_failed"} else 1

    if args.mode == "collect":
        target_source = args.target_source
        if target_source is None and DEFAULT_TARGET_SOURCE.exists():
            target_source = DEFAULT_TARGET_SOURCE
        status = collect_acceptance(
            package_root=package_root,
            run_root=run_root,
            output_root=output_root,
            fixed_points=fixed_points,
            target_source=target_source,
            create_archive=not bool(args.no_archive),
        )
        print(json.dumps(status, indent=2))
        return 0 if status.get("acceptance_status") in {"pass", "pending_hpc_regression"} else 1

    if args.mode == "package-hpc":
        status = package_hpc(fixed_points)
        print(json.dumps(status, indent=2))
        return 0

    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
