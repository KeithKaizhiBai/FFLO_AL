from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch


TARGET_VARIANTS = [
    "baseline",
    "cluster_only",
    "rank_and_cap_k3",
    "rank_and_cap_k2",
    "rank_and_cap_energy_window",
]
DEFAULT_OUTPUT_ROOT = Path("reports/local_refinement_refactor/target_construction_dryrun")
DELTA_EPS = 1.0e-3
FREE_ENERGY_AMBIGUITY_TOL = 1.0e-6
DELTA_REFINE_HALF_WIDTH = 0.03
Q_MAX_ABS = float(np.pi)
MAX_Q_REFINEMENTS = 3
LOCAL_REFINE_ENERGY_WINDOW = max(1.0e-5, 5.0 * FREE_ENERGY_AMBIGUITY_TOL)
MAX_REFINED_MINIMA = 6
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _default_run_root(package_root: Path) -> Path:
    env_run_root = os.environ.get("RUN_ROOT")
    if env_run_root:
        return Path(env_run_root)
    if os.access(package_root, os.W_OK):
        return package_root / "local_refinement_target_construction_dryrun_run"
    fallback = os.environ.get("SCRATCH") or os.environ.get("TMPDIR") or os.environ.get("HOME")
    if fallback:
        return Path(fallback) / "local_refinement_target_construction_dryrun_run"
    return package_root / "local_refinement_target_construction_dryrun_run"


def _package_path(path: Path, package_root: Path) -> Path:
    return path if path.is_absolute() else package_root / path


def _run_path(path: Path, run_root: Path) -> Path:
    return path if path.is_absolute() else run_root / path


def _read_point(points_file: Path, point_id: int) -> dict[str, str]:
    with points_file.open("r", newline="", encoding="utf-8") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            row_id = int(row.get("point_id", idx))
            if row_id == int(point_id):
                row["point_id"] = str(row_id)
                return row
    raise KeyError(f"point_id {point_id} not found in {points_file}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")
    tmp.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        if not fieldnames:
            f.write("")
        else:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    tmp.replace(path)


def _variant_output_paths(run_root: Path, output_root: Path, point_id: int) -> dict[str, Path]:
    point_dir = run_root / output_root / "point_tasks"
    stem = f"point_{int(point_id):03d}"
    return {
        "point_dir": point_dir,
        "summary_csv": point_dir / f"{stem}_summary.csv",
        "candidate_csv": point_dir / f"{stem}_candidates.csv",
        "json": point_dir / f"{stem}.json",
    }


def _row_key(row: dict[str, Any]) -> tuple[str, int]:
    if "basin_id" in row:
        return ("basin", int(row.get("basin_id", 0)))
    return ("rank", int(row.get("minimum_rank", 0)))


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _reasons(row: dict[str, Any]) -> set[str]:
    raw = str(row.get("mandatory_basin_reasons", "none"))
    return {part for part in raw.split(";") if part and part != "none"}


def _reason_count(rows: list[dict[str, Any]], reason: str) -> int:
    return sum(1 for row in rows if reason in _reasons(row))


def _apply_selection_metadata(
    rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    *,
    overflow_policy: str,
) -> list[dict[str, Any]]:
    selected_by_key = {_row_key(row): row for row in selected}
    overflow_count = max((int(row.get("mandatory_overflow_count", 0)) for row in selected), default=0)
    out: list[dict[str, Any]] = []
    for row_in in rows:
        row = dict(row_in)
        selected_row = selected_by_key.get(_row_key(row))
        row["selected_for_refinement"] = int(selected_row is not None)
        row["mandatory_overflow_policy"] = str(overflow_policy)
        row["mandatory_overflow_count"] = int(overflow_count)
        row["mandatory_overflow"] = int(overflow_count > 0)
        if selected_row is None:
            if _truthy(row.get("mandatory_basin", False)) and str(overflow_policy) == "rank_and_cap":
                row["dropped_mandatory_reason"] = "rank_and_cap_overflow"
            elif _truthy(row.get("pruned_by_energy_window", False)):
                row["dropped_mandatory_reason"] = "ordinary_pruned_by_energy_window"
            else:
                row["dropped_mandatory_reason"] = "ordinary_not_in_optional_topk"
            row["rank_and_cap_selection_reason"] = "not_selected"
        else:
            row["dropped_mandatory_reason"] = "not_dropped"
            row["rank_and_cap_selection_reason"] = str(selected_row.get("rank_and_cap_selection_reason", "selected"))
        out.append(row)
    return out


def _run_shared_coarse_scan(kT: float, JA: float, cfg: Any, device: torch.device) -> dict[str, Any]:
    from ml_phase.exact_oracle import (
        _build_branch_candidates,
        _cache_to_scan,
        _diagnose_q_window,
        _expand_cfg_keep_density,
        _incremental_q_strips,
        _merge_q_scan_caches,
        _run_scan_for_q_vec_with_normal,
        _run_scan_with_normal,
        _scan_to_cache,
    )

    timing: dict[str, float] = {
        "base_scan_runtime_sec": 0.0,
        "q_expansion_runtime_sec": 0.0,
        "merge_cache_runtime_sec": 0.0,
    }
    counts: dict[str, int] = {
        "base_q_points_evaluated": 0,
        "added_q_points_evaluated": 0,
        "recomputed_q_points": 0,
        "base_grid_evaluations": 0,
        "incremental_q_grid_evaluations": 0,
        "fallback_full_rescan_grid_evaluations": 0,
        "fallback_full_rescan_used": 0,
        "incremental_expansion_used": 0,
    }

    cfg_current = replace(cfg)
    initial_scan = _run_scan_with_normal(kT, JA, cfg_current, device, q_edge_margin=None)
    timing["base_scan_runtime_sec"] += float(initial_scan.scan_runtime_sec)
    counts["base_q_points_evaluated"] += int(initial_scan.q_points_evaluated)
    counts["base_grid_evaluations"] += int(initial_scan.estimated_grid_evaluations)
    scans = [initial_scan]
    expansion_directions: list[str] = []
    expansion_triggers: list[str] = []
    sym_used = False

    for level in range(MAX_Q_REFINEMENTS):
        current = scans[-1]
        diag = _diagnose_q_window(current, delta_eps=DELTA_EPS, ambiguity_tol=FREE_ENERGY_AMBIGUITY_TOL)
        from ml_phase.exact_oracle import _select_expansion_direction

        direction, trigger = _select_expansion_direction(diag, allow_symmetric_once=(not sym_used))
        if direction == "both":
            sym_used = True
        if direction == "none":
            break
        next_cfg = _expand_cfg_keep_density(cfg_current, direction=direction, q_max_abs=Q_MAX_ABS)
        if (
            float(next_cfg.q_min) == float(cfg_current.q_min)
            and float(next_cfg.q_max) == float(cfg_current.q_max)
            and int(next_cfg.n_q) == int(cfg_current.n_q)
        ):
            break
        expansion_directions.append(direction)
        expansion_triggers.append(trigger)
        cfg_current = next_cfg

        left_q, right_q = _incremental_q_strips(current, cfg_current)
        strip_scans = []
        if left_q.size:
            cfg_left = replace(cfg_current, q_min=float(left_q[0]), q_max=float(left_q[-1]), n_q=int(left_q.size))
            left_scan = _run_scan_for_q_vec_with_normal(
                kT,
                JA,
                cfg_left,
                device,
                q_edge_margin=None,
                q_vec=left_q,
                omega_normal_scalar=float(current.omega_normal_scalar),
            )
            strip_scans.append(left_scan)
            counts["added_q_points_evaluated"] += int(left_scan.q_points_evaluated)
            counts["incremental_q_grid_evaluations"] += int(left_scan.estimated_grid_evaluations)
        if right_q.size:
            cfg_right = replace(cfg_current, q_min=float(right_q[0]), q_max=float(right_q[-1]), n_q=int(right_q.size))
            right_scan = _run_scan_for_q_vec_with_normal(
                kT,
                JA,
                cfg_right,
                device,
                q_edge_margin=None,
                q_vec=right_q,
                omega_normal_scalar=float(current.omega_normal_scalar),
            )
            strip_scans.append(right_scan)
            counts["added_q_points_evaluated"] += int(right_scan.q_points_evaluated)
            counts["incremental_q_grid_evaluations"] += int(right_scan.estimated_grid_evaluations)
        if strip_scans:
            merge_t0 = time.perf_counter()
            merged_cache = _merge_q_scan_caches([_scan_to_cache(current, level)] + [_scan_to_cache(s, level + 1) for s in strip_scans])
            merged_scan = _cache_to_scan(merged_cache, q_edge_margin=None)
            timing["merge_cache_runtime_sec"] += float(time.perf_counter() - merge_t0)
            scans.append(merged_scan)
            counts["incremental_expansion_used"] = 1
        else:
            fallback_t0 = time.perf_counter()
            fallback_scan = _run_scan_with_normal(kT, JA, cfg_current, device, q_edge_margin=None)
            timing["q_expansion_runtime_sec"] += float(time.perf_counter() - fallback_t0)
            counts["recomputed_q_points"] += int(fallback_scan.q_points_evaluated)
            counts["fallback_full_rescan_grid_evaluations"] += int(fallback_scan.estimated_grid_evaluations)
            counts["fallback_full_rescan_used"] = 1
            scans.append(fallback_scan)

    final_scan = scans[-1]
    minima = _build_branch_candidates(final_scan, local_refine_energy_window=LOCAL_REFINE_ENERGY_WINDOW)
    final_diag = _diagnose_q_window(final_scan, delta_eps=DELTA_EPS, ambiguity_tol=FREE_ENERGY_AMBIGUITY_TOL)
    return {
        "initial_scan": initial_scan,
        "final_scan": final_scan,
        "final_diag": final_diag,
        "raw_minima": sorted((dict(row) for row in minima), key=lambda r: float(r.get("DeltaF_local_min", np.inf))),
        "expansion_directions": expansion_directions,
        "expansion_triggers": expansion_triggers,
        "timing": timing,
        "counts": counts,
    }


def _construct_variant_targets(
    raw_minima: list[dict[str, Any]],
    cfg: Any,
    variant: str,
    variant_config: dict[str, Any],
    final_scan: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from ml_phase.exact_oracle import (
        _mandatory_basin_reasons,
        cluster_branch_candidates,
        mark_energy_window_pruning,
        select_local_refine_targets,
    )

    rows = [dict(row) for row in raw_minima]
    raw_candidate_count = int(len(rows))
    if bool(variant_config.get("enable_basin_clustering", False)) and rows:
        coarse_delta = float((float(cfg.delta_max) - float(cfg.delta_min)) / max(1, int(cfg.n_delta) - 1))
        rows = cluster_branch_candidates(
            rows,
            coarse_dq=float(final_scan.dq),
            coarse_dDelta=float(coarse_delta),
            numerical_energy_scale=max(LOCAL_REFINE_ENERGY_WINDOW, FREE_ENERGY_AMBIGUITY_TOL, 1.0e-12),
            delta_eps=DELTA_EPS,
            delta_refine_half_width=DELTA_REFINE_HALF_WIDTH,
        )
    clustered_basin_count = int(len(rows))
    pre_energy_rows = mark_energy_window_pruning(
        rows,
        local_refine_energy_window=LOCAL_REFINE_ENERGY_WINDOW,
        delta_eps=DELTA_EPS,
        delta_refine_half_width=DELTA_REFINE_HALF_WIDTH,
        enabled=False,
    )
    ordinary_before_energy = sum(1 for row in pre_energy_rows if not _truthy(row.get("mandatory_basin", False)))
    rows = mark_energy_window_pruning(
        rows,
        local_refine_energy_window=LOCAL_REFINE_ENERGY_WINDOW,
        delta_eps=DELTA_EPS,
        delta_refine_half_width=DELTA_REFINE_HALF_WIDTH,
        enabled=bool(variant_config.get("energy_window_pruning_enabled", False)),
    )
    selected = select_local_refine_targets(
        rows,
        delta_eps=DELTA_EPS,
        delta_refine_half_width=DELTA_REFINE_HALF_WIDTH,
        max_total_refined_basins=MAX_REFINED_MINIMA,
        enable_selective_refinement=bool(variant_config.get("enable_selective_refinement", False)),
        max_optional_refined_basins=int(variant_config.get("max_optional_refined_basins", 3)),
        mandatory_basins_can_exceed_cap=bool(variant_config.get("mandatory_basins_can_exceed_cap", True)),
        high_risk_overflow_policy=str(variant_config.get("high_risk_overflow_policy", "keep_all")),
        max_edge_risk_basins=int(variant_config.get("max_edge_risk_basins", 1)),
        max_delta_near_eps_basins=int(variant_config.get("max_delta_near_eps_basins", 2)),
        max_near_degenerate_basins=int(variant_config.get("max_near_degenerate_basins", 2)),
    )
    rows = _apply_selection_metadata(
        rows,
        selected,
        overflow_policy=str(variant_config.get("high_risk_overflow_policy", "keep_all")),
    )
    mandatory_rows = [row for row in rows if _truthy(row.get("mandatory_basin", False))]
    ordinary_after_energy = sum(
        1 for row in rows if not _truthy(row.get("mandatory_basin", False)) and not _truthy(row.get("pruned_by_energy_window", False))
    )
    selected_rows = [row for row in rows if int(row.get("selected_for_refinement", 0)) == 1]
    selected_mandatory = [row for row in selected_rows if _truthy(row.get("mandatory_basin", False))]
    selected_optional = [row for row in selected_rows if not _truthy(row.get("mandatory_basin", False))]
    selected_count = int(len(selected_rows))
    energy_pruned = sum(1 for row in rows if _truthy(row.get("pruned_by_energy_window", False)))
    max_total = MAX_REFINED_MINIMA
    summary = {
        "variant": variant,
        "raw_candidate_count": raw_candidate_count,
        "clustered_basin_count": clustered_basin_count,
        "candidate_count_before_clustering": raw_candidate_count,
        "basin_count_after_clustering": clustered_basin_count,
        "global_best_count": _reason_count(rows, "global_best"),
        "edge_risk_basin_count": _reason_count(rows, "edge_risk"),
        "delta_near_eps_basin_count": _reason_count(rows, "Delta_near_epsilon"),
        "near_degenerate_basin_count": _reason_count(rows, "near_degenerate"),
        "mandatory_basin_count": int(len(mandatory_rows)),
        "optional_basin_count": int(len(selected_optional)),
        "ordinary_count_before_energy_window": int(ordinary_before_energy),
        "ordinary_count_after_energy_window": int(ordinary_after_energy),
        "energy_window_enabled": int(bool(variant_config.get("energy_window_pruning_enabled", False))),
        "energy_window_pruned_count": int(energy_pruned),
        "max_optional_refined_basins": variant_config.get("max_optional_refined_basins", ""),
        "max_total_refined_basins": max_total,
        "mandatory_basins_can_exceed_cap": variant_config.get("mandatory_basins_can_exceed_cap", ""),
        "high_risk_overflow_policy": variant_config.get("high_risk_overflow_policy", "keep_all"),
        "selected_refine_target_count": selected_count,
        "selected_mandatory_count": int(len(selected_mandatory)),
        "selected_optional_count": int(len(selected_optional)),
        "mandatory_overflow": int(any(int(row.get("mandatory_overflow", 0)) for row in rows)),
        "mandatory_overflow_count": max((int(row.get("mandatory_overflow_count", 0)) for row in rows), default=0),
        "topk_applied": int(bool(variant_config.get("enable_selective_refinement", False))),
        "topk_pruned_count": int(max(0, ordinary_after_energy - len(selected_optional))),
        "target_count_gate": "pass" if selected_count <= max_total else "fail",
    }
    candidate_rows: list[dict[str, Any]] = []
    for row in rows:
        reasons = _mandatory_basin_reasons(row, DELTA_EPS, DELTA_REFINE_HALF_WIDTH)
        candidate_rows.append(
            {
                "variant": variant,
                "minimum_rank": int(row.get("minimum_rank", 0)),
                "basin_id": int(row.get("basin_id", row.get("minimum_rank", 0))),
                "cluster_size": int(row.get("cluster_size", 1)),
                "merged_branch_ids": str(row.get("merged_branch_ids", row.get("minimum_rank", ""))),
                "q_local_min": float(row.get("q_local_min", np.nan)),
                "Delta_local_min": float(row.get("Delta_local_min", np.nan)),
                "DeltaF_local_min": float(row.get("DeltaF_local_min", np.nan)),
                "energy_above_global": float(row.get("energy_above_global", np.nan)),
                "distance_to_q_edge": float(row.get("distance_to_q_edge", np.nan)),
                "mandatory_basin": int(bool(reasons)),
                "mandatory_basin_reasons": ";".join(sorted(reasons)) if reasons else "none",
                "pruned_by_energy_window": int(_truthy(row.get("pruned_by_energy_window", False))),
                "pruned_reason": str(row.get("pruned_reason", "not_pruned")),
                "selected_for_refinement": int(row.get("selected_for_refinement", 0)),
                "dropped_mandatory_reason": str(row.get("dropped_mandatory_reason", "not_recorded")),
                "rank_and_cap_selection_reason": str(row.get("rank_and_cap_selection_reason", "not_recorded")),
                "mandatory_overflow": int(row.get("mandatory_overflow", 0)),
                "mandatory_overflow_count": int(row.get("mandatory_overflow_count", 0)),
            }
        )
    return summary, candidate_rows


def run_point(
    *,
    package_root: Path,
    run_root: Path,
    points_file: Path,
    point_id: int,
    output_root: Path,
    device: str,
    force: bool,
) -> dict[str, Any]:
    paths = _variant_output_paths(run_root, output_root, point_id)
    if paths["json"].exists() and paths["summary_csv"].exists() and not force:
        existing = json.loads(paths["json"].read_text(encoding="utf-8"))
        if existing.get("status") == "success":
            return existing

    from eta_phase_diagram_cuda import EtaPhaseConfig
    from tfflo_1d_cuda import maybe_set_linalg_backend
    from scripts.run_local_refinement_fixed_point_regression import resolve_variant_config

    point = _read_point(points_file, point_id)
    kT = float(point["kT"])
    JA = float(point["JA"])
    cfg = EtaPhaseConfig()
    cfg_scaled = cfg.scaled()
    maybe_set_linalg_backend(cfg_scaled)
    device_obj = torch.device(device)
    t0 = time.perf_counter()
    status_base = {
        "mode": "target_construction_only",
        "status": "running",
        "point_id": int(point_id),
        "kT": kT,
        "JA": JA,
        "source_category": point.get("category", "unknown"),
        "source_run": point.get("source_run", "unknown"),
        "device": str(device_obj),
        "local_box_scan": "not_run",
        "package_root": str(package_root),
        "run_root": str(run_root),
        "output_summary_csv": str(paths["summary_csv"]),
        "output_candidate_csv": str(paths["candidate_csv"]),
    }
    _write_json(paths["json"], status_base)
    try:
        coarse = _run_shared_coarse_scan(kT, JA, cfg, device_obj)
        summary_rows: list[dict[str, Any]] = []
        candidate_rows: list[dict[str, Any]] = []
        for variant in TARGET_VARIANTS:
            variant_config = resolve_variant_config(variant)
            summary, candidates = _construct_variant_targets(
                coarse["raw_minima"],
                cfg,
                variant,
                variant_config,
                coarse["final_scan"],
            )
            row = {
                **status_base,
                **summary,
                "status": "success",
                "q_expansion_count": int(len(coarse["expansion_directions"])),
                "q_expansion_directions": ";".join(coarse["expansion_directions"]) if coarse["expansion_directions"] else "none",
                "q_expansion_trigger": ";".join(coarse["expansion_triggers"]) if coarse["expansion_triggers"] else "none",
                "initial_q_min": float(coarse["initial_scan"].q_min),
                "initial_q_max": float(coarse["initial_scan"].q_max),
                "final_q_min": float(coarse["final_scan"].q_min),
                "final_q_max": float(coarse["final_scan"].q_max),
                "initial_n_q": int(coarse["initial_scan"].n_q),
                "final_n_q": int(coarse["final_scan"].n_q),
                "final_deltaf_min": float(coarse["final_scan"].deltaf_min),
                "final_q_opt": float(coarse["final_scan"].q_opt),
                "final_delta_opt": float(coarse["final_scan"].delta_opt),
                "base_scan_runtime_sec": float(coarse["timing"]["base_scan_runtime_sec"]),
                "q_expansion_runtime_sec": float(coarse["timing"]["q_expansion_runtime_sec"]),
                "merge_cache_runtime_sec": float(coarse["timing"]["merge_cache_runtime_sec"]),
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
            summary_rows.append(row)
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
        gate_failures = [
            row
            for row in summary_rows
            if str(row["variant"]).startswith("rank_and_cap") and str(row["target_count_gate"]) != "pass"
        ]
        status = {
            **status_base,
            "status": "success",
            "wall_runtime_sec": float(time.perf_counter() - t0),
            "variants": TARGET_VARIANTS,
            "target_count_gate": "pass" if not gate_failures else "fail",
            "gate_failure_count": len(gate_failures),
            "summary_rows": len(summary_rows),
            "candidate_rows": len(candidate_rows),
        }
        _write_json(paths["json"], status)
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
        _write_json(paths["json"], status)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one target-construction-only fixed-point dry-run task.")
    parser.add_argument("--package-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--points-file", type=Path, default=Path("config/validation_points.csv"))
    parser.add_argument("--point-id", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    package_root = args.package_root.resolve()
    run_root = (args.run_root or _default_run_root(package_root)).resolve()
    points_file = _package_path(args.points_file, package_root)
    output_root = args.output_root
    point_id = args.point_id
    if point_id is None:
        point_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    status = run_point(
        package_root=package_root,
        run_root=run_root,
        points_file=points_file,
        point_id=int(point_id),
        output_root=output_root,
        device=args.device,
        force=bool(args.force),
    )
    print(json.dumps(status, indent=2))
    return 0 if status.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
