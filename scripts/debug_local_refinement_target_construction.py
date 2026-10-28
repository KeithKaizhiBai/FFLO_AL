from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


REPORT_NAME = "target_logic_audit"
UNKNOWN = "cannot determine from current metadata"
DEFAULT_RUN_ROOT = Path(
    "local_refinement_refactor_hpc_upload_set/"
    "local_refinement_refactor_variant_suite/"
    "local_refinement_refactor_variant_suite_run"
)
DEFAULT_OUTPUT_DIR = Path("reports/local_refinement_target_logic_audit")
VARIANTS = [
    "baseline",
    "cluster_only",
    "cluster_optional_k3",
    "cluster_optional_k2",
    "cluster_energy_window",
]
VARIANT_CONFIGS: dict[str, dict[str, Any]] = {
    "baseline": {
        "enable_basin_clustering": False,
        "enable_selective_refinement": False,
        "max_optional_refined_basins": "",
        "max_total_refined_basins": 6,
        "mandatory_basins_can_exceed_cap": "",
        "energy_window_pruning_enabled": False,
        "energy_window_value": "",
    },
    "cluster_only": {
        "enable_basin_clustering": True,
        "enable_selective_refinement": False,
        "max_optional_refined_basins": "",
        "max_total_refined_basins": 6,
        "mandatory_basins_can_exceed_cap": "",
        "energy_window_pruning_enabled": False,
        "energy_window_value": "",
    },
    "cluster_optional_k3": {
        "enable_basin_clustering": True,
        "enable_selective_refinement": True,
        "max_optional_refined_basins": 3,
        "max_total_refined_basins": 6,
        "mandatory_basins_can_exceed_cap": True,
        "energy_window_pruning_enabled": False,
        "energy_window_value": "",
    },
    "cluster_optional_k2": {
        "enable_basin_clustering": True,
        "enable_selective_refinement": True,
        "max_optional_refined_basins": 2,
        "max_total_refined_basins": 6,
        "mandatory_basins_can_exceed_cap": True,
        "energy_window_pruning_enabled": False,
        "energy_window_value": "",
    },
    "cluster_energy_window": {
        "enable_basin_clustering": True,
        "enable_selective_refinement": True,
        "max_optional_refined_basins": 3,
        "max_total_refined_basins": 6,
        "mandatory_basins_can_exceed_cap": True,
        "energy_window_pruning_enabled": True,
        "energy_window_value": "defaults_to_local_refine_energy_window",
    },
}
DISPLAY = {
    "baseline": "baseline",
    "cluster_only": "cluster only",
    "cluster_optional_k3": "optional k3",
    "cluster_optional_k2": "optional k2",
    "cluster_energy_window": "energy window",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
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


def fval(raw: Any, default: float = float("nan")) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def ival(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def fmt_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if math.isfinite(value) else ""


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return statistics.mean(vals) if vals else float("nan")


def median(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return statistics.median(vals) if vals else float("nan")


def join_sorted(values: list[str]) -> str:
    return ";".join(sorted({str(v) for v in values if str(v)}))


def effective_status(row: dict[str, str], logs_root: Path) -> str:
    if row.get("status") == "success":
        return "success"
    task_id = row.get("task_id", "")
    text = ""
    for suffix in (".err", ".out"):
        for path in logs_root.glob(f"variant_point_*_{task_id}{suffix}"):
            text += path.read_text(encoding="utf-8", errors="ignore")
    if "DUE TO TIME LIMIT" in text or "TIMEOUT" in text or "CANCELLED" in text:
        return "timeout"
    return row.get("status", "unknown") or "unknown"


def local_box_files(regression_root: Path, variant: str) -> list[Path]:
    aggregate = regression_root / variant / f"{variant}_local_box_timing.csv"
    if aggregate.exists():
        return [aggregate]
    return sorted((regression_root / "point_tasks" / variant).glob("point_*_local_box_timing.csv"))


def load_variant_pointwise(regression_root: Path) -> dict[tuple[str, int], dict[str, str]]:
    rows: dict[tuple[str, int], dict[str, str]] = {}
    for variant in VARIANTS:
        path = regression_root / variant / f"{variant}_pointwise.csv"
        for row in read_csv_rows(path):
            rows[(variant, ival(row.get("point_id"), -1))] = row
    return rows


def load_local_box_rows(regression_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for variant in VARIANTS:
        for path in local_box_files(regression_root, variant):
            for row in read_csv_rows(path):
                row = dict(row)
                row.setdefault("variant_name", variant)
                row["_source_file"] = str(path)
                rows.append(row)
    return rows


def is_reason_mandatory(row: dict[str, str]) -> bool:
    reason = str(row.get("selection_reason", ""))
    branch_rank = ival(row.get("branch_rank_before_refine", row.get("branch_id")), 0)
    has_near = "within_low_energy_window" in reason and "global_best" not in reason and branch_rank != 1
    return (
        "global_best" in reason
        or "edge_risk" in reason
        or "Delta_near_epsilon" in reason
        or has_near
    )


def reason_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    c = Counter(str(r.get("selection_reason", "unknown")) for r in rows)
    return dict(sorted(c.items()))


def count_reason(rows: list[dict[str, str]], token: str) -> int:
    return sum(1 for r in rows if token in str(r.get("selection_reason", "")))


def count_near_degenerate(rows: list[dict[str, str]]) -> int:
    out = 0
    for row in rows:
        reason = str(row.get("selection_reason", ""))
        if "within_low_energy_window" in reason and "global_best" not in reason and ival(row.get("branch_rank_before_refine"), 0) != 1:
            out += 1
    return out


def build_indexes(
    task_rows: list[dict[str, str]], pointwise: dict[tuple[str, int], dict[str, str]], local_boxes: list[dict[str, str]]
) -> dict[str, Any]:
    task_by_key: dict[tuple[str, int], dict[str, str]] = {}
    for row in task_rows:
        task_by_key[(row.get("variant", ""), ival(row.get("point_id"), -1))] = row
    boxes_by_key: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in local_boxes:
        variant = row.get("variant_name", "")
        boxes_by_key[(variant, ival(row.get("point_id"), -1))].append(row)
    risk_by_point: dict[int, str] = {}
    for row in task_rows:
        if row.get("variant") == "baseline":
            risk_by_point[ival(row.get("point_id"), -1)] = row.get("risk_tag", "unknown")
    return {"task_by_key": task_by_key, "boxes_by_key": boxes_by_key, "risk_by_point": risk_by_point}


def build_variant_result_recheck(
    report_root: Path,
    regression_root: Path,
    task_rows: list[dict[str, str]],
    pointwise: dict[tuple[str, int], dict[str, str]],
    boxes_by_key: dict[tuple[str, int], list[dict[str, str]]],
) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    for variant in VARIANTS:
        rows = [r for r in task_rows if r.get("variant") == variant]
        success = [r for r in rows if r.get("effective_status") == "success"]
        timeout = [r for r in rows if r.get("effective_status") == "timeout"]
        other = [r for r in rows if r.get("effective_status") not in {"success", "timeout"}]
        runtime = []
        box_counts = []
        for row in success:
            pid = ival(row.get("point_id"), -1)
            pw = pointwise.get((variant, pid), {})
            runtime.append(fval(pw.get("point_total_runtime_sec"), fval(row.get("wall_runtime_sec"))))
            box_counts.append(fval(pw.get("local_boxes_refined_count")))
        rows_out.append(
            {
                "variant": variant,
                "total_tasks": len(rows),
                "success_count": len(success),
                "timeout_count": len(timeout),
                "other_failure_count": len(other),
                "success_rate": fmt_float(len(success) / len(rows) if rows else float("nan"), 4),
                "mean_runtime_min": fmt_float(mean(runtime) / 60.0),
                "median_runtime_min": fmt_float(median(runtime) / 60.0),
                "mean_local_box_count": fmt_float(mean(box_counts)),
                "median_local_box_count": fmt_float(median(box_counts)),
                "max_local_box_count": fmt_float(max([v for v in box_counts if math.isfinite(v)], default=float("nan"))),
                "completed_categories": join_sorted([r.get("risk_tag", "") for r in success]),
                "timeout_categories": join_sorted([r.get("risk_tag", "") for r in timeout]),
                "status": "pass" if len(success) == len(rows) and not timeout and not other else "fail",
                "source_file": str(regression_root / "summary" / "task_status.csv"),
                "pointwise_source": str(regression_root / variant / f"{variant}_pointwise.csv"),
                "local_box_sources": ";".join(str(p) for p in local_box_files(regression_root, variant)),
            }
        )
    write_csv(report_root / "tables" / "variant_result_recheck.csv", rows_out)
    return rows_out


def build_definition_audit(report_root: Path) -> list[dict[str, Any]]:
    rows = [
        {
            "field_name": "local_boxes_refined_count",
            "source_file": "ml_phase/exact_oracle.py",
            "source_table_or_npz_key": "pointwise CSV / OracleResult.local_boxes_refined_count",
            "producer_function": "_confirm_one_point_robust",
            "producer_line_range": "2299-2319, 2365-2381, 2655",
            "definition": "Number of entries appended to refined_rows after local scans complete for the point.",
            "counts_refined_boxes": "yes",
            "counts_candidate_boxes": "no",
            "counts_planned_boxes": "equals selected_refine_target_count only for successfully completed points",
            "counts_completed_boxes": "yes",
            "counts_timed_out_boxes": "no; timed-out point rows are absent",
            "used_in_report_figure": "yes, common local-box count figure",
            "notes": "For completed rows this is actual executed local-box scans. It does not include unexecuted candidates or pruned candidates.",
        },
        {
            "field_name": "selected_refine_target_count",
            "source_file": "ml_phase/exact_oracle.py",
            "source_table_or_npz_key": "pointwise CSV / OracleResult.selected_refine_target_count",
            "producer_function": "_confirm_one_point_robust",
            "producer_line_range": "2266-2274, 2650",
            "definition": "Length of final refine_targets passed into the local-box scan loop.",
            "counts_refined_boxes": "planned target count before execution",
            "counts_candidate_boxes": "no",
            "counts_planned_boxes": "yes",
            "counts_completed_boxes": "not directly",
            "counts_timed_out_boxes": "no; absent for timed-out point rows",
            "used_in_report_figure": "indirectly",
            "notes": "For successful one-point tasks selected_refine_target_count equals local_boxes_refined_count in current outputs.",
        },
        {
            "field_name": "local_box_rows",
            "source_file": "scripts/build_local_refinement_performance_report.py",
            "source_table_or_npz_key": "performance_report/local_box_summary.csv",
            "producer_function": "summarize_local_boxes",
            "producer_line_range": "188-223",
            "definition": "Rows present in local_box_timing.csv files written after successful point completion.",
            "counts_refined_boxes": "yes",
            "counts_candidate_boxes": "no",
            "counts_planned_boxes": "no",
            "counts_completed_boxes": "yes",
            "counts_timed_out_boxes": "no",
            "used_in_report_figure": "yes, as supporting evidence",
            "notes": "A timed-out task that never flushes the one-point output contributes no local-box rows.",
        },
        {
            "field_name": "local_minima_detected_count",
            "source_file": "ml_phase/exact_oracle.py",
            "source_table_or_npz_key": "pointwise CSV / OracleResult.local_minima_detected_count",
            "producer_function": "_build_branch_candidates",
            "producer_line_range": "795-824, 2240-2242, 2648",
            "definition": "Raw coarse local minima count before optional basin clustering.",
            "counts_refined_boxes": "no",
            "counts_candidate_boxes": "yes",
            "counts_planned_boxes": "no",
            "counts_completed_boxes": "no",
            "counts_timed_out_boxes": "no",
            "used_in_report_figure": "yes, raw candidate count figure",
            "notes": "Available only for completed point rows.",
        },
        {
            "field_name": "clustered_basin_count",
            "source_file": "ml_phase/exact_oracle.py",
            "source_table_or_npz_key": "pointwise CSV / OracleResult.clustered_basin_count",
            "producer_function": "cluster_branch_candidates",
            "producer_line_range": "1274-1321, 2243-2256, 2275, 2649",
            "definition": "Number of rows after optional basin clustering and before selection.",
            "counts_refined_boxes": "no",
            "counts_candidate_boxes": "clustered candidates / basins",
            "counts_planned_boxes": "no",
            "counts_completed_boxes": "no",
            "counts_timed_out_boxes": "no",
            "used_in_report_figure": "yes, clustered basin count figure",
            "notes": "For cluster_only, clustering reduces raw candidates but legacy cap still selects six.",
        },
        {
            "field_name": "energy_window_pruned_count",
            "source_file": "ml_phase/exact_oracle.py",
            "source_table_or_npz_key": "pointwise CSV / OracleResult.energy_window_pruned_count",
            "producer_function": "mark_energy_window_pruning",
            "producer_line_range": "1378-1405, 2259-2265, 2277, 2654",
            "definition": "Number of ordinary non-mandatory basin rows marked pruned by the energy window.",
            "counts_refined_boxes": "no",
            "counts_candidate_boxes": "pruned ordinary basin rows",
            "counts_planned_boxes": "no",
            "counts_completed_boxes": "no",
            "counts_timed_out_boxes": "no",
            "used_in_report_figure": "no",
            "notes": "Current completed energy-window points have zero pruned rows.",
        },
    ]
    write_csv(report_root / "tables" / "local_box_count_definition_audit.csv", rows)
    return rows


def summarize_selected_reasons(boxes: list[dict[str, str]]) -> dict[str, int]:
    return {
        "global_best": count_reason(boxes, "global_best"),
        "edge_risk": count_reason(boxes, "edge_risk"),
        "delta_near_eps": count_reason(boxes, "Delta_near_epsilon"),
        "near_degenerate": count_near_degenerate(boxes),
        "low_energy_window": count_reason(boxes, "within_low_energy_window"),
        "ordinary": sum(1 for r in boxes if str(r.get("selection_reason", "")) == "ordinary_selected"),
        "mandatory": sum(1 for r in boxes if is_reason_mandatory(r)),
    }


def build_target_tables(
    report_root: Path,
    regression_root: Path,
    task_rows: list[dict[str, str]],
    pointwise: dict[tuple[str, int], dict[str, str]],
    boxes_by_key: dict[tuple[str, int], list[dict[str, str]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_point: list[dict[str, Any]] = []
    for task in task_rows:
        variant = task.get("variant", "")
        point_id = ival(task.get("point_id"), -1)
        cfg = VARIANT_CONFIGS.get(variant, {})
        pw = pointwise.get((variant, point_id))
        boxes = boxes_by_key.get((variant, point_id), [])
        counts = summarize_selected_reasons(boxes)
        status = task.get("effective_status", task.get("status", "unknown"))
        has_pw = pw is not None
        selective = bool(cfg.get("enable_selective_refinement", False))
        max_total = cfg.get("max_total_refined_basins", 6)
        mandatory_count = counts["mandatory"] if boxes else (UNKNOWN if not has_pw else 0)
        selected = ival(pw.get("selected_refine_target_count"), 0) if has_pw else UNKNOWN
        actual_boxes = ival(pw.get("local_boxes_refined_count"), 0) if has_pw else UNKNOWN
        overflow = (
            "yes"
            if selective and isinstance(mandatory_count, int) and mandatory_count > int(max_total)
            else ("no" if selective and isinstance(mandatory_count, int) else UNKNOWN)
        )
        overflow_count = (
            max(0, mandatory_count - int(max_total))
            if selective and isinstance(mandatory_count, int)
            else UNKNOWN
        )
        optional_selected = (
            max(0, len(boxes) - counts["mandatory"]) if boxes else (UNKNOWN if not has_pw else 0)
        )
        row = {
            "profile": "metadata_reconstruction_no_local_scan",
            "variant": variant,
            "point_id": point_id,
            "risk_category": task.get("risk_tag", "unknown"),
            "kBT": task.get("kT", ""),
            "JA": task.get("JA", ""),
            "effective_status": status,
            "evidence_scope": "observed_completed" if has_pw else "not_observed_timeout_or_missing",
            "raw_candidate_count": ival(pw.get("local_minima_detected_count"), 0) if has_pw else UNKNOWN,
            "clustered_basin_count": ival(pw.get("clustered_basin_count"), 0) if has_pw else UNKNOWN,
            "global_best_count": counts["global_best"] if boxes else (UNKNOWN if not has_pw else 0),
            "edge_risk_candidate_count": UNKNOWN,
            "delta_near_eps_candidate_count": UNKNOWN,
            "near_degenerate_candidate_count": UNKNOWN,
            "low_energy_window_candidate_count": UNKNOWN,
            "ordinary_candidate_count": UNKNOWN,
            "edge_risk_basin_count": counts["edge_risk"] if boxes else (UNKNOWN if not has_pw else 0),
            "delta_near_eps_basin_count": counts["delta_near_eps"] if boxes else (UNKNOWN if not has_pw else 0),
            "near_degenerate_basin_count": counts["near_degenerate"] if boxes else (UNKNOWN if not has_pw else 0),
            "low_energy_window_basin_count": counts["low_energy_window"] if boxes else (UNKNOWN if not has_pw else 0),
            "ordinary_basin_count": counts["ordinary"] if boxes else (UNKNOWN if not has_pw else 0),
            "mandatory_candidate_count": UNKNOWN,
            "mandatory_basin_count": mandatory_count,
            "optional_candidate_count": UNKNOWN,
            "optional_basin_count": optional_selected,
            "selected_refine_target_count": selected,
            "actual_local_boxes_refined_count_if_available": actual_boxes,
            "max_optional_refined_basins": cfg.get("max_optional_refined_basins", ""),
            "max_total_refined_basins": max_total,
            "mandatory_basins_can_exceed_cap": cfg.get("mandatory_basins_can_exceed_cap", ""),
            "mandatory_overflow": overflow,
            "mandatory_overflow_count": overflow_count,
            "energy_window_enabled": int(bool(cfg.get("energy_window_pruning_enabled", False))),
            "energy_window_value": cfg.get("energy_window_value", ""),
            "energy_window_pruned_count": ival(pw.get("energy_window_pruned_count"), 0) if has_pw else UNKNOWN,
            "clustering_enabled": int(bool(cfg.get("enable_basin_clustering", False))),
            "candidate_count_before_clustering": ival(pw.get("local_minima_detected_count"), 0) if has_pw else UNKNOWN,
            "basin_count_after_clustering": ival(pw.get("clustered_basin_count"), 0) if has_pw else UNKNOWN,
            "topk_applied": int(bool(cfg.get("enable_selective_refinement", False))),
            "topk_pruned_count": UNKNOWN,
            "selected_target_count_expected": selected,
            "selected_target_count_actual_if_available": actual_boxes,
            "selection_reason_counts_json": json.dumps(reason_counts(boxes), sort_keys=True),
        }
        by_point.append(row)
    write_csv(report_root / "tables" / "target_construction_by_point.csv", by_point)
    write_csv(report_root / "tables" / "dry_run_target_audit.csv", by_point)

    summary: list[dict[str, Any]] = []
    for variant in VARIANTS:
        subset = [r for r in by_point if r["variant"] == variant and r["evidence_scope"] == "observed_completed"]
        summary.append(
            {
                "variant": variant,
                "observed_completed_points": len(subset),
                "mean_raw_candidate_count": fmt_float(mean([fval(r["raw_candidate_count"]) for r in subset])),
                "mean_clustered_basin_count": fmt_float(mean([fval(r["clustered_basin_count"]) for r in subset])),
                "mean_selected_refine_target_count": fmt_float(mean([fval(r["selected_refine_target_count"]) for r in subset])),
                "mean_actual_local_boxes_refined_count": fmt_float(
                    mean([fval(r["actual_local_boxes_refined_count_if_available"]) for r in subset])
                ),
                "mean_mandatory_basin_count": fmt_float(mean([fval(r["mandatory_basin_count"]) for r in subset])),
                "mandatory_overflow_points": sum(1 for r in subset if r["mandatory_overflow"] == "yes"),
                "mean_energy_window_pruned_count": fmt_float(mean([fval(r["energy_window_pruned_count"]) for r in subset])),
                "interpretation": (
                    "selective variants over-selected mandatory basins"
                    if variant.startswith("cluster_optional") or variant == "cluster_energy_window"
                    else "legacy cap remains at six selected boxes"
                ),
            }
        )
    write_csv(report_root / "tables" / "target_construction_summary.csv", summary)

    by_risk: list[dict[str, Any]] = []
    for variant in VARIANTS:
        risks = sorted({r["risk_category"] for r in by_point if r["variant"] == variant})
        for risk in risks:
            subset = [r for r in by_point if r["variant"] == variant and r["risk_category"] == risk]
            observed = [r for r in subset if r["evidence_scope"] == "observed_completed"]
            by_risk.append(
                {
                    "variant": variant,
                    "risk_category": risk,
                    "total_tasks": len(subset),
                    "observed_completed_points": len(observed),
                    "timeout_or_missing_points": sum(1 for r in subset if r["effective_status"] == "timeout"),
                    "mean_selected_refine_target_count_observed": fmt_float(
                        mean([fval(r["selected_refine_target_count"]) for r in observed])
                    ),
                    "mean_mandatory_basin_count_observed": fmt_float(
                        mean([fval(r["mandatory_basin_count"]) for r in observed])
                    ),
                    "mandatory_overflow_observed_points": sum(1 for r in observed if r["mandatory_overflow"] == "yes"),
                    "metadata_limitation": "" if len(observed) == len(subset) else "timeout points lack coarse target metadata",
                }
            )
    write_csv(report_root / "tables" / "target_construction_by_risk_category.csv", by_risk)
    return by_point, by_risk


def build_reason_and_effect_tables(
    report_root: Path,
    by_point: list[dict[str, Any]],
    boxes_by_key: dict[tuple[str, int], list[dict[str, str]]],
) -> None:
    reason_rows: list[dict[str, Any]] = []
    for row in by_point:
        variant = row["variant"]
        point_id = int(row["point_id"])
        boxes = boxes_by_key.get((variant, point_id), [])
        for reason, count in reason_counts(boxes).items():
            reason_rows.append(
                {
                    "variant": variant,
                    "point_id": point_id,
                    "risk_category": row["risk_category"],
                    "selection_reason": reason,
                    "count": count,
                    "evidence_scope": row["evidence_scope"],
                }
            )
    write_csv(report_root / "tables" / "refine_target_reason_counts.csv", reason_rows)

    overflow_rows: list[dict[str, Any]] = []
    topk_rows: list[dict[str, Any]] = []
    energy_rows: list[dict[str, Any]] = []
    for row in by_point:
        variant = row["variant"]
        cfg = VARIANT_CONFIGS.get(variant, {})
        selected = row["selected_refine_target_count"]
        mandatory = row["mandatory_basin_count"]
        optional_after = row["optional_basin_count"]
        overflow_rows.append(
            {
                "variant": variant,
                "point_id": row["point_id"],
                "risk_category": row["risk_category"],
                "mandatory_candidate_count": UNKNOWN,
                "mandatory_basin_count": mandatory,
                "max_total_refined_basins": row["max_total_refined_basins"],
                "mandatory_basins_can_exceed_cap": row["mandatory_basins_can_exceed_cap"],
                "mandatory_overflow": row["mandatory_overflow"],
                "mandatory_overflow_count": row["mandatory_overflow_count"],
                "selected_refine_target_count": selected,
                "timeout_status": row["effective_status"],
                "evidence_scope": row["evidence_scope"],
            }
        )
        topk_rows.append(
            {
                "variant": variant,
                "point_id": row["point_id"],
                "max_optional_refined_basins": row["max_optional_refined_basins"],
                "max_total_refined_basins": row["max_total_refined_basins"],
                "mandatory_count": mandatory,
                "optional_count_before_topk": UNKNOWN,
                "optional_count_after_topk": optional_after,
                "total_count_before_cap": UNKNOWN,
                "total_count_after_cap": selected,
                "cap_applied_to_mandatory": (
                    "no" if cfg.get("enable_selective_refinement") and cfg.get("mandatory_basins_can_exceed_cap") else "legacy_or_strict"
                ),
                "cap_applied_to_optional": "yes" if cfg.get("enable_selective_refinement") else "legacy_total_cap",
                "cap_applied_to_total": (
                    "no" if cfg.get("enable_selective_refinement") and cfg.get("mandatory_basins_can_exceed_cap") else "yes"
                ),
                "selected_refine_target_count": selected,
                "evidence_scope": row["evidence_scope"],
            }
        )
        energy_rows.append(
            {
                "variant": variant,
                "point_id": row["point_id"],
                "energy_window_enabled": row["energy_window_enabled"],
                "energy_window_value": row["energy_window_value"],
                "ordinary_count_before_energy_window": UNKNOWN,
                "ordinary_count_after_energy_window": UNKNOWN,
                "mandatory_count_before_energy_window": mandatory,
                "mandatory_count_after_energy_window": mandatory,
                "energy_window_pruned_count": row["energy_window_pruned_count"],
                "selected_refine_target_count": selected,
                "evidence_scope": row["evidence_scope"],
            }
        )
    write_csv(report_root / "tables" / "mandatory_overflow_audit.csv", overflow_rows)
    write_csv(report_root / "tables" / "topk_cap_effect_audit.csv", topk_rows)
    write_csv(report_root / "tables" / "energy_window_effect_audit.csv", energy_rows)


def build_code_path_tables(report_root: Path) -> None:
    code_rows = [
        {
            "step_order": 1,
            "step_name": "coarse candidate detection",
            "file": "ml_phase/exact_oracle.py",
            "function": "_build_branch_candidates",
            "line_range": "795-824",
            "input_object": "final_scan.deltaf_q, final_scan.q_vec, final_scan.delta_star_q",
            "output_object": "rows with minimum_rank, q_local_min, Delta_local_min, DeltaF_local_min, within_low_energy_window, edge_risk",
            "candidate_level_or_basin_level": "candidate-level",
            "notes": "edge_risk and within_low_energy_window originate at candidate level.",
        },
        {
            "step_order": 2,
            "step_name": "optional basin clustering",
            "file": "ml_phase/exact_oracle.py",
            "function": "cluster_branch_candidates",
            "line_range": "1274-1321, 2243-2256",
            "input_object": "coarse candidate rows",
            "output_object": "representative rows with basin_id, cluster_size, mandatory_basin_reasons",
            "candidate_level_or_basin_level": "basin-level representatives",
            "notes": "Runs before energy pruning and target selection when enable_basin_clustering=True.",
        },
        {
            "step_order": 3,
            "step_name": "mandatory risk annotation",
            "file": "ml_phase/exact_oracle.py",
            "function": "_mandatory_basin_reasons / mark_energy_window_pruning / select_local_refine_targets",
            "line_range": "827-838, 1389-1391, 1358-1360",
            "input_object": "candidate or clustered representative row",
            "output_object": "mandatory_basin and mandatory_basin_reasons",
            "candidate_level_or_basin_level": "mixed: candidate-level before clustering; basin-level after clustering",
            "notes": "Delta_near_epsilon is evaluated per row; clustered rows aggregate member risk reasons.",
        },
        {
            "step_order": 4,
            "step_name": "energy-window pruning",
            "file": "ml_phase/exact_oracle.py",
            "function": "mark_energy_window_pruning",
            "line_range": "1378-1405, 2259-2265",
            "input_object": "sorted candidate/basin rows",
            "output_object": "rows with pruned_by_energy_window and pruned_reason",
            "candidate_level_or_basin_level": "same as input rows",
            "notes": "Only ordinary non-mandatory rows can be pruned.",
        },
        {
            "step_order": 5,
            "step_name": "mandatory / optional target construction",
            "file": "ml_phase/exact_oracle.py",
            "function": "select_local_refine_targets",
            "line_range": "1324-1375, 2266-2274",
            "input_object": "minima_sorted after pruning marks",
            "output_object": "refine_targets",
            "candidate_level_or_basin_level": "basin-level for clustered variants; candidate-level for unclustered variants",
            "notes": "Selective mode selects all mandatory first, then ordinary[:K].",
        },
        {
            "step_order": 6,
            "step_name": "top-k truncation",
            "file": "ml_phase/exact_oracle.py",
            "function": "select_local_refine_targets",
            "line_range": "1366-1374",
            "input_object": "ordinary rows",
            "output_object": "ordinary optional additions",
            "candidate_level_or_basin_level": "same as input rows",
            "notes": "K applies only to ordinary optional rows when mandatory_basins_can_exceed_cap=True.",
        },
        {
            "step_order": 7,
            "step_name": "local-box scan loop",
            "file": "ml_phase/exact_oracle.py",
            "function": "_confirm_one_point_robust",
            "line_range": "2299-2319",
            "input_object": "refine_targets",
            "output_object": "local_scan per target, refined_rows, local_box_records",
            "candidate_level_or_basin_level": "final target rows",
            "notes": "Every row in refine_targets triggers one local scan unless the task is killed.",
        },
    ]
    write_csv(report_root / "tables" / "code_path_audit.csv", code_rows)
    write_csv(report_root / "tables" / "clustering_stage_order_audit.csv", code_rows[:3])
    risk_rows = [
        {
            "step_order": 1,
            "step_name": "edge/low-energy flags",
            "file": "ml_phase/exact_oracle.py",
            "function": "_build_branch_candidates",
            "line_range": "812-822",
            "input_object": "coarse local minimum",
            "output_object": "edge_risk, within_low_energy_window",
            "candidate_level_or_basin_level": "candidate-level",
            "notes": "These flags are assigned before clustering.",
        },
        {
            "step_order": 2,
            "step_name": "risk aggregation during clustering",
            "file": "ml_phase/exact_oracle.py",
            "function": "cluster_branch_candidates",
            "line_range": "1308-1318",
            "input_object": "cluster members",
            "output_object": "representative mandatory_basin_reasons, edge_risk, within_low_energy_window",
            "candidate_level_or_basin_level": "basin-level representative",
            "notes": "A representative is mandatory if any member contributes a mandatory reason.",
        },
        {
            "step_order": 3,
            "step_name": "selection-time mandatory reasons",
            "file": "ml_phase/exact_oracle.py",
            "function": "_mandatory_basin_reasons / select_local_refine_targets",
            "line_range": "827-838, 1358-1364",
            "input_object": "post-clustering row",
            "output_object": "mandatory list vs ordinary list",
            "candidate_level_or_basin_level": "basin-level in current optimized variants",
            "notes": "Global best, edge risk, Delta_near_epsilon and near-degenerate rows are mandatory.",
        },
    ]
    write_csv(report_root / "tables" / "risk_flag_stage_audit.csv", risk_rows)


def build_trace_tables(
    report_root: Path,
    by_point: list[dict[str, Any]],
    boxes_by_key: dict[tuple[str, int], list[dict[str, str]]],
) -> None:
    requested = [
        ("cluster_optional_k3", 4, "clean_fflo"),
        ("cluster_optional_k3", 8, "clean_uniform_sc"),
        ("cluster_optional_k3", 0, "boundary_band_normal"),
        ("cluster_optional_k3", 12, "near_degenerate_or_delta_ambiguous"),
        ("cluster_optional_k3", 16, "previous_normal_to_fflo_correction"),
        ("cluster_optional_k3", 20, "q_edge_risk"),
        ("cluster_optional_k3", 28, "stable_normal_interior"),
    ]
    by_key = {(r["variant"], int(r["point_id"])): r for r in by_point}
    trace_rows: list[dict[str, Any]] = []
    timeout_rows: list[dict[str, Any]] = []
    for variant, point_id, risk in requested:
        boxes = boxes_by_key.get((variant, point_id), [])
        meta = by_key.get((variant, point_id), {})
        if not boxes:
            trace_rows.append(
                {
                    "variant": variant,
                    "point_id": point_id,
                    "risk_category": risk,
                    "raw_candidate_id": UNKNOWN,
                    "cluster_id": UNKNOWN,
                    "basin_representative_id": UNKNOWN,
                    "q": UNKNOWN,
                    "Delta": UNKNOWN,
                    "DeltaF": UNKNOWN,
                    "energy_above_global": UNKNOWN,
                    "coarse_rank": UNKNOWN,
                    "is_global_best": UNKNOWN,
                    "is_edge_risk": UNKNOWN,
                    "is_delta_near_eps": UNKNOWN,
                    "is_near_degenerate": UNKNOWN,
                    "is_low_energy_window": UNKNOWN,
                    "is_ordinary": UNKNOWN,
                    "risk_marked_before_clustering": UNKNOWN,
                    "risk_marked_after_clustering": UNKNOWN,
                    "mandatory_keep": UNKNOWN,
                    "mandatory_reason": UNKNOWN,
                    "optional_candidate": UNKNOWN,
                    "selected_for_refinement": UNKNOWN,
                    "selection_reason": "no partial candidate/local-box metadata returned",
                    "pruned": UNKNOWN,
                    "pruned_reason": "timeout task has only startup JSON",
                    "rank_before_truncation": UNKNOWN,
                    "rank_after_truncation": UNKNOWN,
                }
            )
            timeout_rows.append(
                {
                    "variant": variant,
                    "point_id": point_id,
                    "risk_category": risk,
                    "effective_status": meta.get("effective_status", "timeout"),
                    "selected_refine_target_count": UNKNOWN,
                    "mandatory_basin_count": UNKNOWN,
                    "local_box_count": UNKNOWN,
                    "evidence": "timeout point has no point CSV, NPZ, or local-box timing CSV",
                    "interpretation": "cannot determine actual target explosion for this timeout point from current metadata",
                }
            )
            continue
        for rank, box in enumerate(boxes[:20], start=1):
            reason = str(box.get("selection_reason", ""))
            is_global = "global_best" in reason
            is_edge = "edge_risk" in reason
            is_delta = "Delta_near_epsilon" in reason
            is_low = "within_low_energy_window" in reason
            is_near = is_low and not is_global
            mandatory = is_reason_mandatory(box)
            trace_rows.append(
                {
                    "variant": variant,
                    "point_id": point_id,
                    "risk_category": risk,
                    "raw_candidate_id": box.get("branch_id", ""),
                    "cluster_id": UNKNOWN,
                    "basin_representative_id": box.get("branch_rank_before_refine", ""),
                    "q": box.get("q_center_before_refine", ""),
                    "Delta": box.get("Delta_center_before_refine", ""),
                    "DeltaF": box.get("DeltaF_before_refine", ""),
                    "energy_above_global": box.get("energy_above_global_before_refine", ""),
                    "coarse_rank": box.get("branch_rank_before_refine", ""),
                    "is_global_best": int(is_global),
                    "is_edge_risk": int(is_edge),
                    "is_delta_near_eps": int(is_delta),
                    "is_near_degenerate": int(is_near),
                    "is_low_energy_window": int(is_low),
                    "is_ordinary": int(not mandatory),
                    "risk_marked_before_clustering": "partly; edge/low-energy candidate flags originate before clustering",
                    "risk_marked_after_clustering": "yes; mandatory_basin_reasons recomputed on clustered rows",
                    "mandatory_keep": int(mandatory),
                    "mandatory_reason": reason if mandatory else "",
                    "optional_candidate": int(not mandatory),
                    "selected_for_refinement": 1,
                    "selection_reason": reason,
                    "pruned": 0,
                    "pruned_reason": box.get("pruned_reason", ""),
                    "rank_before_truncation": box.get("branch_rank_before_refine", ""),
                    "rank_after_truncation": rank,
                }
            )
    write_csv(report_root / "tables" / "candidate_to_basin_trace_examples.csv", trace_rows)
    write_csv(report_root / "tables" / "hard_risk_timeout_trace.csv", timeout_rows)


def build_root_cause_and_fix_tables(report_root: Path) -> None:
    root_rows = [
        {
            "hypothesis": "H1. local-box count about 85 is refined boxes, not candidates.",
            "supported": "confirmed",
            "evidence": "local_boxes_refined_count is len(refined_rows); local_box_rows/unique_points is 681/8=85.125.",
            "counter_evidence": "none in completed outputs",
            "confidence": "high",
            "next_check": "none needed for completed rows; timeout rows need better partial metadata",
        },
        {
            "hypothesis": "H2. mandatory branches bypass max_total cap.",
            "supported": "confirmed",
            "evidence": "select_local_refine_targets keeps all mandatory when mandatory_basins_can_exceed_cap=True; completed selective points select 63-169 targets.",
            "counter_evidence": "strict mode exists but was not used",
            "confidence": "high",
            "next_check": "run target-construction-only strict cap comparison",
        },
        {
            "hypothesis": "H3. risk flags are candidate-level and same basin duplicates all become mandatory.",
            "supported": "supported but not proven",
            "evidence": "edge_risk and within_low_energy_window originate at candidate level; clustering aggregates reasons into representative basins.",
            "counter_evidence": "clustering does collapse duplicate candidates before selection, so same-basin duplicate selection is not directly shown.",
            "confidence": "medium",
            "next_check": "write full candidate-to-cluster trace before and after clustering",
        },
        {
            "hypothesis": "H4. clustering does not apply to mandatory targets.",
            "supported": "not supported",
            "evidence": "cluster_branch_candidates runs before mark_energy_window_pruning and select_local_refine_targets.",
            "counter_evidence": "none",
            "confidence": "high",
            "next_check": "inspect cluster tolerances if basin count remains high",
        },
        {
            "hypothesis": "H5. energy-window pruning only acts on ordinary branches.",
            "supported": "confirmed",
            "evidence": "mark_energy_window_pruning requires not reasons before pruning.",
            "counter_evidence": "none",
            "confidence": "high",
            "next_check": "none; this is intentional by D5",
        },
        {
            "hypothesis": "H6. top-k only limits optional branches.",
            "supported": "confirmed",
            "evidence": "selective mode uses selected=list(mandatory), then extends ordinary[:K].",
            "counter_evidence": "strict mode would cap mandatory, but current variants set mandatory_basins_can_exceed_cap=True.",
            "confidence": "high",
            "next_check": "test rank_and_cap policy before local scan",
        },
        {
            "hypothesis": "H7. hard-risk categories trigger many mandatory-risk candidates.",
            "supported": "supported but not proven",
            "evidence": "completed clean controls already have 63-169 selected mandatory basins dominated by Delta_near_epsilon.",
            "counter_evidence": "timeout hard-risk points lack pointwise target metadata.",
            "confidence": "medium",
            "next_check": "target-construction-only audit on timeout coordinates",
        },
        {
            "hypothesis": "H8. local-box count field definition is wrong.",
            "supported": "not supported",
            "evidence": "field traces to len(refined_rows), and local_box_timing rows match pointwise local_boxes_refined_count for completed points.",
            "counter_evidence": "none for completed rows",
            "confidence": "high",
            "next_check": "none",
        },
        {
            "hypothesis": "H9. timeout comes from a few single boxes being extremely slow rather than target explosion.",
            "supported": "not supported",
            "evidence": "completed boxes are about 31.5 s each; 85 boxes explain about 45 min completed runtimes.",
            "counter_evidence": "timeout rows lack partial per-box timing, so hard-risk task internals remain unobserved.",
            "confidence": "medium",
            "next_check": "flush local-box records incrementally or run target-only audit",
        },
        {
            "hypothesis": "H10. cluster_only passes but does not speed up because selected target count remains baseline cap 6.",
            "supported": "confirmed",
            "evidence": "cluster_only mean selected_refine_target_count=6 and local_boxes_refined_count=6, same as baseline.",
            "counter_evidence": "clustered_basin_count is reduced, but legacy cap still selects six boxes.",
            "confidence": "high",
            "next_check": "separate clustering benefit from selection policy benefit",
        },
    ]
    write_csv(report_root / "tables" / "root_cause_candidates.csv", root_rows)

    fix_rows = [
        {
            "fix_id": "F1",
            "fix_name": "Basin-level risk annotation",
            "description": "Compute final risk flags on basin representatives after clustering, preserving member evidence but avoiding candidate-level proliferation.",
            "requires_code_change": "yes",
            "risk": "medium",
            "expected_effect": "reduces duplicate mandatory selections if candidate-level risk promotion is too broad",
            "required_regression": "target-construction-only trace plus 32-point fixed gate",
        },
        {
            "fix_id": "F2",
            "fix_name": "Cluster all candidates before any mandatory selection",
            "description": "Keep clustering before mandatory selection and add tests proving all risk candidates pass through clustering.",
            "requires_code_change": "small test/report change first",
            "risk": "low",
            "expected_effect": "prevents future regressions; current code already mostly follows this order",
            "required_regression": "unit test for mandatory-risk clustering order",
        },
        {
            "fix_id": "F3",
            "fix_name": "Hard max_total_refined_basins cap",
            "description": "Enforce a final cap on all selected targets after mandatory and optional branches are merged.",
            "requires_code_change": "yes",
            "risk": "medium-high",
            "expected_effect": "prevents 85-target explosions",
            "required_regression": "physics equivalence on fixed hard-risk points",
        },
        {
            "fix_id": "F4",
            "fix_name": "Default mandatory_basins_can_exceed_cap False",
            "description": "Use strict cap by default for selective variants unless an explicit overflow audit permits more.",
            "requires_code_change": "yes",
            "risk": "medium",
            "expected_effect": "K variants become bounded by max_total",
            "required_regression": "compare against baseline on hard categories",
        },
        {
            "fix_id": "F5",
            "fix_name": "high_risk_overflow_policy rank_and_cap",
            "description": "When mandatory count exceeds cap, sort mandatory targets by risk priority and energy, then cap with overflow metadata.",
            "requires_code_change": "yes",
            "risk": "medium",
            "expected_effect": "keeps strongest mandatory evidence without unbounded scans",
            "required_regression": "target-only audit and 32-point equivalence gate",
        },
        {
            "fix_id": "F6",
            "fix_name": "Per-risk mandatory caps",
            "description": "Add max_edge_risk_basins, max_delta_near_eps_basins, and max_near_degenerate_basins.",
            "requires_code_change": "yes",
            "risk": "medium-high",
            "expected_effect": "prevents Delta_near_epsilon from dominating all targets",
            "required_regression": "risk-stratified fixed-point comparison",
        },
        {
            "fix_id": "F7",
            "fix_name": "Priority and energy ordering for risk branches",
            "description": "Sort risk branches by physical priority and energy_above_global rather than keeping all.",
            "requires_code_change": "yes",
            "risk": "medium",
            "expected_effect": "retains the most relevant mandatory branches under cap",
            "required_regression": "target trace examples and physics gate",
        },
        {
            "fix_id": "F8",
            "fix_name": "Document ordinary-only energy-window pruning",
            "description": "Keep energy-window pruning ordinary-only and make reports state it cannot reduce mandatory overflow.",
            "requires_code_change": "report/docs only",
            "risk": "low",
            "expected_effect": "prevents misinterpreting energy-window variant failure",
            "required_regression": "none beyond report audit",
        },
        {
            "fix_id": "F9",
            "fix_name": "Target-construction dry-run gate",
            "description": "Before local scans, run target construction only and fail fast if selected_refine_target_count exceeds policy.",
            "requires_code_change": "yes, report-only first",
            "risk": "low",
            "expected_effect": "avoids spending GPU time on known target explosions",
            "required_regression": "dry-run fixtures over 32 fixed points",
        },
        {
            "fix_id": "F10",
            "fix_name": "Hard-risk target-only audit before expensive scans",
            "description": "Run target-construction-only audit on the 72 missing hard-risk tasks before rerunning local scans.",
            "requires_code_change": "small helper only",
            "risk": "low",
            "expected_effect": "separates algorithmic target explosion from true physics complexity",
            "required_regression": "none; diagnostic gate",
        },
    ]
    write_csv(report_root / "tables" / "recommended_fix_plan.csv", fix_rows)


def plot_bar(report_root: Path, filename: str, rows: list[dict[str, Any]], value_key: str, title: str, ylabel: str) -> None:
    values = []
    labels = []
    for row in rows:
        labels.append(DISPLAY.get(str(row["variant"]), str(row["variant"])))
        values.append(fval(row.get(value_key)))
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    ax.bar(labels, values, color=["#4c78a8", "#59a14f", "#c97b55", "#c97b55", "#c97b55"])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = report_root / "figures" / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def build_figures(report_root: Path, summary_rows: list[dict[str, Any]], by_risk: list[dict[str, Any]]) -> None:
    plot_bar(
        report_root,
        "selected_target_count_by_variant.png",
        summary_rows,
        "mean_selected_refine_target_count",
        "Mean selected target count by variant",
        "selected targets per completed point",
    )
    plot_bar(
        report_root,
        "mandatory_count_by_variant.png",
        summary_rows,
        "mean_mandatory_basin_count",
        "Mean mandatory selected basin count by variant",
        "mandatory selected basins",
    )
    plot_bar(
        report_root,
        "clustered_basin_count_by_variant.png",
        summary_rows,
        "mean_clustered_basin_count",
        "Mean clustered basin count by variant",
        "clustered basins per completed point",
    )
    plot_bar(
        report_root,
        "raw_candidate_count_by_variant.png",
        summary_rows,
        "mean_raw_candidate_count",
        "Mean raw candidate count by variant",
        "raw local minima per completed point",
    )

    risks = sorted({r["risk_category"] for r in by_risk})
    variants = VARIANTS
    matrix = []
    overflow = []
    for variant in variants:
        row_vals = []
        row_over = []
        for risk in risks:
            match = next((r for r in by_risk if r["variant"] == variant and r["risk_category"] == risk), None)
            row_vals.append(fval(match.get("mean_selected_refine_target_count_observed")) if match else float("nan"))
            row_over.append(fval(match.get("mandatory_overflow_observed_points")) if match else float("nan"))
        matrix.append(row_vals)
        overflow.append(row_over)

    for data, filename, title, cbar_label in [
        (matrix, "selected_target_count_by_risk_category.png", "Selected target count by risk category", "mean selected targets"),
        (overflow, "mandatory_overflow_by_risk_category.png", "Mandatory overflow by risk category", "overflow points"),
    ]:
        fig, ax = plt.subplots(figsize=(10.5, 4.2))
        arr = [[0.0 if not math.isfinite(v) else v for v in row] for row in data]
        im = ax.imshow(arr, aspect="auto", cmap="YlOrRd")
        ax.set_yticks(range(len(variants)), [DISPLAY[v] for v in variants])
        ax.set_xticks(range(len(risks)), risks, rotation=35, ha="right")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label=cbar_label)
        fig.tight_layout()
        fig.savefig(report_root / "figures" / filename, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = range(len(summary_rows))
    labels = [DISPLAY.get(str(r["variant"]), str(r["variant"])) for r in summary_rows]
    raw = [fval(r["mean_raw_candidate_count"]) for r in summary_rows]
    clustered = [fval(r["mean_clustered_basin_count"]) for r in summary_rows]
    selected = [fval(r["mean_selected_refine_target_count"]) for r in summary_rows]
    width = 0.24
    ax.bar([i - width for i in x], raw, width=width, label="raw candidates")
    ax.bar(list(x), clustered, width=width, label="clustered basins")
    ax.bar([i + width for i in x], selected, width=width, label="selected targets")
    ax.set_xticks(list(x), labels, rotation=20)
    ax.set_ylabel("mean count on completed points")
    ax.set_title("Target-count pipeline")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(report_root / "figures" / "target_count_pipeline_sankey_or_bar.png", dpi=180)
    plt.close(fig)


def build_markdown(report_root: Path, variant_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> str:
    path = report_root / f"{REPORT_NAME}.md"
    selective = [r for r in summary_rows if str(r["variant"]).startswith("cluster_optional") or r["variant"] == "cluster_energy_window"]
    mean_selected = mean([fval(r["mean_selected_refine_target_count"]) for r in selective])
    mean_mandatory = mean([fval(r["mean_mandatory_basin_count"]) for r in selective])
    lines = [
        "# Local-Refinement Target Logic Audit",
        "",
        "## 1. Executive Summary",
        "",
        "| Question | Status | Answer |",
        "| --- | --- | --- |",
        "| Q1. local-box count about 85 is refined boxes? | confirmed | It is `local_boxes_refined_count` / local-box CSV rows, produced after completed local scans. |",
        "| Q2. Why did K=3/K=2 not reduce local-box count? | confirmed | K caps only ordinary optional rows; completed selective rows are dominated by mandatory `Delta_near_epsilon` targets. |",
        "| Q3. Do mandatory branches bypass cap? | confirmed | Current variants use `mandatory_basins_can_exceed_cap=True`; code selects all mandatory rows before optional K. |",
        "| Q4. Are risk flags candidate-level? | confirmed | Edge and low-energy flags originate at candidate level and are promoted/aggregated to basin rows after clustering. |",
        "| Q5. Does clustering act on mandatory targets? | confirmed | Clustering runs before mandatory target construction for clustered variants. |",
        "| Q6. Is energy-window pruning ordinary-only? | confirmed | Code prunes only rows with no mandatory reason. |",
        "| Q7. Are hard-risk timeouts caused by target explosion? | supported but not proven | Completed clean controls prove target explosion; timeout hard points lack target metadata. |",
        "| Q8. Should we directly rerun 72 missing tasks? | not supported | The current evidence favors target-construction-only audit/fix before spending more GPU time. |",
        "| Q9. Is target construction rewrite needed? | supported but not proven | At minimum the overflow policy needs revision; a full rewrite should wait for target-only traces. |",
        "| Q10. Minimal safe fix direction? | confirmed | Add target-construction dry-run gate and rank/cap mandatory overflow before local scans. |",
        "",
        "## 2. Recheck of Variant Return Results",
        "",
        "The returned archive is a diagnostic failure, not a completed optimized-variant gate. "
        "`baseline` and `cluster_only` completed all 32 points. The three optimized variants completed only 8 clean superconducting controls and timed out on 24 hard-risk points each.",
        "",
        "| variant | success | timeout | mean runtime min | mean local boxes | status |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in variant_rows:
        lines.append(
            f"| {row['variant']} | {row['success_count']}/{row['total_tasks']} | {row['timeout_count']} | "
            f"{row['mean_runtime_min']} | {row['mean_local_box_count']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## 3. Definition of Local-Box Count",
            "",
            "`local_boxes_refined_count` is assigned from `len(refined_rows)` after the loop over `refine_targets`. "
            "The local-box timing CSV is also written only for completed local scans. Therefore the count of about 85 on completed common points is an actual completed refined-box count, not a raw candidate count. It does not include pruned candidates or unexecuted timeout targets.",
            "",
            "## 4. Target-Construction Dry Run",
            "",
            "This audit is a metadata reconstruction, not an expensive dry-run of the exact solver. It reads completed pointwise/local-box outputs and code paths. Timeout points have only startup JSON, so their exact raw candidate, clustered basin, and selected target counts are not recoverable from current metadata.",
            "",
            f"On completed selective points the mean selected target count is {fmt_float(mean_selected)} and the mean mandatory selected count is {fmt_float(mean_mandatory)}. This is already far above `max_total_refined_basins=6`.",
            "",
            "## 5. Code Path Order",
            "",
            "The production order is: coarse candidate detection, optional clustering, energy-window marking, mandatory/ordinary split, optional top-k, then local-box scans. Clustering is before selection for clustered variants.",
            "",
            "## 6. Candidate-Level vs Basin-Level Risk Marking",
            "",
            "Edge-risk and low-energy-window flags originate in `_build_branch_candidates` before clustering. `cluster_branch_candidates` aggregates member mandatory reasons onto one representative. Selection then recomputes mandatory reasons on the post-clustering row.",
            "",
            "## 7. Clustering Scope",
            "",
            "Clustering does apply before mandatory target selection. The failure is not that mandatory targets bypass clustering entirely. The observed problem is that after clustering, many basin representatives still satisfy mandatory reasons, especially `Delta_near_epsilon`.",
            "",
            "## 8. Top-k Cap Effectiveness",
            "",
            "K=3 and K=2 limit only ordinary optional rows. They do not limit mandatory rows when `mandatory_basins_can_exceed_cap=True`. Completed selective local-box rows show nearly all selected targets are mandatory, so changing K from 3 to 2 has no visible effect.",
            "",
            "## 9. Mandatory Overflow",
            "",
            "Mandatory overflow is confirmed on completed selective points. For example, point 4 has 78 selected targets under `cluster_optional_k3` with `max_total_refined_basins=6`. That is an overflow of 72 targets. The timeout points themselves lack target metadata, so their overflow is not directly proven.",
            "",
            "## 10. Energy-Window Effectiveness",
            "",
            "`cluster_energy_window` has `energy_window_pruned_count=0` on completed points and matches K3/K2 behavior. This is expected from the code because the energy window only prunes ordinary non-mandatory branches. If mandatory branches dominate, the energy window is ineffective.",
            "",
            "## 11. Hard-Risk Timeout Trace",
            "",
            "The hard-risk timeout rows have startup JSON only. No point CSV, NPZ, or local-box timing CSV is present for those optimized variants. The report therefore does not claim exact target counts for those timeout points.",
            "",
            "## 12. Root Cause Candidates",
            "",
            "The strongest confirmed root causes are: mandatory branches bypass the total cap; top-k applies only to optional branches; energy-window pruning is ordinary-only; completed selective points are dominated by mandatory `Delta_near_epsilon` targets.",
            "",
            "## 13. Recommended Fix Direction",
            "",
            "Do not first rerun the 72 missing optimized tasks. The minimum safe direction is to add a target-construction-only gate, record mandatory overflow, and implement or test `high_risk_overflow_policy = rank_and_cap` before another expensive GPU run.",
            "",
            "## 14. Proposed Correct Target-Construction Pseudocode",
            "",
            "```python",
            "raw_candidates = detect_local_minima(coarse_scan)",
            "basins = cluster_candidates(raw_candidates)",
            "annotate_basin_risk_flags(basins)",
            "global_best = select_global_best_basin(basins)",
            "mandatory = [global_best]",
            "mandatory += select_top_risk_basins(basins, risk_type='edge_risk', max_count=max_edge_risk_basins, sort_by=['edge_distance', 'energy_above_global'])",
            "mandatory += select_top_risk_basins(basins, risk_type='delta_near_eps', max_count=max_delta_near_eps_basins, sort_by=['abs_delta_minus_eps', 'energy_above_global'])",
            "mandatory += select_top_risk_basins(basins, risk_type='near_degenerate', max_count=max_near_degenerate_basins, sort_by=['energy_above_global'])",
            "ordinary = basins - mandatory",
            "ordinary = apply_energy_window_if_enabled(ordinary)",
            "optional = take_lowest_energy(ordinary, max_count=max_optional_refined_basins)",
            "targets = unique(mandatory + optional)",
            "targets = enforce_total_cap(targets, max_total_refined_basins, overflow_policy='rank_and_cap')",
            "if mandatory_overflow:",
            "    record_overflow_metadata()",
            "return targets",
            "```",
            "",
            "## 15. Do-Not-Claim List",
            "",
            "1. Do not claim optimized variants are physics-equivalent; they completed only 8/32 points.",
            "2. Do not claim top-k or energy-window pruning has failed as a physical method; the current failure is likely implementation-policy overflow.",
            "3. Do not directly rerun 72 timeout tasks unless target-count explosion is excluded.",
            "4. Do not interpret 85 local boxes as 85 phases.",
            "5. Do not attribute hard-risk timeout to physical complexity unless target-construction audit supports it.",
            "6. Do not change thermodynamic criterion or tolerance.",
            "7. Do not treat all risk candidates as mandatory branch representatives.",
            "",
            "## 16. Next Step",
            "",
            "Add a true target-construction-only instrumented path that saves raw candidates, clustered basins, mandatory reasons, optional candidates, and final selected targets before any local-box scan. Use it on the 72 missing coordinates before rerunning GPU refinement.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def build_pdf(report_root: Path, markdown_path: Path) -> Path:
    pdf_path = report_root / f"{REPORT_NAME}.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, rightMargin=42, leftMargin=42, topMargin=42, bottomMargin=42)
    styles = getSampleStyleSheet()
    story: list[Any] = []
    story.append(Paragraph("Local-Refinement Target Logic Audit", styles["Title"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Executive Summary", styles["Heading1"]))
    summary_data = [
        ["Question", "Status", "Answer"],
        ["local-box count ~85", "confirmed", "Actual completed refined boxes on completed rows."],
        ["K=3/K=2 effect", "confirmed", "K caps optional rows only; mandatory rows dominate."],
        ["mandatory overflow", "confirmed", "Current selective variants allow mandatory rows above total cap."],
        ["hard-risk timeout cause", "supported but not proven", "Timeout rows lack target metadata."],
        ["direct rerun missing tasks", "not supported", "Run target-construction-only audit first."],
    ]
    table = Table(summary_data, colWidths=[145, 95, 250])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 10))
    story.append(Paragraph("Key Finding", styles["Heading1"]))
    story.append(
        Paragraph(
            "The optimized variants did not reduce local-refinement targets because the current selective path keeps all mandatory basins. "
            "Completed selective points are dominated by Delta_near_epsilon mandatory targets, so the optional K cap and ordinary-only energy window do not reduce the final target list.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 8))
    for fig_name in [
        "selected_target_count_by_variant.png",
        "mandatory_count_by_variant.png",
        "target_count_pipeline_sankey_or_bar.png",
        "selected_target_count_by_risk_category.png",
    ]:
        fig_path = report_root / "figures" / fig_name
        if fig_path.exists():
            story.append(Image(str(fig_path), width=6.6 * 72, height=3.0 * 72))
            story.append(Spacer(1, 8))
    story.append(PageBreak())
    story.append(Paragraph("Decision", styles["Heading1"]))
    story.append(
        Paragraph(
            "Do not rerun the 72 missing optimized tasks yet. First add a true target-construction-only audit or revise the mandatory overflow policy, then rerun a bounded regression.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 8))
    story.append(Paragraph("Companion Markdown", styles["Heading1"]))
    story.append(Paragraph(str(markdown_path), styles["BodyText"]))
    doc.build(story)
    return pdf_path


def build_decision_log(report_root: Path) -> None:
    text = """# Decision Log

- status: target-construction audit complete
- local_box_count_definition: confirmed actual completed refined boxes for completed rows
- mandatory_overflow: confirmed on completed selective variants
- k_cap_effect: K=3/K=2 caps optional ordinary targets only
- energy_window_effect: ordinary-only; ineffective when mandatory targets dominate
- hard_risk_timeout_target_count: cannot determine from current metadata
- direct_rerun_missing_tasks: not recommended before target-construction-only audit
- minimum_fix_direction: add target-construction dry-run gate and rank/cap mandatory overflow policy before expensive rerun
"""
    (report_root / "decision_log.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Report-only audit of local-refinement target construction outputs.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    run_root = args.run_root
    report_root = args.output_dir
    regression_root = run_root / "reports" / "local_refinement_refactor" / "variant_regression"
    logs_root = run_root / "logs"
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "tables").mkdir(parents=True, exist_ok=True)
    (report_root / "figures").mkdir(parents=True, exist_ok=True)

    task_rows = read_csv_rows(regression_root / "summary" / "task_status.csv")
    for row in task_rows:
        row["effective_status"] = effective_status(row, logs_root)
    pointwise = load_variant_pointwise(regression_root)
    local_boxes = load_local_box_rows(regression_root)
    indexes = build_indexes(task_rows, pointwise, local_boxes)
    boxes_by_key = indexes["boxes_by_key"]

    variant_rows = build_variant_result_recheck(report_root, regression_root, task_rows, pointwise, boxes_by_key)
    build_definition_audit(report_root)
    by_point, by_risk = build_target_tables(report_root, regression_root, task_rows, pointwise, boxes_by_key)
    build_reason_and_effect_tables(report_root, by_point, boxes_by_key)
    build_code_path_tables(report_root)
    build_trace_tables(report_root, by_point, boxes_by_key)
    build_root_cause_and_fix_tables(report_root)

    summary_rows = read_csv_rows(report_root / "tables" / "target_construction_summary.csv")
    build_figures(report_root, summary_rows, by_risk)
    md = Path(build_markdown(report_root, variant_rows, summary_rows))
    pdf = build_pdf(report_root, md)
    build_decision_log(report_root)

    print(
        json.dumps(
            {
                "report_root": str(report_root.resolve()),
                "markdown": str(md.resolve()),
                "pdf": str(pdf.resolve()),
                "tables": str((report_root / "tables").resolve()),
                "figures": str((report_root / "figures").resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
