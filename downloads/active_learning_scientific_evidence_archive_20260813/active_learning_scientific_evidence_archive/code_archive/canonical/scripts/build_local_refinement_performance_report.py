from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_local_refinement_fixed_point_regression import supported_variant_names


DEFAULT_OUTPUT_DIR = Path("reports/local_refinement_refactor/variant_regression/performance_report")
VARIANT_REGRESSION_ROOT = Path("reports/local_refinement_refactor/variant_regression")
POINTWISE_COUNT_COLUMNS = [
    "trusted_exact",
    "training_eligible_exact",
    "rerun_required",
]
POINTWISE_SUM_COLUMNS = [
    "local_minima_detected_count",
    "clustered_basin_count",
    "selected_refine_target_count",
    "basin_clustering_merged_count",
    "energy_window_pruned_count",
    "local_boxes_refined_count",
    "local_refinement_reused_count",
]


def _default_run_root() -> Path:
    env_run_root = os.environ.get("RUN_ROOT")
    if env_run_root:
        return Path(env_run_root)
    if os.access(ROOT, os.W_OK):
        return ROOT
    fallback = os.environ.get("SCRATCH") or os.environ.get("TMPDIR") or os.environ.get("HOME")
    if fallback:
        return Path(fallback) / "local_refinement_refactor_variant_suite_run"
    return ROOT


def _run_path(path: Path, run_root: Path) -> Path:
    return path if path.is_absolute() else run_root / path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float_value(raw: Any, default: float = float("nan")) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _int_value(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        text = str(raw).strip().lower()
        if text in {"true", "yes"}:
            return 1
        if text in {"false", "no"}:
            return 0
        return int(default)


def _sum_column(rows: list[dict[str, str]], column: str) -> float:
    return float(sum(_float_value(row.get(column), 0.0) for row in rows))


def _count_truthy(rows: list[dict[str, str]], column: str) -> int:
    return int(sum(1 for row in rows if _int_value(row.get(column), 0) != 0))


def _mean(total: float, count: int) -> float:
    return float(total / count) if count else float("nan")


def _format_float(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.10g}"


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


def _variant_dir(result_root: Path, variant: str) -> Path:
    return result_root / VARIANT_REGRESSION_ROOT / variant


def _pointwise_path(result_root: Path, variant: str) -> Path:
    return _variant_dir(result_root, variant) / f"{variant}_pointwise.csv"


def _local_box_path(result_root: Path, variant: str) -> Path:
    return _variant_dir(result_root, variant) / f"{variant}_local_box_timing.csv"


def _manifest_path(result_root: Path, variant: str) -> Path:
    return _variant_dir(result_root, variant) / "regression_manifest.json"


def _phase_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("phase_candidate", "unknown"))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _runtime_summary_for_variant(result_root: Path, variant: str) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    pointwise = _pointwise_path(result_root, variant)
    manifest_file = _manifest_path(result_root, variant)
    if not pointwise.exists():
        failures.append(f"missing pointwise CSV for {variant}: {pointwise}")
    if not manifest_file.exists():
        failures.append(f"missing regression manifest for {variant}: {manifest_file}")
    if failures:
        return {"variant": variant, "status": "missing"}, failures

    rows = _read_csv_rows(pointwise)
    manifest = _read_json(manifest_file)
    n_points = len(rows)
    total_runtime_rows = _sum_column(rows, "point_total_runtime_sec")
    local_runtime_rows = _sum_column(rows, "local_refinement_runtime_sec")
    total_runtime_manifest = _float_value(manifest.get("total_runtime_sec"), 0.0)
    local_runtime_manifest = _float_value(manifest.get("local_refinement_runtime_sec"), 0.0)

    summary: dict[str, Any] = {
        "variant": variant,
        "status": "present",
        "n_points": n_points,
        "manifest_n_points": int(manifest.get("n_points", -1)),
        "phase_counts_json": json.dumps(_phase_counts(rows), sort_keys=True),
        "total_runtime_sec_sum": total_runtime_rows,
        "local_refinement_runtime_sec_sum": local_runtime_rows,
        "manifest_total_runtime_sec": total_runtime_manifest,
        "manifest_local_refinement_runtime_sec": local_runtime_manifest,
        "mean_point_runtime_sec": _mean(total_runtime_rows, n_points),
        "mean_local_refinement_runtime_sec": _mean(local_runtime_rows, n_points),
        "local_runtime_fraction": float(local_runtime_rows / total_runtime_rows)
        if total_runtime_rows > 0.0
        else float("nan"),
    }
    for column in POINTWISE_COUNT_COLUMNS:
        summary[f"{column}_count"] = _count_truthy(rows, column)
    for column in POINTWISE_SUM_COLUMNS:
        total = _sum_column(rows, column)
        summary[f"{column}_sum"] = total
        summary[f"{column}_mean"] = _mean(total, n_points)
    return summary, failures


def _box_summary_for_variant(result_root: Path, variant: str) -> dict[str, Any]:
    local_box = _local_box_path(result_root, variant)
    if not local_box.exists():
        return {
            "variant": variant,
            "has_local_box_csv": 0,
            "local_box_rows": 0,
            "unique_points": 0,
            "boxes_per_point": float("nan"),
            "box_runtime_sec_sum": 0.0,
            "box_runtime_sec_mean": float("nan"),
            "box_runtime_sec_min": float("nan"),
            "box_runtime_sec_max": float("nan"),
            "changed_global_minimum_rows": 0,
            "changed_phase_label_rows": 0,
            "near_degenerate_after_refine_rows": 0,
            "reused_from_previous_scan_rows": 0,
            "pruned_rows": 0,
            "selection_reason_counts_json": "{}",
        }
    rows = _read_csv_rows(local_box)
    runtimes = [_float_value(row.get("box_runtime_sec"), 0.0) for row in rows]
    point_ids = {str(row.get("point_id", "")) for row in rows}
    point_ids.discard("")
    selection_counts: dict[str, int] = {}
    pruned_rows = 0
    for row in rows:
        reason = str(row.get("selection_reason", "unknown"))
        selection_counts[reason] = selection_counts.get(reason, 0) + 1
        if str(row.get("pruned_reason", "")).strip():
            pruned_rows += 1
    runtime_sum = float(sum(runtimes))
    return {
        "variant": variant,
        "has_local_box_csv": 1,
        "local_box_rows": len(rows),
        "unique_points": len(point_ids),
        "boxes_per_point": float(len(rows) / len(point_ids)) if point_ids else float("nan"),
        "box_runtime_sec_sum": runtime_sum,
        "box_runtime_sec_mean": _mean(runtime_sum, len(rows)),
        "box_runtime_sec_min": min(runtimes) if runtimes else float("nan"),
        "box_runtime_sec_max": max(runtimes) if runtimes else float("nan"),
        "changed_global_minimum_rows": _count_truthy(rows, "changed_global_minimum"),
        "changed_phase_label_rows": _count_truthy(rows, "changed_phase_label"),
        "near_degenerate_after_refine_rows": _count_truthy(rows, "near_degenerate_after_refine"),
        "reused_from_previous_scan_rows": _count_truthy(rows, "reused_from_previous_scan"),
        "pruned_rows": pruned_rows,
        "selection_reason_counts_json": json.dumps(selection_counts, sort_keys=True),
    }


def _add_speedup_columns(runtime_rows: list[dict[str, Any]]) -> None:
    baseline = next((row for row in runtime_rows if row.get("variant") == "baseline"), None)
    if baseline is None or baseline.get("status") != "present":
        return
    baseline_total = _float_value(baseline.get("total_runtime_sec_sum"), 0.0)
    baseline_local = _float_value(baseline.get("local_refinement_runtime_sec_sum"), 0.0)
    baseline_boxes = _float_value(baseline.get("local_boxes_refined_count_sum"), 0.0)
    for row in runtime_rows:
        total = _float_value(row.get("total_runtime_sec_sum"), 0.0)
        local = _float_value(row.get("local_refinement_runtime_sec_sum"), 0.0)
        boxes = _float_value(row.get("local_boxes_refined_count_sum"), 0.0)
        row["total_runtime_speedup_vs_baseline"] = float(baseline_total / total) if total > 0 else float("nan")
        row["local_runtime_speedup_vs_baseline"] = float(baseline_local / local) if local > 0 else float("nan")
        row["local_boxes_ratio_vs_baseline"] = float(boxes / baseline_boxes) if baseline_boxes > 0 else float("nan")
        row["total_runtime_pct_change_vs_baseline"] = (
            float(100.0 * (total - baseline_total) / baseline_total) if baseline_total > 0 else float("nan")
        )
        row["local_runtime_pct_change_vs_baseline"] = (
            float(100.0 * (local - baseline_local) / baseline_local) if baseline_local > 0 else float("nan")
        )


def _write_markdown_report(
    path: Path,
    result_root: Path,
    runtime_rows: list[dict[str, Any]],
    box_rows: list[dict[str, Any]],
    failures: list[str],
) -> None:
    lines = [
        "# Local-refinement Performance Report",
        "",
        f"- result_root: `{result_root}`",
        f"- status: {'pass' if not failures else 'fail'}",
        "",
        "## Runtime Summary",
        "",
        "| variant | n_points | total_runtime_sec | local_runtime_sec | local_fraction | total_speedup_vs_baseline | local_speedup_vs_baseline | boxes_sum | pruned_sum |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in runtime_rows:
        lines.append(
            "| {variant} | {n_points} | {total} | {local} | {fraction} | {total_speedup} | {local_speedup} | {boxes} | {pruned} |".format(
                variant=row.get("variant", ""),
                n_points=row.get("n_points", ""),
                total=_format_float(_float_value(row.get("total_runtime_sec_sum"))),
                local=_format_float(_float_value(row.get("local_refinement_runtime_sec_sum"))),
                fraction=_format_float(_float_value(row.get("local_runtime_fraction"))),
                total_speedup=_format_float(_float_value(row.get("total_runtime_speedup_vs_baseline"))),
                local_speedup=_format_float(_float_value(row.get("local_runtime_speedup_vs_baseline"))),
                boxes=_format_float(_float_value(row.get("local_boxes_refined_count_sum"))),
                pruned=_format_float(_float_value(row.get("energy_window_pruned_count_sum"))),
            )
        )
    lines.extend(
        [
            "",
            "## Local Box Summary",
            "",
            "| variant | rows | unique_points | boxes_per_point | box_runtime_sec_sum | changed_global_minimum_rows | changed_phase_label_rows | pruned_rows |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in box_rows:
        lines.append(
            "| {variant} | {rows} | {points} | {boxes_per_point} | {runtime} | {changed_min} | {changed_phase} | {pruned_rows} |".format(
                variant=row.get("variant", ""),
                rows=row.get("local_box_rows", ""),
                points=row.get("unique_points", ""),
                boxes_per_point=_format_float(_float_value(row.get("boxes_per_point"))),
                runtime=_format_float(_float_value(row.get("box_runtime_sec_sum"))),
                changed_min=row.get("changed_global_minimum_rows", ""),
                changed_phase=row.get("changed_phase_label_rows", ""),
                pruned_rows=row.get("pruned_rows", ""),
            )
        )
    lines.extend(["", "## Failures", ""])
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("- none")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_performance_report(
    result_root: Path,
    output_dir: Path,
    *,
    variants: list[str] | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    variants = variants or supported_variant_names()
    runtime_rows: list[dict[str, Any]] = []
    box_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for variant in variants:
        runtime_summary, variant_failures = _runtime_summary_for_variant(result_root, variant)
        failures.extend(variant_failures)
        runtime_rows.append(runtime_summary)
        box_rows.append(_box_summary_for_variant(result_root, variant))
    _add_speedup_columns(runtime_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "runtime_summary.csv", runtime_rows)
    _write_csv(output_dir / "local_box_summary.csv", box_rows)
    _write_markdown_report(output_dir / "performance_report.md", result_root, runtime_rows, box_rows, failures)
    summary = {
        "status": "pass" if not failures else "fail",
        "result_root": str(result_root),
        "output_dir": str(output_dir),
        "variants": variants,
        "runtime_summary_csv": str(output_dir / "runtime_summary.csv"),
        "local_box_summary_csv": str(output_dir / "local_box_summary.csv"),
        "performance_report_md": str(output_dir / "performance_report.md"),
        "failures": failures,
    }
    (output_dir / "performance_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if strict and failures:
        raise FileNotFoundError("; ".join(failures))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build runtime and local-box summaries for local-refinement variants.")
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variant", action="append", dest="variants", default=None)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--run-root", type=Path, default=None)
    args = parser.parse_args()
    run_root = args.run_root or _default_run_root()
    result_root = args.result_root if args.result_root is not None else run_root
    result_root = _run_path(result_root, run_root)
    output_dir = _run_path(args.output_dir, run_root)
    summary = build_performance_report(
        result_root,
        output_dir,
        variants=args.variants,
        strict=not bool(args.allow_missing),
    )
    print(json.dumps(summary, indent=2))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
