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

from scripts.build_local_refinement_performance_report import build_performance_report
from scripts.compare_local_refinement_variants import compare
from scripts.run_local_refinement_fixed_point_regression import resolve_variant_config


RUNNABLE_VARIANTS = [
    "baseline",
    "cluster_only",
    "cluster_optional_k3",
    "cluster_optional_k2",
    "cluster_energy_window",
]
COMPARISON_VARIANTS = [variant for variant in RUNNABLE_VARIANTS if variant != "baseline"]
DEFAULT_OUTPUT_ROOT = Path("reports/local_refinement_refactor/variant_regression")
DEFAULT_TOLERANCES = {
    "max_q_opt_abs_diff": 1.0e-10,
    "max_delta_opt_abs_diff": 1.0e-10,
    "max_deltaf_abs_diff": 1.0e-8,
}


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


def _package_path(path: Path, package_root: Path) -> Path:
    return path if path.is_absolute() else package_root / path


def _run_path(path: Path, run_root: Path) -> Path:
    return path if path.is_absolute() else run_root / path


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


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


def _load_tolerances(path: Path | None) -> dict[str, float]:
    tolerances = dict(DEFAULT_TOLERANCES)
    if path is None or not path.exists():
        return tolerances
    raw = _read_json(path)
    for key in tolerances:
        if key in raw:
            tolerances[key] = float(raw[key])
    return tolerances


def _load_task_matrix(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _point_task_paths(output_root: Path, task: dict[str, str]) -> dict[str, Path]:
    variant = str(task["variant"])
    point_id = int(task["point_id"])
    task_dir = output_root / "point_tasks" / variant
    stem = f"point_{point_id:03d}"
    return {
        "json": task_dir / f"{stem}.json",
        "csv": task_dir / f"{stem}.csv",
        "local_box": task_dir / f"{stem}_local_box_timing.csv",
    }


def _sort_point_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: int(float(row.get("point_id", 0))))


def _variant_output_dir(output_root: Path, variant: str) -> Path:
    return output_root / variant


def _summarize_variant(
    *,
    output_root: Path,
    variant: str,
    tasks: list[dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    point_rows: list[dict[str, Any]] = []
    local_box_rows: list[dict[str, Any]] = []
    task_status_rows: list[dict[str, Any]] = []

    for task in tasks:
        paths = _point_task_paths(output_root, task)
        status = "missing"
        error = ""
        wall_runtime = float("nan")
        point_rows_found = 0
        if paths["json"].exists():
            try:
                point_status = _read_json(paths["json"])
                status = str(point_status.get("status", "unknown"))
                error = str(point_status.get("error", ""))
                wall_runtime = float(point_status.get("wall_runtime_sec", float("nan")))
            except Exception as exc:
                status = "bad_status_json"
                error = repr(exc)
        if status == "success":
            rows = _read_csv_rows(paths["csv"])
            point_rows_found = len(rows)
            if len(rows) == 1:
                point_rows.extend(rows)
            else:
                status = "bad_point_csv"
                error = f"expected one row, got {len(rows)}"
        if paths["local_box"].exists() and paths["local_box"].stat().st_size > 0:
            local_box_rows.extend(_read_csv_rows(paths["local_box"]))
        task_status_rows.append(
            {
                "task_id": int(task["task_id"]),
                "variant": variant,
                "point_id": int(task["point_id"]),
                "kT": task.get("kT", ""),
                "JA": task.get("JA", ""),
                "risk_tag": task.get("category", "unknown"),
                "status": status,
                "point_rows_found": point_rows_found,
                "wall_runtime_sec": wall_runtime,
                "error": error,
                "status_json": str(paths["json"]),
                "point_csv": str(paths["csv"]),
            }
        )

    complete = all(row["status"] == "success" for row in task_status_rows)
    variant_dir = _variant_output_dir(output_root, variant)
    point_rows = _sort_point_rows(point_rows)
    _write_csv(variant_dir / f"{variant}_pointwise.csv", point_rows)
    _write_csv(variant_dir / f"{variant}_local_box_timing.csv", local_box_rows)

    phase_counts: dict[str, int] = {}
    for row in point_rows:
        phase = str(row.get("phase_candidate", "unknown"))
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
    total_runtime = sum(float(row.get("point_total_runtime_sec", 0.0)) for row in point_rows)
    local_runtime = sum(float(row.get("local_refinement_runtime_sec", 0.0)) for row in point_rows)
    manifest = {
        "mode": "exact" if complete else "incomplete",
        "status": "pass" if complete else "fail",
        "variant_name": variant,
        "n_points": len(point_rows),
        "expected_points": len(tasks),
        "task_success_count": sum(1 for row in task_status_rows if row["status"] == "success"),
        "task_failure_count": sum(1 for row in task_status_rows if row["status"] not in {"success"}),
        "phase_counts": phase_counts,
        "trusted_count": sum(int(float(row.get("trusted_exact", 0))) for row in point_rows),
        "training_eligible_count": sum(int(float(row.get("training_eligible_exact", 0))) for row in point_rows),
        "rerun_required_count": sum(int(float(row.get("rerun_required", 0))) for row in point_rows),
        "total_runtime_sec": float(total_runtime),
        "local_refinement_runtime_sec": float(local_runtime),
        "variant_config": resolve_variant_config(variant),
        "array_task_mode": True,
        "point_tasks_root": str(output_root / "point_tasks" / variant),
    }
    _write_json(variant_dir / "regression_manifest.json", manifest)
    _write_text(
        variant_dir / "regression_summary.md",
        "# Local-refinement fixed-point regression\n\n"
        + "\n".join(f"- {key}: {value}" for key, value in manifest.items())
        + "\n",
    )
    return manifest, point_rows, task_status_rows


def _comparison_status(value: float, threshold: float) -> str:
    if not math.isfinite(value):
        return "fail"
    return "pass" if value <= threshold else "fail"


def aggregate_suite(
    *,
    package_root: Path,
    run_root: Path,
    task_matrix: Path,
    output_root: Path,
    tolerances: dict[str, float],
) -> dict[str, Any]:
    task_rows = _load_task_matrix(task_matrix)
    tasks_by_variant: dict[str, list[dict[str, str]]] = {variant: [] for variant in RUNNABLE_VARIANTS}
    for task in task_rows:
        tasks_by_variant.setdefault(str(task["variant"]), []).append(task)

    output_root.mkdir(parents=True, exist_ok=True)
    all_task_status_rows: list[dict[str, Any]] = []
    variant_summaries: dict[str, Any] = {}
    for variant in RUNNABLE_VARIANTS:
        manifest, _point_rows, task_status_rows = _summarize_variant(
            output_root=output_root,
            variant=variant,
            tasks=tasks_by_variant.get(variant, []),
        )
        variant_summaries[variant] = manifest
        all_task_status_rows.extend(task_status_rows)

    summary_dir = output_root / "summary"
    failed_task_rows = [row for row in all_task_status_rows if row["status"] != "success"]
    _write_csv(summary_dir / "task_status.csv", all_task_status_rows)
    _write_csv(summary_dir / "missing_or_failed_tasks.csv", failed_task_rows)

    comparison_rows: list[dict[str, Any]] = []
    baseline_csv = _variant_output_dir(output_root, "baseline") / "baseline_pointwise.csv"
    for variant in COMPARISON_VARIANTS:
        candidate_csv = _variant_output_dir(output_root, variant) / f"{variant}_pointwise.csv"
        comparison_dir = output_root / "comparisons" / f"baseline_vs_{variant}"
        if baseline_csv.exists() and candidate_csv.exists():
            comparison_summary = compare(baseline_csv, candidate_csv, comparison_dir)
            row = {
                "variant": variant,
                **comparison_summary,
            }
            for key, threshold in tolerances.items():
                row[f"{key}_threshold"] = threshold
                row[f"{key}_status"] = _comparison_status(float(row.get(key, float("nan"))), threshold)
            row["flag_status"] = "pass" if int(row.get("flag_mismatch_count", -1)) == 0 else "fail"
            row["status"] = (
                "pass"
                if row["flag_status"] == "pass"
                and all(row[f"{key}_status"] == "pass" for key in tolerances)
                and int(row.get("n_missing_in_candidate", -1)) == 0
                and int(row.get("n_extra_in_candidate", -1)) == 0
                else "fail"
            )
            comparison_rows.append(row)
        else:
            comparison_rows.append(
                {
                    "variant": variant,
                    "status": "missing",
                    "baseline_csv": str(baseline_csv),
                    "candidate_csv": str(candidate_csv),
                }
            )
    _write_csv(summary_dir / "equivalence_matrix.csv", comparison_rows)

    performance_summary = build_performance_report(
        run_root,
        output_root / "performance_report",
        variants=RUNNABLE_VARIANTS,
        strict=False,
    )
    task_status = "pass" if not failed_task_rows else "fail"
    comparison_status = "pass" if comparison_rows and all(row.get("status") == "pass" for row in comparison_rows) else "fail"
    performance_status = str(performance_summary.get("status", "fail"))
    status = "pass" if task_status == "pass" and comparison_status == "pass" and performance_status == "pass" else "fail"
    summary = {
        "status": status,
        "package_root": str(package_root),
        "run_root": str(run_root),
        "task_matrix": str(task_matrix),
        "output_root": str(output_root),
        "variants": RUNNABLE_VARIANTS,
        "comparison_variants": COMPARISON_VARIANTS,
        "expected_tasks": len(task_rows),
        "successful_tasks": sum(1 for row in all_task_status_rows if row["status"] == "success"),
        "failed_or_missing_tasks": len(failed_task_rows),
        "task_status": task_status,
        "comparison_status": comparison_status,
        "performance_status": performance_status,
        "variant_summaries": variant_summaries,
        "tolerances": tolerances,
        "task_status_csv": str(summary_dir / "task_status.csv"),
        "missing_or_failed_tasks_csv": str(summary_dir / "missing_or_failed_tasks.csv"),
        "equivalence_matrix_csv": str(summary_dir / "equivalence_matrix.csv"),
        "performance_summary_json": str(output_root / "performance_report" / "performance_summary.json"),
        "failures": [
            f"{len(failed_task_rows)} point tasks failed or are missing" if failed_task_rows else "",
            "one or more baseline comparisons failed" if comparison_status != "pass" else "",
            "performance report failed" if performance_status != "pass" else "",
        ],
    }
    summary["failures"] = [failure for failure in summary["failures"] if failure]
    _write_json(summary_dir / "array_suite_status.json", summary)
    _write_text(
        output_root / "decision_log.md",
        "# Local-refinement Variant Array Suite Decision Log\n\n"
        f"- status: {status}\n"
        f"- successful_tasks: {summary['successful_tasks']} / {summary['expected_tasks']}\n"
        f"- comparison_status: {comparison_status}\n"
        f"- performance_status: {performance_status}\n"
        f"- next_action: {'accept gate and import return archive' if status == 'pass' else 'inspect summary/missing_or_failed_tasks.csv and slurm logs'}\n",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate point-wise local-refinement variant array outputs.")
    parser.add_argument("--package-root", type=Path, default=ROOT)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--task-matrix", type=Path, default=Path("config/task_matrix.csv"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tolerances", type=Path, default=Path("config/equivalence_tolerances.json"))
    parser.add_argument("--fail-if-incomplete", action="store_true")
    args = parser.parse_args()

    package_root = args.package_root.resolve()
    run_root = (args.run_root if args.run_root is not None else _default_run_root(package_root)).resolve()
    task_matrix = _package_path(args.task_matrix, package_root)
    output_root = _run_path(args.output_root, run_root)
    tolerances_path = _package_path(args.tolerances, package_root)
    summary = aggregate_suite(
        package_root=package_root,
        run_root=run_root,
        task_matrix=task_matrix,
        output_root=output_root,
        tolerances=_load_tolerances(tolerances_path),
    )
    print(json.dumps(summary, indent=2))
    if args.fail_if_incomplete and summary["status"] != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
