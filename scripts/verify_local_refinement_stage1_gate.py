from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _default_run_root() -> Path:
    env_run_root = os.environ.get("RUN_ROOT")
    if env_run_root:
        return Path(env_run_root)
    if os.access(ROOT, os.W_OK):
        return ROOT / "local_refinement_refactor_stage1_run"
    fallback = os.environ.get("SCRATCH") or os.environ.get("TMPDIR") or os.environ.get("HOME")
    if fallback:
        return Path(fallback) / "local_refinement_refactor_stage1_run"
    return ROOT / "local_refinement_refactor_stage1_run"


def _run_path(path: Path, run_root: Path) -> Path:
    return path if path.is_absolute() else run_root / path


REQUIRED_FLAG_COLUMNS = [
    "phase_candidate",
    "trusted_exact",
    "training_eligible_exact",
    "q_unresolved",
    "delta_unresolved",
    "rerun_required",
]
FLOAT_DIFF_COLUMNS = {
    "q_opt": "q_opt_abs_diff",
    "delta_opt": "delta_opt_abs_diff",
    "DeltaF": "DeltaF_abs_diff",
}
REQUIRED_LOCAL_BOX_COLUMNS = [
    "point_id",
    "branch_id",
    "selection_reason",
    "box_q_min",
    "box_q_max",
    "box_Delta_min",
    "box_Delta_max",
    "box_runtime_sec",
    "refined_q",
    "refined_Delta",
    "refined_DeltaF",
    "pruned_reason",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _file_status(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


def _max_float(rows: list[dict[str, str]], column: str) -> float:
    values: list[float] = []
    for row in rows:
        raw = row.get(column, "")
        try:
            value = float(raw)
        except ValueError:
            continue
        if math.isfinite(value):
            values.append(value)
    return max(values) if values else float("nan")


def verify_gate(
    baseline_dir: Path,
    instrumented_dir: Path,
    comparison_dir: Path,
    local_box_csv: Path,
    expected_points: int,
    max_q_opt_diff: float,
    max_delta_opt_diff: float,
    max_deltaf_diff: float,
) -> dict[str, Any]:
    baseline_csv = baseline_dir / "baseline_pointwise.csv"
    candidate_csv = instrumented_dir / "baseline_pointwise.csv"
    baseline_manifest = baseline_dir / "regression_manifest.json"
    instrumented_manifest = instrumented_dir / "regression_manifest.json"
    comparison_summary = comparison_dir / "variant_summary.json"
    pointwise_comparison = comparison_dir / "pointwise_comparison.csv"
    mismatch_points = comparison_dir / "mismatch_points.csv"

    required_files = {
        "baseline_csv": _file_status(baseline_csv),
        "candidate_csv": _file_status(candidate_csv),
        "baseline_manifest": _file_status(baseline_manifest),
        "instrumented_manifest": _file_status(instrumented_manifest),
        "comparison_summary": _file_status(comparison_summary),
        "pointwise_comparison": _file_status(pointwise_comparison),
        "mismatch_points": _file_status(mismatch_points),
        "local_box_csv": _file_status(local_box_csv),
    }
    missing_files = [name for name, status in required_files.items() if not bool(status["exists"])]
    failures: list[str] = []
    if missing_files:
        failures.append("missing required files: " + ", ".join(missing_files))

    summary: dict[str, Any] = {
        "status": "fail",
        "expected_points": int(expected_points),
        "required_files": required_files,
        "missing_files": missing_files,
        "checks": {},
        "failures": failures,
    }
    if missing_files:
        return summary

    baseline_rows = _read_csv_rows(baseline_csv)
    candidate_rows = _read_csv_rows(candidate_csv)
    comparison_rows = _read_csv_rows(pointwise_comparison)
    local_box_rows = _read_csv_rows(local_box_csv)
    variant_summary = _read_json(comparison_summary)
    baseline_manifest_data = _read_json(baseline_manifest)
    instrumented_manifest_data = _read_json(instrumented_manifest)

    checks: dict[str, Any] = {
        "baseline_rows": len(baseline_rows),
        "candidate_rows": len(candidate_rows),
        "comparison_rows": len(comparison_rows),
        "local_box_rows": len(local_box_rows),
        "variant_summary": variant_summary,
        "baseline_manifest": baseline_manifest_data,
        "instrumented_manifest": instrumented_manifest_data,
    }

    if len(baseline_rows) != int(expected_points):
        failures.append(f"baseline point count {len(baseline_rows)} != expected {expected_points}")
    if len(candidate_rows) != int(expected_points):
        failures.append(f"candidate point count {len(candidate_rows)} != expected {expected_points}")
    if int(variant_summary.get("n_common_points", -1)) != int(expected_points):
        failures.append(
            f"comparison common point count {variant_summary.get('n_common_points')} != expected {expected_points}"
        )
    if int(variant_summary.get("n_missing_in_candidate", -1)) != 0:
        failures.append(f"candidate is missing {variant_summary.get('n_missing_in_candidate')} baseline points")
    if int(variant_summary.get("n_extra_in_candidate", -1)) != 0:
        failures.append(f"candidate has {variant_summary.get('n_extra_in_candidate')} extra points")
    if int(variant_summary.get("flag_mismatch_count", -1)) != 0:
        failures.append(f"flag_mismatch_count is {variant_summary.get('flag_mismatch_count')}, expected 0")

    flag_mismatch_counts: dict[str, int] = {}
    for flag in REQUIRED_FLAG_COLUMNS:
        match_col = f"{flag}_match"
        mismatch_count = sum(1 for row in comparison_rows if str(row.get(match_col, "")) != "1")
        flag_mismatch_counts[flag] = mismatch_count
        if mismatch_count:
            failures.append(f"{flag} mismatch count is {mismatch_count}, expected 0")
    checks["flag_mismatch_counts"] = flag_mismatch_counts

    max_diffs = {name: _max_float(comparison_rows, diff_col) for name, diff_col in FLOAT_DIFF_COLUMNS.items()}
    checks["max_float_diffs"] = max_diffs
    thresholds = {
        "q_opt": float(max_q_opt_diff),
        "delta_opt": float(max_delta_opt_diff),
        "DeltaF": float(max_deltaf_diff),
    }
    checks["float_diff_thresholds"] = thresholds
    for name, max_diff in max_diffs.items():
        threshold = thresholds[name]
        if not math.isfinite(max_diff):
            failures.append(f"{name} max difference is not finite")
        elif max_diff > threshold:
            failures.append(f"{name} max difference {max_diff} exceeds threshold {threshold}")

    if str(baseline_manifest_data.get("mode")) != "exact":
        failures.append(f"baseline manifest mode is {baseline_manifest_data.get('mode')!r}, expected 'exact'")
    if str(instrumented_manifest_data.get("mode")) != "exact":
        failures.append(f"instrumented manifest mode is {instrumented_manifest_data.get('mode')!r}, expected 'exact'")
    if bool(baseline_manifest_data.get("enable_local_box_instrumentation")):
        failures.append("baseline manifest unexpectedly has local-box instrumentation enabled")
    if not bool(instrumented_manifest_data.get("enable_local_box_instrumentation")):
        failures.append("instrumented manifest does not record local-box instrumentation enabled")

    local_box_columns = list(local_box_rows[0].keys()) if local_box_rows else []
    checks["local_box_columns"] = local_box_columns
    checks["required_local_box_columns"] = REQUIRED_LOCAL_BOX_COLUMNS
    if not local_box_rows:
        failures.append("local-box instrumentation CSV has no rows")
    missing_local_box_columns = [col for col in REQUIRED_LOCAL_BOX_COLUMNS if col not in local_box_columns]
    if missing_local_box_columns:
        failures.append("local-box CSV missing columns: " + ", ".join(missing_local_box_columns))

    mismatch_text = mismatch_points.read_text(encoding="utf-8").strip()
    checks["mismatch_points_empty"] = mismatch_text == ""
    if mismatch_text:
        failures.append("mismatch_points.csv is not empty")

    summary["checks"] = checks
    summary["failures"] = failures
    summary["status"] = "pass" if not failures else "fail"
    return summary


def write_gate_report(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stage1_gate_status.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Local-refinement Stage 1 Gate Status",
        "",
        f"- status: {summary['status']}",
        f"- expected_points: {summary['expected_points']}",
        f"- missing_files: {', '.join(summary['missing_files']) if summary['missing_files'] else 'none'}",
        "",
        "## Failures",
        "",
    ]
    failures = summary.get("failures", [])
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("- none")
    lines.append("")
    (output_dir / "stage1_gate_status.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the local-refinement Stage 1 fixed-point regression gate.")
    parser.add_argument("--baseline-dir", type=Path, default=Path("reports/local_refinement_refactor/stage_00_baseline/regression_gpu_baseline"))
    parser.add_argument("--instrumented-dir", type=Path, default=Path("reports/local_refinement_refactor/stage_01_instrumentation/regression_gpu_instrumented"))
    parser.add_argument("--comparison-dir", type=Path, default=Path("reports/local_refinement_refactor/stage_01_instrumentation/baseline_vs_instrumented"))
    parser.add_argument(
        "--local-box-csv",
        type=Path,
        default=Path("reports/local_refinement_refactor/stage_01_instrumentation/regression_gpu_instrumented/baseline_local_box_timing.csv"),
    )
    parser.add_argument("--expected-points", type=int, default=32)
    parser.add_argument("--max-q-opt-diff", type=float, default=1.0e-12)
    parser.add_argument("--max-delta-opt-diff", type=float, default=1.0e-12)
    parser.add_argument("--max-deltaf-diff", type=float, default=1.0e-10)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/local_refinement_refactor/stage_01_instrumentation/baseline_vs_instrumented"))
    parser.add_argument("--run-root", type=Path, default=None)
    args = parser.parse_args()
    run_root = args.run_root or _default_run_root()

    summary = verify_gate(
        baseline_dir=_run_path(args.baseline_dir, run_root),
        instrumented_dir=_run_path(args.instrumented_dir, run_root),
        comparison_dir=_run_path(args.comparison_dir, run_root),
        local_box_csv=_run_path(args.local_box_csv, run_root),
        expected_points=args.expected_points,
        max_q_opt_diff=args.max_q_opt_diff,
        max_delta_opt_diff=args.max_delta_opt_diff,
        max_deltaf_diff=args.max_deltaf_diff,
    )
    write_gate_report(summary, _run_path(args.output_dir, run_root))
    print(json.dumps(summary, indent=2))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
