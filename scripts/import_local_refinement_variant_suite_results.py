from __future__ import annotations

import argparse
import csv
import json
import math
import tarfile
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_local_refinement_fixed_point_regression import resolve_variant_config
from scripts.build_local_refinement_performance_report import build_performance_report


RUNNABLE_VARIANTS = [
    "baseline",
    "cluster_only",
    "cluster_optional_k3",
    "cluster_optional_k2",
    "cluster_energy_window",
]
COMPARISON_VARIANTS = [variant for variant in RUNNABLE_VARIANTS if variant != "baseline"]
FLOAT_DIFF_KEYS = {
    "q_opt": "max_q_opt_abs_diff",
    "delta_opt": "max_delta_opt_abs_diff",
    "DeltaF": "max_deltaf_abs_diff",
}
PERFORMANCE_OUTPUT_REL = Path("reports/local_refinement_refactor/variant_regression/performance_report")


def _default_import_root(archive: Path) -> Path:
    return archive.parent / "imported_results"


def _default_extract_dir(archive: Path, import_root: Path) -> Path:
    stem = archive.name
    for suffix in (".tar.gz", ".tgz", ".tar"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return import_root / stem


def _safe_members(tar: tarfile.TarFile, destination: Path) -> list[tarfile.TarInfo]:
    dest = destination.resolve()
    members: list[tarfile.TarInfo] = []
    for member in tar.getmembers():
        target = (destination / member.name).resolve()
        if not str(target).startswith(str(dest)):
            raise ValueError(f"Unsafe archive path: {member.name}")
        members.append(member)
    return members


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


def _is_close_config(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return actual == expected


def _variant_dir(root: Path, variant: str) -> Path:
    return root / "reports" / "local_refinement_refactor" / "variant_regression" / variant


def _comparison_dir(root: Path, variant: str) -> Path:
    return root / "reports" / "local_refinement_refactor" / "variant_regression" / "comparisons" / f"baseline_vs_{variant}"


def _write_gate_report(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "variant_suite_gate_status.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Local-refinement Variant Suite Gate Status",
        "",
        f"- status: {summary['status']}",
        f"- expected_points: {summary['expected_points']}",
        f"- variants: {', '.join(summary['variants'])}",
        f"- missing_files: {', '.join(summary['missing_files']) if summary['missing_files'] else 'none'}",
        "",
        "## Variant Checks",
        "",
    ]
    for variant, checks in summary.get("variant_checks", {}).items():
        lines.append(
            f"- {variant}: rows={checks.get('pointwise_rows')}, "
            f"manifest_mode={checks.get('manifest_mode')}, "
            f"manifest_variant={checks.get('manifest_variant')}"
        )
    lines.extend(["", "## Comparison Checks", ""])
    for variant, checks in summary.get("comparison_checks", {}).items():
        lines.append(
            f"- baseline_vs_{variant}: flag_mismatch_count={checks.get('flag_mismatch_count')}, "
            f"max_q_opt_abs_diff={checks.get('max_q_opt_abs_diff')}, "
            f"max_delta_opt_abs_diff={checks.get('max_delta_opt_abs_diff')}, "
            f"max_deltaf_abs_diff={checks.get('max_deltaf_abs_diff')}"
        )
    lines.extend(["", "## Failures", ""])
    failures = summary.get("failures", [])
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("- none")
    lines.append("")
    (output_dir / "variant_suite_gate_status.md").write_text("\n".join(lines), encoding="utf-8")


def verify_variant_suite(
    root: Path,
    *,
    expected_points: int = 32,
    max_q_opt_diff: float = 1.0e-10,
    max_delta_opt_diff: float = 1.0e-10,
    max_deltaf_diff: float = 1.0e-8,
) -> dict[str, Any]:
    failures: list[str] = []
    required_files: dict[str, dict[str, Any]] = {
        "run_manifest": _file_status(root / "RUN_MANIFEST.json"),
        "fixed_points": _file_status(root / "fixed_points" / "fixed_point_regression_points.csv"),
    }

    for variant in RUNNABLE_VARIANTS:
        variant_root = _variant_dir(root, variant)
        required_files[f"{variant}_pointwise"] = _file_status(variant_root / f"{variant}_pointwise.csv")
        required_files[f"{variant}_manifest"] = _file_status(variant_root / "regression_manifest.json")
    for variant in COMPARISON_VARIANTS:
        comparison_root = _comparison_dir(root, variant)
        required_files[f"baseline_vs_{variant}_summary"] = _file_status(comparison_root / "variant_summary.json")
        required_files[f"baseline_vs_{variant}_pointwise"] = _file_status(comparison_root / "pointwise_comparison.csv")
        required_files[f"baseline_vs_{variant}_mismatch"] = _file_status(comparison_root / "mismatch_points.csv")

    missing_files = [name for name, status in required_files.items() if not bool(status["exists"])]
    if missing_files:
        failures.append("missing required files: " + ", ".join(missing_files))

    summary: dict[str, Any] = {
        "status": "fail",
        "root": str(root),
        "expected_points": int(expected_points),
        "variants": RUNNABLE_VARIANTS,
        "comparison_variants": COMPARISON_VARIANTS,
        "required_files": required_files,
        "missing_files": missing_files,
        "variant_checks": {},
        "comparison_checks": {},
        "thresholds": {
            "max_q_opt_diff": float(max_q_opt_diff),
            "max_delta_opt_diff": float(max_delta_opt_diff),
            "max_deltaf_diff": float(max_deltaf_diff),
        },
        "failures": failures,
    }
    if missing_files:
        return summary

    run_manifest = _read_json(root / "RUN_MANIFEST.json")
    if run_manifest.get("package_name") != "local_refinement_refactor_variant_suite":
        failures.append(f"RUN_MANIFEST package_name mismatch: {run_manifest.get('package_name')}")
    if run_manifest.get("active_learning") != "not_run":
        failures.append(f"RUN_MANIFEST active_learning should be not_run, got {run_manifest.get('active_learning')}")
    if run_manifest.get("variants") != RUNNABLE_VARIANTS:
        failures.append(f"RUN_MANIFEST variants mismatch: {run_manifest.get('variants')}")
    if int(run_manifest.get("expected_fixed_points", -1)) != int(expected_points):
        failures.append(f"RUN_MANIFEST expected_fixed_points mismatch: {run_manifest.get('expected_fixed_points')}")

    fixed_rows = _read_csv_rows(root / "fixed_points" / "fixed_point_regression_points.csv")
    if len(fixed_rows) != int(expected_points):
        failures.append(f"fixed-point row count {len(fixed_rows)} != expected {expected_points}")
    summary["fixed_point_rows"] = len(fixed_rows)

    for variant in RUNNABLE_VARIANTS:
        variant_root = _variant_dir(root, variant)
        pointwise = variant_root / f"{variant}_pointwise.csv"
        manifest_path = variant_root / "regression_manifest.json"
        rows = _read_csv_rows(pointwise)
        manifest = _read_json(manifest_path)
        expected_config = resolve_variant_config(variant)
        actual_config = manifest.get("variant_config", {})
        checks = {
            "pointwise_rows": len(rows),
            "manifest_mode": manifest.get("mode"),
            "manifest_variant": manifest.get("variant_name"),
            "manifest_n_points": manifest.get("n_points"),
            "variant_config": actual_config,
            "expected_variant_config": expected_config,
        }
        summary["variant_checks"][variant] = checks
        if len(rows) != int(expected_points):
            failures.append(f"{variant} pointwise row count {len(rows)} != expected {expected_points}")
        if manifest.get("mode") != "exact":
            failures.append(f"{variant} manifest mode is {manifest.get('mode')!r}, expected 'exact'")
        if manifest.get("variant_name") != variant:
            failures.append(f"{variant} manifest variant mismatch: {manifest.get('variant_name')}")
        if int(manifest.get("n_points", -1)) != int(expected_points):
            failures.append(f"{variant} manifest n_points {manifest.get('n_points')} != expected {expected_points}")
        if not _is_close_config(actual_config, expected_config):
            failures.append(f"{variant} manifest variant_config mismatch")

    thresholds = {
        "max_q_opt_abs_diff": float(max_q_opt_diff),
        "max_delta_opt_abs_diff": float(max_delta_opt_diff),
        "max_deltaf_abs_diff": float(max_deltaf_diff),
    }
    for variant in COMPARISON_VARIANTS:
        comparison_root = _comparison_dir(root, variant)
        variant_summary = _read_json(comparison_root / "variant_summary.json")
        mismatch_text = (comparison_root / "mismatch_points.csv").read_text(encoding="utf-8").strip()
        checks = {
            "n_common_points": int(variant_summary.get("n_common_points", -1)),
            "n_missing_in_candidate": int(variant_summary.get("n_missing_in_candidate", -1)),
            "n_extra_in_candidate": int(variant_summary.get("n_extra_in_candidate", -1)),
            "flag_mismatch_count": int(variant_summary.get("flag_mismatch_count", -1)),
            "max_deltaf_abs_diff": float(variant_summary.get("max_deltaf_abs_diff", float("nan"))),
            "max_q_opt_abs_diff": float(variant_summary.get("max_q_opt_abs_diff", float("nan"))),
            "max_delta_opt_abs_diff": float(variant_summary.get("max_delta_opt_abs_diff", float("nan"))),
            "mismatch_points_empty": mismatch_text == "",
        }
        summary["comparison_checks"][variant] = checks
        if checks["n_common_points"] != int(expected_points):
            failures.append(f"baseline_vs_{variant} common point count {checks['n_common_points']} != expected {expected_points}")
        if checks["n_missing_in_candidate"] != 0:
            failures.append(f"baseline_vs_{variant} missing candidate points: {checks['n_missing_in_candidate']}")
        if checks["n_extra_in_candidate"] != 0:
            failures.append(f"baseline_vs_{variant} extra candidate points: {checks['n_extra_in_candidate']}")
        if checks["flag_mismatch_count"] != 0:
            failures.append(f"baseline_vs_{variant} flag_mismatch_count {checks['flag_mismatch_count']} != 0")
        if not checks["mismatch_points_empty"]:
            failures.append(f"baseline_vs_{variant} mismatch_points.csv is not empty")
        for key, threshold in thresholds.items():
            value = float(checks[key])
            if not math.isfinite(value):
                failures.append(f"baseline_vs_{variant} {key} is not finite")
            elif value > threshold:
                failures.append(f"baseline_vs_{variant} {key} {value} exceeds threshold {threshold}")

    summary["failures"] = failures
    summary["status"] = "pass" if not failures else "fail"
    return summary


def import_and_verify(
    archive: Path,
    import_root: Path,
    extract_dir: Path | None = None,
    *,
    expected_points: int = 32,
    max_q_opt_diff: float = 1.0e-10,
    max_delta_opt_diff: float = 1.0e-10,
    max_deltaf_diff: float = 1.0e-8,
) -> dict[str, Any]:
    if not archive.exists():
        raise FileNotFoundError(f"Missing return archive: {archive}")
    target_dir = extract_dir if extract_dir is not None else _default_extract_dir(archive, import_root)
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tar:
        tar.extractall(target_dir, members=_safe_members(tar, target_dir))

    gate_output_dir = target_dir / "reports" / "local_refinement_refactor" / "variant_regression" / "import_gate"
    gate_summary = verify_variant_suite(
        target_dir,
        expected_points=expected_points,
        max_q_opt_diff=max_q_opt_diff,
        max_delta_opt_diff=max_delta_opt_diff,
        max_deltaf_diff=max_deltaf_diff,
    )
    _write_gate_report(gate_summary, gate_output_dir)

    performance_summary: dict[str, Any] | None = None
    performance_failures: list[str] = []
    if gate_summary["status"] == "pass":
        try:
            performance_summary = build_performance_report(
                target_dir,
                target_dir / PERFORMANCE_OUTPUT_REL,
                variants=RUNNABLE_VARIANTS,
                strict=True,
            )
        except Exception as exc:
            performance_failures.append(f"performance report build failed: {exc!r}")

    failures = list(gate_summary.get("failures", [])) + performance_failures
    import_status = "pass" if gate_summary["status"] == "pass" and not performance_failures else "fail"
    manifest = {
        "archive": str(archive),
        "extract_dir": str(target_dir),
        "import_status": import_status,
        "gate_status": gate_summary["status"],
        "gate_status_json": str(gate_output_dir / "variant_suite_gate_status.json"),
        "gate_status_md": str(gate_output_dir / "variant_suite_gate_status.md"),
        "performance_report_status": performance_summary["status"] if performance_summary is not None else "not_built",
        "performance_summary_json": str(target_dir / PERFORMANCE_OUTPUT_REL / "performance_summary.json"),
        "performance_report_md": str(target_dir / PERFORMANCE_OUTPUT_REL / "performance_report.md"),
        "runtime_summary_csv": str(target_dir / PERFORMANCE_OUTPUT_REL / "runtime_summary.csv"),
        "local_box_summary_csv": str(target_dir / PERFORMANCE_OUTPUT_REL / "local_box_summary.csv"),
        "missing_files": gate_summary.get("missing_files", []),
        "failures": failures,
    }
    import_root.mkdir(parents=True, exist_ok=True)
    (target_dir / "import_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (import_root / "latest_variant_suite_import_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Import and verify a returned local-refinement variant-suite HPC result bundle.")
    parser.add_argument("archive", type=Path, help="Returned local_refinement_refactor_variant_suite_results.tar.gz")
    parser.add_argument(
        "--import-root",
        type=Path,
        default=None,
        help="Import destination. Defaults to imported_results/ next to the returned archive.",
    )
    parser.add_argument("--extract-dir", type=Path, default=None)
    parser.add_argument("--expected-points", type=int, default=32)
    parser.add_argument("--max-q-opt-diff", type=float, default=1.0e-10)
    parser.add_argument("--max-delta-opt-diff", type=float, default=1.0e-10)
    parser.add_argument("--max-deltaf-diff", type=float, default=1.0e-8)
    args = parser.parse_args()
    import_root = args.import_root if args.import_root is not None else _default_import_root(args.archive)

    manifest = import_and_verify(
        args.archive,
        import_root,
        args.extract_dir,
        expected_points=args.expected_points,
        max_q_opt_diff=args.max_q_opt_diff,
        max_delta_opt_diff=args.max_delta_opt_diff,
        max_deltaf_diff=args.max_deltaf_diff,
    )
    print(json.dumps(manifest, indent=2))
    if manifest["import_status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
