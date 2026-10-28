from __future__ import annotations

import argparse
import csv
import json
import math
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = Path("reports/local_refinement_refactor/target_construction_dryrun")
RESULT_ARCHIVE = "local_refinement_target_construction_dryrun_results.tar.gz"
TARGET_VARIANTS = [
    "baseline",
    "cluster_only",
    "rank_and_cap_k3",
    "rank_and_cap_k2",
    "rank_and_cap_energy_window",
]


def _default_run_root(package_root: Path) -> Path:
    return package_root / "local_refinement_target_construction_dryrun_run"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")


def _fval(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def _fmt(value: float) -> str:
    return "" if not math.isfinite(value) else f"{value:.6g}"


def _load_point_statuses(point_dir: Path) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    for path in sorted(point_dir.glob("point_*.json")):
        try:
            statuses.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            statuses.append({"status": "failed", "status_file": str(path), "error": f"json decode failed: {exc}"})
    return statuses


def aggregate(package_root: Path, run_root: Path, output_root: Path, create_archive: bool = True) -> dict[str, Any]:
    report_root = run_root / output_root
    point_dir = report_root / "point_tasks"
    summary_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for path in sorted(point_dir.glob("point_*_summary.csv")):
        summary_rows.extend(_read_csv(path))
    for path in sorted(point_dir.glob("point_*_candidates.csv")):
        candidate_rows.extend(_read_csv(path))

    tables_dir = report_root / "tables"
    _write_csv(tables_dir / "target_construction_by_point.csv", summary_rows)
    _write_csv(tables_dir / "target_construction_candidates.csv", candidate_rows)

    by_variant: list[dict[str, Any]] = []
    for variant in TARGET_VARIANTS:
        subset = [row for row in summary_rows if row.get("variant") == variant]
        by_variant.append(
            {
                "variant": variant,
                "completed_points": len({row.get("point_id") for row in subset}),
                "rows": len(subset),
                "mean_raw_candidate_count": _fmt(_mean([_fval(row.get("raw_candidate_count")) for row in subset])),
                "mean_clustered_basin_count": _fmt(_mean([_fval(row.get("clustered_basin_count")) for row in subset])),
                "mean_selected_refine_target_count": _fmt(
                    _mean([_fval(row.get("selected_refine_target_count")) for row in subset])
                ),
                "max_selected_refine_target_count": _fmt(
                    max([_fval(row.get("selected_refine_target_count")) for row in subset], default=float("nan"))
                ),
                "mean_mandatory_basin_count": _fmt(_mean([_fval(row.get("mandatory_basin_count")) for row in subset])),
                "mean_ordinary_before_energy_window": _fmt(
                    _mean([_fval(row.get("ordinary_count_before_energy_window")) for row in subset])
                ),
                "mean_ordinary_after_energy_window": _fmt(
                    _mean([_fval(row.get("ordinary_count_after_energy_window")) for row in subset])
                ),
                "mean_energy_window_pruned_count": _fmt(
                    _mean([_fval(row.get("energy_window_pruned_count")) for row in subset])
                ),
                "gate_fail_count": sum(1 for row in subset if str(row.get("target_count_gate")) != "pass"),
                "cap": 6,
            }
        )
    _write_csv(tables_dir / "target_construction_summary.csv", by_variant)

    by_risk: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        grouped[(str(row.get("variant", "")), str(row.get("source_category", row.get("risk_category", "unknown"))))].append(row)
    for (variant, risk), subset in sorted(grouped.items()):
        by_risk.append(
            {
                "variant": variant,
                "risk_category": risk,
                "completed_points": len({row.get("point_id") for row in subset}),
                "mean_selected_refine_target_count": _fmt(
                    _mean([_fval(row.get("selected_refine_target_count")) for row in subset])
                ),
                "max_selected_refine_target_count": _fmt(
                    max([_fval(row.get("selected_refine_target_count")) for row in subset], default=float("nan"))
                ),
                "mean_mandatory_basin_count": _fmt(_mean([_fval(row.get("mandatory_basin_count")) for row in subset])),
                "mean_energy_window_pruned_count": _fmt(
                    _mean([_fval(row.get("energy_window_pruned_count")) for row in subset])
                ),
                "gate_fail_count": sum(1 for row in subset if str(row.get("target_count_gate")) != "pass"),
            }
        )
    _write_csv(tables_dir / "target_construction_by_risk_category.csv", by_risk)

    statuses = _load_point_statuses(point_dir)
    expected_points = 0
    validation_points = package_root / "config" / "validation_points.csv"
    if validation_points.exists():
        expected_points = len(_read_csv(validation_points))
    completed_point_ids = {int(row["point_id"]) for row in summary_rows if str(row.get("status")) == "success"}
    missing_points = [idx for idx in range(expected_points) if idx not in completed_point_ids]
    failed_statuses = [row for row in statuses if row.get("status") != "success"]
    gate_failures = [
        row
        for row in summary_rows
        if str(row.get("variant", "")).startswith("rank_and_cap") and str(row.get("target_count_gate")) != "pass"
    ]
    status = {
        "status": "pass" if not missing_points and not failed_statuses and not gate_failures else "fail",
        "package_root": str(package_root),
        "run_root": str(run_root),
        "output_root": str(report_root),
        "expected_points": expected_points,
        "completed_points": len(completed_point_ids),
        "summary_rows": len(summary_rows),
        "candidate_rows": len(candidate_rows),
        "missing_points": missing_points,
        "failed_status_count": len(failed_statuses),
        "rank_and_cap_gate_failure_count": len(gate_failures),
        "tables": {
            "target_construction_by_point": str(tables_dir / "target_construction_by_point.csv"),
            "target_construction_candidates": str(tables_dir / "target_construction_candidates.csv"),
            "target_construction_summary": str(tables_dir / "target_construction_summary.csv"),
            "target_construction_by_risk_category": str(tables_dir / "target_construction_by_risk_category.csv"),
        },
    }
    summary_dir = report_root / "summary"
    _write_json(summary_dir / "target_construction_gate_status.json", status)
    _write_csv(summary_dir / "failed_statuses.csv", failed_statuses)
    _write_csv(summary_dir / "rank_and_cap_gate_failures.csv", gate_failures)
    _write_csv(summary_dir / "missing_points.csv", [{"point_id": point_id} for point_id in missing_points])

    decision = [
        "# Target-Construction Dry-Run Decision Log",
        "",
        f"- status: {status['status']}",
        f"- expected_points: {expected_points}",
        f"- completed_points: {len(completed_point_ids)}",
        f"- rank_and_cap_gate_failure_count: {len(gate_failures)}",
        "",
        "This package ran target construction only: coarse scan, q-window expansion, candidate detection, basin clustering, risk annotation, energy-window marking, and final target selection.",
        "",
        "It did not run local refinement boxes and did not run active learning.",
    ]
    (report_root / "decision_log.md").write_text("\n".join(decision) + "\n", encoding="utf-8", newline="\n")

    if create_archive:
        archive = run_root / RESULT_ARCHIVE
        if archive.exists():
            archive.unlink()
        with tarfile.open(archive, "w:gz") as tar:
            for rel in ["RUN_MANIFEST.json", "README.md", "config"]:
                src = package_root / rel
                if src.exists():
                    tar.add(src, arcname=rel)
            if (run_root / "logs").exists():
                tar.add(run_root / "logs", arcname="logs")
            if report_root.exists():
                tar.add(report_root, arcname=str(output_root))
        status["return_archive"] = str(archive)
        _write_json(summary_dir / "target_construction_gate_status.json", status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate target-construction dry-run point tasks.")
    parser.add_argument("--package-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--no-archive", action="store_true")
    args = parser.parse_args()
    package_root = args.package_root.resolve()
    run_root = (args.run_root or _default_run_root(package_root)).resolve()
    status = aggregate(package_root, run_root, args.output_root, create_archive=not args.no_archive)
    print(json.dumps(status, indent=2))
    if status["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
