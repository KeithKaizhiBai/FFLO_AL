from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports" / "local_refinement_refactor"

EXPECTED_STAGES = (
    "stage_00_baseline",
    "stage_01_instrumentation",
    "stage_02_basin_clustering",
    "stage_03_selective_refinement",
    "stage_04_energy_pruning",
    "stage_05_branch_reuse",
    "stage_06_adaptive_box_skeleton",
    "stage_07_hpc_packaging",
)

REQUIRED_STAGE_REPORTS = (
    "plan.md",
    "implementation_summary.md",
    "test_summary.md",
    "regression_summary.md",
    "decision_log.md",
)


def audit_stage_reports(report_root: Path = REPORT_ROOT, output_path: Path | None = None) -> dict[str, object]:
    report_root = Path(report_root)
    rows: list[dict[str, object]] = []
    missing: list[dict[str, str]] = []

    for stage in EXPECTED_STAGES:
        stage_dir = report_root / stage
        stage_missing: list[str] = []
        for report_name in REQUIRED_STAGE_REPORTS:
            path = stage_dir / report_name
            exists = path.is_file()
            rows.append(
                {
                    "stage": stage,
                    "report": report_name,
                    "exists": exists,
                    "path": str(path),
                }
            )
            if not exists:
                stage_missing.append(report_name)
                missing.append({"stage": stage, "report": report_name, "path": str(path)})

        if not stage_dir.is_dir():
            missing.append({"stage": stage, "report": "<stage_dir>", "path": str(stage_dir)})

    summary: dict[str, object] = {
        "status": "pass" if not missing else "fail",
        "stage_count": len(EXPECTED_STAGES),
        "required_report_count": len(REQUIRED_STAGE_REPORTS),
        "checked_file_count": len(rows),
        "missing_count": len(missing),
        "missing": missing,
        "rows": rows,
    }

    if output_path is None:
        output_path = report_root / "stage_report_completeness.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    summary = audit_stage_reports()
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
