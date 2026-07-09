from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "tests"
REPORT_ROOT = ROOT / "reports" / "local_refinement_refactor"

EXPECTED_RUNBOOK_TESTS = (
    "test_local_box_instrumentation.py",
    "test_basin_clustering.py",
    "test_mandatory_branch_keep.py",
    "test_selective_refinement.py",
    "test_energy_window_pruning.py",
    "test_branch_reuse.py",
    "test_report_template_path.py",
    "test_feature_flag_baseline_equivalence.py",
)


def audit_runbook_tests(test_root: Path = TEST_ROOT, output_path: Path | None = None) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for name in EXPECTED_RUNBOOK_TESTS:
        path = Path(test_root) / name
        exists = path.is_file()
        rows.append({"test": name, "exists": exists, "path": str(path)})
        if not exists:
            missing.append(name)

    summary: dict[str, object] = {
        "status": "pass" if not missing else "fail",
        "expected_test_count": len(EXPECTED_RUNBOOK_TESTS),
        "missing_count": len(missing),
        "missing": missing,
        "rows": rows,
    }

    if output_path is None:
        output_path = REPORT_ROOT / "runbook_test_matrix.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    summary = audit_runbook_tests()
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
