from __future__ import annotations

from scripts.audit_local_refinement_stage_reports import (
    EXPECTED_STAGES,
    REQUIRED_STAGE_REPORTS,
    audit_stage_reports,
)


def test_local_refinement_stage_reports_are_complete(tmp_path):
    summary = audit_stage_reports(output_path=tmp_path / "stage_report_completeness.json")

    assert summary["status"] == "pass"
    assert summary["stage_count"] == 8
    assert summary["required_report_count"] == 5
    assert summary["checked_file_count"] == len(EXPECTED_STAGES) * len(REQUIRED_STAGE_REPORTS)
    assert summary["missing"] == []
    assert (tmp_path / "stage_report_completeness.json").exists()
