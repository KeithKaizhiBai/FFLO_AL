from __future__ import annotations

from scripts.audit_local_refinement_runbook_tests import EXPECTED_RUNBOOK_TESTS, audit_runbook_tests


def test_runbook_named_tests_exist(tmp_path):
    summary = audit_runbook_tests(output_path=tmp_path / "runbook_test_matrix.json")

    assert summary["status"] == "pass"
    assert summary["expected_test_count"] == len(EXPECTED_RUNBOOK_TESTS)
    assert summary["missing"] == []
    assert (tmp_path / "runbook_test_matrix.json").exists()
