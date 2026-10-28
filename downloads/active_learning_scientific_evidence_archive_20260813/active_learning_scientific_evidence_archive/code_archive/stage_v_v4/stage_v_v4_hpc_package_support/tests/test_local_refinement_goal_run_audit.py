from __future__ import annotations

import csv
import json

from scripts import audit_local_refinement_refactor_goal_run as audit_script


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_goal_run_audit_reflects_current_local_refinement_state(tmp_path):
    summary = audit_script.build_goal_run_audit(tmp_path / "goal_run")

    assert summary["status"] == "stage2_3_4_gpu_variant_pending"
    assert summary["stage1_gate_status"] == "pass"
    assert summary["stage1_import_status"] == "pass"
    assert summary["variant_preflight_status"] == "pass"
    assert summary["variant_return_gate_status"] == "pending"
    assert summary["variant_return_import_status"] == "pending"
    assert summary["variant_performance_report_status"] == "pending"
    assert summary["upload_set_status"] == "present"
    assert summary["upload_set_verify_status"] == "pass"
    assert summary["upload_set_nested_verify_status"] == "pass"

    stage_rows = _read_csv(tmp_path / "goal_run" / "tables" / "stage_status.csv")
    stage_status = {row["stage"]: row["status"] for row in stage_rows}
    assert stage_status["0"] == "completed"
    assert stage_status["1"] == "completed"
    assert stage_status["2"] == "local_minimal_complete_gpu_variant_pending"
    assert stage_status["3"] == "local_minimal_complete_gpu_variant_pending"
    assert stage_status["4"] == "local_minimal_complete_gpu_variant_pending"
    assert stage_status["5"] == "prototype_local_complete_integration_pending"
    assert stage_status["6"] == "skeleton_local_complete_integration_pending"
    assert stage_status["7"] == "package_handoff_ready"

    evidence_rows = _read_csv(tmp_path / "goal_run" / "tables" / "evidence_matrix.csv")
    artifacts = {row["artifact"]: row["status"] for row in evidence_rows}
    assert artifacts["Stage 1 imported gate status"] == "pass"
    assert artifacts["Variant-suite returned gate"] == "pending"
    assert artifacts["Variant-suite returned import"] == "pending"
    assert artifacts["Variant-suite performance report"] == "pending"
    assert artifacts["Upload-set handoff manifest"] == "present"
    assert artifacts["Upload-set verifier"] == "pass"
    assert artifacts["Upload-set nested package entrypoints"] == "pass"
    assert artifacts["Variant-suite return readiness checker"] == "pass"
    assert artifacts["Variant-suite HPC status checker"] == "pass"
    interpretations = {row["artifact"]: row["interpretation"] for row in evidence_rows}
    assert "nested_packages=2" in interpretations["Upload-set nested package entrypoints"]
    assert "alias_count=4" in interpretations["Upload-set nested package entrypoints"]
    assert "missing_required_paths=0" in interpretations["Upload-set nested package entrypoints"]
    assert "shell_policy_violations=0" in interpretations["Upload-set nested package entrypoints"]
    assert "gpu_script_count=7" in interpretations["Upload-set nested package entrypoints"]
    assert "gpu_policy_violations=0" in interpretations["Upload-set nested package entrypoints"]
    assert "checker_present=1" in interpretations["Variant-suite return readiness checker"]
    assert "checklist_mentions_checker=1" in interpretations["Variant-suite return readiness checker"]
    assert "package_present=1" in interpretations["Variant-suite HPC status checker"]
    assert "manifest_mentions_checker=1" in interpretations["Variant-suite HPC status checker"]

    audit_summary = json.loads((tmp_path / "goal_run" / "goal_run_audit_summary.json").read_text(encoding="utf-8"))
    assert audit_summary["status"] == summary["status"]


def test_goal_run_audit_requires_upload_set_verifier_for_handoff_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(audit_script, "UPLOAD_SET_VERIFY", tmp_path / "missing_verify_upload_set.py")

    summary = audit_script.build_goal_run_audit(tmp_path / "goal_run")

    assert summary["upload_set_status"] == "present"
    assert summary["upload_set_verify_status"] == "missing"
    assert summary["upload_set_nested_verify_status"] == "missing"

    stage_rows = _read_csv(tmp_path / "goal_run" / "tables" / "stage_status.csv")
    stage_status = {row["stage"]: row["status"] for row in stage_rows}
    assert stage_status["7"] == "package_handoff_incomplete"

    evidence_rows = _read_csv(tmp_path / "goal_run" / "tables" / "evidence_matrix.csv")
    artifacts = {row["artifact"]: row["status"] for row in evidence_rows}
    assert artifacts["Upload-set verifier"] == "missing"
    assert artifacts["Upload-set nested package entrypoints"] == "missing"
    assert artifacts["Variant-suite return readiness checker"] == "pass"
    assert artifacts["Variant-suite HPC status checker"] == "pass"


def test_goal_run_audit_requires_variant_import_and_performance_status(tmp_path, monkeypatch):
    variant_manifest = tmp_path / "latest_variant_suite_import_manifest.json"
    variant_manifest.write_text(
        json.dumps(
            {
                "gate_status": "pass",
                "import_status": "fail",
                "performance_report_status": "not_built",
                "performance_summary_json": "synthetic/performance_summary.json",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(audit_script, "VARIANT_IMPORT_MANIFEST", variant_manifest)

    summary = audit_script.build_goal_run_audit(tmp_path / "goal_run")

    assert summary["status"] == "stage2_3_4_gpu_variant_import_incomplete"
    assert summary["variant_return_gate_status"] == "pass"
    assert summary["variant_return_import_status"] == "fail"
    assert summary["variant_performance_report_status"] == "not_built"

    stage_rows = _read_csv(tmp_path / "goal_run" / "tables" / "stage_status.csv")
    stage_status = {row["stage"]: row["status"] for row in stage_rows}
    assert stage_status["2"] == "gpu_variant_import_incomplete"
    assert stage_status["3"] == "gpu_variant_import_incomplete"
    assert stage_status["4"] == "gpu_variant_import_incomplete"
