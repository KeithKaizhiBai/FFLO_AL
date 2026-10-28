from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_local_refinement_runbook_tests import audit_runbook_tests
from scripts.audit_local_refinement_stage_reports import audit_stage_reports
from scripts.audit_local_refinement_refactor_goal_run import (
    STAGE1_IMPORTED_GATE,
    VARIANT_PACKAGE_ROOT,
    VARIANT_IMPORT_MANIFEST,
    VARIANT_PREFLIGHT,
    UPLOAD_SET_ROOT,
    UPLOAD_SET_VERIFY,
    build_goal_run_audit,
)


REPORT_ROOT = ROOT / "reports" / "local_refinement_refactor_goal_run"
DEFAULT_JSON = REPORT_ROOT / "twophase_completion_audit.json"
DEFAULT_CSV = REPORT_ROOT / "tables" / "twophase_completion_requirements.csv"
DEFAULT_MD = REPORT_ROOT / "twophase_completion_audit.md"
RETURN_READINESS_CHECKER = ROOT / "scripts" / "check_local_refinement_variant_suite_return.py"
HPC_STATUS_CHECKER = ROOT / "scripts" / "check_variant_suite_hpc_status.py"
UPLOAD_SET_RETURN_CHECKLIST = UPLOAD_SET_ROOT / "RETURN_CHECKLIST.md"

RUNBOOK = ROOT / "project_history" / "plans_and_runbooks" / "TwoPhase_Optimization.md"
GOAL_REPORT_VALIDATION = REPORT_ROOT / "goal_run_report_validation.json"

PROJECT_DOCS = (
    ROOT / "docs" / "PROJECT_SUMMARY.md",
    ROOT / "docs" / "LOCAL_REFINEMENT_REFACTOR_MASTER_PLAN.md",
    ROOT / "docs" / "LOCAL_REFINEMENT_REFACTOR_DECISION_LOG.md",
    ROOT / "docs" / "LOCAL_REFINEMENT_REFACTOR_STATUS.md",
    ROOT / "docs" / "LOCAL_REFINEMENT_REFACTOR_CALL_GRAPH.md",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _row(requirement: str, status: str, evidence: str, next_action: str) -> dict[str, str]:
    return {
        "requirement": requirement,
        "status": status,
        "evidence": evidence,
        "next_action": next_action,
    }


def _run_goal_report_verifier() -> dict[str, Any]:
    verifier = ROOT / "scripts" / "verify_local_refinement_goal_run_report.py"
    if not verifier.exists():
        return {"status": "missing", "errors": [f"missing verifier: {_display(verifier)}"]}
    completed = subprocess.run(
        [sys.executable, str(verifier)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {
            "status": "fail",
            "errors": ["goal report verifier did not emit JSON", completed.stderr.strip()],
            "returncode": completed.returncode,
        }
    if completed.returncode != 0 and result.get("status") == "pass":
        result["status"] = "fail"
        result.setdefault("errors", []).append(f"verifier returned {completed.returncode}")
    return result


def _run_upload_set_verifier() -> dict[str, Any]:
    if not UPLOAD_SET_VERIFY.exists():
        return {"status": "missing", "failures": [f"missing verifier: {_display(UPLOAD_SET_VERIFY)}"]}
    completed = subprocess.run(
        [sys.executable, str(UPLOAD_SET_VERIFY), "--upload-root", str(UPLOAD_SET_ROOT)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {
            "status": "fail",
            "failures": ["upload-set verifier did not emit JSON", completed.stderr.strip()],
            "returncode": completed.returncode,
        }
    if completed.returncode != 0 and result.get("status") == "pass":
        result["status"] = "fail"
        result.setdefault("failures", []).append(f"verifier returned {completed.returncode}")
    return result


def _nested_shell_policy_summary(upload_verify: dict[str, Any]) -> tuple[str, str]:
    checked = upload_verify.get("checked", {})
    packages = checked.get("packages", {})
    if not isinstance(packages, dict) or not packages:
        return "missing", "upload-set verifier did not report nested packages"

    nested_count = 0
    shell_script_count = 0
    violation_count = 0
    for package_check in packages.values():
        if not isinstance(package_check, dict):
            continue
        nested = package_check.get("nested_package", {})
        if not isinstance(nested, dict):
            continue
        nested_count += 1
        policy = nested.get("shell_output_policy", {})
        if isinstance(policy, dict):
            shell_script_count += int(policy.get("shell_script_count", 0) or 0)
            violation_count += len(policy.get("violations", []) or [])
            if policy.get("status") != "pass":
                violation_count += 1

    status = "pass" if nested_count > 0 and violation_count == 0 else "fail"
    evidence = (
        f"nested_packages={nested_count}; shell_script_count={shell_script_count}; "
        f"shell_policy_violations={violation_count}"
    )
    return status, evidence


def _nested_gpu_slurm_policy_summary(upload_verify: dict[str, Any]) -> tuple[str, str]:
    checked = upload_verify.get("checked", {})
    packages = checked.get("packages", {})
    if not isinstance(packages, dict) or not packages:
        return "missing", "upload-set verifier did not report nested packages"

    nested_count = 0
    gpu_script_count = 0
    violation_count = 0
    for package_check in packages.values():
        if not isinstance(package_check, dict):
            continue
        nested = package_check.get("nested_package", {})
        if not isinstance(nested, dict):
            continue
        nested_count += 1
        policy = nested.get("gpu_slurm_policy", {})
        if isinstance(policy, dict):
            gpu_script_count += int(policy.get("gpu_script_count", 0) or 0)
            violation_count += len(policy.get("violations", []) or [])
            if policy.get("status") != "pass":
                violation_count += 1

    status = "pass" if nested_count > 0 and gpu_script_count > 0 and violation_count == 0 else "fail"
    evidence = (
        f"nested_packages={nested_count}; gpu_script_count={gpu_script_count}; "
        f"gpu_policy_violations={violation_count}"
    )
    return status, evidence


def _return_readiness_checker_summary() -> tuple[str, str]:
    checklist_text = UPLOAD_SET_RETURN_CHECKLIST.read_text(encoding="utf-8") if UPLOAD_SET_RETURN_CHECKLIST.exists() else ""
    checker_present = RETURN_READINESS_CHECKER.exists()
    checklist_present = UPLOAD_SET_RETURN_CHECKLIST.exists()
    checklist_mentions_checker = "check_local_refinement_variant_suite_return.py" in checklist_text
    checklist_mentions_importer = "import_local_refinement_variant_suite_results.py" in checklist_text
    status = (
        "pass"
        if checker_present and checklist_present and checklist_mentions_checker and checklist_mentions_importer
        else "missing"
    )
    evidence = (
        f"checker_present={int(checker_present)}; checklist_present={int(checklist_present)}; "
        f"checklist_mentions_checker={int(checklist_mentions_checker)}; "
        f"checklist_mentions_importer={int(checklist_mentions_importer)}"
    )
    return status, evidence


def _hpc_status_checker_summary() -> tuple[str, str]:
    package_checker = VARIANT_PACKAGE_ROOT / "scripts" / "check_variant_suite_hpc_status.py"
    run_manifest = _read_json(VARIANT_PACKAGE_ROOT / "RUN_MANIFEST.json")
    readme_path = VARIANT_PACKAGE_ROOT / "README.md"
    readme_text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    source_present = HPC_STATUS_CHECKER.exists()
    package_present = package_checker.exists()
    manifest_mentions_checker = run_manifest.get("hpc_status_check") == "scripts/check_variant_suite_hpc_status.py"
    readme_mentions_checker = "check_variant_suite_hpc_status.py" in readme_text
    status = (
        "pass"
        if source_present and package_present and manifest_mentions_checker and readme_mentions_checker
        else "missing"
    )
    evidence = (
        f"source_present={int(source_present)}; package_present={int(package_present)}; "
        f"manifest_mentions_checker={int(manifest_mentions_checker)}; "
        f"readme_mentions_checker={int(readme_mentions_checker)}"
    )
    return status, evidence


def _variant_state(variant_import: dict[str, Any]) -> str:
    gate = variant_import.get("gate_status", "pending")
    imported = variant_import.get("import_status", "pending")
    performance = variant_import.get("performance_report_status", "pending")
    if gate == "pass" and imported == "pass" and performance == "pass":
        return "pass"
    if gate == "fail" or imported == "fail" or performance == "fail":
        return "fail"
    return "pending_hpc"


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["requirement", "status", "evidence", "next_action"])
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, summary: dict[str, Any], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# TwoPhase Optimization Completion Audit",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        f"Overall status: `{summary['status']}`",
        "",
        "## Requirement Matrix",
        "",
        "| Requirement | Status | Evidence | Next action |",
        "|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["requirement"].replace("|", "\\|"),
                    f"`{row['status']}`",
                    row["evidence"].replace("|", "\\|"),
                    row["next_action"].replace("|", "\\|"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Blocking Gate",
            "",
            "The full runbook objective is not complete until the Stage 2-4 "
            "variant-suite GPU return archive is imported with "
            "`gate_status=pass`, `import_status=pass`, and "
            "`performance_report_status=pass`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_twophase_completion(
    json_path: Path = DEFAULT_JSON,
    csv_path: Path = DEFAULT_CSV,
    md_path: Path = DEFAULT_MD,
) -> dict[str, Any]:
    rows: list[dict[str, str]] = []

    missing_docs = [_display(path) for path in PROJECT_DOCS if not path.exists()]
    rows.append(
        _row(
            "project docs and runbook are present",
            "pass" if RUNBOOK.exists() and not missing_docs else "missing",
            f"runbook={_display(RUNBOOK)}; missing_docs={missing_docs}",
            "restore missing planning/status docs" if missing_docs or not RUNBOOK.exists() else "none",
        )
    )

    stage_report = audit_stage_reports()
    rows.append(
        _row(
            "all Stage 0-7 report files exist",
            str(stage_report.get("status", "missing")),
            f"checked_file_count={stage_report.get('checked_file_count')}; missing_count={stage_report.get('missing_count')}",
            "create missing stage report files" if stage_report.get("status") != "pass" else "none",
        )
    )

    runbook_tests = audit_runbook_tests()
    rows.append(
        _row(
            "runbook-named unit/synthetic tests exist",
            str(runbook_tests.get("status", "missing")),
            f"expected_test_count={runbook_tests.get('expected_test_count')}; missing_count={runbook_tests.get('missing_count')}",
            "add missing runbook tests" if runbook_tests.get("status") != "pass" else "none",
        )
    )

    build_goal_run_audit()
    goal_report = _run_goal_report_verifier()
    rows.append(
        _row(
            "goal-run report protocol validates",
            str(goal_report.get("status", "missing")),
            f"errors={goal_report.get('errors', [])}",
            "regenerate goal-run report and fix verifier errors" if goal_report.get("status") != "pass" else "none",
        )
    )

    stage1_gate = _read_json(STAGE1_IMPORTED_GATE)
    rows.append(
        _row(
            "Stage 1 imported GPU gate passes",
            "pass" if stage1_gate.get("status") == "pass" else "missing",
            f"stage1_gate_status={stage1_gate.get('status', 'missing')}",
            "import Stage 1 return bundle or rerun Stage 1 gate" if stage1_gate.get("status") != "pass" else "none",
        )
    )

    variant_preflight = _read_json(VARIANT_PREFLIGHT)
    rows.append(
        _row(
            "Stage 2-4 variant-suite local preflight passes",
            "pass" if variant_preflight.get("status") == "pass" else "missing",
            f"variant_preflight_status={variant_preflight.get('status', 'missing')}; fixed_point_count={variant_preflight.get('checked', {}).get('fixed_point_count', 'missing')}",
            "rerun variant-suite preflight" if variant_preflight.get("status") != "pass" else "none",
        )
    )

    upload_verify = _run_upload_set_verifier()
    rows.append(
        _row(
            "HPC upload-set verifier passes",
            str(upload_verify.get("status", "missing")),
            f"package_count={upload_verify.get('checked', {}).get('package_count', 'missing')}; required_next={upload_verify.get('checked', {}).get('required_next_package', 'missing')}",
            "fix upload-set verifier failures" if upload_verify.get("status") != "pass" else "none",
        )
    )

    shell_status, shell_evidence = _nested_shell_policy_summary(upload_verify)
    rows.append(
        _row(
            "nested package shell outputs stay under RUN_ROOT",
            shell_status,
            shell_evidence,
            "fix shell scripts that write logs/reports without RUN_ROOT" if shell_status != "pass" else "none",
        )
    )

    gpu_status, gpu_evidence = _nested_gpu_slurm_policy_summary(upload_verify)
    rows.append(
        _row(
            "GPU Slurm scripts exclude gpuh01 and probe CUDA runtime",
            gpu_status,
            gpu_evidence,
            "fix GPU Slurm scripts before uploading to the cluster" if gpu_status != "pass" else "none",
        )
    )

    readiness_status, readiness_evidence = _return_readiness_checker_summary()
    rows.append(
        _row(
            "variant-suite return readiness checker is available",
            readiness_status,
            readiness_evidence,
            "add checker and checklist command before returning HPC results" if readiness_status != "pass" else "none",
        )
    )

    hpc_status, hpc_evidence = _hpc_status_checker_summary()
    rows.append(
        _row(
            "variant-suite HPC status checker is packaged",
            hpc_status,
            hpc_evidence,
            "add package-local HPC status checker before upload" if hpc_status != "pass" else "none",
        )
    )

    variant_import = _read_json(VARIANT_IMPORT_MANIFEST)
    variant_status = _variant_state(variant_import)
    rows.append(
        _row(
            "Stage 2-4 GPU variant-suite return is imported and passed",
            variant_status,
            (
                f"gate_status={variant_import.get('gate_status', 'pending')}; "
                f"import_status={variant_import.get('import_status', 'pending')}; "
                f"performance_report_status={variant_import.get('performance_report_status', 'pending')}"
            ),
            (
                "upload/run variant-suite package, return results archive, then import/check locally"
                if variant_status == "pending_hpc"
                else "inspect failed variant import"
                if variant_status == "fail"
                else "none"
            ),
        )
    )

    deferred_evidence = "deferred until Stage 2-4 GPU variant-suite return passes"
    for requirement in (
        "Stage 5 branch reuse production integration",
        "Stage 6 adaptive local-box production integration",
        "Stage 7 GPU batching/Hamiltonian cache production integration",
    ):
        rows.append(_row(requirement, "deferred", deferred_evidence, "wait for Stage 2-4 GPU gate"))

    statuses = {row["status"] for row in rows}
    if "fail" in statuses or "missing" in statuses:
        overall = "incomplete"
    elif "pending_hpc" in statuses:
        overall = "pending_hpc"
    elif "deferred" in statuses:
        overall = "deferred_after_gpu_gate"
    else:
        overall = "pass"

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": overall,
        "requirement_count": len(rows),
        "status_counts": {status: sum(1 for row in rows if row["status"] == status) for status in sorted(statuses)},
        "json_path": _display(json_path),
        "csv_path": _display(csv_path),
        "md_path": _display(md_path),
        "rows": rows,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(csv_path, rows)
    _write_markdown(md_path, summary, rows)
    return summary


def main() -> int:
    summary = audit_twophase_completion()
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] in {"pass", "pending_hpc", "deferred_after_gpu_gate"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
