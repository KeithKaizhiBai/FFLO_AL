from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports" / "local_refinement_refactor_goal_run"
STAGE1_REPORT_ROOT = ROOT / "reports" / "local_refinement_refactor" / "stage_01_instrumentation"
STAGE1_IMPORTED_MANIFEST = ROOT / "local_refinement_refactor_stage01_instrumentation" / "imported_results" / "latest_import_manifest.json"
STAGE1_IMPORTED_GATE = (
    ROOT
    / "local_refinement_refactor_stage01_instrumentation"
    / "imported_results"
    / "stage1_regression_results"
    / "reports"
    / "local_refinement_refactor"
    / "stage_01_instrumentation"
    / "baseline_vs_instrumented"
    / "stage1_gate_status.json"
)
PACKAGE_VALIDATION = STAGE1_REPORT_ROOT / "package_validation.json"
PREFLIGHT = STAGE1_REPORT_ROOT / "stage1_runtime_preflight_local_package.json"
VARIANT_PREFLIGHT = (
    ROOT
    / "hpc_packages"
    / "local_refinement_refactor_variant_suite"
    / "local_refinement_refactor_variant_suite_run"
    / "reports"
    / "local_refinement_refactor"
    / "variant_regression"
    / "preflight.json"
)
VARIANT_IMPORT_MANIFEST = ROOT / "local_refinement_refactor_variant_suite" / "imported_results" / "latest_variant_suite_import_manifest.json"
VARIANT_PACKAGE_ROOT = ROOT / "hpc_packages" / "local_refinement_refactor_variant_suite"
UPLOAD_SET_ROOT = ROOT / "hpc_packages" / "local_refinement_refactor_hpc_upload_set"
UPLOAD_SET_MANIFEST = UPLOAD_SET_ROOT / "UPLOAD_MANIFEST.json"
UPLOAD_SET_VERIFY = UPLOAD_SET_ROOT / "verify_upload_set.py"
RETURN_READINESS_CHECKER = ROOT / "scripts" / "check_local_refinement_variant_suite_return.py"
HPC_STATUS_CHECKER = ROOT / "scripts" / "check_variant_suite_hpc_status.py"
UPLOAD_SET_RETURN_CHECKLIST = UPLOAD_SET_ROOT / "RETURN_CHECKLIST.md"
VALIDATION_JSON = REPORT_ROOT / "goal_run_report_validation.json"


REQUIRED_OUTPUTS = {
    "goal_run_summary_md": "goal_run_summary.md",
    "goal_run_summary_tex": "goal_run_summary.tex",
    "goal_run_summary_pdf": "goal_run_summary.pdf",
    "decision_log": "decision_log.md",
    "stage_status_csv": "tables/stage_status.csv",
    "evidence_matrix_csv": "tables/evidence_matrix.csv",
    "stage_gate_status_png": "figures/stage_gate_status.png",
}

REQUIRED_EVIDENCE_ARTIFACTS = {
    "Stage 1 imported gate manifest",
    "Stage 1 imported gate status",
    "Stage 1 package validation",
    "Stage 1 runtime preflight",
    "Stage 1 package archive",
    "Stage 1 package SHA256 sidecar",
    "Variant-suite package archive",
    "Variant-suite local preflight",
    "Variant-suite returned gate",
    "Variant-suite returned import",
    "Variant-suite performance report",
    "Upload-set handoff manifest",
    "Upload-set verifier",
    "Upload-set nested package entrypoints",
    "Variant-suite return readiness checker",
    "Variant-suite HPC status checker",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _run_upload_set_verifier() -> dict[str, Any]:
    if not UPLOAD_SET_VERIFY.exists():
        return {"status": "missing", "checked": {}, "failures": [f"missing verifier: {_relative(UPLOAD_SET_VERIFY)}"]}
    try:
        completed = subprocess.run(
            [sys.executable, str(UPLOAD_SET_VERIFY), "--upload-root", str(UPLOAD_SET_ROOT)],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception as exc:
        return {"status": "fail", "checked": {}, "failures": [repr(exc)]}
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {
            "status": "fail",
            "checked": {"returncode": completed.returncode},
            "failures": ["verifier did not write JSON", completed.stderr.strip()],
        }
    if completed.returncode != 0 and result.get("status") == "pass":
        result["status"] = "fail"
        result.setdefault("failures", []).append(f"verifier returned {completed.returncode}")
    return result


def _check_required_outputs(report_root: Path, summary: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    outputs = summary.get("outputs", {})
    checked: dict[str, Any] = {}
    for key, rel_path in REQUIRED_OUTPUTS.items():
        expected_path = report_root / rel_path
        reported_path = outputs.get(key)
        exists = expected_path.exists()
        size = expected_path.stat().st_size if exists else 0
        checked[key] = {
            "expected_path": _relative(expected_path),
            "reported_path": reported_path,
            "exists": exists,
            "size_bytes": size,
        }
        if reported_path is None:
            errors.append(f"summary outputs is missing key: {key}")
        elif Path(reported_path) != Path(_relative(expected_path)):
            errors.append(f"summary output path mismatch for {key}: {reported_path}")
        if not exists:
            errors.append(f"required report output is missing: {_relative(expected_path)}")
        elif size <= 0:
            errors.append(f"required report output is empty: {_relative(expected_path)}")
    return checked


def _check_stage_status(report_root: Path, errors: list[str]) -> dict[str, Any]:
    rows = _read_csv(report_root / "tables" / "stage_status.csv")
    stage_by_id = {row.get("stage", ""): row for row in rows}
    expected_stages = {str(index) for index in range(8)}
    present_stages = set(stage_by_id)
    missing_stages = sorted(expected_stages - present_stages)
    extra_stages = sorted(present_stages - expected_stages)

    if missing_stages:
        errors.append(f"stage_status.csv is missing stages: {', '.join(missing_stages)}")
    if extra_stages:
        errors.append(f"stage_status.csv has unexpected stages: {', '.join(extra_stages)}")

    variant_import = _read_json(VARIANT_IMPORT_MANIFEST)
    gate_status = variant_import.get("gate_status", "pending")
    import_status = variant_import.get("import_status", "pending")
    performance_status = variant_import.get("performance_report_status", "pending")
    if gate_status == "pass" and import_status == "pass" and performance_status == "pass":
        stage2_4_status = "gpu_variant_passed"
    elif gate_status == "pass":
        stage2_4_status = "gpu_variant_import_incomplete"
    elif gate_status == "fail" or import_status == "fail":
        stage2_4_status = "gpu_variant_failed"
    else:
        stage2_4_status = "local_minimal_complete_gpu_variant_pending"

    expected_statuses = {
        "0": "completed",
        "1": "completed",
        "2": stage2_4_status,
        "3": stage2_4_status,
        "4": stage2_4_status,
        "5": "prototype_local_complete_integration_pending",
        "6": "skeleton_local_complete_integration_pending",
        "7": "package_handoff_ready" if _run_upload_set_verifier().get("status") == "pass" else "package_handoff_incomplete",
    }
    observed_statuses: dict[str, str | None] = {}
    for stage, expected_status in expected_statuses.items():
        observed = stage_by_id.get(stage, {}).get("status")
        observed_statuses[stage] = observed
        if observed != expected_status:
            errors.append(f"Stage {stage} status {observed!r} != expected {expected_status!r}")

    return {
        "row_count": len(rows),
        "present_stages": sorted(present_stages),
        "observed_statuses": observed_statuses,
    }


def _check_evidence_matrix(report_root: Path, errors: list[str]) -> dict[str, Any]:
    rows = _read_csv(report_root / "tables" / "evidence_matrix.csv")
    artifacts = {row.get("artifact", ""): row for row in rows}
    present = set(artifacts)
    missing = sorted(REQUIRED_EVIDENCE_ARTIFACTS - present)
    if missing:
        errors.append(f"evidence_matrix.csv is missing artifacts: {', '.join(missing)}")

    readiness = artifacts.get("Variant-suite return readiness checker", {})
    if readiness and readiness.get("status") != "pass":
        errors.append("variant-suite return readiness checker evidence should be pass")
    hpc_status = artifacts.get("Variant-suite HPC status checker", {})
    if hpc_status and hpc_status.get("status") != "pass":
        errors.append("variant-suite HPC status checker evidence should be pass")

    return {
        "row_count": len(rows),
        "present_artifacts": sorted(present),
    }


def _check_status_consistency(summary: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    stage1_manifest = _read_json(STAGE1_IMPORTED_MANIFEST)
    stage1_gate = _read_json(STAGE1_IMPORTED_GATE)
    package_validation = _read_json(PACKAGE_VALIDATION)
    preflight = _read_json(PREFLIGHT)
    variant_preflight = _read_json(VARIANT_PREFLIGHT)
    variant_import = _read_json(VARIANT_IMPORT_MANIFEST)
    upload_set = _read_json(UPLOAD_SET_MANIFEST)

    stage1_import_status = stage1_manifest.get("gate_status", "missing")
    stage1_gate_status = stage1_gate.get("status", "missing")
    package_status = package_validation.get("status", "missing")
    preflight_status = preflight.get("status", "missing")
    variant_preflight_status = variant_preflight.get("status", "missing")
    variant_return_status = variant_import.get("gate_status", "pending")
    variant_import_status = variant_import.get("import_status", "pending")
    variant_performance_status = variant_import.get("performance_report_status", "pending")
    upload_set_status = "present" if UPLOAD_SET_MANIFEST.exists() else "missing"
    upload_set_verify = _run_upload_set_verifier()
    upload_set_verify_status = upload_set_verify.get("status", "missing")
    packages = upload_set_verify.get("checked", {}).get("packages", {})
    nested_status = "missing"
    if isinstance(packages, dict) and packages:
        nested_status = "pass"
        for package_check in packages.values():
            nested = package_check.get("nested_package", {}) if isinstance(package_check, dict) else {}
            if nested.get("status") != "pass":
                nested_status = "fail"
                break
    if variant_return_status == "pass" and variant_import_status == "pass" and variant_performance_status == "pass":
        expected_goal_status = "stage2_3_4_gpu_variant_passed"
    elif variant_return_status == "pass":
        expected_goal_status = "stage2_3_4_gpu_variant_import_incomplete"
    elif variant_return_status == "fail" or variant_import_status == "fail":
        expected_goal_status = "stage2_3_4_gpu_variant_failed"
    else:
        expected_goal_status = "stage2_3_4_gpu_variant_pending"

    checks = {
        "stage1_import_status": {
            "summary": summary.get("stage1_import_status"),
            "source": stage1_import_status,
        },
        "stage1_gate_status": {
            "summary": summary.get("stage1_gate_status"),
            "source": stage1_gate_status,
        },
        "package_validation_status": {
            "summary": summary.get("package_validation_status"),
            "source": package_status,
        },
        "preflight_status": {
            "summary": summary.get("preflight_status"),
            "source": preflight_status,
        },
        "variant_preflight_status": {
            "summary": summary.get("variant_preflight_status"),
            "source": variant_preflight_status,
        },
        "variant_return_gate_status": {
            "summary": summary.get("variant_return_gate_status"),
            "source": variant_return_status,
        },
        "variant_return_import_status": {
            "summary": summary.get("variant_return_import_status"),
            "source": variant_import_status,
        },
        "variant_performance_report_status": {
            "summary": summary.get("variant_performance_report_status"),
            "source": variant_performance_status,
        },
        "upload_set_status": {
            "summary": summary.get("upload_set_status"),
            "source": upload_set_status,
        },
        "upload_set_verify_status": {
            "summary": summary.get("upload_set_verify_status"),
            "source": upload_set_verify_status,
        },
        "upload_set_nested_verify_status": {
            "summary": summary.get("upload_set_nested_verify_status"),
            "source": nested_status,
        },
        "status": {
            "summary": summary.get("status"),
            "expected": expected_goal_status,
        },
    }

    for key, values in checks.items():
        observed = values["summary"]
        expected = values.get("source", values.get("expected"))
        if observed != expected:
            errors.append(f"summary {key}={observed!r} does not match expected {expected!r}")

    if stage1_gate_status != "pass":
        errors.append("Stage 1 imported GPU gate should be pass before using the current stage status audit")
    if package_status != "pass":
        errors.append("Stage 1 package validation should be pass")
    if variant_preflight_status != "pass":
        errors.append("Variant-suite preflight should be pass")
    if upload_set.get("required_next_package") != "local_refinement_refactor_variant_suite":
        errors.append(f"upload set required_next_package mismatch: {upload_set.get('required_next_package')}")
    if upload_set_verify_status != "pass":
        errors.append(f"upload set verifier status should be pass, got {upload_set_verify_status!r}")
    if nested_status != "pass":
        errors.append(f"upload set nested package entrypoint status should be pass, got {nested_status!r}")
    checklist_text = UPLOAD_SET_RETURN_CHECKLIST.read_text(encoding="utf-8") if UPLOAD_SET_RETURN_CHECKLIST.exists() else ""
    if not RETURN_READINESS_CHECKER.exists():
        errors.append(f"missing return readiness checker: {_relative(RETURN_READINESS_CHECKER)}")
    if "check_local_refinement_variant_suite_return.py" not in checklist_text:
        errors.append("upload-set RETURN_CHECKLIST does not mention the return readiness checker")
    variant_manifest = _read_json(VARIANT_PACKAGE_ROOT / "RUN_MANIFEST.json")
    variant_readme = (VARIANT_PACKAGE_ROOT / "README.md").read_text(encoding="utf-8") if (VARIANT_PACKAGE_ROOT / "README.md").exists() else ""
    if not HPC_STATUS_CHECKER.exists():
        errors.append(f"missing HPC status checker source: {_relative(HPC_STATUS_CHECKER)}")
    if not (VARIANT_PACKAGE_ROOT / "scripts" / "check_variant_suite_hpc_status.py").exists():
        errors.append("variant-suite package is missing scripts/check_variant_suite_hpc_status.py")
    if variant_manifest.get("hpc_status_check") != "scripts/check_variant_suite_hpc_status.py":
        errors.append("variant-suite RUN_MANIFEST does not record hpc_status_check")
    if "check_variant_suite_hpc_status.py" not in variant_readme:
        errors.append("variant-suite README does not mention the HPC status checker")

    return {
        "stage1_import_status": stage1_import_status,
        "stage1_gate_status": stage1_gate_status,
        "package_validation_status": package_status,
        "preflight_status": preflight_status,
        "variant_preflight_status": variant_preflight_status,
        "variant_return_gate_status": variant_return_status,
        "variant_return_import_status": variant_import_status,
        "variant_performance_report_status": variant_performance_status,
        "upload_set_status": upload_set_status,
        "upload_set_verify_status": upload_set_verify_status,
        "upload_set_nested_verify_status": nested_status,
        "expected_goal_status": expected_goal_status,
    }


def verify_goal_run_report(report_root: Path = REPORT_ROOT) -> dict[str, Any]:
    errors: list[str] = []
    summary_path = report_root / "goal_run_audit_summary.json"
    summary = _read_json(summary_path)
    if not summary:
        errors.append(f"missing or empty audit summary: {_relative(summary_path)}")

    checked = {
        "required_outputs": _check_required_outputs(report_root, summary, errors),
        "stage_status": _check_stage_status(report_root, errors),
        "evidence_matrix": _check_evidence_matrix(report_root, errors),
        "status_consistency": _check_status_consistency(summary, errors),
    }
    result = {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "checked": checked,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify local-refinement refactor goal-run report outputs.")
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    parser.add_argument("--output-json", type=Path, default=VALIDATION_JSON)
    args = parser.parse_args()

    result = verify_goal_run_report(args.report_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
