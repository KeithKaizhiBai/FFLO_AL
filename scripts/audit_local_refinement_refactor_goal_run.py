from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports" / "local_refinement_refactor_goal_run"
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
PACKAGE_VALIDATION = ROOT / "reports" / "local_refinement_refactor" / "stage_01_instrumentation" / "package_validation.json"
PREFLIGHT = ROOT / "reports" / "local_refinement_refactor" / "stage_01_instrumentation" / "stage1_runtime_preflight_local_package.json"
PACKAGE_ARCHIVE = ROOT / "hpc_packages" / "local_refinement_refactor_stage01_instrumentation.tar.gz"
PACKAGE_SHA256 = ROOT / "hpc_packages" / "local_refinement_refactor_stage01_instrumentation.tar.gz.sha256"
VARIANT_PACKAGE_ARCHIVE = ROOT / "hpc_packages" / "local_refinement_refactor_variant_suite.tar.gz"
VARIANT_PACKAGE_METADATA = ROOT / "hpc_packages" / "local_refinement_refactor_variant_suite.tar.gz.metadata.json"
VARIANT_PACKAGE_ROOT = ROOT / "hpc_packages" / "local_refinement_refactor_variant_suite"
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
UPLOAD_SET_ROOT = ROOT / "hpc_packages" / "local_refinement_refactor_hpc_upload_set"
UPLOAD_SET_MANIFEST = UPLOAD_SET_ROOT / "UPLOAD_MANIFEST.json"
UPLOAD_SET_VERIFY = UPLOAD_SET_ROOT / "verify_upload_set.py"
UPLOAD_SET_ARCHIVE = ROOT / "hpc_packages" / "local_refinement_refactor_hpc_upload_set.tar.gz"
RETURN_READINESS_CHECKER = ROOT / "scripts" / "check_local_refinement_variant_suite_return.py"
HPC_STATUS_CHECKER = ROOT / "scripts" / "check_variant_suite_hpc_status.py"
UPLOAD_SET_RETURN_CHECKLIST = UPLOAD_SET_ROOT / "RETURN_CHECKLIST.md"

STAGE_REPORTS = {
    "2": ROOT / "reports" / "local_refinement_refactor" / "stage_02_basin_clustering" / "test_summary.md",
    "3": ROOT / "reports" / "local_refinement_refactor" / "stage_03_selective_refinement" / "test_summary.md",
    "4": ROOT / "reports" / "local_refinement_refactor" / "stage_04_energy_pruning" / "test_summary.md",
    "5": ROOT / "reports" / "local_refinement_refactor" / "stage_05_branch_reuse" / "test_summary.md",
    "6": ROOT / "reports" / "local_refinement_refactor" / "stage_06_adaptive_box_skeleton" / "test_summary.md",
    "7": ROOT / "reports" / "local_refinement_refactor" / "stage_07_hpc_packaging" / "test_summary.md",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _exists_status(path: Path) -> str:
    return "present" if path.exists() else "missing"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _status_from_path(path: Path, present_status: str) -> str:
    return present_status if path.exists() else "missing"


def _run_upload_set_verifier() -> dict[str, Any]:
    if not UPLOAD_SET_VERIFY.exists():
        return {
            "status": "missing",
            "checked": {"upload_root": str(UPLOAD_SET_ROOT)},
            "failures": [f"missing upload-set verifier: {UPLOAD_SET_VERIFY}"],
        }
    try:
        completed = subprocess.run(
            [sys.executable, str(UPLOAD_SET_VERIFY), "--upload-root", str(UPLOAD_SET_ROOT)],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception as exc:
        return {
            "status": "fail",
            "checked": {"upload_root": str(UPLOAD_SET_ROOT)},
            "failures": [f"upload-set verifier execution failed: {exc!r}"],
        }
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {
            "status": "fail",
            "checked": {"upload_root": str(UPLOAD_SET_ROOT), "returncode": completed.returncode},
            "failures": ["upload-set verifier did not write JSON output", completed.stderr.strip()],
        }
    result["returncode"] = completed.returncode
    if completed.returncode != 0 and result.get("status") == "pass":
        result["status"] = "fail"
        result.setdefault("failures", []).append(f"upload-set verifier returned {completed.returncode}")
    return result


def _variant_validation_state(variant_import: dict[str, Any]) -> str:
    gate_status = variant_import.get("gate_status", "pending")
    import_status = variant_import.get("import_status", "pending")
    performance_status = variant_import.get("performance_report_status", "pending")
    if import_status == "pass" and gate_status == "pass" and performance_status == "pass":
        return "gpu_variant_passed"
    if gate_status == "pass":
        return "gpu_variant_import_incomplete"
    if gate_status == "fail" or import_status == "fail":
        return "gpu_variant_failed"
    return "gpu_variant_pending"


def _local_or_variant_stage_status(path: Path, variant_state: str) -> str:
    if not path.exists():
        return "missing"
    if variant_state in {"gpu_variant_passed", "gpu_variant_import_incomplete", "gpu_variant_failed"}:
        return variant_state
    return "local_minimal_complete_gpu_variant_pending"


def _upload_stage_status(upload_set_verify: dict[str, Any]) -> str:
    if not UPLOAD_SET_MANIFEST.exists():
        return "missing"
    if upload_set_verify.get("status") == "pass":
        return "package_handoff_ready"
    return "package_handoff_incomplete"


def _upload_set_nested_entrypoint_check(upload_set_verify: dict[str, Any]) -> dict[str, str]:
    checked = upload_set_verify.get("checked", {})
    packages = checked.get("packages", {})
    if upload_set_verify.get("status") == "missing":
        return {
            "status": "missing",
            "interpretation": "upload-set verifier is missing; nested package entry points were not checked",
        }
    if not isinstance(packages, dict) or not packages:
        return {
            "status": "missing",
            "interpretation": "upload-set verifier did not report nested package checks",
        }

    nested_count = 0
    nested_pass = 0
    alias_count = 0
    missing_required_paths = 0
    run_root_env_failures = 0
    shell_script_count = 0
    shell_policy_violations = 0
    gpu_script_count = 0
    gpu_policy_violations = 0
    for package_check in packages.values():
        if not isinstance(package_check, dict):
            continue
        nested = package_check.get("nested_package", {})
        if not isinstance(nested, dict):
            continue
        nested_count += 1
        if nested.get("status") == "pass":
            nested_pass += 1
        alias_count += int(nested.get("alias_count", 0) or 0)
        missing_required_paths += len(nested.get("missing_required_paths", []) or [])
        if nested.get("run_manifest_writable_run_root_env") != "RUN_ROOT":
            run_root_env_failures += 1
        shell_output_policy = nested.get("shell_output_policy", {})
        if isinstance(shell_output_policy, dict):
            shell_script_count += int(shell_output_policy.get("shell_script_count", 0) or 0)
            shell_policy_violations += len(shell_output_policy.get("violations", []) or [])
            if shell_output_policy.get("status") != "pass":
                shell_policy_violations += 1
        gpu_slurm_policy = nested.get("gpu_slurm_policy", {})
        if isinstance(gpu_slurm_policy, dict):
            gpu_script_count += int(gpu_slurm_policy.get("gpu_script_count", 0) or 0)
            gpu_policy_violations += len(gpu_slurm_policy.get("violations", []) or [])
            if gpu_slurm_policy.get("status") != "pass":
                gpu_policy_violations += 1

    status = (
        "pass"
        if (
            nested_count > 0
            and nested_count == nested_pass
            and missing_required_paths == 0
            and run_root_env_failures == 0
            and shell_policy_violations == 0
            and gpu_policy_violations == 0
        )
        else "fail"
    )
    return {
        "status": status,
        "interpretation": (
            f"nested_packages={nested_count}; nested_pass={nested_pass}; "
            f"alias_count={alias_count}; missing_required_paths={missing_required_paths}; "
            f"run_root_env_failures={run_root_env_failures}; "
            f"shell_script_count={shell_script_count}; shell_policy_violations={shell_policy_violations}; "
            f"gpu_script_count={gpu_script_count}; gpu_policy_violations={gpu_policy_violations}"
        ),
    }


def _return_readiness_checker_status() -> dict[str, str]:
    checklist_text = UPLOAD_SET_RETURN_CHECKLIST.read_text(encoding="utf-8") if UPLOAD_SET_RETURN_CHECKLIST.exists() else ""
    checker_present = RETURN_READINESS_CHECKER.exists()
    checklist_present = UPLOAD_SET_RETURN_CHECKLIST.exists()
    checklist_mentions_checker = "check_local_refinement_variant_suite_return.py" in checklist_text
    checklist_mentions_importer = "import_local_refinement_variant_suite_results.py" in checklist_text
    status = (
        "pass"
        if (
            checker_present
            and checklist_present
            and checklist_mentions_checker
            and checklist_mentions_importer
        )
        else "missing"
    )
    return {
        "status": status,
        "interpretation": (
            f"checker_present={int(checker_present)}; checklist_present={int(checklist_present)}; "
            f"checklist_mentions_checker={int(checklist_mentions_checker)}; "
            f"checklist_mentions_importer={int(checklist_mentions_importer)}"
        ),
    }


def _hpc_status_checker_package_status() -> dict[str, str]:
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
    return {
        "status": status,
        "interpretation": (
            f"source_present={int(source_present)}; package_present={int(package_present)}; "
            f"manifest_mentions_checker={int(manifest_mentions_checker)}; "
            f"readme_mentions_checker={int(readme_mentions_checker)}"
        ),
    }


def _stage_rows(
    stage1_gate: dict[str, Any],
    variant_import: dict[str, Any],
    upload_set_verify: dict[str, Any],
) -> list[dict[str, str]]:
    stage1_status = "completed" if stage1_gate.get("status") == "pass" else "gpu_gate_missing"
    variant_status = _variant_validation_state(variant_import)
    return [
        {
            "stage": "0",
            "name": "baseline freeze",
            "status": "completed",
            "evidence": "Stage 0 baseline exact regression is included in the imported Stage 1 gate.",
            "next_gate": "none",
        },
        {
            "stage": "1",
            "name": "box-level instrumentation",
            "status": stage1_status,
            "evidence": f"imported_stage1_gate={stage1_gate.get('status', 'missing')}",
            "next_gate": "none" if stage1_status == "completed" else "baseline-vs-instrumented GPU equivalence gate",
        },
        {
            "stage": "2",
            "name": "basin clustering",
            "status": _local_or_variant_stage_status(STAGE_REPORTS["2"], variant_status),
            "evidence": f"feature-flagged local implementation and synthetic tests; variant_state={variant_status}",
            "next_gate": "none" if variant_status == "gpu_variant_passed" else "variant-suite GPU fixed-point regression and import",
        },
        {
            "stage": "3",
            "name": "mandatory-risk keep and selective refinement",
            "status": _local_or_variant_stage_status(STAGE_REPORTS["3"], variant_status),
            "evidence": f"feature-flagged local implementation and synthetic tests; variant_state={variant_status}",
            "next_gate": "none" if variant_status == "gpu_variant_passed" else "variant-suite GPU fixed-point regression and import",
        },
        {
            "stage": "4",
            "name": "energy-window pruning",
            "status": _local_or_variant_stage_status(STAGE_REPORTS["4"], variant_status),
            "evidence": f"feature-flagged local implementation and synthetic tests; variant_state={variant_status}",
            "next_gate": "none" if variant_status == "gpu_variant_passed" else "variant-suite GPU fixed-point regression and import",
        },
        {
            "stage": "5",
            "name": "branch reuse prototype",
            "status": _status_from_path(STAGE_REPORTS["5"], "prototype_local_complete_integration_pending"),
            "evidence": "decision helpers and synthetic tests only; not integrated into production exact loop",
            "next_gate": "production reuse diagnostics and later GPU regression",
        },
        {
            "stage": "6",
            "name": "adaptive local box skeleton",
            "status": _status_from_path(STAGE_REPORTS["6"], "skeleton_local_complete_integration_pending"),
            "evidence": "geometry/proxy helpers and synthetic tests only; fixed boxes remain production default",
            "next_gate": "adaptive-box integration design and later GPU regression",
        },
        {
            "stage": "7",
            "name": "GPU batching and Hamiltonian cache sketch",
            "status": _upload_stage_status(upload_set_verify),
            "evidence": f"variant-suite package and upload-set verifier={upload_set_verify.get('status', 'missing')}; stage2_3_4={variant_status}",
            "next_gate": "upload and return variant-suite GPU results",
        },
    ]


def _evidence_rows(
    stage1_manifest: dict[str, Any],
    stage1_gate: dict[str, Any],
    package_validation: dict[str, Any],
    preflight: dict[str, Any],
    variant_metadata: dict[str, Any],
    variant_preflight: dict[str, Any],
    variant_import: dict[str, Any],
    upload_set: dict[str, Any],
    upload_set_verify: dict[str, Any],
) -> list[dict[str, str]]:
    checked = package_validation.get("checked", {})
    preflight_checked = preflight.get("checked", {})
    variant_checked = variant_preflight.get("checked", {})
    upload_packages = upload_set.get("packages", [])
    performance_status = variant_import.get("performance_report_status", "pending")
    nested_entrypoints = _upload_set_nested_entrypoint_check(upload_set_verify)
    return_readiness = _return_readiness_checker_status()
    hpc_status_checker = _hpc_status_checker_package_status()
    return [
        {
            "artifact": "Stage 1 imported gate manifest",
            "path": str(STAGE1_IMPORTED_MANIFEST.relative_to(ROOT)),
            "status": stage1_manifest.get("gate_status", "missing"),
            "interpretation": "returned Stage 1 bundle imported and verified",
        },
        {
            "artifact": "Stage 1 imported gate status",
            "path": str(STAGE1_IMPORTED_GATE.relative_to(ROOT)),
            "status": stage1_gate.get("status", "missing"),
            "interpretation": "baseline-vs-instrumented GPU equivalence passed" if stage1_gate.get("status") == "pass" else "Stage 1 imported GPU gate missing or failed",
        },
        {
            "artifact": "Stage 1 package validation",
            "path": str(PACKAGE_VALIDATION.relative_to(ROOT)),
            "status": package_validation.get("status", "missing"),
            "interpretation": f"archive_file_count={checked.get('archive_file_count')}; sha256={checked.get('archive_sha256')}",
        },
        {
            "artifact": "Stage 1 runtime preflight",
            "path": str(PREFLIGHT.relative_to(ROOT)),
            "status": preflight.get("status", "missing"),
            "interpretation": f"fixed_point_count={preflight_checked.get('fixed_point_count')}; syntax_check_count={preflight_checked.get('syntax_check_count')}",
        },
        {
            "artifact": "Stage 1 package archive",
            "path": str(PACKAGE_ARCHIVE.relative_to(ROOT)),
            "status": _exists_status(PACKAGE_ARCHIVE),
            "interpretation": "upload artifact for target HPC fixed-point regression",
        },
        {
            "artifact": "Stage 1 package SHA256 sidecar",
            "path": str(PACKAGE_SHA256.relative_to(ROOT)),
            "status": _exists_status(PACKAGE_SHA256),
            "interpretation": "HPC-side upload integrity check input",
        },
        {
            "artifact": "Variant-suite package archive",
            "path": str(VARIANT_PACKAGE_ARCHIVE.relative_to(ROOT)),
            "status": _exists_status(VARIANT_PACKAGE_ARCHIVE),
            "interpretation": f"sha256={variant_metadata.get('archive_sha256', 'missing')}",
        },
        {
            "artifact": "Variant-suite local preflight",
            "path": str(VARIANT_PREFLIGHT.relative_to(ROOT)),
            "status": variant_preflight.get("status", "missing"),
            "interpretation": f"fixed_point_count={variant_checked.get('fixed_point_count')}; variants={variant_checked.get('variants')}",
        },
        {
            "artifact": "Variant-suite returned gate",
            "path": _display_path(VARIANT_IMPORT_MANIFEST),
            "status": variant_import.get("gate_status", "pending"),
            "interpretation": "physics-equivalence gate from returned variant-suite archive",
        },
        {
            "artifact": "Variant-suite returned import",
            "path": _display_path(VARIANT_IMPORT_MANIFEST),
            "status": variant_import.get("import_status", "pending"),
            "interpretation": "combined return import; requires gate pass and performance companion files",
        },
        {
            "artifact": "Variant-suite performance report",
            "path": str(variant_import.get("performance_summary_json", _display_path(VARIANT_IMPORT_MANIFEST))),
            "status": performance_status,
            "interpretation": "runtime/local-box diagnostics required after physics gate passes",
        },
        {
            "artifact": "Upload-set handoff manifest",
            "path": str(UPLOAD_SET_MANIFEST.relative_to(ROOT)),
            "status": _exists_status(UPLOAD_SET_MANIFEST),
            "interpretation": f"packages={len(upload_packages)}; required_next={upload_set.get('required_next_package', 'missing')}",
        },
        {
            "artifact": "Upload-set verifier",
            "path": _display_path(UPLOAD_SET_VERIFY),
            "status": upload_set_verify.get("status", "missing"),
            "interpretation": f"package_count={upload_set_verify.get('checked', {}).get('package_count')}; failures={len(upload_set_verify.get('failures', []))}",
        },
        {
            "artifact": "Upload-set nested package entrypoints",
            "path": _display_path(UPLOAD_SET_VERIFY),
            "status": nested_entrypoints["status"],
            "interpretation": nested_entrypoints["interpretation"],
        },
        {
            "artifact": "Variant-suite return readiness checker",
            "path": _display_path(RETURN_READINESS_CHECKER),
            "status": return_readiness["status"],
            "interpretation": return_readiness["interpretation"],
        },
        {
            "artifact": "Variant-suite HPC status checker",
            "path": _display_path(HPC_STATUS_CHECKER),
            "status": hpc_status_checker["status"],
            "interpretation": hpc_status_checker["interpretation"],
        },
    ]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_stage_figure(path: Path, stage_rows: list[dict[str, str]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on local plotting environment.
        raise RuntimeError("matplotlib is required to write the goal-run figure") from exc

    status_colors = {
        "completed": "#7cbf7c",
        "local_minimal_complete_gpu_variant_pending": "#f2b84b",
        "prototype_local_complete_integration_pending": "#d9c46f",
        "skeleton_local_complete_integration_pending": "#d9c46f",
        "package_handoff_ready": "#8fb6d9",
        "gpu_variant_passed": "#5ab878",
        "gpu_variant_import_incomplete": "#f2b84b",
        "gpu_variant_failed": "#d47777",
        "missing": "#c9c9c9",
    }
    labels = [f"Stage {row['stage']}" for row in stage_rows]
    colors = [status_colors.get(row["status"], "#8fbf88") for row in stage_rows]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.2, 3.2))
    ax.bar(labels, [1] * len(stage_rows), color=colors, edgecolor="#333333", linewidth=0.8)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([])
    ax.set_title("Local-Refinement Refactor Stage Gate Status")
    ax.text(0.5, 1.08, "Stage 2-4 local logic is ready; GPU variant regression is pending", ha="center", transform=ax.transAxes)
    for index, row in enumerate(stage_rows):
        ax.text(index, 0.5, row["status"].replace("_", "\n"), ha="center", va="center", fontsize=7)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.spines["bottom"].set_color("#333333")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(row[col] for col in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def _latex_cell_text(text: str) -> str:
    if "sha256=" in text:
        prefix, digest = text.split("sha256=", 1)
        return prefix + "sha256=" + digest[:16] + "..."
    return text


def _write_markdown(
    path: Path,
    stage_rows: list[dict[str, str]],
    evidence_rows: list[dict[str, str]],
    generated_at: str,
    figure_rel_path: str,
) -> None:
    variant_import = next(row for row in evidence_rows if row["artifact"] == "Variant-suite returned import")
    text = f"""# Local-Refinement Refactor Goal-Run Summary

Generated: {generated_at}

## Current Conclusion

The refactor is not complete.  Stage 1 has passed the returned target GPU/CUDA
baseline-vs-instrumented gate.  Stages 2, 3, and 4 are locally implemented and
feature-flagged, but their fixed-point GPU variant regression has not yet
completed a successful local import with required performance companion files.

## Stage Status

![Stage gate status]({figure_rel_path})

{_markdown_table(stage_rows, ["stage", "name", "status", "next_gate"])}

## Evidence Matrix

{_markdown_table(evidence_rows, ["artifact", "status", "interpretation"])}

## Active Blocker

{variant_import["interpretation"]}

## Next Action

Upload `hpc_packages/local_refinement_refactor_hpc_upload_set.tar.gz` or
`hpc_packages/local_refinement_refactor_variant_suite.tar.gz`, run the
variant-suite workflow on the target GPU/CUDA environment, return
`local_refinement_refactor_variant_suite_results.tar.gz`, and import it locally
with `scripts/import_local_refinement_variant_suite_results.py`.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _write_decision_log(path: Path, generated_at: str) -> None:
    text = f"""# Local-Refinement Refactor Goal-Run Decision Log

Generated: {generated_at}

## Decision

Keep Stage 5 branch reuse, Stage 6 adaptive boxes, and Stage 7 GPU
batching/cache out of the production exact loop until the Stage 2/3/4
variant-suite fixed-point GPU regression is returned and checked.

## Reason

The TwoPhase runbook requires feature-flag baseline equivalence and forbids
using local mock checks, package QA, or preflight checks as proof of physics
equivalence.  Stage 1 equivalence is now proven, but Stage 2/3/4 variants still
need the target GPU/CUDA fixed-point comparison gate.

## Consequence

Current local work may improve auditability, packaging, upload checks, and
handoff documentation.  Production branch reuse, adaptive boxes, GPU batching,
and Hamiltonian cache remain deferred until their own explicit integration
designs and GPU regression gates exist.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _write_latex(
    path: Path,
    stage_rows: list[dict[str, str]],
    evidence_rows: list[dict[str, str]],
    generated_at: str,
    figure_rel_path: str,
) -> None:
    stage_lines = "\n".join(
        rf"{_latex_escape(row['stage'])} & {_latex_escape(row['name'])} & {_latex_escape(row['status'])} \\" for row in stage_rows
    )
    evidence_lines = "\n".join(
        rf"{_latex_escape(row['artifact'])} & {_latex_escape(row['status'])} & {_latex_escape(_latex_cell_text(row['interpretation']))} \\" for row in evidence_rows
    )
    text = rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.7in]{{geometry}}
\usepackage{{longtable}}
\usepackage{{array}}
\usepackage{{graphicx}}
\begin{{document}}
\section*{{Local-Refinement Refactor Goal-Run Summary}}
Generated: {_latex_escape(generated_at)}

The refactor is not complete. Stage 1 has passed the returned target GPU/CUDA
baseline-vs-instrumented gate. Stages 2, 3, and 4 are locally implemented and
feature-flagged, but their fixed-point GPU variant regression is pending.

\begin{{center}}
\includegraphics[width=0.95\linewidth]{{{_latex_escape(figure_rel_path)}}}
\end{{center}}

\section*{{Stage Status}}
\begin{{longtable}}{{p{{0.08\linewidth}}p{{0.32\linewidth}}p{{0.46\linewidth}}}}
Stage & Name & Status \\
\hline
{stage_lines}
\end{{longtable}}

\section*{{Evidence Matrix}}
\begin{{longtable}}{{p{{0.24\linewidth}}p{{0.12\linewidth}}p{{0.56\linewidth}}}}
Artifact & Status & Interpretation \\
\hline
{evidence_lines}
\end{{longtable}}

\section*{{Next Action}}
Upload the local-refinement HPC upload set or the variant-suite package, run
the fixed-point variant workflow on the target GPU/CUDA environment, return the
variant-suite result bundle, and import it locally.
\end{{document}}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def build_goal_run_audit(report_root: Path = REPORT_ROOT) -> dict[str, Any]:
    generated_at = datetime.now().isoformat(timespec="seconds")
    stage1_manifest = _read_json(STAGE1_IMPORTED_MANIFEST)
    stage1_gate = _read_json(STAGE1_IMPORTED_GATE)
    package_validation = _read_json(PACKAGE_VALIDATION)
    preflight = _read_json(PREFLIGHT)
    variant_metadata = _read_json(VARIANT_PACKAGE_METADATA)
    variant_preflight = _read_json(VARIANT_PREFLIGHT)
    variant_import = _read_json(VARIANT_IMPORT_MANIFEST)
    upload_set = _read_json(UPLOAD_SET_MANIFEST)
    upload_set_verify = _run_upload_set_verifier()
    nested_entrypoints = _upload_set_nested_entrypoint_check(upload_set_verify)
    stage_rows = _stage_rows(stage1_gate, variant_import, upload_set_verify)
    evidence_rows = _evidence_rows(
        stage1_manifest,
        stage1_gate,
        package_validation,
        preflight,
        variant_metadata,
        variant_preflight,
        variant_import,
        upload_set,
        upload_set_verify,
    )

    tables_dir = report_root / "tables"
    figures_dir = report_root / "figures"
    stage_figure = figures_dir / "stage_gate_status.png"
    _write_csv(tables_dir / "stage_status.csv", stage_rows)
    _write_csv(tables_dir / "evidence_matrix.csv", evidence_rows)
    _write_stage_figure(stage_figure, stage_rows)
    figure_rel_path = "figures/stage_gate_status.png"
    _write_markdown(report_root / "goal_run_summary.md", stage_rows, evidence_rows, generated_at, figure_rel_path)
    _write_decision_log(report_root / "decision_log.md", generated_at)
    _write_latex(report_root / "goal_run_summary.tex", stage_rows, evidence_rows, generated_at, figure_rel_path)

    variant_state = _variant_validation_state(variant_import)
    if variant_state == "gpu_variant_passed":
        goal_status = "stage2_3_4_gpu_variant_passed"
    elif variant_state == "gpu_variant_import_incomplete":
        goal_status = "stage2_3_4_gpu_variant_import_incomplete"
    elif variant_state == "gpu_variant_failed":
        goal_status = "stage2_3_4_gpu_variant_failed"
    else:
        goal_status = "stage2_3_4_gpu_variant_pending"

    summary = {
        "generated_at": generated_at,
        "status": goal_status,
        "stage1_gate_status": stage1_gate.get("status", "missing"),
        "stage1_import_status": stage1_manifest.get("gate_status", "missing"),
        "package_validation_status": package_validation.get("status", "missing"),
        "preflight_status": preflight.get("status", "missing"),
        "variant_package_status": _exists_status(VARIANT_PACKAGE_ARCHIVE),
        "variant_preflight_status": variant_preflight.get("status", "missing"),
        "variant_return_gate_status": variant_import.get("gate_status", "pending"),
        "variant_return_import_status": variant_import.get("import_status", "pending"),
        "variant_performance_report_status": variant_import.get("performance_report_status", "pending"),
        "upload_set_status": _exists_status(UPLOAD_SET_MANIFEST),
        "upload_set_verify_status": upload_set_verify.get("status", "missing"),
        "upload_set_nested_verify_status": nested_entrypoints["status"],
        "outputs": {
            "goal_run_summary_md": _display_path(report_root / "goal_run_summary.md"),
            "goal_run_summary_tex": _display_path(report_root / "goal_run_summary.tex"),
            "goal_run_summary_pdf": _display_path(report_root / "goal_run_summary.pdf"),
            "decision_log": _display_path(report_root / "decision_log.md"),
            "stage_status_csv": _display_path(tables_dir / "stage_status.csv"),
            "evidence_matrix_csv": _display_path(tables_dir / "evidence_matrix.csv"),
            "stage_gate_status_png": _display_path(stage_figure),
        },
    }
    (report_root / "goal_run_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    summary = build_goal_run_audit()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
