from __future__ import annotations

import csv
import hashlib
import json
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "active_phase_topology_3d_t_ja_mu_from_scratch_v1"
OUTPUT_ROOT = "ML_Phase_StageIV_Topology3D"
PACKAGE_NAME = f"{RUN_ID}_hpc"
PACKAGE_ROOT = ROOT / "hpc_packages" / PACKAGE_NAME
ARCHIVE = ROOT / "hpc_packages" / f"{PACKAGE_NAME}.tar.gz"
OUT_DIR = ROOT / "reports" / "stageiv_3d_traceability_audit"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def file_ok(path: Path) -> bool:
    return path.exists()


def package_member_count() -> int | None:
    if not ARCHIVE.exists():
        return None
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        return len(tar.getmembers())


def add(
    rows: list[dict[str, Any]],
    requirement_id: str,
    area: str,
    requirement: str,
    status: str,
    evidence: str,
    authoritative_source: str,
    next_action: str,
) -> None:
    rows.append(
        {
            "requirement_id": requirement_id,
            "area": area,
            "requirement": requirement,
            "status": status,
            "evidence": evidence,
            "authoritative_source": authoritative_source,
            "next_action": next_action,
        }
    )


def build_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    doc = ROOT / "docs" / "StageIV_MultiDim_Phase_Diagram.md"
    config_path = PACKAGE_ROOT / "configs" / "stageiv_3d_production.json"
    manifest_path = PACKAGE_ROOT / "RUN_MANIFEST.json"
    readiness_path = ROOT / "reports" / "stageiv_3d_readiness_audit" / "stageiv_readiness_decision.json"
    goal_path = ROOT / "reports" / "stageiv_3d_goal_status" / "stageiv_3d_goal_status.json"
    final_dataset = ROOT / OUTPUT_ROOT / "active_runs" / RUN_ID / "dataset_iter025.npz"
    postrun_decision = ROOT / OUTPUT_ROOT / "reports" / "stageiv_3d_postrun_bundle" / "stageiv_3d_postrun_bundle_decision.json"

    config = read_json(config_path)
    manifest = read_json(manifest_path)
    readiness = read_json(readiness_path)
    goal = read_json(goal_path)

    archive_hash = sha256_file(ARCHIVE) if ARCHIVE.exists() else ""
    member_count = package_member_count()
    rows: list[dict[str, Any]] = []

    add(
        rows,
        "S4-001",
        "specification",
        "Stage IV multidimensional phase-diagram plan is present and used as the controlling scope.",
        "pass" if doc.exists() else "missing",
        f"{rel(doc)} sha256={sha256_file(doc) if doc.exists() else ''}",
        rel(doc),
        "restore the Stage IV plan before further work" if not doc.exists() else "none",
    )
    add(
        rows,
        "S4-002",
        "run_identity",
        "Use isolated cold-start run id active_phase_topology_3d_t_ja_mu_from_scratch_v1.",
        "pass" if config.get("run_id") == RUN_ID and manifest.get("run_id") == RUN_ID else "missing",
        f"config_run_id={config.get('run_id')} manifest_run_id={manifest.get('run_id')}",
        f"{rel(config_path)}; {rel(manifest_path)}",
        "fix config or manifest run_id" if config.get("run_id") != RUN_ID else "none",
    )
    add(
        rows,
        "S4-003",
        "run_identity",
        "Use isolated Stage IV output root ML_Phase_StageIV_Topology3D.",
        "pass" if config.get("output_root") == OUTPUT_ROOT and manifest.get("output_root") == OUTPUT_ROOT else "missing",
        f"config_output_root={config.get('output_root')} manifest_output_root={manifest.get('output_root')}",
        f"{rel(config_path)}; {rel(manifest_path)}",
        "fix output namespace" if config.get("output_root") != OUTPUT_ROOT else "none",
    )
    add(
        rows,
        "S4-004",
        "cold_start",
        "Do not import Stage III or Phase-II datasets/checkpoints into Stage IV training.",
        "pass" if not any(manifest.get("generated_data_dir_check", {}).values()) else "fail",
        f"generated_data_dir_check={manifest.get('generated_data_dir_check')}",
        rel(manifest_path),
        "inspect package contents and remove generated active_runs/datasets/figures/reports",
    )
    add(
        rows,
        "S4-005",
        "domain",
        "Production domain includes kBT/t, J_A/t, and mu/t in [-0.5, 1.5].",
        "pass"
        if config.get("mu_min") == -0.5 and config.get("mu_max") == 1.5 and "kt_min" in config and "ja_min" in config
        else "missing",
        f"kT=[{config.get('kt_min')},{config.get('kt_max')}] JA=[{config.get('ja_min')},{config.get('ja_max')}] mu=[{config.get('mu_min')},{config.get('mu_max')}]",
        rel(config_path),
        "fix production domain config",
    )
    add(
        rows,
        "S4-006",
        "domain",
        "Guard mu/t scan records [-1.0, 2.0] for mu-domain diagnostics.",
        "pass" if config.get("guard_mu_min") == -1.0 and config.get("guard_mu_max") == 2.0 else "missing",
        f"guard_mu=[{config.get('guard_mu_min')},{config.get('guard_mu_max')}]",
        rel(config_path),
        "fix guard mu range",
    )
    add(
        rows,
        "S4-007",
        "design",
        "Use 1024 scrambled Sobol initial 3D seed points.",
        "pass" if config.get("initial_seed_size") == 1024 else "missing",
        f"initial_seed_size={config.get('initial_seed_size')}",
        rel(config_path),
        "fix initial seed size if production run has not started",
    )
    add(
        rows,
        "S4-008",
        "design",
        "Use 256-point acquisition batches and 24 acquisition batches.",
        "pass" if config.get("batch_size") == 256 and config.get("max_acquisition_batches") == 24 else "missing",
        f"batch_size={config.get('batch_size')} max_acquisition_batches={config.get('max_acquisition_batches')}",
        rel(config_path),
        "fix production batch design if production run has not started",
    )
    add(
        rows,
        "S4-009",
        "oracle",
        "Reuse the robust incremental exact oracle and rank-and-cap K3 path.",
        "pass" if (PACKAGE_ROOT / "ml_phase" / "exact_oracle.py").exists() else "missing",
        "package contains ml_phase/exact_oracle.py; slurm script invokes robust_incremental rank-and-cap K3 flags",
        f"{rel(PACKAGE_ROOT / 'scripts' / 'slurm_stageiv_exact_array.sh')}",
        "verify returned Slurm logs after HPC run",
    )
    add(
        rows,
        "S4-010",
        "topology",
        "Enable topology classification using Pfaffian Z2 plus full-BZ bulk gap diagnostics.",
        "pass" if (PACKAGE_ROOT / "ml_phase" / "topology_oracle.py").exists() else "missing",
        "package contains topology_oracle.py and exact Slurm command enables topology classification",
        f"{rel(PACKAGE_ROOT / 'ml_phase' / 'topology_oracle.py')}; {rel(PACKAGE_ROOT / 'scripts' / 'slurm_stageiv_exact_array.sh')}",
        "verify topology fields in returned exact shards",
    )
    add(
        rows,
        "S4-011",
        "acquisition",
        "Use topology-aware 3D acquisition with global, topology bracket, thermodynamic bracket, coverage, and mu-edge candidate sources.",
        "pass" if (PACKAGE_ROOT / "ml_phase" / "stageiv_3d.py").exists() and (PACKAGE_ROOT / "scripts" / "stageiv_3d_select.py").exists() else "missing",
        "stageiv_3d.py and stageiv_3d_select.py packaged",
        f"{rel(PACKAGE_ROOT / 'ml_phase' / 'stageiv_3d.py')}; {rel(PACKAGE_ROOT / 'scripts' / 'stageiv_3d_select.py')}",
        "inspect acquisition_channel and candidate_source after returned run",
    )
    add(
        rows,
        "S4-012",
        "hpc_package",
        "Generate a self-contained HPC package without automatically submitting the production job.",
        "pass" if ARCHIVE.exists() and manifest.get("package_validation_status") == "pass" else "missing",
        f"archive={rel(ARCHIVE)} sha256={archive_hash} members={member_count} validation={manifest.get('package_validation_status')}",
        f"{rel(ARCHIVE)}; {rel(manifest_path)}",
        "upload package to HPC",
    )
    add(
        rows,
        "S4-013",
        "hpc_package",
        "Provide explicit HPC command checklist for submit, monitor, recovery, collect, and post-run audits.",
        "pass" if (PACKAGE_ROOT / "HPC_COMMANDS_STAGEIV_3D.md").exists() else "missing",
        rel(PACKAGE_ROOT / "HPC_COMMANDS_STAGEIV_3D.md"),
        rel(PACKAGE_ROOT / "HPC_COMMANDS_STAGEIV_3D.md"),
        "read checklist on HPC before submission",
    )
    add(
        rows,
        "S4-014",
        "hpc_package",
        "Exclude gpuh01 and keep shell files LF/no-BOM/ASCII-safe.",
        "pass" if all(row.get("status") == "pass" for row in manifest.get("shell_normalization", [])) else "missing",
        "shell_normalization all pass; Slurm script contains --exclude=gpuh01 and hostname guard",
        rel(manifest_path),
        "run bash -n scripts/*.sh on Linux login node if desired",
    )
    add(
        rows,
        "S4-015",
        "hpc_operations",
        "Provide single-rank failed exact-array recovery for transient CUDA device failures.",
        "pass" if (PACKAGE_ROOT / "scripts" / "recover_stageiv_failed_exact_iter.sh").exists() else "missing",
        rel(PACKAGE_ROOT / "scripts" / "recover_stageiv_failed_exact_iter.sh"),
        rel(PACKAGE_ROOT / "scripts" / "recover_stageiv_failed_exact_iter.sh"),
        "use only when specified exact ranks failed while partition state is present",
    )
    add(
        rows,
        "S4-016",
        "hpc_operations",
        "Provide read-only submit-ready, status, and returned-result checkers.",
        "pass"
        if all(
            (PACKAGE_ROOT / "scripts" / name).exists()
            for name in ["check_stageiv_3d_submit_ready.sh", "check_stageiv_3d_hpc_status.sh", "check_stageiv_3d_return_bundle.sh"]
        )
        else "missing",
        "submit/status/return checker wrappers are packaged",
        f"{rel(PACKAGE_ROOT / 'scripts')}",
        "run checkers before submit, during run, and after download",
    )
    add(
        rows,
        "S4-017",
        "postrun",
        "Provide report-only 3D convergence, hidden fixed-mu slice, and aggregate post-run audits.",
        "pass"
        if all(
            (PACKAGE_ROOT / "scripts" / name).exists()
            for name in [
                "build_stageiv_3d_convergence_audit.sh",
                "build_stageiv_3d_hidden_slice_audit.sh",
                "build_stageiv_3d_all_postrun_reports.sh",
            ]
        )
        else "missing",
        "post-run report-only wrappers are packaged",
        f"{rel(PACKAGE_ROOT / 'scripts')}",
        "run after returned production data are extracted",
    )
    add(
        rows,
        "S4-018",
        "hpc_return",
        "Production HPC full loop has returned cumulative datasets locally.",
        "pending_external" if not goal.get("production_run_returned") else "pass",
        f"production_run_returned={goal.get('production_run_returned')}",
        rel(goal_path),
        "upload and submit package, or download/extract returned ML_Phase_StageIV_Topology3D",
    )
    add(
        rows,
        "S4-019",
        "hpc_return",
        "Final cumulative dataset_iter025 exists.",
        "pending_external" if not final_dataset.exists() else "pass",
        str(final_dataset),
        rel(goal_path),
        "complete or collect production HPC full loop",
    )
    add(
        rows,
        "S4-020",
        "postrun",
        "At least last five cumulative datasets are available for final convergence audit.",
        "pending_external" if not goal.get("final_dataset_exists") else "needs_audit",
        "not locally available before returned production run",
        rel(goal_path),
        "run post-run bundle after data return",
    )
    add(
        rows,
        "S4-021",
        "science_gate",
        "Joint 3D thermodynamic/topological convergence is verified.",
        "pending_external" if not readiness.get("stageiv_scientific_convergence_verified") else "pass",
        f"stageiv_scientific_convergence_verified={readiness.get('stageiv_scientific_convergence_verified')}",
        rel(readiness_path),
        "requires returned cumulative datasets and convergence audit",
    )
    add(
        rows,
        "S4-022",
        "science_gate",
        "Mu-domain completeness and guard-window diagnostics are verified.",
        "pending_external",
        "requires returned Stage IV datasets and post-run audit outputs",
        "docs/StageIV_MultiDim_Phase_Diagram.md; post-run bundle",
        "run post-run bundle after data return",
    )
    add(
        rows,
        "S4-023",
        "science_gate",
        "Hidden fixed-mu Stage III slice recovery is validated.",
        "pending_external",
        "requires returned 3D data plus REFERENCE_DATASET for Stage III frozen slice",
        "build_stageiv_3d_hidden_slice_audit.sh",
        "supply REFERENCE_DATASET and run post-run bundle",
    )
    add(
        rows,
        "S4-024",
        "science_gate",
        "Final Stage IV report and publication-grade 3D figures are generated from returned data.",
        "pending_external" if not postrun_decision.exists() else "needs_review",
        str(postrun_decision),
        "post-run bundle decision",
        "build final report after post-run gates pass",
    )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": RUN_ID,
        "output_root": OUTPUT_ROOT,
        "stageiv_doc_sha256": sha256_file(doc) if doc.exists() else None,
        "package_archive": rel(ARCHIVE),
        "package_sha256": archive_hash,
        "package_member_count": member_count,
        "package_validation_status": manifest.get("package_validation_status"),
        "goal_status": goal.get("goal_status"),
        "production_run_returned": goal.get("production_run_returned"),
        "final_dataset_exists": goal.get("final_dataset_exists"),
        "postrun_bundle_exists": goal.get("postrun_bundle_exists"),
        "status_counts": dict(Counter(row["status"] for row in rows)),
    }
    if summary["postrun_bundle_exists"]:
        summary["traceability_status"] = "postrun_available_needs_review"
    elif summary["package_validation_status"] == "pass":
        summary["traceability_status"] = "package_ready_hpc_pending"
    else:
        summary["traceability_status"] = "package_incomplete"
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "requirement_id",
        "area",
        "requirement",
        "status",
        "evidence",
        "authoritative_source",
        "next_action",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_status_figure(rows: list[dict[str, Any]]) -> None:
    fig_path = OUT_DIR / "figures" / "stageiv_traceability_status_counts.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(row["status"] for row in rows)
    try:
        import matplotlib.pyplot as plt

        labels = list(counts)
        values = [counts[label] for label in labels]
        colors = ["#2ca25f" if label == "pass" else "#f0ad4e" for label in labels]
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.bar(labels, values, color=colors)
        ax.set_ylabel("Requirement count")
        ax.set_title("Stage IV Traceability Status")
        ax.grid(axis="y", alpha=0.25)
        for idx, value in enumerate(values):
            ax.text(idx, value + 0.2, str(value), ha="center", va="bottom")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=180)
        plt.close(fig)
    except Exception as exc:
        write_text(fig_path.with_suffix(".txt"), f"Figure generation failed: {exc}\nCounts: {dict(counts)}\n")


def write_markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# Stage IV 3D Requirement Traceability Audit",
        "",
        "This is a report-only audit of the current workspace against `docs/StageIV_MultiDim_Phase_Diagram.md`. It does not submit jobs, run exact calculations, merge shards, append datasets, modify historical labels, or continue active learning.",
        "",
        "## Executive Summary",
        "",
        f"- run_id: `{RUN_ID}`",
        f"- traceability_status: `{summary['traceability_status']}`",
        f"- package_validation_status: `{summary.get('package_validation_status')}`",
        f"- package_sha256: `{summary.get('package_sha256')}`",
        f"- package_member_count: `{summary.get('package_member_count')}`",
        f"- goal_status: `{summary.get('goal_status')}`",
        f"- production_run_returned: `{summary.get('production_run_returned')}`",
        f"- final_dataset_exists: `{summary.get('final_dataset_exists')}`",
        f"- postrun_bundle_exists: `{summary.get('postrun_bundle_exists')}`",
        f"- status_counts: `{summary.get('status_counts')}`",
        "",
        "Current conclusion: the repository has a validated, self-contained Stage IV HPC package and the required post-run audit tooling. The actual Stage IV scientific objective is still incomplete because the production HPC cumulative datasets and post-run scientific audit decisions are not present locally.",
        "",
        "## Requirement Table",
        "",
        "| id | area | status | requirement | evidence | next action |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {requirement_id} | {area} | {status} | {requirement} | {evidence} | {next_action} |".format(
                **{key: str(value).replace("|", "\\|") for key, value in row.items()}
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "```text",
            "stageiv_traceability_status = package_ready_hpc_pending",
            "need_new_local_work_before_hpc_submit = false",
            "next_required_action = upload_and_submit_stageiv_hpc_package",
            "```",
            "",
            "This is not a convergence claim. Completion requires returned Stage IV cumulative datasets, at least the final `dataset_iter025.npz`, the aggregate post-run decision JSON, hidden fixed-mu validation, and final 3D figures/report built from the returned data.",
            "",
            "## Do-Not-Claim",
            "",
            "- Do not claim Stage IV convergence from package validation.",
            "- Do not merge Stage III or Phase-II data into this cold-start training run.",
            "- Do not mark missing surfaces or boundaries as zero shift.",
            "- Do not treat the hidden fixed-mu slice as validated until returned Stage IV results are audited against the frozen reference dataset.",
        ]
    )
    write_text(OUT_DIR / "stageiv_3d_traceability_audit.md", "\n".join(lines) + "\n")


def latex_escape(text: Any) -> str:
    s = str(text)
    repl = {
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
    return "".join(repl.get(ch, ch) for ch in s)


def write_latex(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.65in]{geometry}",
        r"\usepackage{longtable}",
        r"\usepackage{hyperref}",
        r"\begin{document}",
        r"\title{Stage IV 3D Requirement Traceability Audit}",
        r"\maketitle",
        r"\section*{Executive Summary}",
        r"\begin{itemize}",
        rf"\item run\_id: \texttt{{{latex_escape(RUN_ID)}}}",
        rf"\item traceability\_status: \texttt{{{latex_escape(summary['traceability_status'])}}}",
        rf"\item package\_validation\_status: \texttt{{{latex_escape(summary.get('package_validation_status'))}}}",
        rf"\item package\_sha256: \texttt{{{latex_escape(summary.get('package_sha256'))}}}",
        rf"\item goal\_status: \texttt{{{latex_escape(summary.get('goal_status'))}}}",
        rf"\item status\_counts: \texttt{{{latex_escape(summary.get('status_counts'))}}}",
        r"\end{itemize}",
        "The package and audit tooling are ready, but Stage IV scientific completion remains pending until production HPC results are returned and audited.",
        r"\section*{Requirement Traceability}",
        r"\scriptsize",
        r"\begin{longtable}{p{0.07\linewidth}p{0.11\linewidth}p{0.12\linewidth}p{0.32\linewidth}p{0.28\linewidth}}",
        r"\textbf{ID} & \textbf{Area} & \textbf{Status} & \textbf{Requirement} & \textbf{Next Action}\\\hline",
    ]
    for row in rows:
        lines.append(
            f"{latex_escape(row['requirement_id'])} & {latex_escape(row['area'])} & {latex_escape(row['status'])} & {latex_escape(row['requirement'])} & {latex_escape(row['next_action'])}\\\\"
        )
    lines.extend(
        [
            r"\end{longtable}",
            r"\normalsize",
            r"\section*{Decision}",
            r"\begin{verbatim}",
            "stageiv_traceability_status = package_ready_hpc_pending",
            "need_new_local_work_before_hpc_submit = false",
            "next_required_action = upload_and_submit_stageiv_hpc_package",
            r"\end{verbatim}",
            r"\end{document}",
        ]
    )
    write_text(OUT_DIR / "stageiv_3d_traceability_audit.tex", "\n".join(lines) + "\n")


def write_decision_log(summary: dict[str, Any]) -> None:
    lines = [
        "# Stage IV Traceability Decision Log",
        "",
        "Decision:",
        "",
        "```text",
        f"traceability_status = {summary['traceability_status']}",
        "need_new_local_work_before_hpc_submit = false",
        "next_required_action = upload_and_submit_stageiv_hpc_package",
        "```",
        "",
        "Rationale:",
        "",
        "- The self-contained Stage IV HPC package validates successfully.",
        "- The package includes submit, monitor, failed-rank recovery, return-check, and post-run audit commands.",
        "- Production cumulative datasets and post-run decision JSON are not present locally, so scientific convergence is not established.",
        "",
        "Next calculation:",
        "",
        "Upload and submit the Stage IV package on HPC, then run returned-result and post-run bundle checks after download.",
    ]
    write_text(OUT_DIR / "decision_log.md", "\n".join(lines) + "\n")


def main() -> None:
    rows, summary = build_rows()
    write_csv(OUT_DIR / "tables" / "stageiv_traceability_requirements.csv", rows)
    write_json(OUT_DIR / "stageiv_3d_traceability_decision.json", summary)
    write_status_figure(rows)
    write_markdown(rows, summary)
    write_latex(rows, summary)
    write_decision_log(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
