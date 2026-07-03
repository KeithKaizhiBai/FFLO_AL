from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "active_phase_topology_3d_t_ja_mu_from_scratch_v1"
OUTPUT_ROOT = "ML_Phase_StageIV_Topology3D"
PACKAGE_NAME = f"{RUN_ID}_hpc"
ARCHIVE = ROOT / "hpc_packages" / f"{PACKAGE_NAME}.tar.gz"
PACKAGE_ROOT = ROOT / "hpc_packages" / PACKAGE_NAME
READINESS_DIR = ROOT / "reports" / "stageiv_3d_readiness_audit"
HANDOFF_DIR = ROOT / "reports" / "stageiv_3d_hpc_handoff"
OUT_DIR = ROOT / "reports" / "stageiv_3d_goal_status"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_status(path: Path) -> tuple[str, str]:
    if path.exists():
        return "pass", str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
    return "missing", str(path)


def add_row(
    rows: list[dict[str, Any]],
    *,
    area: str,
    requirement: str,
    status: str,
    evidence: str,
    next_action: str,
) -> None:
    rows.append(
        {
            "area": area,
            "requirement": requirement,
            "status": status,
            "evidence": evidence,
            "next_action": next_action,
        }
    )


def build_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    readiness = read_json(READINESS_DIR / "stageiv_readiness_decision.json")
    handoff = read_json(HANDOFF_DIR / "stageiv_3d_hpc_handoff.json")
    manifest = read_json(PACKAGE_ROOT / "RUN_MANIFEST.json")
    stageiv_doc = ROOT / "docs" / "StageIV_MultiDim_Phase_Diagram.md"
    run_dir = ROOT / OUTPUT_ROOT / "active_runs" / RUN_ID
    postrun_bundle = ROOT / OUTPUT_ROOT / "reports" / "stageiv_3d_postrun_bundle" / "stageiv_3d_postrun_bundle_decision.json"
    final_dataset = run_dir / "dataset_iter025.npz"
    archive_hash = sha256_file(ARCHIVE) if ARCHIVE.exists() else ""

    rows: list[dict[str, Any]] = []
    status, evidence = file_status(stageiv_doc)
    add_row(
        rows,
        area="specification",
        requirement="Stage IV multidimensional phase-diagram plan is present",
        status=status,
        evidence=evidence,
        next_action="none" if status == "pass" else "restore docs/StageIV_MultiDim_Phase_Diagram.md",
    )
    add_row(
        rows,
        area="package",
        requirement="Self-contained HPC archive exists",
        status="pass" if ARCHIVE.exists() else "missing",
        evidence=f"{ARCHIVE.relative_to(ROOT)} sha256={archive_hash}" if ARCHIVE.exists() else str(ARCHIVE),
        next_action="upload archive to HPC" if ARCHIVE.exists() else "rebuild package",
    )
    add_row(
        rows,
        area="package",
        requirement="Package manifest validation passes",
        status="pass" if manifest.get("package_validation_status") == "pass" else "missing",
        evidence=f"RUN_MANIFEST package_validation_status={manifest.get('package_validation_status')}",
        next_action="none" if manifest.get("package_validation_status") == "pass" else "inspect RUN_MANIFEST",
    )
    add_row(
        rows,
        area="package",
        requirement="Archive hash in readiness decision matches current archive",
        status="pass" if readiness.get("package_sha256") == archive_hash and archive_hash else "missing",
        evidence=f"readiness={readiness.get('package_sha256')} archive={archive_hash}",
        next_action="none" if readiness.get("package_sha256") == archive_hash and archive_hash else "update readiness decision",
    )
    for rel in [
        "README.md",
        "README_STAGEIV_3D_HPC.md",
        "ENVIRONMENT_STAGEIV_3D_HPC.md",
        "HPC_COMMANDS_STAGEIV_3D.md",
        "scripts/submit_stageiv_3d_full_loop.sh",
        "scripts/check_stageiv_3d_submit_ready.sh",
        "scripts/resume_stageiv_3d_full_loop.sh",
        "scripts/recover_stageiv_failed_exact_iter.sh",
        "scripts/collect_stageiv_3d_results.sh",
        "scripts/check_stageiv_3d_hpc_status.sh",
        "scripts/check_stageiv_3d_return_bundle.sh",
        "scripts/stageiv_3d_return_check.py",
        "scripts/build_stageiv_3d_all_postrun_reports.sh",
    ]:
        path = PACKAGE_ROOT / rel
        status, evidence = file_status(path)
        add_row(
            rows,
            area="package_entrypoint",
            requirement=f"Package contains {rel}",
            status=status,
            evidence=evidence,
            next_action="none" if status == "pass" else "rebuild package",
        )
    add_row(
        rows,
        area="readiness",
        requirement="Readiness status separates package readiness from scientific completion",
        status="pass" if readiness.get("stageiv_readiness_status") == "package_ready_hpc_pending" else "missing",
        evidence=f"stageiv_readiness_status={readiness.get('stageiv_readiness_status')}",
        next_action="none" if readiness.get("stageiv_readiness_status") == "package_ready_hpc_pending" else "refresh readiness audit",
    )
    add_row(
        rows,
        area="handoff",
        requirement="Operational HPC handoff report exists",
        status="pass" if (HANDOFF_DIR / "stageiv_3d_hpc_handoff.md").exists() else "missing",
        evidence=str((HANDOFF_DIR / "stageiv_3d_hpc_handoff.md").relative_to(ROOT))
        if (HANDOFF_DIR / "stageiv_3d_hpc_handoff.md").exists()
        else str(HANDOFF_DIR / "stageiv_3d_hpc_handoff.md"),
        next_action="none" if (HANDOFF_DIR / "stageiv_3d_hpc_handoff.md").exists() else "generate HPC handoff report",
    )
    add_row(
        rows,
        area="handoff",
        requirement="Handoff status is ready for HPC submission",
        status="pass" if handoff.get("handoff_status") == "ready_for_hpc_submit" else "missing",
        evidence=f"handoff_status={handoff.get('handoff_status')}",
        next_action="none" if handoff.get("handoff_status") == "ready_for_hpc_submit" else "refresh HPC handoff report",
    )
    add_row(
        rows,
        area="handoff",
        requirement="Handoff command table exists for upload, submit, recovery, collect, and post-run audit",
        status="pass" if (HANDOFF_DIR / "tables" / "stageiv_handoff_commands.csv").exists() else "missing",
        evidence=str((HANDOFF_DIR / "tables" / "stageiv_handoff_commands.csv").relative_to(ROOT))
        if (HANDOFF_DIR / "tables" / "stageiv_handoff_commands.csv").exists()
        else str(HANDOFF_DIR / "tables" / "stageiv_handoff_commands.csv"),
        next_action="none"
        if (HANDOFF_DIR / "tables" / "stageiv_handoff_commands.csv").exists()
        else "regenerate HPC handoff report",
    )
    add_row(
        rows,
        area="hpc_return",
        requirement="Production run directory exists locally",
        status="pass" if run_dir.exists() else "pending",
        evidence=str(run_dir.relative_to(ROOT) if run_dir.exists() else run_dir),
        next_action="download/extract returned Stage IV output" if not run_dir.exists() else "run postrun bundle",
    )
    add_row(
        rows,
        area="hpc_return",
        requirement="Final cumulative dataset_iter025 exists",
        status="pass" if final_dataset.exists() else "pending",
        evidence=str(final_dataset.relative_to(ROOT) if final_dataset.exists() else final_dataset),
        next_action="complete or collect production HPC full loop" if not final_dataset.exists() else "audit final dataset",
    )
    add_row(
        rows,
        area="postrun_audit",
        requirement="Post-run bundle decision exists for production run",
        status="pass" if postrun_bundle.exists() else "pending",
        evidence=str(postrun_bundle.relative_to(ROOT) if postrun_bundle.exists() else postrun_bundle),
        next_action="run build_stageiv_3d_all_postrun_reports.sh after data return" if not postrun_bundle.exists() else "inspect decision JSON",
    )
    add_row(
        rows,
        area="scientific_gate",
        requirement="3D thermodynamic/topological convergence verified on returned cumulative history",
        status="pending" if not readiness.get("stageiv_scientific_convergence_verified") else "pass",
        evidence=f"stageiv_scientific_convergence_verified={readiness.get('stageiv_scientific_convergence_verified')}",
        next_action="requires returned cumulative datasets and convergence audit",
    )
    add_row(
        rows,
        area="scientific_gate",
        requirement="Hidden fixed-mu Stage III slice validation completed",
        status="pending",
        evidence="No production hidden-slice audit decision found under ML_Phase_StageIV_Topology3D",
        next_action="provide REFERENCE_DATASET and run postrun bundle after return",
    )
    add_row(
        rows,
        area="scientific_gate",
        requirement="Publication-grade Stage IV 3D report and figures completed",
        status="pending",
        evidence="Production post-run bundle and returned-data report are not available locally",
        next_action="build final Stage IV report after postrun bundle passes",
    )

    summary = {
        "run_id": RUN_ID,
        "output_root": OUTPUT_ROOT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stageiv_doc": str(stageiv_doc),
        "stageiv_doc_sha256": sha256_file(stageiv_doc) if stageiv_doc.exists() else None,
        "package_archive": str(ARCHIVE),
        "package_sha256": archive_hash,
        "package_validation_status": manifest.get("package_validation_status"),
        "stageiv_readiness_status": readiness.get("stageiv_readiness_status"),
        "handoff_status": handoff.get("handoff_status"),
        "production_run_returned": bool(run_dir.exists()),
        "final_dataset_exists": bool(final_dataset.exists()),
        "postrun_bundle_exists": bool(postrun_bundle.exists()),
        "status_counts": dict(Counter(str(row["status"]) for row in rows)),
    }
    if summary["postrun_bundle_exists"]:
        bundle = read_json(postrun_bundle)
        summary["postrun_bundle_status"] = bundle.get("postrun_bundle_status")
        summary["postrun_decision_class"] = bundle.get("decision_class")
    else:
        summary["postrun_bundle_status"] = None
        summary["postrun_decision_class"] = None
    summary["goal_status"] = "package_ready_hpc_pending" if not summary["postrun_bundle_exists"] else "postrun_available_needs_review"
    return rows, summary


def write_markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# Stage IV 3D Goal Status Audit",
        "",
        "This audit records current evidence for the persistent Stage IV objective. It does not submit jobs, run exact calculations, merge shards, append datasets, or modify historical results.",
        "",
        "## Executive Summary",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- package_validation_status: `{summary['package_validation_status']}`",
        f"- package_sha256: `{summary['package_sha256']}`",
        f"- stageiv_readiness_status: `{summary['stageiv_readiness_status']}`",
        f"- handoff_status: `{summary['handoff_status']}`",
        f"- production_run_returned: `{summary['production_run_returned']}`",
        f"- final_dataset_exists: `{summary['final_dataset_exists']}`",
        f"- postrun_bundle_exists: `{summary['postrun_bundle_exists']}`",
        f"- goal_status: `{summary['goal_status']}`",
        "",
        "Current conclusion: the local package and report-only audit tooling are ready, but the full Stage IV objective is not complete because production HPC cumulative datasets and the post-run scientific audits are not present locally.",
        "",
        "## Requirement Status",
        "",
        "| area | requirement | status | evidence | next_action |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                str(row[key]).replace("|", "/")
                for key in ["area", "requirement", "status", "evidence", "next_action"]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Next Required Action",
            "",
            "Upload and submit the current Stage IV HPC archive, or if it has already run externally, download/extract `ML_Phase_StageIV_Topology3D` and run `scripts/build_stageiv_3d_all_postrun_reports.sh` with the Stage III frozen reference dataset supplied through `REFERENCE_DATASET`.",
            "",
        ]
    )
    write_text(OUT_DIR / "stageiv_3d_goal_status.md", "\n".join(lines))


def tex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def write_tex(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    counts = Counter(str(row["status"]) for row in rows)
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{longtable}",
        r"\usepackage{hyperref}",
        r"\title{Stage IV 3D Goal Status Audit}",
        r"\date{2026-06-23}",
        r"\begin{document}",
        r"\maketitle",
        r"\section*{Executive Summary}",
        r"\begin{itemize}",
        rf"\item run id: \texttt{{{tex_escape(summary['run_id'])}}}",
        rf"\item package validation: \texttt{{{tex_escape(summary['package_validation_status'])}}}",
        rf"\item readiness status: \texttt{{{tex_escape(summary['stageiv_readiness_status'])}}}",
        rf"\item handoff status: \texttt{{{tex_escape(summary['handoff_status'])}}}",
        rf"\item production run returned: \texttt{{{tex_escape(summary['production_run_returned'])}}}",
        rf"\item final dataset exists: \texttt{{{tex_escape(summary['final_dataset_exists'])}}}",
        rf"\item post-run bundle exists: \texttt{{{tex_escape(summary['postrun_bundle_exists'])}}}",
        rf"\item goal status: \texttt{{{tex_escape(summary['goal_status'])}}}",
        r"\end{itemize}",
        r"The package and report-only audit tooling are ready, but the full Stage IV objective is still pending returned production HPC data and post-run scientific audits.",
        r"\section*{Status Counts}",
        r"\begin{tabular}{lr}",
        r"status & count \\ \hline",
    ]
    for key, value in sorted(counts.items()):
        lines.append(rf"{tex_escape(key)} & {int(value)} \\")
    lines.extend(
        [
            r"\end{tabular}",
            r"\section*{Requirement Evidence}",
            r"\begin{longtable}{p{0.16\linewidth}p{0.34\linewidth}p{0.12\linewidth}p{0.28\linewidth}}",
            r"area & requirement & status & evidence \\ \hline",
        ]
    )
    for row in rows:
        lines.append(
            rf"{tex_escape(row['area'])} & {tex_escape(row['requirement'])} & {tex_escape(row['status'])} & {tex_escape(row['evidence'])} \\"
        )
    lines.extend([r"\end{longtable}", r"\end{document}", ""])
    write_text(OUT_DIR / "stageiv_3d_goal_status.tex", "\n".join(lines))


def write_figure(rows: list[dict[str, Any]]) -> None:
    figures_dir = OUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter(str(row["status"]) for row in rows)
    try:
        import matplotlib.pyplot as plt

        labels = sorted(counts)
        values = [counts[label] for label in labels]
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        colors = ["#4c78a8" if label == "pass" else "#f58518" if label == "pending" else "#e45756" for label in labels]
        ax.bar(labels, values, color=colors)
        ax.set_ylabel("Requirement count")
        ax.set_title("Stage IV Goal Status")
        for i, value in enumerate(values):
            ax.text(i, value + 0.05, str(value), ha="center", va="bottom")
        fig.tight_layout()
        fig.savefig(figures_dir / "stageiv_goal_status_counts.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        write_text(figures_dir / "stageiv_goal_status_counts.txt", f"figure generation failed: {exc}\n")


def main() -> None:
    rows, summary = build_rows()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUT_DIR / "stageiv_3d_goal_status.json", summary)
    write_csv(
        OUT_DIR / "tables" / "stageiv_goal_requirements.csv",
        rows,
        ["area", "requirement", "status", "evidence", "next_action"],
    )
    write_markdown(rows, summary)
    write_tex(rows, summary)
    write_figure(rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
