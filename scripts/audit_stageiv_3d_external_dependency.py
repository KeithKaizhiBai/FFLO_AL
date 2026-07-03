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
PACKAGE_ARCHIVE = ROOT / "hpc_packages" / f"{PACKAGE_NAME}.tar.gz"
PACKAGE_ROOT = ROOT / "hpc_packages" / PACKAGE_NAME
OUT_DIR = ROOT / "reports" / "stageiv_3d_external_dependency_audit"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


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


def add_row(
    rows: list[dict[str, Any]],
    *,
    requirement_id: str,
    requirement: str,
    evidence_type: str,
    status: str,
    authoritative_evidence: str,
    missing_evidence: str,
    next_action: str,
) -> None:
    rows.append(
        {
            "requirement_id": requirement_id,
            "requirement": requirement,
            "evidence_type": evidence_type,
            "status": status,
            "authoritative_evidence": authoritative_evidence,
            "missing_evidence": missing_evidence,
            "next_action": next_action,
        }
    )


def build_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    readiness = read_json(ROOT / "reports" / "stageiv_3d_readiness_audit" / "stageiv_readiness_decision.json")
    handoff = read_json(ROOT / "reports" / "stageiv_3d_hpc_handoff" / "stageiv_3d_hpc_handoff.json")
    goal = read_json(ROOT / "reports" / "stageiv_3d_goal_status" / "stageiv_3d_goal_status.json")
    traceability = read_json(ROOT / "reports" / "stageiv_3d_traceability_audit" / "stageiv_3d_traceability_decision.json")
    run_dir = ROOT / OUTPUT_ROOT / "active_runs" / RUN_ID
    final_dataset = run_dir / "dataset_iter025.npz"
    postrun_decision = ROOT / OUTPUT_ROOT / "reports" / "stageiv_3d_postrun_bundle" / "stageiv_3d_postrun_bundle_decision.json"

    package_hash = sha256_file(PACKAGE_ARCHIVE) if PACKAGE_ARCHIVE.exists() else ""
    rows: list[dict[str, Any]] = []

    add_row(
        rows,
        requirement_id="local-001",
        requirement="Validated Stage IV HPC package exists.",
        evidence_type="local_package",
        status="pass" if PACKAGE_ARCHIVE.exists() and readiness.get("package_validation_status") == "pass" else "missing",
        authoritative_evidence=f"{rel(PACKAGE_ARCHIVE)} sha256={package_hash}",
        missing_evidence="",
        next_action="none" if PACKAGE_ARCHIVE.exists() else "rebuild Stage IV package",
    )
    add_row(
        rows,
        requirement_id="local-002",
        requirement="Package hash is synchronized across readiness, handoff, and goal-status decisions.",
        evidence_type="local_consistency",
        status="pass"
        if package_hash
        and readiness.get("package_sha256") == package_hash
        and handoff.get("package_sha256") == package_hash
        and goal.get("package_sha256") == package_hash
        else "missing",
        authoritative_evidence=(
            f"archive={package_hash}; readiness={readiness.get('package_sha256')}; "
            f"handoff={handoff.get('package_sha256')}; goal={goal.get('package_sha256')}"
        ),
        missing_evidence="",
        next_action="refresh package/readiness/handoff/goal-status if hashes diverge",
    )
    add_row(
        rows,
        requirement_id="local-003",
        requirement="Operational HPC command checklist exists inside the package.",
        evidence_type="local_handoff",
        status="pass" if (PACKAGE_ROOT / "HPC_COMMANDS_STAGEIV_3D.md").exists() else "missing",
        authoritative_evidence=rel(PACKAGE_ROOT / "HPC_COMMANDS_STAGEIV_3D.md"),
        missing_evidence="",
        next_action="rebuild package if missing",
    )
    add_row(
        rows,
        requirement_id="local-004",
        requirement="Operational handoff report is ready for submit/recovery/collect/post-run sequence.",
        evidence_type="local_handoff",
        status="pass" if handoff.get("handoff_status") == "ready_for_hpc_submit" else "missing",
        authoritative_evidence=f"handoff_status={handoff.get('handoff_status')}",
        missing_evidence="",
        next_action="refresh handoff report if missing",
    )
    add_row(
        rows,
        requirement_id="external-001",
        requirement="Production Stage IV run directory returned locally.",
        evidence_type="external_hpc_return",
        status="pending_external" if not run_dir.exists() else "pass",
        authoritative_evidence=rel(run_dir) if run_dir.exists() else "",
        missing_evidence=rel(run_dir),
        next_action="upload/submit package or download/extract returned run directory",
    )
    add_row(
        rows,
        requirement_id="external-002",
        requirement="Final cumulative dataset_iter025 exists.",
        evidence_type="external_hpc_return",
        status="pending_external" if not final_dataset.exists() else "pass",
        authoritative_evidence=rel(final_dataset) if final_dataset.exists() else "",
        missing_evidence=rel(final_dataset),
        next_action="complete production full loop and collect returned results",
    )
    add_row(
        rows,
        requirement_id="external-003",
        requirement="Last-five cumulative datasets are available for final transition audits.",
        evidence_type="external_hpc_return",
        status="pending_external",
        authoritative_evidence="",
        missing_evidence=f"{OUTPUT_ROOT}/active_runs/{RUN_ID}/dataset_iter021.npz ... dataset_iter025.npz",
        next_action="requires returned production cumulative history",
    )
    add_row(
        rows,
        requirement_id="external-004",
        requirement="Joint 3D thermodynamic/topology convergence audit passes.",
        evidence_type="external_postrun_audit",
        status="pending_external" if not traceability.get("production_run_returned") else "needs_review",
        authoritative_evidence="",
        missing_evidence=f"{OUTPUT_ROOT}/reports/stageiv_3d_convergence_audit/stageiv_3d_convergence_decision.json",
        next_action="run post-run convergence audit after data return",
    )
    add_row(
        rows,
        requirement_id="external-005",
        requirement="Hidden fixed-mu Stage III slice validation passes.",
        evidence_type="external_postrun_audit",
        status="pending_external",
        authoritative_evidence="",
        missing_evidence=f"{OUTPUT_ROOT}/reports/stageiv_3d_hidden_slice_audit/stageiv_3d_hidden_slice_decision.json",
        next_action="run hidden-slice validation after data return with REFERENCE_DATASET",
    )
    add_row(
        rows,
        requirement_id="external-006",
        requirement="Post-run bundle decision exists and passes.",
        evidence_type="external_postrun_audit",
        status="pending_external" if not postrun_decision.exists() else "pass",
        authoritative_evidence=rel(postrun_decision) if postrun_decision.exists() else "",
        missing_evidence=rel(postrun_decision),
        next_action="run build_stageiv_3d_all_postrun_reports.sh after data return",
    )
    add_row(
        rows,
        requirement_id="external-007",
        requirement="Publication-grade Stage IV 3D report and figures completed from returned production data.",
        evidence_type="external_final_report",
        status="pending_external",
        authoritative_evidence="",
        missing_evidence=f"{OUTPUT_ROOT}/reports/stageiv_3d_final_report/",
        next_action="build final Stage IV report after post-run bundle passes",
    )

    counts = Counter(str(row["status"]) for row in rows)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": RUN_ID,
        "output_root": OUTPUT_ROOT,
        "package_sha256": package_hash,
        "package_validation_status": readiness.get("package_validation_status"),
        "handoff_status": handoff.get("handoff_status"),
        "goal_status": goal.get("goal_status"),
        "status_counts": dict(counts),
        "remaining_external_dependency_count": counts.get("pending_external", 0),
        "production_run_returned": bool(run_dir.exists()),
        "final_dataset_exists": bool(final_dataset.exists()),
        "postrun_bundle_exists": bool(postrun_decision.exists()),
        "can_complete_stageiv_without_external_hpc_return": False,
        "blocking_condition": "production_hpc_outputs_not_returned",
        "blocked_next_action": "submit_or_return_stageiv_hpc_production_results",
    }
    return rows, summary


def write_markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# Stage IV 3D External Dependency Audit",
        "",
        "This report-only audit separates local pre-HPC readiness from the evidence that can only be supplied by the external Stage IV production run. It does not submit jobs, run exact calculations, merge shards, append datasets, or modify historical results.",
        "",
        "## Executive Summary",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- package_validation_status: `{summary['package_validation_status']}`",
        f"- handoff_status: `{summary['handoff_status']}`",
        f"- goal_status: `{summary['goal_status']}`",
        f"- production_run_returned: `{summary['production_run_returned']}`",
        f"- final_dataset_exists: `{summary['final_dataset_exists']}`",
        f"- postrun_bundle_exists: `{summary['postrun_bundle_exists']}`",
        f"- remaining_external_dependency_count: `{summary['remaining_external_dependency_count']}`",
        f"- blocking_condition: `{summary['blocking_condition']}`",
        "",
        "Conclusion: local package/readiness/handoff work is complete enough for upload and submission. The Stage IV scientific objective cannot be completed or verified locally until the production HPC outputs are returned.",
        "",
        "## Dependency Table",
        "",
        "| id | requirement | evidence_type | status | authoritative_evidence | missing_evidence | next_action |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                str(row[key]).replace("|", "/")
                for key in [
                    "requirement_id",
                    "requirement",
                    "evidence_type",
                    "status",
                    "authoritative_evidence",
                    "missing_evidence",
                    "next_action",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Do-Not-Claim",
            "",
            "- Do not claim Stage IV convergence from package validation, preflight success, or submit-readiness checks.",
            "- Do not claim missing surfaces or boundaries have zero shift.",
            "- Do not merge Stage III, Phase-II, or topology-derived datasets into this cold-start Stage IV training run.",
            "- Do not modify thermodynamic criteria, topology formulas, acquisition rules, StopController thresholds, or exact-oracle tolerances to satisfy this audit.",
            "",
        ]
    )
    write_text(OUT_DIR / "stageiv_3d_external_dependency_audit.md", "\n".join(lines) + "\n")


def tex_escape(value: Any) -> str:
    text = str(value)
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
    return "".join(repl.get(ch, ch) for ch in text)


def write_tex(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    counts = Counter(str(row["status"]) for row in rows)
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{longtable}",
        r"\usepackage{hyperref}",
        r"\title{Stage IV 3D External Dependency Audit}",
        r"\date{2026-06-23}",
        r"\begin{document}",
        r"\maketitle",
        r"\section*{Executive Summary}",
        r"\begin{itemize}",
        rf"\item run id: \texttt{{{tex_escape(summary['run_id'])}}}",
        rf"\item package validation: \texttt{{{tex_escape(summary['package_validation_status'])}}}",
        rf"\item handoff status: \texttt{{{tex_escape(summary['handoff_status'])}}}",
        rf"\item production run returned: \texttt{{{tex_escape(summary['production_run_returned'])}}}",
        rf"\item final dataset exists: \texttt{{{tex_escape(summary['final_dataset_exists'])}}}",
        rf"\item post-run bundle exists: \texttt{{{tex_escape(summary['postrun_bundle_exists'])}}}",
        rf"\item remaining external dependency count: \texttt{{{tex_escape(summary['remaining_external_dependency_count'])}}}",
        r"\end{itemize}",
        r"Local package, readiness, and handoff evidence is synchronized. The remaining Stage IV requirements require external HPC production outputs.",
        r"\section*{Status Counts}",
        r"\begin{tabular}{lr}",
        r"status & count \\ \hline",
    ]
    for key, value in sorted(counts.items()):
        lines.append(rf"{tex_escape(key)} & {int(value)} \\")
    lines.extend(
        [
            r"\end{tabular}",
            r"\section*{Dependency Evidence}",
            r"\begin{longtable}{p{0.12\linewidth}p{0.38\linewidth}p{0.15\linewidth}p{0.12\linewidth}p{0.17\linewidth}}",
            r"id & requirement & evidence type & status & missing evidence \\ \hline",
        ]
    )
    for row in rows:
        lines.append(
            rf"{tex_escape(row['requirement_id'])} & {tex_escape(row['requirement'])} & {tex_escape(row['evidence_type'])} & {tex_escape(row['status'])} & {tex_escape(row['missing_evidence'])} \\"
        )
    lines.extend([r"\end{longtable}", r"\end{document}", ""])
    write_text(OUT_DIR / "stageiv_3d_external_dependency_audit.tex", "\n".join(lines))


def write_figure(rows: list[dict[str, Any]]) -> None:
    figures_dir = OUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter(str(row["status"]) for row in rows)
    try:
        import matplotlib.pyplot as plt

        labels = sorted(counts)
        values = [counts[label] for label in labels]
        colors = ["#2ca25f" if label == "pass" else "#f0ad4e" for label in labels]
        fig, ax = plt.subplots(figsize=(6, 3.3))
        ax.bar(labels, values, color=colors)
        ax.set_ylabel("Requirement count")
        ax.set_title("Stage IV External Dependency State")
        for idx, value in enumerate(values):
            ax.text(idx, value + 0.04, str(value), ha="center")
        fig.tight_layout()
        fig.savefig(figures_dir / "stageiv_external_dependency_status.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        write_text(figures_dir / "stageiv_external_dependency_status.txt", f"figure generation failed: {exc}\n")


def main() -> None:
    rows, summary = build_rows()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUT_DIR / "stageiv_3d_external_dependency_decision.json", summary)
    write_csv(
        OUT_DIR / "tables" / "stageiv_external_dependencies.csv",
        rows,
        [
            "requirement_id",
            "requirement",
            "evidence_type",
            "status",
            "authoritative_evidence",
            "missing_evidence",
            "next_action",
        ],
    )
    write_markdown(rows, summary)
    write_tex(rows, summary)
    write_figure(rows)
    write_text(
        OUT_DIR / "decision_log.md",
        "\n".join(
            [
                "# Stage IV 3D External Dependency Decision Log",
                "",
                "Decision:",
                "",
                "```text",
                f"external_dependency_status = {'external_hpc_required' if summary['remaining_external_dependency_count'] else 'none'}",
                f"blocking_condition = {summary['blocking_condition']}",
                f"next_action = {summary['blocked_next_action']}",
                "```",
                "",
                "The local package, readiness, and handoff evidence is synchronized. Remaining scientific proof requires returned production HPC data.",
                "",
            ]
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
