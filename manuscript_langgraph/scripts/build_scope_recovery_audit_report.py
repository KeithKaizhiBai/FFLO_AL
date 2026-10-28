from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = ROOT / "manuscript_langgraph"
OUT = ROOT / "output/pdf/manuscript_evidence_recovery_progress_audit_20260813"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORT_STEM = "manuscript_evidence_recovery_progress_audit_20260813"

RECOVERED_ARTIFACT_IDS = {"ART-008", "ART-011", "ART-012", "ART-015", "ART-016", "ART-017"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str] | None = None,
) -> None:
    if not rows and not fieldnames:
        raise ValueError(f"fieldnames are required for an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_tables() -> tuple[list[dict[str, str]], dict[str, object]]:
    manifest = read_csv(ML_ROOT / "audits/manuscript_artifact_manifest.csv")
    data_presence = read_csv(ML_ROOT / "audits/transferred_package_data_presence.csv")
    state = json.loads((ML_ROOT / "audits/current_workflow_state.json").read_text(encoding="utf-8"))
    present = sum(row["status"] == "present" for row in manifest)
    missing = sum(row["status"] == "missing" for row in manifest)

    status_rows = [
        {"metric": "manifest_rows", "value": len(manifest), "interpretation": "curated manuscript artifacts"},
        {"metric": "present_rows", "value": present, "interpretation": "locally verified and hashed"},
        {"metric": "missing_rows", "value": missing, "interpretation": "Stage II--IV frozen recovery targets"},
    ]
    missing_rows = [
        {
            "artifact_id": row["artifact_id"],
            "source_stage": row["source_stage"],
            "run_id": row["run_id"],
            "path": row["path"],
            "status": row["status"],
            "provenance_status": row["provenance_status"],
        }
        for row in manifest
        if row["status"] == "missing"
    ]
    recovered_rows = [
        {
            "artifact_id": row["artifact_id"],
            "source_stage": row["source_stage"],
            "run_id": row["run_id"],
            "path": row["path"],
            "sha256": row["sha256"],
            "status": row["status"],
        }
        for row in manifest
        if row["artifact_id"] in RECOVERED_ARTIFACT_IDS
    ]
    gate_rows = [
        {
            "gate_id": gate["gate_id"],
            "status": gate["status"],
            "finding": "; ".join(gate["findings"]),
        }
        for gate in state["gate_results"]
    ]
    write_csv(TABLES / "artifact_status_summary.csv", status_rows)
    write_csv(
        TABLES / "missing_recovery_targets.csv",
        missing_rows,
        ["artifact_id", "source_stage", "run_id", "path", "status", "provenance_status"],
    )
    write_csv(TABLES / "recovered_artifacts.csv", recovered_rows)
    write_csv(TABLES / "current_gate_results.csv", gate_rows)
    write_csv(TABLES / "transferred_package_data_presence.csv", data_presence)
    return manifest, state


def build_figure(manifest: list[dict[str, str]], state: dict[str, object]) -> Path:
    present = sum(row["status"] == "present" for row in manifest)
    missing = sum(row["status"] == "missing" for row in manifest)
    gates = state["gate_results"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    axes[0].bar(["Present", "Missing"], [present, missing], color=["#2b8c6b", "#c9504d"])
    axes[0].set_ylabel("Manifest rows")
    axes[0].set_title("Artifact/provenance audit")
    axes[0].set_ylim(0, max(present + 5, 10))
    for idx, value in enumerate([present, missing]):
        axes[0].text(idx, value + 0.7, str(value), ha="center", fontweight="bold")

    labels = [gate["gate_id"].replace("PUBLICATION_READINESS_GATE", "READINESS") for gate in gates]
    statuses = [gate["status"] for gate in gates]
    palette = {"PASS": "#2b8c6b", "BLOCK": "#c9504d", "HUMAN_WAIT": "#d69e2e"}
    y = list(range(len(labels)))
    axes[1].barh(y, [1] * len(y), color=[palette.get(status, "#718096") for status in statuses])
    axes[1].set_yticks(y, labels)
    axes[1].set_xticks([])
    axes[1].invert_yaxis()
    axes[1].set_title("Current workflow stop")
    for idx, status in enumerate(statuses):
        axes[1].text(0.5, idx, status, ha="center", va="center", color="white", fontweight="bold", fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    path = FIGURES / "gate_and_artifact_status.png"
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)
    return path


def build_markdown(manifest: list[dict[str, str]], state: dict[str, object]) -> str:
    present = sum(row["status"] == "present" for row in manifest)
    missing_rows = [row for row in manifest if row["status"] == "missing"]
    missing_lines = "\n".join(
        f"- `{row['source_stage']}` / `{row['run_id']}`: `{row['path']}`"
        for row in missing_rows
    ) or "- None in the curated 51-row manuscript manifest."
    gate_lines = "\n".join(
        f"- `{gate['gate_id']}`: `{gate['status']}`"
        for gate in state["gate_results"]
    )
    return f"""# Manuscript evidence-recovery and progress audit

Date: 2026-08-13

## Decision-level outcome

`GATE_SCOPE_APPROVAL = approved` and `GATE_CLAIM_APPROVAL = approved` for the
reliability-aware Stage I--IV narrative. `GATE_FIGURE_APPROVAL = pending`.
The supplied scientific-evidence archive closes the requested Stage II--IV raw
artifact gaps. The project is now **data-ready for manuscript figure redraw**,
but it is not figure-approved or submission-ready. No physics was recomputed,
and no frozen artifact was modified.

## Approved scope

- Stage I: proof-of-concept thermodynamic phase discovery.
- Stage II: exact-oracle reliability and numerical validation, enabling
  scalable active learning and the 3D extension.
- Stage III: topology-aware phase discovery and convergence.
- Stage IV: supported-with-caveat main-text higher-dimensional exploration.
- Stage V: Discussion-only exploratory learned-acquisition studies, with no
  superiority claim.
- Stage VI: Outlook-only future learned world-model/EIG direction.

## Artifact/provenance audit

The rebuilt manifest contains {len(manifest)} rows: {present} present and
{len(missing_rows)} missing. Present files were hashed by SHA-256.

{missing_lines}

The ZIP contains 2,397 safe members and its 2,396-entry internal checksum list
verified exactly after normalizing CRLF line endings. The archive itself has
SHA-256 `8f3402bb7dce090eac3504adff3cf3cfd55ab24564a309c6042d94bf3311027d`.

Recovered endpoints:

- Stage II `dataset_iter035.npz`: SHA-256
  `f2d751d0cfae8a3c63da786109903ffd1969a3f90c20a17d92665d8f252e2700`;
  7,434 samples; all unresolved numerical-status counts are zero.
- Stage III `dataset_iter018.npz`: SHA-256
  `6583230495ad9e3eb45b0cff034d5525fa443bd43b1b31a754defeaabdfb5ab4`;
  4,345 samples; the full 37-file convergence/figure bundle is present.
- Stage IV `dataset_iter025.npz`: SHA-256
  `7acab805115bcfcae56e94e2541bfdaa727de7767f2d7dd34330f163e2695842`;
  7,081 samples; return report, audits, tables, figures, and reproduction
  manifest are present.

For all three endpoints, every archived NPZ array matches the paired CSV.
Stage III retains a provenance-metadata note: the frozen decision JSON labels
its source as NPZ while the stored SHA-256 is the archived CSV hash. The input
provenance table names that CSV correctly, and NPZ/CSV numerical arrays agree.
The frozen JSON was retained unchanged.

## Current workflow gates

{gate_lines}

The current workflow status is `{state['workflow_status']}` and the active stop
is `{state['current_stop']}`. Claim approval is human-recorded; figure approval
was not auto-passed. External Introduction retrieval was not run.

## Scientific consequence

Figures 2--4 are no longer blocked by missing evidence. They remain pending
human figure-contract approval and manuscript-ready redraw. The Stage III
vertical slice retains Decision A, the $k=8$ audit pass, and the $k=6$
sensitivity deviation; boundary brackets remain diagnostics only. Candidate
Figure 4 must state that the 3D topology surface is not formally closed by the
strict convergence criteria used for the 2D topology result.

The archive intentionally omits redundant intermediate cumulative NPZ
checkpoints. It retains the derived convergence tables, boundary data, original
figures, report, decision record, and source needed for manuscript redraw and
claim audit, but not a ground-up rerun of every historical convergence step.

## Next decision

Human review should now decide `GATE_FIGURE_APPROVAL` for the Stage III
vertical slice and revised Figures 1--4 contracts. After approval, expand the
Results-by-figure workflow and redraw Figures 1--4 from the recovered sources.
External Introduction retrieval remains forbidden until separate permission is
given.
"""


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#5a6573"))
    canvas.drawString(18 * mm, 11 * mm, "AL_FFLO manuscript workflow - provenance audit")
    canvas.drawRightString(192 * mm, 11 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(markdown: str, figure_path: Path, manifest: list[dict[str, str]], state: dict[str, object]) -> Path:
    pdf_path = OUT / f"{REPORT_STEM}.pdf"
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#17324d"), spaceAfter=14))
    styles.add(ParagraphStyle(name="SubtitleCenter", parent=styles["BodyText"], alignment=TA_CENTER, textColor=colors.HexColor("#5a6573"), fontSize=10, leading=13, spaceAfter=8))
    styles.add(ParagraphStyle(name="SectionBlue", parent=styles["Heading2"], textColor=colors.HexColor("#24577a"), spaceBefore=10, spaceAfter=7))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.7, leading=11.5))
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=18 * mm,
        title="Manuscript Evidence-Recovery and Progress Audit",
        author="AL_FFLO evidence-first manuscript workflow",
    )
    story = [
        Paragraph("Manuscript Evidence-Recovery and Progress Audit", styles["TitleCenter"]),
        Paragraph("Evidence-first PRL workflow | 2026-08-13", styles["SubtitleCenter"]),
        Spacer(1, 7),
        Paragraph("Decision-level outcome", styles["SectionBlue"]),
        Paragraph(
            "GATE_SCOPE_APPROVAL and GATE_CLAIM_APPROVAL are approved for the "
            "reliability-aware Stage I-IV narrative. GATE_FIGURE_APPROVAL remains pending. "
            "The supplied archive closes all requested Stage II-IV raw-evidence gaps. "
            "The manuscript is data-ready, but figures are not yet approved or redrawn.",
            styles["BodyText"],
        ),
        Spacer(1, 10),
        Image(str(figure_path), width=170 * mm, height=69.6 * mm),
        Paragraph("Approved scope", styles["SectionBlue"]),
    ]
    scope_rows = [
        ["Stage", "Approved manuscript role"],
        ["I", "Proof-of-concept thermodynamic phase discovery"],
        ["II", "Exact-oracle reliability enabling scalable active learning"],
        ["III", "Topology-aware phase discovery and convergence"],
        ["IV", "Main text: caveated higher-dimensional 3D extension"],
        ["V", "Discussion: exploratory learned acquisition; no superiority"],
        ["VI", "Outlook: future learned world model / EIG direction"],
    ]
    scope_table = Table(scope_rows, colWidths=[20 * mm, 145 * mm], repeatRows=1)
    scope_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dceaf4")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17324d")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9aa7b5")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.6),
        ("LEADING", (0, 0), (-1, -1), 11),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fb")]),
    ]))
    story.extend([scope_table, PageBreak(), Paragraph("Frozen-artifact recovery", styles["SectionBlue"])])
    recovered = [row for row in manifest if row["artifact_id"] in RECOVERED_ARTIFACT_IDS]
    cell_style = ParagraphStyle(
        name="TableCellSmall",
        parent=styles["Small"],
        fontSize=7.4,
        leading=9.1,
        splitLongWords=True,
    )

    display_names = {
        "dataset_iter035.npz": "Stage II final endpoint — dataset iteration 035",
        "dataset_iter018.npz": "Stage III final endpoint — dataset iteration 018",
        "topology_convergence_audit_report.pdf": "Stage III convergence report / figure bundle",
        "dataset_iter025.npz": "Stage IV final endpoint — dataset iteration 025",
        "stageiv_3d_return_report.pdf": "Stage IV return report / table bundle",
        "phase_3d_jakt_view.png": "Stage IV three-dimensional source figure",
    }
    recovered_rows = [["Stage", "Recovered artifact", "Verification"]] + [
        [
            row["source_stage"],
            Paragraph(display_names.get(Path(row["path"]).name, Path(row["path"]).name), cell_style),
            Paragraph(f"Present; SHA-256 {row['sha256'][:12]}...", cell_style),
        ]
        for row in recovered
    ]
    recovered_table = Table(recovered_rows, colWidths=[17 * mm, 91 * mm, 57 * mm], repeatRows=1)
    recovered_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9eee6")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9aa7b5")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8.2),
        ("LEADING", (0, 0), (-1, 0), 10.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 1), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
    ]))
    story.extend([
        Paragraph(
            "The supplied archive passed ZIP safety checks and its 2,396-file internal "
            "checksum list verified exactly. The rebuilt curated manifest contains 51 "
            "present rows and zero missing rows. No physics was recomputed.",
            styles["BodyText"],
        ),
        Spacer(1, 8),
        recovered_table,
        Spacer(1, 8),
        Paragraph(
            "Endpoint checks: Stage II has 7,434 samples, Stage III has 4,345 samples, "
            "and Stage IV has 7,081 samples. Every NPZ array agrees with its paired CSV. "
            "The Stage III frozen decision JSON carries a label/hash mismatch: it names "
            "the NPZ but stores the paired CSV hash. The provenance table is correct and "
            "the numerical arrays agree, so this is documented as metadata rather than "
            "silently edited.",
            styles["Small"],
        ),
        Paragraph("Current human gates", styles["SectionBlue"]),
    ])
    gate_rows = [["Gate", "Status"]] + [[gate["gate_id"], gate["status"]] for gate in state["gate_results"]]
    gate_table = Table(gate_rows, colWidths=[112 * mm, 53 * mm], repeatRows=1)
    gate_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dceaf4")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9aa7b5")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fb")]),
    ]))
    story.extend([
        gate_table,
        Spacer(1, 9),
        Paragraph(
            "The active workflow stop is GATE_FIGURE_APPROVAL. Claim approval is recorded "
            "as a human decision; figure approval remains pending. External Introduction "
            "retrieval was not run.",
            styles["BodyText"],
        ),
        Paragraph("Scientific consequence", styles["SectionBlue"]),
        Paragraph(
            "Figures 2-4 are no longer blocked by missing evidence. They remain pending "
            "human figure-contract approval and manuscript-ready redraw. Stage III retains "
            "the k=8 pass, k=6 sensitivity deviation, and diagnostic-only brackets. Stage "
            "IV remains explicitly non-closed under the strict 2D topology criteria.",
            styles["BodyText"],
        ),
        Paragraph("Next decision", styles["SectionBlue"]),
        Paragraph(
            "Review the Stage III vertical slice and revised Figures 1-4 contracts at "
            "GATE_FIGURE_APPROVAL. Candidate Figure 4 is the caveated Stage IV extension; "
            "after approval, redraw Figures 1-4 from the recovered sources. External "
            "Introduction retrieval remains unauthorized.",
            styles["BodyText"],
        ),
    ])
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return pdf_path


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    manifest, state = build_tables()
    figure = build_figure(manifest, state)
    markdown = build_markdown(manifest, state)
    (OUT / f"{REPORT_STEM}.md").write_text(markdown, encoding="utf-8")
    (OUT / "decision_log.md").write_text(
        "# Decision log\n\n"
        "- GATE_SCOPE_APPROVAL: approved with the Stage I-IV revision.\n"
        "- GATE_CLAIM_APPROVAL: approved for all six bounded claims.\n"
        "- Frozen artifact recovery: complete for the requested Stage II-IV groups.\n"
        "- Artifact manifest: 51 present, 0 missing after SHA-256 rebuild.\n"
        "- Stage III provenance: frozen decision JSON has an NPZ label with the paired CSV hash; arrays match and the metadata note is retained.\n"
        "- GATE_FIGURE_APPROVAL: pending and reached; not auto-passed.\n"
        "- Figures 2-4: evidence recovered; manuscript-ready redraw awaits figure approval.\n"
        "- Figure 4: main-text candidate with strict-convergence caveat.\n"
        "- External Introduction retrieval: not authorized.\n"
        "- Next action: scientific review of the Stage III slice and Figures 1-4 contracts.\n",
        encoding="utf-8",
    )
    pdf = build_pdf(markdown, figure, manifest, state)
    print(f"wrote {pdf.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
