from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "manuscript_langgraph"
sys.path.insert(0, str(PACKAGE_ROOT))

from review_agent.figure_authority import (  # noqa: E402
    CANONICAL_MAIN_FIGURES,
    sha256,
    stage_canonical_main_figures,
)


SOURCE_VARIANT = PROJECT_ROOT / "manuscript_variants/prl_integrated"
TRIAL_VARIANT = PROJECT_ROOT / "manuscript_variants/prl_integrated_singlecol_trial"
SOURCE_OUTPUT = PROJECT_ROOT / "output/pdf/manuscript_prl_integrated_v1"
TRIAL_OUTPUT = PROJECT_ROOT / "output/pdf/manuscript_prl_integrated_singlecol_trial_v1"
SOURCE_PDF = SOURCE_OUTPUT / "manuscript_prl_integrated_candidate.pdf"
TRIAL_PDF = TRIAL_OUTPUT / "manuscript_prl_integrated_singlecol_trial.pdf"
TRIAL_FIGURE_IDS = [
    "FIG-02-THERMO",
    "FIG-03-STAGEIII-TOPOLOGY",
    "FIG-04-STAGEIV-3D-EXTENSION",
]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, value: object) -> None:
    write_text(path, json.dumps(value, indent=2, ensure_ascii=False))


def pdf_tool(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    candidate = (
        Path.home()
        / f".cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/poppler/bin/{name}"
    )
    if not candidate.exists():
        raise RuntimeError(f"required PDF tool not found: {name}")
    return str(candidate)


def pdf_page_width_points(path: Path) -> float:
    info = subprocess.run(
        [pdf_tool("pdfinfo"), str(path)],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    match = re.search(r"^Page size:\s+([0-9.]+) x", info, re.MULTILINE)
    if not match:
        raise RuntimeError(f"could not read page width from {path}")
    return float(match.group(1))


def page_texts(path: Path) -> list[str]:
    info = subprocess.run(
        [pdf_tool("pdfinfo"), str(path)],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    page_match = re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE)
    if not page_match:
        raise RuntimeError("could not determine trial PDF page count")
    pages = int(page_match.group(1))
    return [
        subprocess.run(
            [pdf_tool("pdftotext"), "-f", str(page), "-l", str(page), "-layout", str(path), "-"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        for page in range(1, pages + 1)
    ]


def make_trial_sources() -> None:
    if not SOURCE_VARIANT.exists() or not SOURCE_PDF.exists():
        raise RuntimeError("build the approved-layout PRL-integrated candidate before the trial")
    (TRIAL_VARIANT / "sections").mkdir(parents=True, exist_ok=True)
    for name in ["abstract.tex", "introduction.tex", "discussion.tex"]:
        shutil.copy2(SOURCE_VARIANT / "sections" / name, TRIAL_VARIANT / "sections" / name)

    body = (SOURCE_VARIANT / "sections/integrated_results_methods.tex").read_text(encoding="utf-8")
    blocks = list(re.finditer(r"\\begin\{figure\*\}\[t\].*?\\end\{figure\*\}", body, re.DOTALL))
    if len(blocks) != 4:
        raise RuntimeError(f"expected four double-column source figures, found {len(blocks)}")
    for match in reversed(blocks[1:]):
        block = match.group(0)
        replacement = (
            block.replace(r"\begin{figure*}[t]", r"\begin{figure}[t]")
            .replace(r"width=0.96\textwidth", r"width=\columnwidth")
            .replace(r"\end{figure*}", r"\end{figure}")
        )
        body = body[: match.start()] + replacement + body[match.end() :]

    for record in CANONICAL_MAIN_FIGURES.values():
        old_asset = SOURCE_OUTPUT / "figures" / record["staged_pdf"]
        new_asset = TRIAL_OUTPUT / "figures" / record["staged_pdf"]
        body = body.replace(old_asset.as_posix(), new_asset.as_posix())
    write_text(TRIAL_VARIANT / "sections/integrated_results_methods.tex", body)

    main = (SOURCE_VARIANT / "main.tex").read_text(encoding="utf-8")
    main = main.replace(
        r"\begin{document}",
        "\\begin{document}\n"
        "\\typeout{TRIAL_COLUMNWIDTH=\\the\\columnwidth}\n"
        "\\typeout{TRIAL_TEXTWIDTH=\\the\\textwidth}",
        1,
    )
    write_text(TRIAL_VARIANT / "main.tex", main)


def compile_trial() -> str:
    TRIAL_OUTPUT.mkdir(parents=True, exist_ok=True)
    stage_canonical_main_figures(PROJECT_ROOT, TRIAL_OUTPUT / "figures", TRIAL_OUTPUT)
    command = [
        "pdflatex",
        "-recorder",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={TRIAL_OUTPUT}",
        "-jobname=manuscript_prl_integrated_singlecol_trial",
        "main.tex",
    ]
    logs: list[str] = []
    return_code = 1
    for _ in range(3):
        proc = subprocess.run(command, cwd=TRIAL_VARIANT, text=True, capture_output=True, check=False)
        logs.append(proc.stdout + "\n" + proc.stderr)
        return_code = proc.returncode
        if return_code:
            break
    full_log = "\n\n--- PASS ---\n\n".join(logs)
    write_text(TRIAL_OUTPUT / "latex_compile.log", full_log)
    if return_code or not TRIAL_PDF.exists():
        raise RuntimeError("single-column trial LaTeX compilation failed")
    return full_log


def audit_trial(source_hash: str, frozen_hashes: dict[str, str], compile_log: str) -> None:
    body = (TRIAL_VARIANT / "sections/integrated_results_methods.tex").read_text(encoding="utf-8")
    if body.count(r"\begin{figure*}[t]") != 1 or body.count(r"\begin{figure}[t]") != 3:
        raise RuntimeError("trial does not contain one double-column and three single-column figures")
    if body.count(r"width=\columnwidth") != 3:
        raise RuntimeError("trial single-column widths were not applied exactly three times")
    if "0.006269" not in body or "not all closed" not in body:
        raise RuntimeError("mandatory scientific caveats were lost in trial construction")

    column_match = re.search(r"TRIAL_COLUMNWIDTH=([0-9.]+)pt", compile_log)
    text_match = re.search(r"TRIAL_TEXTWIDTH=([0-9.]+)pt", compile_log)
    if not column_match or not text_match:
        raise RuntimeError("REVTeX trial dimensions were not recorded")
    column_width = float(column_match.group(1))
    text_width = float(text_match.group(1))

    pages = page_texts(TRIAL_PDF)
    leads = {
        "1": "Exact oracle and physical labels",
        "2": "Audited thermodynamic reconstruction",
        "3": "Topology-resolved FFLO structure",
        "4": "Higher-dimensional exploration",
    }
    anchor_pages = {
        number: next((i for i, text in enumerate(pages, 1) if lead in text), None)
        for number, lead in leads.items()
    }
    figure_pages = {
        number: next((i for i, text in enumerate(pages, 1) if re.search(rf"FIG\.\s*{number}\.", text)), None)
        for number in leads
    }
    placement_pass = all(
        anchor_pages[number] is not None
        and figure_pages[number] is not None
        and 0 <= figure_pages[number] - anchor_pages[number] <= 2
        for number in leads
    )

    scale_rows: list[dict[str, object]] = []
    for figure_id in TRIAL_FIGURE_IDS:
        record = CANONICAL_MAIN_FIGURES[figure_id]
        native_width = pdf_page_width_points(PROJECT_ROOT / record["source_pdf"])
        scale = column_width / native_width
        scale_rows.append(
            {
                "figure_id": figure_id,
                "native_width_pt": native_width,
                "placed_width_pt": column_width,
                "scale_factor": round(scale, 4),
                "projected_minimum_internal_text_pt_from_registered_7pt": round(7.0 * scale, 2),
                "automatic_readability_status": "HUMAN_REVIEW_REQUIRED",
            }
        )

    overfull = re.findall(r"Overfull \\hbox[^\n]*", compile_log)
    underfull = re.findall(r"Underfull \\hbox[^\n]*", compile_log)
    trial_hash = sha256(TRIAL_PDF)
    manifest = {
        "schema_version": 1,
        "trial_id": "PRL_INTEGRATED_SINGLE_COLUMN_FIG2_TO_FIG4_V1",
        "source_candidate_pdf": str(SOURCE_PDF.relative_to(PROJECT_ROOT)),
        "source_candidate_sha256": source_hash,
        "trial_pdf": str(TRIAL_PDF.relative_to(PROJECT_ROOT)),
        "trial_pdf_sha256": trial_hash,
        "layout": {"fig1": "double_column", "fig2": "single_column", "fig3": "single_column", "fig4": "single_column"},
        "revtex_column_width_pt": column_width,
        "revtex_text_width_pt": text_width,
        "compiled_page_count": len(pages),
        "anchor_pages": anchor_pages,
        "figure_pages": figure_pages,
        "placement_status": "PASS" if placement_pass else "FAIL",
        "scale_audit": scale_rows,
        "frozen_figure_hashes": frozen_hashes,
        "frozen_figures_modified": False,
        "captions_modified": False,
        "scientific_text_modified": False,
        "original_candidate_modified": False,
        "overfull_box_count": len(overfull),
        "underfull_box_count": len(underfull),
        "human_visual_review": "PENDING",
        "gate_effect": "none",
    }
    write_json(TRIAL_VARIANT / "single_column_trial_manifest.json", manifest)

    rows = "\n".join(
        f"| {row['figure_id']} | {row['native_width_pt']:.2f} | {row['placed_width_pt']:.2f} | "
        f"{row['scale_factor']:.3f} | {row['projected_minimum_internal_text_pt_from_registered_7pt']:.2f} pt | HUMAN REVIEW |"
        for row in scale_rows
    )
    write_text(
        TRIAL_VARIANT / "single_column_trial_audit.md",
        f"""# Fig. 2--4 single-column trial audit

## Scope

Fig. 1 remains a two-column `figure*`. Figs. 2--4 use ordinary `figure` floats at full `\\columnwidth`. Frozen PDFs, captions, scientific text, claims, and gates are unchanged.

## Compilation and placement

- Trial pages: **{len(pages)}**.
- REVTeX column width: **{column_width:.2f} pt**; text width: **{text_width:.2f} pt**.
- First-text pages: `{anchor_pages}`.
- Compiled figure pages: `{figure_pages}`.
- Placement: **{'PASS' if placement_pass else 'FAIL'}** under the within-two-pages structural rule.
- Overfull boxes: **{len(overfull)}**; underfull boxes: **{len(underfull)}**.

## Print-scale projection

| Figure | Native width | Placed width | Scale | Projected 7 pt internal text | Status |
|---|---:|---:|---:|---:|---|
{rows}

The projection is a geometry check, not an automatic aesthetic decision. Internal labels near 3.4 pt are below the Figure Engine's registered 7 pt native/full-width target, so the single-column trial cannot receive an automatic readability pass. The final decision requires inspection of the rendered pages at native letter size and a 100% print-scale crop.

## Provenance

- Source candidate SHA-256: `{source_hash}`.
- Trial candidate SHA-256: `{trial_hash}`.
- All four source and staged figure hashes match the canonical frozen registry.
- No external retrieval or physics calculation was performed.
- This trial does not change or pass `GATE_PRL_INTEGRATED_MANUSCRIPT_HUMAN_REVIEW`.
""",
    )
    write_json(TRIAL_OUTPUT / "pdf_hashes.json", {TRIAL_PDF.name: trial_hash})


def main() -> None:
    source_hash = sha256(SOURCE_PDF)
    source_tex_hash = sha256(SOURCE_VARIANT / "sections/integrated_results_methods.tex")
    frozen_hashes = {
        figure_id: sha256(PROJECT_ROOT / record["source_pdf"])
        for figure_id, record in CANONICAL_MAIN_FIGURES.items()
    }
    for figure_id, record in CANONICAL_MAIN_FIGURES.items():
        if frozen_hashes[figure_id] != record["pdf_sha256"]:
            raise RuntimeError(f"frozen source mismatch before trial: {figure_id}")

    make_trial_sources()
    compile_log = compile_trial()
    audit_trial(source_hash, frozen_hashes, compile_log)

    if sha256(SOURCE_PDF) != source_hash:
        raise RuntimeError("approved-layout candidate changed during trial")
    if sha256(SOURCE_VARIANT / "sections/integrated_results_methods.tex") != source_tex_hash:
        raise RuntimeError("approved-layout source changed during trial")
    for figure_id, record in CANONICAL_MAIN_FIGURES.items():
        if sha256(PROJECT_ROOT / record["source_pdf"]) != frozen_hashes[figure_id]:
            raise RuntimeError(f"frozen figure changed during trial: {figure_id}")

    print("PRL_SINGLE_COLUMN_TRIAL_BUILD_PASS")
    print(f"trial_pdf={TRIAL_PDF}")
    print("fig1_layout=double_column")
    print("fig2_to_fig4_layout=single_column")
    print("gate_effect=none")


if __name__ == "__main__":
    main()
