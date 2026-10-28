from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from review_agent.figure_authority import (
    CANONICAL_MAIN_FIGURES_STAGEIV_REVISION,
    sha256,
    stage_stageiv_revision_main_figures,
)


RUN_ID = "active_learning_manuscript_stageiv_revision_v3"
PARENT_RUN_ID = "active_learning_prl_integrated_cited_v2"
PROFILE_ID = "INTEGRATED_CITED_STAGEIV_SCOPE_REVISION_V3"
TITLE = "Reliability-aware active learning for FFLO phase spaces"


def _root(state: dict[str, Any]) -> Path:
    return Path(state.get("project_root", Path.cwd())).resolve()


def _variant(root: Path) -> Path:
    return root / "manuscript_variants/integrated_cited_stageiv_revision_v3"


def _output(root: Path) -> Path:
    return root / "output/pdf/manuscript_integrated_cited_stageiv_revision_v3"


def _run(root: Path) -> Path:
    return root / f"manuscript_langgraph/runs/{RUN_ID}"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _json(path: Path, value: Any) -> None:
    _write(path, json.dumps(value, indent=2, ensure_ascii=False))


def figure_scope_gate(state: dict[str, Any]) -> dict[str, Any]:
    root = _root(state)
    fig4 = root / "figures/fig4_strict_data_v2"
    sm = root / "figures/sm_stageiv_numerical_diagnostics"
    reports = [
        json.loads((fig4 / "figure_validation_report.json").read_text(encoding="utf-8")),
        json.loads((fig4 / "specific_validation.json").read_text(encoding="utf-8")),
        json.loads((sm / "figure_validation_report.json").read_text(encoding="utf-8")),
        json.loads((sm / "specific_validation.json").read_text(encoding="utf-8")),
    ]
    passed = all(item.get("status") == "pass" for item in reports)
    contracts = [
        (fig4 / "figure_contract.yaml").read_text(encoding="utf-8"),
        (sm / "figure_contract.yaml").read_text(encoding="utf-8"),
    ]
    semantics = (
        "threshold_line_visible: false" in contracts[1]
        and "pass_fail_labels_visible: false" in contracts[1]
        and "strict_3d_nonclosure" not in contracts[0]
    )
    run = _run(root)
    payload = {
        "gate_id": "GATE_STAGEIV_SCOPE_REVISION_FIGURE_INPUT",
        "status": "PASS" if passed and semantics else "BLOCK",
        "figure_engine_reports": [item.get("figure_id") for item in reports],
        "threshold_based_physical_convergence_claim_authorized": False,
        "panel_d_authorized": False,
        "main_figure_panels": ["a", "b", "c"],
        "supplement_mode": "raw_numerical_diagnostics_without_threshold_lines",
    }
    _json(run / "gate_records/GATE_STAGEIV_SCOPE_REVISION_FIGURE_INPUT.json", payload)
    if payload["status"] != "PASS":
        raise RuntimeError("Stage-IV revision figure/scope gate failed")
    return {
        "stageiv_scope_gate_passed": True,
        "run_id": RUN_ID,
        "parent_run_id": PARENT_RUN_ID,
        "publication_profile_id": PROFILE_ID,
        "workflow_status": "stageiv_scope_gate_passed",
        "next_action": "prepare versioned manuscript and Supplement sources",
    }


def _replace_stageiv_text(variant: Path, output: Path) -> None:
    abstract = variant / "sections/abstract.tex"
    text = abstract.read_text(encoding="utf-8")
    abstract_replacement = (
        "The same framework exposes thermodynamic and topology-aware structure in "
        r"$(k_BT/t,J_A/t,\mu/t)$ across a finite sampled domain; the exact support is "
        "reported discretely, without inferring a continuous three-dimensional topology surface or a complete atlas."
    )
    text = re.sub(r"The same framework exposes.*$", lambda _: abstract_replacement, text, flags=re.DOTALL)
    _write(abstract, text)

    intro = variant / "sections/introduction.tex"
    text = intro.read_text(encoding="utf-8")
    text = text.replace(
        "The extension reveals thermodynamic and topology-aware structure, while its strict topology convergence criteria remain open; it is not presented as a complete three-dimensional topology atlas.",
        "The extension reveals thermodynamic and topology-aware structure across a finite sampled domain; its exact support is reported discretely and is not promoted to a continuous three-dimensional topology surface or a complete atlas.",
    )
    _write(intro, text)

    discussion = variant / "sections/discussion.tex"
    text = discussion.read_text(encoding="utf-8")
    text = text.replace(
        "The two-dimensional thermodynamic and configured topology audits show what closes under registered criteria, while the three-dimensional extension demonstrates why selective exact sampling becomes more consequential as the domain grows. That extension is deliberately bounded: its discrete support and fixed-$\\mu$ slices reveal structure, but failed strict topology criteria prevent a claim of a converged surface or complete atlas.",
        "The two-dimensional thermodynamic and configured topology audits establish the low-dimensional results, while the three-dimensional extension demonstrates why selective exact sampling becomes more consequential as the domain grows. That extension is deliberately bounded by its finite sampled interval, discrete exact support, and four fixed-$\\mu$ slices; these data reveal structure but do not define a continuous topology surface or complete atlas.",
    )
    _write(discussion, text)

    body = variant / "sections/integrated_results_methods.tex"
    text = body.read_text(encoding="utf-8")
    old_caption = re.search(
        r"\\caption\{\\textbf\{Active-learning-enabled higher-dimensional exploration\.\}.*?\}\n\\label\{fig:3d\}",
        text,
        flags=re.DOTALL,
    )
    if not old_caption:
        raise RuntimeError("could not locate the inherited Figure-4 caption")
    new_caption = r"""\caption{\textbf{Active-learning-enabled higher-dimensional exploration.} \textbf{(a)} Exact three-dimensional thermodynamic point cloud in $(k_BT/t,J_A/t,\mu/t)$ over $\mu/t\in[-0.5,1.5]$: 7,081 observations containing 2,350 Normal, 141 uniform-SC, and 4,590 FFLO states. \textbf{(b)} Trusted FFLO-only topology support comprises 3,145 cFFLO and 1,445 tFFLO points. \textbf{(c)} Four narrow fixed-$\mu$ bins, centered at $\mu/t=-0.45,0.15,0.85,$ and $1.45$ with half-width 0.025, show exact slice samples without interpolation between slices. The finite-domain support is discrete; no continuous topology surface or complete three-dimensional atlas is inferred.}
\label{fig:3d}"""
    text = text[:old_caption.start()] + new_caption + text[old_caption.end():]
    old_paragraph = re.search(
        r"The same oracle--scheduler separation extends the sampled coordinates.*?(?=\n\n\\FloatBarrier)",
        text,
        flags=re.DOTALL,
    )
    if not old_paragraph:
        raise RuntimeError("could not locate the inherited Stage-IV Results paragraph")
    new_paragraph = r"""The same oracle--scheduler separation extends the sampled coordinates to $x=(k_BT/t,J_A/t,\mu/t)$ over the finite interval $\mu/t\in[-0.5,1.5]$. The endpoint contains 7,081 exact observations: 2,350 Normal, 141 uniform-SC, and 4,590 FFLO states (Fig.~\ref{fig:3d}). Trusted FFLO-only topology support comprises 3,145 cFFLO and 1,445 tFFLO points and remains discrete rather than a fitted surface. Four explicit fixed-$\mu$ bins centered at $\mu/t=-0.45,0.15,0.85,$ and $1.45$, each with half-width 0.025, expose exact slice-to-slice structure without interpolation between slices. These point clouds and slices demonstrate higher-dimensional exploration and reveal thermodynamic and topology-aware structures across the sampled domain. Archived iteration histories are reported in the Supplemental Material as numerical stability diagnostics; no operational cutoff is promoted to a physical convergence criterion. Consequently, the main claim is bounded by the finite domain and discrete exact support, and does not imply a continuous topology surface or complete three-dimensional atlas (Fig.~\ref{fig:3d})."""
    text = text[:old_paragraph.start()] + new_paragraph + text[old_paragraph.end():]
    old_output = re.search(r"/Users/baikaizhi/.+?/output/pdf/manuscript_prl_integrated_cited_v2/figures", text)
    if old_output:
        text = text.replace(old_output.group(0), (output / "figures").as_posix())
    _write(body, text)


def _supplement_source(root: Path, output: Path) -> str:
    inherited = (root / "output/pdf/manuscript_methods_sm_discussion_review_v1/supplement_candidate.tex").read_text(encoding="utf-8")
    replacement = r"""\section{Stage-IV finite-domain diagnostics}
The three-dimensional run samples $(k_BT/t,J_A/t,\mu/t)$ over the finite interval
$\mu/t\in[-0.5,1.5]$ and returns 7,081 unique exact observations.  The main-text
figure reports the exact thermodynamic cloud, trusted FFLO topology support,
and four explicit fixed-$\mu$ bins.  These are discrete observations; no
continuous topology surface is fitted or inferred.

Figure~\ref{fig:s-stageiv-diagnostics} records archived numerical histories.
The map-change fraction measures changes between successive proxy label maps;
the proxy-surface displacement is a normalized bidirectional $p_{95}$ distance;
and trusted surprise is the fraction of new trusted labels that differ from the
previous proxy prediction.  These quantities test stability of a particular
sampling and reconstruction procedure.  They are not thermodynamic observables
or universal physical scales.  We therefore show their raw values without a
cutoff line and do not convert them into a physical convergence declaration.
The archive additionally records one final proxy-coverage value, 0.015318, and
component counts $3\to4\to3$ over the last three iterations; neither supplies a
standalone physical closure criterion.

\begin{figure*}[t]
\centering
\includegraphics[width=0.96\textwidth]{\detokenize{""" + (output / "figures/SuppFig_StageIV_diagnostics.pdf").as_posix() + r"""}}
\caption{\textbf{Archived Stage-IV numerical stability diagnostics.}
\textbf{(a)} Total exact, trusted-exact, and trusted-topology counts.
\textbf{(b)} Thermodynamic-label fractions and \textbf{(c)} applicable-SC
topology-label fractions versus iteration.  \textbf{(d)--(f)} Last-three raw
histories of topology-map change, proxy-surface $p_{95}$ displacement, and
trusted-topology surprise.  No threshold line is drawn and no panel is used as
a physical convergence criterion.}
\label{fig:s-stageiv-diagnostics}
\end{figure*}"""
    pattern = r"\\section\{Higher-dimensional extension and non-closure\}.*?(?=\\section\{Exploratory acquisition)"
    revised, count = re.subn(pattern, lambda _: replacement + "\n\n", inherited, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError("could not replace inherited Stage-IV Supplement section")
    return revised


def prepare_revision_sources(state: dict[str, Any]) -> dict[str, Any]:
    if not state.get("stageiv_scope_gate_passed"):
        raise RuntimeError("scope gate must pass before manuscript source generation")
    root = _root(state)
    variant = _variant(root)
    output = _output(root)
    run = _run(root)
    if variant.exists():
        shutil.rmtree(variant)
    shutil.copytree(root / "manuscript_variants/prl_integrated_cited_v2", variant)
    output.mkdir(parents=True, exist_ok=True)
    stage_stageiv_revision_main_figures(root, output / "figures", output)
    sm_source = root / "figures/sm_stageiv_numerical_diagnostics/sm_stageiv_numerical_diagnostics.pdf"
    shutil.copy2(sm_source, output / "figures/SuppFig_StageIV_diagnostics.pdf")
    shutil.copy2(root / "figures/sm_stageiv_numerical_diagnostics/sm_stageiv_numerical_diagnostics.png", output / "figures/SuppFig_StageIV_diagnostics.png")
    _replace_stageiv_text(variant, output)
    supplement = _supplement_source(root, output)
    _write(variant / "sections/supplement.tex", supplement)

    main_tex = (variant / "main.tex").read_text(encoding="utf-8")
    main_tex = re.sub(r"\\title\{.*?\}", rf"\\title{{{TITLE}}}", main_tex, count=1)
    main_tex = main_tex.replace("\\date{August 28, 2026}", "\\date{August 31, 2026}")
    _write(variant / "main.tex", main_tex)
    sm_tex = rf"""\documentclass[aps,prl,preprint,superscriptaddress,floatfix]{{revtex4-2}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern,microtype,amsmath,amssymb,graphicx,booktabs,hyperref}}
\hypersetup{{colorlinks=true,allcolors=blue}}
\begin{{document}}
\title{{Supplemental Material: {TITLE}}}
\author{{Author list pending human approval}}
\date{{August 31, 2026}}
\maketitle
\setcounter{{figure}}{{0}}
\renewcommand{{\thefigure}}{{S\arabic{{figure}}}}
\input{{sections/supplement.tex}}
\end{{document}}
"""
    _write(variant / "supplement.tex", sm_tex)
    _json(run / "checkpoints/revision_sources.json", {
        "status": "PASS",
        "parent_run_id": PARENT_RUN_ID,
        "variant_root": str(variant),
        "title": TITLE,
        "external_retrieval_executed": False,
        "fig4_panel_d_removed": True,
    })
    return {
        "variant_root": str(variant),
        "variant_output_root": str(output),
        "revision_sources_ready": True,
        "workflow_status": "stageiv_revision_sources_ready",
        "next_action": "compile main manuscript and Supplemental Material",
    }


def _compile(tex: str, job: str, variant: Path, output: Path, *, bibliography: bool) -> Path:
    logs: list[str] = []
    command = ["pdflatex", "-recorder", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={output}", f"-jobname={job}", tex]
    first = subprocess.run(command, cwd=variant, text=True, capture_output=True, check=False)
    logs.append(first.stdout + "\n" + first.stderr)
    if first.returncode:
        _write(output / f"{job}_compile.log", "\n".join(logs)); raise RuntimeError(f"first LaTeX pass failed: {job}")
    if bibliography:
        bib = subprocess.run(["bibtex", job], cwd=output, text=True, capture_output=True, check=False)
        logs.append(bib.stdout + "\n" + bib.stderr)
        if bib.returncode:
            _write(output / f"{job}_compile.log", "\n".join(logs)); raise RuntimeError(f"BibTeX failed: {job}")
        passes = 3
    else:
        passes = 2
    last = first
    for _ in range(passes):
        last = subprocess.run(command, cwd=variant, text=True, capture_output=True, check=False)
        logs.append(last.stdout + "\n" + last.stderr)
        if last.returncode:
            break
    _write(output / f"{job}_compile.log", "\n\n--- PASS ---\n\n".join(logs))
    pdf = output / f"{job}.pdf"
    if last.returncode or not pdf.is_file():
        raise RuntimeError(f"LaTeX compilation failed: {job}")
    return pdf


def compile_revision(state: dict[str, Any]) -> dict[str, Any]:
    if not state.get("revision_sources_ready"):
        raise RuntimeError("revision sources are not ready")
    root = _root(state); variant = Path(state["variant_root"]); output = Path(state["variant_output_root"])
    shutil.copy2(variant / "references.bib", output / "references.bib")
    main_pdf = _compile("main.tex", "manuscript_candidate", variant, output, bibliography=True)
    sm_pdf = _compile("supplement.tex", "supplement_candidate", variant, output, bibliography=False)
    main_md = "\n\n".join((variant / path).read_text(encoding="utf-8") for path in ["sections/abstract.tex", "sections/introduction.tex", "sections/integrated_results_methods.tex", "sections/discussion.tex"])
    _write(output / "manuscript_candidate.md", f"# {TITLE}\n\n{main_md}")
    _write(output / "supplement_candidate.md", (variant / "sections/supplement.tex").read_text(encoding="utf-8"))
    tables = output / "tables"; tables.mkdir(exist_ok=True)
    shutil.copy2(root / "figures/sm_stageiv_numerical_diagnostics/canonical/stageiv_iteration_history.csv", tables / "stageiv_iteration_history.csv")
    shutil.copy2(root / "figures/sm_stageiv_numerical_diagnostics/canonical/stageiv_last3_raw_diagnostics.csv", tables / "stageiv_last3_raw_diagnostics.csv")
    return {
        "revised_manuscript_pdf": str(main_pdf),
        "revised_supplement_pdf": str(sm_pdf),
        "revision_compiled": True,
        "workflow_status": "stageiv_revision_compiled",
        "next_action": "audit compiled PDFs and Stage-IV semantics",
    }


def _pdf_pages(path: Path) -> int:
    output = subprocess.run(["pdfinfo", str(path)], text=True, capture_output=True, check=True).stdout
    return int(re.search(r"^Pages:\s+(\d+)", output, flags=re.MULTILINE).group(1))


def audit_revision(state: dict[str, Any]) -> dict[str, Any]:
    root = _root(state); variant = Path(state["variant_root"]); output = Path(state["variant_output_root"])
    main_pdf = Path(state["revised_manuscript_pdf"]); sm_pdf = Path(state["revised_supplement_pdf"])
    main_source = (variant / "main.tex").read_text(encoding="utf-8")
    body = (variant / "sections/integrated_results_methods.tex").read_text(encoding="utf-8")
    abstract = (variant / "sections/abstract.tex").read_text(encoding="utf-8")
    intro = (variant / "sections/introduction.tex").read_text(encoding="utf-8")
    discussion = (variant / "sections/discussion.tex").read_text(encoding="utf-8")
    sm_source = (variant / "sections/supplement.tex").read_text(encoding="utf-8")
    title = re.search(r"\\title\{([^}]+)\}", main_source).group(1)
    title_clean = not re.search(r"\b(PRL|Physical Review|Nature|Science)\b", title, flags=re.IGNORECASE)
    stageiv_text = "\n".join([abstract, intro, body, discussion])
    removed = all(phrase not in stageiv_text for phrase in [
        "strict three-dimensional topology convergence gates",
        "final strict audit nevertheless exceeds",
        "failed strict topology criteria",
    ])
    fig4_caption = re.search(r"\\caption\{\\textbf\{Active-learning-enabled higher-dimensional exploration\.\}.*?\}\n\\label\{fig:3d\}", body, flags=re.DOTALL).group(0)
    caption_three_panels = "\\textbf{(d)}" not in fig4_caption and all(f"\\textbf{{({letter})}}" in fig4_caption for letter in "abc")
    sm_semantics = all(token in sm_source for token in ["numerical stability diagnostics", "without a\ncutoff line", "not thermodynamic observables"])
    main_log = (output / "manuscript_candidate.log").read_text(encoding="utf-8", errors="replace")
    sm_log = (output / "supplement_candidate.log").read_text(encoding="utf-8", errors="replace")
    no_latex_errors = not re.search(r"Undefined|Overfull \\hbox", main_log + sm_log)
    bbl = (output / "manuscript_candidate.bbl").read_text(encoding="utf-8", errors="replace")
    references = len(re.findall(r"^\\bibitem", bbl, flags=re.MULTILINE))
    figure_hash = sha256(output / "figures/Fig4_candidate.pdf")
    sm_hash = sha256(output / "figures/SuppFig_StageIV_diagnostics.pdf")
    hashes_ok = (
        figure_hash == CANONICAL_MAIN_FIGURES_STAGEIV_REVISION["FIG-04-STAGEIV-3D-EXTENSION"]["pdf_sha256"]
        and sm_hash == "5042dc9235a5c31f01377dda1c8dc652f3379aadbda42207b53977bb2458f9eb"
    )
    checks = {
        "neutral_manuscript_title": title_clean,
        "figure4_panel_d_removed": caption_three_panels,
        "stageiv_threshold_closure_claims_removed": removed,
        "supplement_diagnostic_semantics": sm_semantics,
        "fifty_references_preserved": references == 50,
        "latex_logs_clean": no_latex_errors,
        "figure_hashes_match_authority": hashes_ok,
        "main_pdf_nonempty": _pdf_pages(main_pdf) >= 6,
        "supplement_pdf_nonempty": _pdf_pages(sm_pdf) >= 2,
    }
    passed = all(checks.values())
    audit = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "title": title,
        "checks": checks,
        "main_pages": _pdf_pages(main_pdf),
        "supplement_pages": _pdf_pages(sm_pdf),
        "bibliography_items": references,
        "external_retrieval_executed": False,
        "physics_recomputed": False,
    }
    _json(output / "stageiv_scope_revision_audit.json", audit)
    _write(output / "stageiv_scope_revision_audit.md", "# Stage-IV scope revision audit\n\n" + f"Status: **{audit['status']}**.\n\n" + "\n".join(f"- {name}: {'PASS' if value else 'FAIL'}" for name, value in checks.items()))
    _write(output / "decision_log.md", """# Decision log

- Figure 4(d) and its operational-threshold scoreboard were removed.
- Stage-IV raw histories moved to a dedicated Supplemental figure without cutoff lines or pass/fail labels.
- Stage-IV claims are now bounded by a finite sampled domain, discrete exact support, and the absence of an inferred continuous topology surface.
- Figures 1--3, the physical model, numerical oracle, and 50-source literature registry were not changed.
- Both revised figures and the manuscript package remain at human science/visual review gates.
""")
    if not passed:
        raise RuntimeError(f"Stage-IV manuscript revision audit failed: {checks}")
    return {
        "revision_audit_passed": True,
        "workflow_status": "stageiv_revision_audit_passed",
        "next_action": "human review of revised Figure 4, SM diagnostics, manuscript, and Supplement",
    }


def revision_human_gate(state: dict[str, Any]) -> dict[str, Any]:
    root = _root(state); run = _run(root)
    record = {
        "gate_id": "GATE_STAGEIV_SCOPE_REVISION_HUMAN_REVIEW",
        "status": "HUMAN_WAIT",
        "items": ["Figure 4 v2", "Stage-IV SM diagnostics", "main-text wording", "Supplement wording"],
        "automatic_approval": False,
    }
    _json(run / "gate_records/GATE_STAGEIV_SCOPE_REVISION_HUMAN_REVIEW.json", record)
    _json(run / "checkpoints/final_state.json", {**state, "gate": record})
    return {
        "workflow_status": "stageiv_revision_human_review_wait",
        "next_action": "user reviews the versioned candidate package",
    }
