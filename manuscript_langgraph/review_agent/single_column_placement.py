from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from review_agent.publication_reproducibility import TITLE, _compile, _freeze_hashes, _sha
from review_agent.scientific_depth_revision import _layout_metrics, _result_blocks


RUN_ID = "active_learning_prl_single_column_v6"
PARENT_RUN_ID = "active_learning_prl_scientific_depth_v5"
GATE_ID = "GATE_SINGLE_COLUMN_FIGURE_PLACEMENT_HUMAN_REVIEW"


def _root(state: dict[str, Any]) -> Path:
    return Path(state.get("project_root", Path.cwd())).resolve()


def _run(root: Path) -> Path:
    return root / "manuscript_langgraph/runs" / RUN_ID


def _variant(root: Path) -> Path:
    return root / "manuscript_variants/scientific_depth_single_column_v6"


def _output(root: Path) -> Path:
    return root / "output/pdf/manuscript_scientific_depth_single_column_v6"


def _parent_variant(root: Path) -> Path:
    return root / "manuscript_variants/scientific_depth_v5"


def _parent_run(root: Path) -> Path:
    return root / "manuscript_langgraph/runs" / PARENT_RUN_ID


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _json(path: Path, value: Any) -> None:
    _write(path, json.dumps(value, indent=2, ensure_ascii=False))


def repository_discovery(state: dict[str, Any]) -> dict[str, Any]:
    root = _root(state)
    run = _run(root)
    parent_state_path = _parent_run(root) / "final_state.json"
    required = [
        parent_state_path,
        _parent_variant(root) / "sections/abstract.tex",
        _parent_variant(root) / "sections/introduction.tex",
        _parent_variant(root) / "sections/discussion.tex",
        _parent_variant(root) / "references.bib",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Single-column continuation inputs missing: {missing}")
    parent_state = json.loads(parent_state_path.read_text(encoding="utf-8"))
    if parent_state.get("status") != "HUMAN_WAIT":
        raise RuntimeError("Parent scientific-depth run is not at its recorded human gate")
    hashes = {str(path.relative_to(root)): _sha(path) for path in required}
    _json(run / "checkpoints/repository_discovery.json", {
        "status": "PASS",
        "parent_run_id": PARENT_RUN_ID,
        "parent_gate": parent_state.get("gate_id"),
        "parent_hashes": hashes,
        "frozen_figure_hashes": _freeze_hashes(root),
    })
    return {
        "run_id": RUN_ID,
        "parent_run_id": PARENT_RUN_ID,
        "repository_discovery_complete": True,
        "parent_source_hashes_before": hashes,
        "frozen_figure_hashes_before": _freeze_hashes(root),
        "workflow_status": "repository_discovery_complete",
    }


def _results(output: Path, barrier: bool) -> str:
    pieces = [r"\section{Results}"]
    for first, second, figure in _result_blocks(output, "B"):
        pieces.extend([first, figure, second])
        if barrier:
            pieces.append(r"\FloatBarrier")
    if not barrier:
        pieces.append(r"\FloatBarrier")
    return "\n\n".join(pieces)


def build_sources(state: dict[str, Any]) -> dict[str, Any]:
    if not state.get("repository_discovery_complete"):
        raise RuntimeError("Repository discovery must complete first")
    root = _root(state)
    variant = _variant(root)
    output = _output(root)
    if variant.exists():
        shutil.rmtree(variant)
    variant.mkdir(parents=True)
    (variant / "sections").mkdir()
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)

    parent = _parent_variant(root)
    for name in ["abstract.tex", "introduction.tex", "discussion.tex"]:
        shutil.copy2(parent / "sections" / name, variant / "sections" / name)
    shutil.copy2(parent / "references.bib", variant / "references.bib")
    figure_sources = {
        1: root / "figures/fig1_engine_v3/fig1_engine_v3.pdf",
        2: root / "figures/fig2_strict_data/fig2_strict_data.pdf",
        3: root / "figures/fig3_strict_data/fig3_strict_data.pdf",
        4: root / "figures/fig4_strict_data_v2/fig4_strict_data_v2.pdf",
    }
    for number, source in figure_sources.items():
        shutil.copy2(source, output / f"figures/Fig{number}_candidate.pdf")

    _write(variant / "sections/results_direct.tex", _results(output, False))
    _write(variant / "sections/results_barrier.tex", _results(output, True))
    preamble = r"""\documentclass[aps,prl,reprint,superscriptaddress,floatfix]{revtex4-2}
\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{microtype}
\usepackage{amsmath,amssymb,graphicx,xcolor,booktabs,hyperref,placeins}
\hypersetup{colorlinks=true,allcolors=blue}
\begin{document}
\title{Reliability-aware active learning for FFLO phase spaces}
\author{Author list pending human approval}
\date{September 3, 2026}
\begin{abstract}\input{sections/abstract.tex}\end{abstract}
\maketitle
\input{sections/introduction.tex}
"""
    ending = r"""\input{sections/discussion.tex}
\bibliographystyle{apsrev4-2}
\bibliography{references}
\end{document}"""
    _write(variant / "main_direct.tex", preamble + "\\input{sections/results_direct.tex}\n" + ending)
    _write(variant / "main_barrier.tex", preamble + "\\input{sections/results_barrier.tex}\n" + ending)
    return {
        "variant_root": str(variant),
        "variant_output_root": str(output),
        "single_column_sources_ready": True,
        "workflow_status": "single_column_sources_ready",
    }


def compile_and_select(state: dict[str, Any]) -> dict[str, Any]:
    if not state.get("single_column_sources_ready"):
        raise RuntimeError("Single-column sources must be ready")
    root = _root(state)
    run = _run(root)
    variant = Path(state["variant_root"])
    output = Path(state["variant_output_root"])
    tmp = root / "tmp/pdfs/scientific_depth_single_column_v6"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    candidates: dict[str, dict[str, Any]] = {}
    for name in ["direct", "barrier"]:
        out = tmp / name
        out.mkdir()
        shutil.copy2(variant / "references.bib", out / "references.bib")
        pdf = _compile(f"main_{name}.tex", name, variant, out, True)
        metrics = _layout_metrics(root, pdf)
        log = (out / f"{name}.log").read_text(encoding="utf-8", errors="replace")
        metrics.update({
            "float_warnings": len(re.findall(r"float is stuck|Deferred float stuck|Too many unprocessed floats", log, re.I)),
            "overfull_boxes": len(re.findall(r"Overfull \\hbox|Overfull \\vbox", log)),
            "undefined_references": len(re.findall(r"undefined references|undefined citations", log, re.I)),
            "pdf": str(pdf),
        })
        candidates[name] = metrics

    def score(item: tuple[str, dict[str, Any]]) -> tuple[int, int, int, int]:
        _, metric = item
        distances = [value for value in metric["page_distances"] if value is not None]
        return (
            metric["overfull_boxes"] + metric["undefined_references"] + metric["float_warnings"],
            max(distances, default=99),
            metric["max_consecutive_figure_pages"],
            metric["page_count"],
        )

    selected, selected_metrics = min(candidates.items(), key=score)
    other = "barrier" if selected == "direct" else "direct"
    candidate_pdf = output / "manuscript_single_column_candidate.pdf"
    alternative_pdf = Path(candidates[other]["pdf"])
    shutil.copy2(Path(selected_metrics["pdf"]), candidate_pdf)
    shutil.copy2(variant / f"main_{selected}.tex", output / "manuscript_single_column_candidate.tex")
    shutil.copy2(variant / "references.bib", output / "references.bib")

    comparison = {
        "selected": selected,
        "selected_metrics": selected_metrics,
        "alternative": other,
        "alternative_metrics": candidates[other],
        "human_instruction": "Fig. 1 full width; Figs. 2--4 single column",
        "readability_caveat": "Direct scaling of 178-mm registered artwork to one column reduces internal text size; accepted here as an explicit human layout choice.",
    }
    _json(run / "reports/single_column_layout_comparison.json", comparison)
    _write(run / "reports/single_column_layout_comparison.md", "# Single-column layout comparison\n\n```yaml\n" + __import__("yaml").safe_dump(comparison, sort_keys=False) + "```\n")
    return {
        "selected_single_column_layout": selected,
        "single_column_layout_metrics": candidates,
        "single_column_manuscript_pdf": str(candidate_pdf),
        "single_column_alternative_pdf": str(alternative_pdf),
        "single_column_compiled": True,
        "workflow_status": "single_column_compiled",
    }


def audit_and_gate(state: dict[str, Any]) -> dict[str, Any]:
    if not state.get("single_column_compiled"):
        raise RuntimeError("Single-column manuscript must compile first")
    root = _root(state)
    run = _run(root)
    output = _output(root)
    variant = _variant(root)
    selected = state["selected_single_column_layout"]
    results = (variant / f"sections/results_{selected}.tex").read_text(encoding="utf-8")
    figure_modes = {
        number: ("figure*" if number == 1 else "figure")
        for number in range(1, 5)
    }
    mode_pass = results.count(r"\begin{figure*}") == 1 and results.count(r"\begin{figure}") == 3
    frozen_pass = state["frozen_figure_hashes_before"] == _freeze_hashes(root)
    parent_required = [
        _parent_run(root) / "final_state.json",
        _parent_variant(root) / "sections/abstract.tex",
        _parent_variant(root) / "sections/introduction.tex",
        _parent_variant(root) / "sections/discussion.tex",
        _parent_variant(root) / "references.bib",
    ]
    current_parent_hashes = {str(path.relative_to(root)): _sha(path) for path in parent_required}
    parent_pass = state["parent_source_hashes_before"] == current_parent_hashes
    metrics = state["single_column_layout_metrics"][selected]
    compile_pass = metrics["overfull_boxes"] == 0 and metrics["undefined_references"] == 0
    if not all([mode_pass, frozen_pass, parent_pass, compile_pass]):
        raise RuntimeError("Single-column placement audit failed")

    audit = {
        "status": "PASS_WITH_HUMAN_READABILITY_CAVEAT",
        "figure_modes": figure_modes,
        "selected_layout": selected,
        "compiled_metrics": metrics,
        "scientific_text_changed": False,
        "captions_changed": False,
        "references_changed": False,
        "supplement_changed": False,
        "frozen_figures_changed": False,
        "direct_scale_warning": True,
    }
    _json(run / "reports/single_column_placement_audit.json", audit)
    review_tex = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=22mm]{{geometry}}\usepackage{{hyperref,pdfpages}}
\begin{{document}}\section*{{Single-column figure-placement review}}
Fig. 1 remains double column. Figures 2--4 are ordinary single-column floats
placed after their first substantive paragraphs. Scientific prose, captions,
references, frozen artwork, and the Supplemental Material are unchanged.
The 178-mm assets are directly scaled to one column by explicit human request;
review internal figure text at print size before publication freeze.
\includepdf[pages=-]{{\detokenize{{{Path(state['single_column_manuscript_pdf']).as_posix()}}}}}
\end{{document}}"""
    _write(output / "single_column_placement_review.tex", review_tex)
    review_pdf = _compile("single_column_placement_review.tex", "single_column_placement_review", output, output, False)
    gate = {
        "gate_id": GATE_ID,
        "status": "HUMAN_WAIT",
        "run_id": RUN_ID,
        "parent_run_id": PARENT_RUN_ID,
        "selected_layout": selected,
        "manuscript_pdf": state["single_column_manuscript_pdf"],
        "review_package_pdf": str(review_pdf),
        "scientific_freeze": False,
        "publication_freeze": False,
        "submission_ready": False,
    }
    _json(run / f"gate_records/{GATE_ID}.json", gate)
    _json(run / "final_state.json", gate)
    _json(run / "run_state.json", gate)
    _write(output / "decision_log.md", """# Decision log

- Fig. 1 remains a double-column figure.
- Figures 2--4 are inserted as single-column figures after their corresponding first substantive Results paragraphs.
- No scientific prose, caption, reference, Supplement, frozen artwork, or physical result changed.
- Direct single-column scaling remains a print-readability caveat pending human inspection.
- The run stops at a human layout-review gate.
""")
    return {
        "compiled_review_package_pdf": str(review_pdf),
        "workflow_status": "human_review_wait",
        "current_gate": GATE_ID,
        "gate_status": "HUMAN_WAIT",
    }
