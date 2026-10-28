"""Paired author-review profile of the existing four-node placement graph.

Only manuscript assembly is performed here: no physics or artwork builders.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from review_agent.figure_authority import CANONICAL_MAIN_FIGURES_STAGEIV_REVISION as FIGURES
from review_agent.publication_reproducibility import TITLE

RUN = "active_learning_synchronized_review_v7"
VARIANT = "manuscript_variants/synchronized_review_v7"
OUTPUT = "output/pdf/manuscript_synchronized_review_v7"
ARCHIVE = "downloads/active_learning_scientific_evidence_archive_20260813/active_learning_scientific_evidence_archive"
JOBS = ("manuscript_double_column", "manuscript_single_column_review", "supplement")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def record(path, value):
    write(path, json.dumps(value, indent=2, ensure_ascii=False))


def verify_figures(root):
    for item in FIGURES.values():
        if sha(root / item["source_pdf"]) != item["pdf_sha256"]:
            raise RuntimeError("Wrong or changed approved artwork: " + item["source_pdf"])


def repository_discovery(state):
    root = Path(state["project_root"])
    verify_figures(root)
    paths = []
    for parent in ("manuscript_variants/scientific_depth_v5", "manuscript_variants/scientific_depth_single_column_v6"):
        paths.extend(p for p in (root / parent).rglob("*") if p.is_file())
    paths.extend(root / item["source_pdf"] for item in FIGURES.values())
    paths.extend((root / "figures/fig4_strict_data_v2/canonical").glob("*.csv"))
    paths.extend((root / "figures/fig3_strict_data/canonical").glob("*.csv"))
    paths.extend((root / "figures/supplementary_publication_methods").glob("S*/*.pdf"))
    paths.extend((root / "figures/supplementary_publication_methods_v7").glob("S*/*.pdf"))
    paths.append(root / "figures/supplementary_publication_methods_v7/figure_manifest.json")
    paths.append(root / "figures/build_readable_supplement_methods.py")
    paths.append(root / "figures/sm_stageiv_numerical_diagnostics/sm_stageiv_numerical_diagnostics.pdf")
    paths.extend((root / ARCHIVE / "data_archive").glob("stage_[iiv]*/**/run_config.json"))
    paths += [root / "output/pdf/manuscript_scientific_depth_single_column_v6/manuscript_single_column_candidate.pdf",
              root / "output/pdf/manuscript_scientific_depth_v5/supplement_scientific_depth_candidate.pdf",
              root / "tmp/pdfs/scientific_depth_single_column_v6/barrier/barrier.fls",
              root / "output/pdf/manuscript_scientific_depth_v5/supplement_scientific_depth_candidate.fls",
              root / "manuscript_langgraph/runs/active_learning_prl_v1/gate_records/manuscript_figure_source_repair_approval.json"]
    for stage in ("canonical", "stage_iii", "stage_iv"):
        for name in ("models", "acquisition", "config", "topology_oracle", "exact_oracle", "stageiv_3d"):
            p = root / ARCHIVE / "code_archive" / stage / "ml_phase" / (name + ".py")
            if p.exists():
                paths.append(p)
    hashes = {str(p.relative_to(root)): sha(p) for p in paths}
    out = root / OUTPUT
    record(out / "input_manifest.json", {"parents": ["scientific_depth_v5", "scientific_depth_single_column_v6"],
           "hashes": hashes, "figures": FIGURES, "physics_recomputed": False,
           "main_source": "manuscript_variants/scientific_depth_single_column_v6/main_barrier.tex",
           "supplement_source": "manuscript_variants/scientific_depth_v5/supplement.tex"})
    return {"repository_discovery_complete": True, "parent_source_hashes_before": hashes,
            "variant_root": str(root / VARIANT), "variant_output_root": str(out), "run_id": RUN}


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError("Parent content changed; review replacement: " + old[:70])
    return text.replace(old, new, 1)


def main_sections(root, source):
    parent = root / "manuscript_variants/scientific_depth_v5/sections"
    intro = (parent / "introduction.tex").read_text()
    intro = intro.replace("using 8.89\\% of the prescribed candidate locations", "from a sparse subset of the prescribed candidate locations")
    intro = intro.replace("fixed-$\\mu$ slices", "narrow-$\\mu$ bins")
    intro += "\nThe detailed model, algorithms, counting conventions, and numerical diagnostics are provided in the Supplemental Material~\\cite{SupplementalMaterial}.\n"
    intro = intro.replace("The search for FFLO structure", "\\LayoutAudit\nThe search for FFLO structure")
    results = (root / "manuscript_variants/scientific_depth_single_column_v6/sections/results_barrier.tex").read_text()
    results = results.replace(r"\FloatBarrier", "")
    results = results.replace(r"\begin{figure}[tbp]", r"\begin{figure*}[t]").replace(r"\end{figure}", r"\end{figure*}")
    results = results.replace(r"\begin{figure*}[tbp]", r"\begin{figure*}[t]")
    results = re.sub(r"width=0\.(?:96\\textwidth|98\\columnwidth)", r"width=\\textwidth", results)
    results = re.sub(r"\\detokenize\{[^}]+/figures/(Fig[1-4]_candidate.pdf)\}", r"figures/\1", results)
    results = results.replace("Model, physical hierarchy, and exact authority", "Model and physical hierarchy")
    results = results.replace("retain exact authority", "retain direct BdG evaluation")
    results = results.replace("The altermagnetic and spin-orbit BdG model generates", "Schematic illustration, not computed phase data: the altermagnetic and spin-orbit BdG model generates")
    results = results.replace(" Numerical histories. The 8.89\\% and 11.24-fold headline refer specifically to the Stage-I endpoint and prescribed candidate grid.", " Numerical histories; the location-count comparison uses Stage I, not the Stage-II continuation.")
    results = results.replace("Numerical histories and neighborhood sensitivity are retained in the artwork; their detailed values are reported in the Supplemental Material.", r"Numerical histories and neighborhood sensitivity (Supplemental Material, Sec.~\ref{sm-sec:topology}).")
    results = results.replace("The four exact fixed-$\\mu$ cuts", "The four narrow-$\\mu$ bins of exact observations")
    results = results.replace("Four registered fixed-$\\mu$ bins without interpolation.", r"Four narrow-$\mu$ bins (centers $-0.45,0.15,0.85,1.45$, half-width 0.025) without interpolation; see Supplemental Material, Sec.~\ref{sm-sec:3d}.")
    results = results.replace("a comparison with matched random sampling.", r"a comparison with matched random sampling. Selected requests and retained coordinates are distinguished in Supplemental Material, Table~\ref{sm-tab:counts}; neither is a complete execution ledger.")
    results = results.replace(r"\begin{equation}(\Delta", r"\begin{equation}\label{eq:minimum}(\Delta")
    results = results.replace(r"\begin{equation}(-1)", r"\begin{equation}\label{eq:invariant}(-1)")
    # Queue each ordinary wide float immediately before its discussion, with no barriers.
    blocks = results.split(r"\textit{")
    reordered = [blocks[0]]
    for block in blocks[1:]:
        fig = re.search(r"\\begin\{figure\*\}.*?\\end\{figure\*\}", block, re.S)
        if fig:
            reordered.append(fig.group(0) + "\n\n" + r"\textit{" + block.replace(fig.group(0), ""))
        else:
            reordered.append(r"\textit{" + block)
    results = "\n".join(reordered)
    discussion = (parent / "discussion.tex").read_text()
    discussion = discussion.replace("because the exact calculation retains authority at every layer", "because physical labels are obtained from the BdG calculation at every layer")
    discussion = replace_once(discussion, "The Stage-I endpoint evaluates 8.89\\% of the prescribed two-dimensional grid, an 11.24-fold reduction in evaluated locations.  That result", "The location-count economy")
    discussion = discussion.replace("exact many-body-inspired numerical objectives", "direct BdG numerical objectives")
    discussion = discussion.replace("Exploratory learned-acquisition studies have not established an overall winner, and their candidate nodes are not new phases.", "Exploratory learned-acquisition studies have not established superiority over the production acquisition.")
    discussion += "\n\\paragraph{Data availability.} The numerical data and source configurations are preserved in the project archive. No public repository or persistent identifier has been established for this review version.\n"
    abstract = (parent / "abstract.tex").read_text().replace("fixed-$\\mu$ slices", "narrow-$\\mu$ bins")
    abstract = abstract.replace("Extending the same procedure", "Extending this hierarchy")
    abstract = replace_once(abstract, r"while evaluating 6,880 of 77,361 prescribed candidate locations, an exact-location sampling fraction of 8.89\% (11.24-fold reduction)", "with a sparse allocation of exact calculations")
    for name, text in {"abstract": abstract, "introduction": intro, "results": results, "discussion": discussion}.items():
        write(source / "sections" / (name + ".tex"), text)


def algorithm(number, title, inputs, outputs, steps):
    return (rf"\refstepcounter{{algorithm}}\label{{alg:{number}}}\paragraph{{Algorithm \thealgorithm: {title}.}}" + "\n" +
            rf"\textbf{{Input:}} {inputs}\par\textbf{{Output:}} {outputs}" + "\n\\begin{enumerate}\n" +
            "\n".join(r"\item " + step for step in steps) + "\n\\end{enumerate}\n")


def supplement_section(root, source):
    sm = (root / "manuscript_variants/scientific_depth_v5/sections/supplement.tex").read_text()
    # Keep the complete first section (including Hamiltonian, F_impl, and discussion) verbatim.
    sm = sm.replace(r"\section{Topology diagnosis and reconstruction sensitivity}", r"\section{Topology diagnosis and reconstruction sensitivity}\label{sec:topology}")
    sm = sm.replace(r"\section{Three-dimensional extension}", r"\section{Three-dimensional extension}\label{sec:3d}")
    sm = sm.replace(r"\section{Sampling efficiency}", r"\section{Sampling efficiency}\label{sec:counts}")
    sm = sm.replace(r"\caption{Stage-resolved exact-call accounting.}", r"\caption{Stage-resolved selection and retained-coordinate accounting.}\label{tab:counts}")
    sm = sm.replace("displayed in Fig. 2", r"displayed in main-text Fig.~\ref{main-fig:thermo}")
    sm = sm.replace("Frozen candidate-universe", "Stage-I candidate-universe")
    alg1 = algorithm("oracle", "Thermodynamic and topology evaluation", "requested coordinate, numerical configuration, and enabled oracle mode.", "observables, phase label, numerical status, and applicable topology fields.", [
        r"For each coordinate, construct the BdG spectra on the configured $(\Delta,q)$ grid and evaluate the unchanged objective in Sec.~\ref{sec:model}.",
        r"Locate candidate minima. In the adaptive-box path, test enclosure; while allowed and necessary, expand the $q$ window directionally (at most three expansions). Refine selected global, nearly degenerate, and edge-risk basins using the configured grids (at least 800 $q$ and 600 $\Delta$ points in this path).",
        r"Apply the near-zero-$\Delta$ comparison. Assign thermodynamic identity separately from training eligibility and unresolved/rerun flags; the incremental oracle follows its recorded refinement configuration.",
        r"If the topology calculation is enabled and thermodynamic/numerical applicability holds, scan the full-zone gap. Only a resolved gapped FFLO state receives the Pfaffian index; otherwise return the relevant applicability or unresolved status.",
        "Return records with numerical metadata. Coordinate reconciliation occurs at dataset merge; it is not a physical classification rule."])
    alg2 = algorithm("loop", "Stage-resolved active-learning loop", r"stage configuration and initial or resumed exact dataset (Sec.~\ref{sec:stage-settings}).", "updated records, trained models, selected-call tables, and diagnostic history.", [
        "For a cold start, evaluate the stage-specific initial design; for a continuation, load the recorded resume dataset without drawing another seed.",
        "At each iteration, construct the eligible training view and phase-stratified training/validation split. Reinitialize and train both ensembles; standardize inputs and regression targets using training statistics.",
        r"Generate the stage-specific candidate domain, predict observables/probabilities, and score the candidates using Secs.~\ref{sec:acq-definitions} and \ref{sec:stage-selectors}.",
        r"Select a batch using Algorithm~\ref{alg:batch}; write the point shards. Evaluate the BdG oracle and preserve failed or unresolved returns as explicit records.",
        "Merge the completed shards, reconcile duplicate coordinate keys, and update the eligible learning view without overwriting reliability flags.",
        "Recompute the recorded monitoring diagnostics. Continue unless the configured operational stop or budget is reached; save the dataset, configuration, selections, and diagnostics before returning."])
    alg3 = algorithm("batch", "Stage-specific repelled selection", "candidate coordinates, component scores, existing observations, and stage configuration.", "ordered selected batch, probabilities, and selection metadata.", [
        "In 2D, apply exact-coordinate and cooldown masks, interior penalties, and the active-pool threshold. Initialize batch repulsion to one and include observation repulsion.",
        "In 3D, use the proposal pool and channel quotas; use the weighted combined score for exhausted-channel backfill. Apply the recorded duplicate handling.",
        r"While the requested batch is incomplete and positive finite weights exist: form the appropriate score times repulsion, raise to the stage-specific power, normalize, and sample one index.",
        r"Record its probability, remove the selected index, and multiply the remaining batch factors by $K(d;\ell,r)$. Return early if no eligible weight remains; do not invent extra oracle evaluations."])
    for n, body in [(1, alg1), (2, alg2), (3, alg3)]:
        sm, count = re.subn(r"\\paragraph\{Algorithm S" + str(n) + r":.*?(?=\n\\(?:section|begin))", lambda m: body + "\n", sm, count=1, flags=re.S)
        if count != 1:
            raise RuntimeError("Algorithm source mismatch")
    sm = sm.replace(r"\section{Physical model and free-energy functional}", r"\section{Physical model and free-energy functional}\label{sec:model}")
    # Retain the verified 2D description, explicitly limit the architecture illustration.
    sm = sm.replace("Each of five members is", "For Stages I--III, each of five members is")
    sm = sm.replace("Production surrogate architecture.", "Stage-specific surrogate architecture. Stages I--III have two inputs and 64-unit hidden layers; Stage IV has three inputs and 96-unit hidden layers.")
    sm = sm.replace("Surrogate and training hyperparameters.", "Two-dimensional surrogate and training hyperparameters (Stages I--III).")
    sm = sm.replace("The production profile is", "The Stage-I/II full profile is")
    sm = sm.replace(r"$A_{\rm select}=A_0R_{\rm obs}R_{\rm batch}$", r"$A_{\rm select}=A_{\rm pool}R_{\rm obs}R_{\rm batch}$")
    sm = sm.replace("Stage-IV", "Stage-IV")
    details = (root / VARIANT / "method_details.tex").read_text()
    sm = sm.replace(r"\section{Sampling efficiency}", details + "\n\\section{Sampling efficiency}")
    sm = sm.split(r"\section{Reproducibility, code, data, and environment}")[0]
    sm += r"""
\begin{table}[tb]\centering\caption{Main-text Fig.~\ref{main-fig:3d}(c) bin accounting. Each window has half-width 0.025. Trusted here is the numerical-reliability subset, not the gapped-FFLO topology subset.}\label{tab:bins}
\begin{tabular}{rrrrrr}\toprule $\mu/t$ center & Retained & Normal & Uniform SC & FFLO & Trusted\\\midrule
$-0.45$&272&95&1&176&255\\$0.15$&154&53&4&97&149\\$0.85$&223&57&18&148&208\\$1.45$&142&104&3&35&139\\\bottomrule\end{tabular}\end{table}
Main-text Fig.~\ref{main-fig:topology}(a) contains 3,213 trusted gapped FFLO observations (2,290 cFFLO and 923 tFFLO). The main-text reconstruction retains the $k=8$ operational-audit pass and the $k=6$ sensitivity deviation; neither is a universal physical convergence criterion. The four three-dimensional bin totals in Table~\ref{tab:bins} count retained observations, not estimates of phase-space volume.
\begin{figure}[tb]\centering\includegraphics[width=\textwidth]{figures/sm_stageiv_numerical_diagnostics.pdf}
\caption{Stage-IV numerical histories, without threshold lines. (a) Retained and trusted sample growth. (b) Thermodynamic composition. (c) Topology composition. (d) Available topology-map changes. (e) Proxy-surface displacement. (f) Trusted-label surprise. These are iteration diagnostics, not a formally converged three-dimensional topology surface.}\label{fig:3d-diagnostics}\end{figure}
"""
    sm += (root / VARIANT / "reproducibility.tex").read_text()
    sm = re.sub(r"\\detokenize\{[^}]+/figures/(S[1-4]_[^}]+\.pdf)\}", r"figures/\1", sm)
    sm = sm.replace(r"\begin{figure}[h]", r"\begin{figure}[tb]")
    sm = sm.replace(r"\begin{table}[h]", r"\begin{table}[tb]")
    # Full-size tables in a single-column supplement; no resize-to-fit table text.
    sm = sm.replace(r"\resizebox{\columnwidth}{!}{\begin{tabular}", r"\begin{tabular}")
    sm = sm.replace(r"\resizebox{\textwidth}{!}{\begin{tabular}", r"\begin{tabular}")
    sm = sm.replace(r"\end{tabular}}", r"\end{tabular}")
    sm = sm.replace(r"\centering\small", r"\centering")
    sm = sm.replace(r"\centering\begin{tabular}", r"\centering\setlength{\tabcolsep}{5pt}\begin{tabular}")
    sm = sm.replace("\\centering\n\\caption{Stage-specific", "\\centering\\setlength{\\tabcolsep}{5pt}\n\\caption{Stage-specific")
    sm = sm.replace(r"$\Psi_k=", r"\[\Psi_k=").replace(r"})^T$ with", r"})^T\] with")
    sm = sm.replace(r"\resizebox{0.98\columnwidth}{!}{$H", "H").replace(r"\end{pmatrix}$}.", r"\end{pmatrix}.")
    sm = sm.replace(r"\begin{tabular}{lllll}", r"\begin{tabular}{p{0.16\textwidth}p{0.20\textwidth}p{0.18\textwidth}p{0.17\textwidth}p{0.15\textwidth}}")
    sm = sm.replace("No interpolation is performed between slices.", "No interpolation is performed between these bins.")
    write(source / "sections/supplement.tex", sm)


def build_sources(state):
    root = Path(state["project_root"]); out = root / OUTPUT; source = out / "source"
    source.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root))
    from figures.figure_engine.builders.figure_build import preflight
    for item in FIGURES.values():
        p = root / item["source_pdf"]
        contract = p.parent / {"1": "fig1_contract.yaml", "2": "fig2_panel_contract.yaml", "3": "fig3_panel_contract.yaml", "4": "figure_contract.yaml"}[item["number"]]
        preflight(contract, record_path=out / "checks" / ("fig" + item["number"] + "_preflight.json"))
        (source / "figures").mkdir(exist_ok=True)
        shutil.copy2(p, source / "figures" / item["staged_pdf"])
    method_manifest = root / "figures/supplementary_publication_methods_v7/figure_manifest.json"
    if not method_manifest.exists():
        raise RuntimeError("Build the engine-validated native-width SM derivatives first")
    method_hashes = {r["key"]: r["sha256"] for r in json.loads(method_manifest.read_text())["figures"]}
    for folder in sorted((root / "figures/supplementary_publication_methods_v7").glob("S[1-4]_*")):
        if sha(folder / (folder.name + ".pdf")) != method_hashes[folder.name]:
            raise RuntimeError("SM derivative differs from the validated manifest")
        preflight(folder / "figure_contract.yaml", record_path=out / "checks" / (folder.name + "_preflight.json"))
        shutil.copy2(folder / (folder.name + ".pdf"), source / "figures" / (folder.name + ".pdf"))
    diag = root / "figures/sm_stageiv_numerical_diagnostics"
    preflight(diag / "figure_contract.yaml", record_path=out / "checks/stageiv_diagnostics_preflight.json")
    shutil.copy2(diag / "sm_stageiv_numerical_diagnostics.pdf", source / "figures/sm_stageiv_numerical_diagnostics.pdf")
    main_sections(root, source)
    supplement_section(root, source)
    bib = (root / "manuscript_variants/scientific_depth_v5/references.bib").read_text()
    bib += '\n@misc{SupplementalMaterial, note={See Supplemental Material at [URL will be inserted by publisher] for the model, stage-specific algorithms, numerical diagnostics, and selection accounting.}}\n'
    write(source / "references.bib", bib)
    preamble = r"""\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{microtype}
\usepackage{amsmath,amssymb,graphicx,booktabs,array,url,xr}
\usepackage{hyperref}
\hypersetup{colorlinks=true,allcolors=blue}
\input{metadata.tex}
\newcommand{\LayoutAudit}{\typeout{LAYOUT-TEXTWIDTH=\the\textwidth}\typeout{LAYOUT-COLUMNWIDTH=\the\columnwidth}}
"""
    write(source / "metadata.tex", rf"\newcommand{{\ManuscriptTitle}}{{{TITLE}}}" + "\n% Author/affiliation block intentionally absent: not yet confirmed.\n")
    write(source / "preamble.tex", preamble)
    body = r"""\begin{document}\title{\ManuscriptTitle}
\begin{abstract}\input{sections/abstract.tex}\end{abstract}\maketitle
\input{sections/introduction.tex}
\input{sections/results.tex}
\input{sections/discussion.tex}
\bibliographystyle{apsrev4-2}\bibliography{references}
\end{document}"""
    write(source / "main_content.tex", body)
    for job, mode in [(JOBS[0], "reprint"), (JOBS[1], "reprint,onecolumn,11pt,tightenlines")]:
        write(source / (job + ".tex"), rf"\documentclass[aps,prl,{mode},superscriptaddress,floatfix]{{revtex4-2}}" + "\n\\input{preamble.tex}\n\\externaldocument[sm-]{supplement}\n\\input{main_content.tex}\n")
    write(source / "supplement.tex", r"""\documentclass[aps,prl,reprint,onecolumn,11pt,tightenlines,floatfix]{revtex4-2}
\input{preamble.tex}
\externaldocument[main-]{manuscript_double_column}
\setcounter{secnumdepth}{3}
\renewcommand{\thesection}{S\arabic{section}}
\renewcommand{\thesubsection}{\thesection.\arabic{subsection}}
\makeatletter\def\p@subsection{}\makeatother
\renewcommand{\theequation}{S\arabic{equation}}
\renewcommand{\thefigure}{S\arabic{figure}}
\renewcommand{\thetable}{S\arabic{table}}
\newcounter{algorithm}\renewcommand{\thealgorithm}{S\arabic{algorithm}}
\begin{document}\title{Supplemental Material: \ManuscriptTitle}\maketitle\LayoutAudit
\input{sections/supplement.tex}
\end{document}""")
    shutil.copy2(root / VARIANT / "build.py", source / "build.py")
    write(source / "README.txt", "Paired review sources. Run: python3 build.py\nRequires REVTeX 4.2, pdfLaTeX and BibTeX. Builds all three PDFs in a new temporary directory before copying the complete set to built/. No physics runs. No prior aux/bbl required.\n")
    record(out / "source_manifest.json", {str(p.relative_to(source)): sha(p) for p in source.rglob("*") if p.is_file() and not any(x in p.relative_to(source).parts for x in ("built", "built_logs", "__pycache__"))})
    return {"publication_sources_ready": True, "workflow_status": "paired_sources_ready"}


def compile_and_select(state):
    root = Path(state["project_root"]); out = root / OUTPUT
    subprocess.run([sys.executable, "build.py"], cwd=out / "source", check=True)
    for job in JOBS:
        shutil.copy2(out / "source/built" / (job + ".pdf"), out / (job + ".pdf"))
    return {"revision_compiled": True, "workflow_status": "paired_compiled"}


def audit_and_gate(state):
    root = Path(state["project_root"]); out = root / OUTPUT
    verify_figures(root)
    changed = [p for p, h in state["parent_source_hashes_before"].items() if sha(root / p) != h]
    if changed:
        raise RuntimeError("Protected parent inputs changed: " + repr(changed))
    result = {"workflow_status": "HUMAN_WAIT", "current_gate": "GATE_SYNCHRONIZED_MANUSCRIPT_AUTHOR_REVIEW", "gate_status": "HUMAN_WAIT",
              "revised_manuscript_pdf": str(out / (JOBS[0] + ".pdf")), "single_column_manuscript_pdf": str(out / (JOBS[1] + ".pdf")),
              "revised_supplement_pdf": str(out / "supplement.pdf"), "physics_changed": False, "protected_inputs_unchanged": True}
    record(out / "build_state.json", result)
    record(root / "manuscript_langgraph/runs" / RUN / "final_state.json", result)
    return result
