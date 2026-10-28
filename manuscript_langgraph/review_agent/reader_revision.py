"""R01--R14 reader revision: existing four-node graph, immutable v7 parent."""
from pathlib import Path
import csv
import difflib
import json
import re
import shutil
import subprocess
import sys

from review_agent.synchronized_review import sha, write, record, replace_once, JOBS, ARCHIVE

PARENT = "output/pdf/manuscript_synchronized_review_v7"
OUTPUT = "output/pdf/manuscript_reader_revision_v8"
VARIANT = "manuscript_variants/reader_revision_v8"
RUN = "active_learning_reader_revision_v8"


def repository_discovery(state):
    root = Path(state["project_root"])
    expected = {"manuscript_single_column_review.pdf": "1be63af165dfd6fd3a55458f0414435caaac1ce4a4ab8b7caa87240f08a9d820",
                "supplement.pdf": "e6a27f69a09569f84a5a07a820b356c573c2e44d5a2b7b41d37989eff2e75966"}
    for name, digest in expected.items():
        if sha(root/PARENT/name) != digest:
            raise RuntimeError("Reviewed v7 input changed: " + name)
    protected = {str(p.relative_to(root)): sha(p) for p in (root/PARENT).rglob("*") if p.is_file()}
    old = json.loads((root/PARENT/"input_manifest.json").read_text())
    for item in old["figures"].values():
        protected[item["source_pdf"]] = sha(root/item["source_pdf"])
    record(root/OUTPUT/"input_manifest.json", {"reviewed_parent": PARENT, "hashes": protected, "figures": old["figures"], "physics_recomputed": False})
    return {"repository_discovery_complete": True, "parent_source_hashes_before": protected,
            "variant_root": str(root/VARIANT), "variant_output_root": str(root/OUTPUT), "run_id": RUN}


def build_sources(state):
    root=Path(state["project_root"]); out=root/OUTPUT; source=out/"source"
    shutil.copytree(root/PARENT/"source",source,dirs_exist_ok=True,ignore=shutil.ignore_patterns("built","built_logs","__pycache__"))
    fig=root/"figures/fig2_reader_revision_v8"
    audit=json.loads((fig/"figure_validation_report.json").read_text())
    if audit["status"] != "pass": raise RuntimeError("Fig.2 engine lint failed")
    shutil.copy2(fig/"fig2_strict_data.pdf",source/"figures/Fig2_candidate.pdf")
    manifest=json.loads((out/"input_manifest.json").read_text())
    manifest["supplement_derivatives"] = []
    for key in ("S2_surrogate_architecture", "S3_acquisition_decomposition"):
        folder=root/"figures/supplementary_publication_methods_v8"/key
        if json.loads((folder/"figure_validation_report.json").read_text())["status"] != "pass":
            raise RuntimeError("Compact method figure failed lint: "+key)
        shutil.copy2(folder/(key+".pdf"),source/"figures"/(key+".pdf"))
        manifest["supplement_derivatives"].append({"key":key,"source_pdf":str((folder/(key+".pdf")).relative_to(root)),"pdf_sha256":sha(folder/(key+".pdf")),"approval":"HUMAN_WAIT"})
    item=next(x for x in manifest["figures"].values() if str(x["number"])=="2")
    item.update(source_pdf=str((fig/"fig2_strict_data.pdf").relative_to(root)),source_png=str((fig/"fig2_strict_data.png").relative_to(root)),pdf_sha256=sha(fig/"fig2_strict_data.pdf"),approval="v8_candidate_HUMAN_WAIT")
    record(out/"input_manifest.json",manifest)
    results=(source/"sections/results.tex").read_text()
    results=replace_once(results,"The single-particle sector is",r"We use $\lambda_y/t=\lambda_z/t=0.6$ and $U/t=1.6$; the two-dimensional results fix $\mu/t=0.55$. The single-particle sector is")
    results=results.replace("Exact-call allocation across the Stage-I discovery and Stage-II reliability continuation.","Exact-call allocation across Stage I and its Stage-II continuation; darker red denotes later iterations (1--34), not a phase or probability.")
    results=results.replace("Numerical histories; the location-count comparison", "Numerical histories; dashed lines are the saved operational references described in the Supplemental Material, not physical scales. The location-count comparison")
    # Keep author-dependent topology/Ansatz statements unchanged pending the specific gates.
    discussion=(source/"sections/discussion.tex").read_text()
    discussion=discussion.replace("no matched exhaustive timing baseline is archived", "no matched exhaustive timing baseline was established")
    discussion=discussion.replace("the candidate-wide boundary denominator is unavailable", "boundary membership was not recorded for the complete candidate set")
    start=discussion.index("The broader significance")
    end=discussion.index(r"\paragraph{Data availability.}", start)
    discussion=discussion[:start]+"The method combines adaptive sampling with direct BdG evaluation while keeping numerical reliability separate from phase identity. The results establish finite thermodynamic and topology-resolved sampled supports; determining continuous boundaries requires additional evidence beyond those observations.\n\n"+discussion[end:]
    sm=(source/"sections/supplement.tex").read_text()
    sm=sm.replace("(at most three expansions)","(up to the invocation-specific limit)")
    sm=sm.replace("the incremental oracle follows its recorded refinement configuration", "incremental scanning reuses previously evaluated momentum intervals in the same robust core")
    sm=sm.replace(r"\section{Reliability-state hierarchy}",(root/VARIANT/"oracle_details.tex").read_text()+"\n\\section{Reliability-state hierarchy}")
    sm=sm.replace("and topology is restricted to reliable gapped FFLO states.","and the main-figure topology subset is restricted to reliable gapped FFLO states.")
    sm=sm.replace("Only a resolved gapped FFLO state receives the Pfaffian index", "Only a resolved gapped state meeting that invocation's applicability conditions receives the Pfaffian index; the online superconducting scope and the FFLO-only main-figure subset are distinguished below")
    sm=sm.replace("Topology-inapplicable includes all non-FFLO or numerically unreliable states", "For the FFLO-only main-figure presentation, topology-inapplicable includes non-FFLO or numerically unreliable states; this display restriction is not the broader online superconducting filter described above")
    sm=sm.replace(r"and $\tau=k_BT/t$, $j=J_A/t$",r"and $j=J_A/t$")
    sm=sm.replace("the active pool is set by the larger of a scheduled quantile and $0.7p_{95}(A_0)$",r"the active pool uses quantiles of the penalized $A_{\rm pool}$ over eligible hard-unseen candidates, as defined in Sec.~\ref{sec:acq-definitions}")
    sm=sm.replace("In 2D, apply exact-coordinate and cooldown masks, interior penalties, and the active-pool threshold.","In 2D discovery, apply exact-coordinate exclusion, interior penalties, and the active-pool threshold; the recent and boundary-band cooldown masks are disabled in this mode.")
    sm=sm.replace("configured recent-selection and normal-boundary-band cooldowns also apply.","recent-selection and normal-boundary-band cooldowns are disabled by the discovery branch, despite their configuration fields.")
    sm=sm.replace(r"\partial_{T/t}",r"\partial_\tau")
    sm=sm.replace("In this equation the temperature coordinate denotes $k_BT/t$.",r"Here $\tau=k_BT/t$.")
    sm=sm.replace(r"\section{Sampling efficiency}",(root/VARIANT/"selector_details.tex").read_text()+"\n\\section{Sampling efficiency}")
    sm=sm.replace(r"\section{Topology diagnosis and reconstruction sensitivity}",(root/VARIANT/"diagnostic_details.tex").read_text()+"\n\\section{Topology diagnosis and reconstruction sensitivity}")
    sm=sm.replace(r"\caption{Operational diagnostics.}",r"\caption{Two-dimensional online thermodynamic operational diagnostics; not the Stage-III offline topology or Stage-IV controller.}")
    sm=sm.replace("For neighborhoods $k=6,8,12$","For neighbor counts $k_{\\rm nn}=6,8,12$")
    sm=sm.replace("$k=8$ operational-audit pass and the $k=6$",r"$k_{\rm nn}=8$ operational-audit pass and the $k_{\rm nn}=6$")
    # Keep mathematical units/algorithms atomic, without blanket float barriers.
    sm=re.sub(r"(\\refstepcounter\{algorithm\}.*?\\end\{enumerate\})",r"\\noindent\\begin{minipage}{\\textwidth}\n\1\n\\end{minipage}\\par",sm,flags=re.S)
    sm=sm.replace(r"$B_{UF}=P_{SC}\exp[-||\hat q|-q_\epsilon|/0.04]$",r"\mbox{$B_{UF}=P_{SC}\exp[-||\hat q|-q_\epsilon|/0.04]$}")
    sm=sm.replace("and sampling probability proportional to",r"and \mbox{sampling probability proportional to}")
    sm=sm.replace(r"Let $U_{\rm cls}=",r"\begin{samepage}"+"\n"+r"Let $U_{\rm cls}=")
    sm=sm.replace("Existing exact coordinates are excluded before selection.","Existing exact coordinates are excluded before selection.\n\\end{samepage}")
    # In the one-column SM, place each existing figure/table at its semantic
    # anchor. Atomic blocks are not forced onto separate pages and cannot jump
    # into a preceding sentence. Main wide floats remain untouched.
    sm=re.sub(r"\\begin\{figure\}(?:\[[^]]*\])?",r"\\begin{AnchoredFigure}",sm)
    sm=sm.replace(r"\end{figure}",r"\end{AnchoredFigure}")
    sm=re.sub(r"\\begin\{table\*?\}(?:\[[^]]*\])?",r"\\begin{AnchoredTable}",sm)
    sm=re.sub(r"\\end\{table\*?\}",r"\\end{AnchoredTable}",sm)
    wrapper=(source/"supplement.tex").read_text()
    environments=r"""\makeatletter
\newenvironment{AnchoredFigure}{\par\addvspace{10pt}\noindent\begin{minipage}{\textwidth}\def\@captype{figure}}{\end{minipage}\par\addvspace{10pt}}
\newenvironment{AnchoredTable}{\par\addvspace{8pt}\noindent\begin{minipage}{\textwidth}\def\@captype{table}}{\end{minipage}\par\addvspace{8pt}}
\makeatother
"""
    wrapper=wrapper.replace(r"\begin{document}",environments+r"\begin{document}\raggedbottom")
    write(source/"supplement.tex",wrapper)
    for name,text in [("results",results),("discussion",discussion),("supplement",sm)]:write(source/"sections"/(name+".tex"),text)
    # Save a literal source diff, not only a success summary.
    diffs=[]
    for name in ("results","discussion","supplement"):
        a=(root/PARENT/"source/sections"/(name+".tex")).read_text().splitlines(True)
        b=(source/"sections"/(name+".tex")).read_text().splitlines(True)
        diffs.extend(difflib.unified_diff(a,b,fromfile="v7/"+name,tofile="v8/"+name))
    write(out/"source_changes.diff","".join(diffs))
    return {"publication_sources_ready": True,"workflow_status":"reader_revision_sources_ready"}


def compile_and_select(state):
    out=Path(state["project_root"])/OUTPUT
    subprocess.run([sys.executable,"build.py"],cwd=out/"source",check=True)
    for job in JOBS:shutil.copy2(out/"source/built"/(job+".pdf"),out/(job+".pdf"))
    return {"revision_compiled":True,"workflow_status":"reader_revision_compiled"}


def audit_and_gate(state):
    root=Path(state["project_root"]);out=root/OUTPUT
    changed=[p for p,h in state["parent_source_hashes_before"].items() if sha(root/p)!=h]
    if changed:raise RuntimeError("Protected input changed: "+repr(changed))
    def protected_energy(path):
        text=path.read_text();a=text.index("The production code evaluates");b=text.index("with energies in units of $t$.",a)
        return text[a:b+len("with energies in units of $t$.")]
    if protected_energy(root/PARENT/"source/sections/supplement.tex") != protected_energy(out/"source/sections/supplement.tex"):
        raise RuntimeError("Protected free-energy passage changed")
    result={"workflow_status":"HUMAN_WAIT","current_gate":"GATE_READER_REVISION_SCIENCE_CHOICES","gate_status":"HUMAN_WAIT",
            "revised_manuscript_pdf":str(out/(JOBS[0]+".pdf")),"single_column_manuscript_pdf":str(out/(JOBS[1]+".pdf")),"revised_supplement_pdf":str(out/"supplement.pdf"),
            "physics_changed":False,"protected_inputs_unchanged":True,"free_energy_passage_unchanged":True,
            "author_dependent_items":["R01","R02","R03 Ansatz/basis","R11","R12","R13"],"method_validation":"A revised; B scoped source match; C not reproduced"}
    record(out/"build_state.json",result);record(root/"manuscript_langgraph/runs"/RUN/"final_state.json",result)
    return result
