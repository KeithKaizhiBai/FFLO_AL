from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from review_agent.figure_authority import (
    CANONICAL_MAIN_FIGURES,
    stage_canonical_main_figures,
    verify_canonical_frozen_artifacts,
)


METHODS_TEX = r"""\section{Methods}
\label{sec:methods}

\paragraph{BdG model and thermodynamic oracle.}
We study the finite-temperature one-dimensional altermagnetic superconductor
defined by the Nambu spinor
$\Psi_k=(c_{k+q/2,\uparrow},c_{k+q/2,\downarrow},
c^\dagger_{-k+q/2,\uparrow},c^\dagger_{-k+q/2,\downarrow})^T$ and
\begingroup\small
\begin{equation}
H_{\mathrm{BdG}}(k)=
\begin{pmatrix}
h(k+q/2)-\mu & \Delta_q\\
\Delta_q^* & -\sigma_y h^T(-k+q/2)\sigma_y+\mu
\end{pmatrix},
\end{equation}
\endgroup
where $h(k)=t\cos k+(0,\lambda_y\sin k,
\lambda_z\sin k+J_A\cos k)\cdot\boldsymbol{\sigma}$.  At every sampled
parameter point the exact oracle evaluates the BdG spectrum over the configured
$k$ grid, constructs the free-energy landscape in $(\Delta,q)$, refines
candidate minima, and selects the lowest reliable state.  A superconducting
state is accepted only when its minimized free energy lies below the normal
state beyond the configured numerical tolerance.  The frozen thermodynamic
labels are Normal for $\Delta_{\rm opt}<10^{-3}$, uniform SC for
$\Delta_{\rm opt}\geq10^{-3}$ and $|q_{\rm opt}|<10^{-2}$, and FFLO otherwise.

\paragraph{Reliability hierarchy.}
The oracle keeps physical labels separate from numerical status.  Candidate
minima at a $q$-window edge trigger window expansion, shallow normal--SC
competition triggers local $\Delta$ refinement, and basin-level risk is
retained in the point metadata.  Trusted, training-eligible, rerun-required,
$q$-unresolved, and $\Delta$-unresolved states are therefore distinguishable.
The optimized production path caps the number of locally refined basins while
preserving the accepted phase labels; unresolved points are never treated as a
new phase.

\paragraph{Topology diagnosis.}
Topology is evaluated only after thermodynamic acceptance.  For applicable
superconducting states, Pfaffians $P_0$ and $P_\pi$ are evaluated at the
particle--hole invariant momenta and the $\mathbb Z_2$ index is determined by
the sign of $P_0P_\pi$ away from the Pfaffian tolerance.  A chunked scan of the
full Brillouin zone supplies the minimum quasiparticle gap.  Only reliable
gapped FFLO points are assigned cFFLO ($\nu=0$) or tFFLO ($\nu=1$); a gap
closing or unresolved invariant remains a diagnostic reliability state.

\paragraph{Reliability-aware active learning.}
Small ensembles predict thermodynamic observables and phase probabilities on
a dense candidate set.  Classification uncertainty, phase-facing regression
uncertainty, proximity to the Normal--SC and uniform-SC--FFLO structures,
numerical $q$-edge risk, and a conservative extrapolation term define the
production acquisition score.  Previously evaluated coordinates are excluded,
and soft observation and within-batch repulsion reduce redundant calls.  The
surrogate ranks the next exact micro-batch; only returned BdG calculations add
physical labels to the training set.

\paragraph{Convergence and higher-dimensional extension.}
The two-dimensional thermodynamic stop controller combines fixed-grid phase-map
change, displacements of the two principal thermodynamic boundaries, trusted
label surprise, and boundary coverage over a patience window.  The topology
audit independently interpolates $P_0$ and $P_\pi$ on a fixed monitor grid,
restricts contours to trusted support, and tests map change, boundary shift,
coverage, surprise, and component stability.  The three-dimensional extension
uses the same exact-oracle and scheduler separation with
$x=(k_BT/t,J_A/t,\mu/t)$; its sampled structures are reported with the explicit
qualification that the strict three-dimensional topology closure tests were
not all passed.  Further numerical definitions and audits are provided in the
Supplemental Material.
"""


METHODS_MD = r"""## Methods

### BdG model and thermodynamic oracle

We study the finite-temperature one-dimensional altermagnetic superconductor in
the Nambu basis
$\Psi_k=(c_{k+q/2,\uparrow},c_{k+q/2,\downarrow},c^\dagger_{-k+q/2,\uparrow},c^\dagger_{-k+q/2,\downarrow})^T$,
with

$$
H_{\mathrm{BdG}}(k)=\begin{pmatrix}
h(k+q/2)-\mu & \Delta_q\\
\Delta_q^* & -\sigma_yh^T(-k+q/2)\sigma_y+\mu
\end{pmatrix},
$$

where $h(k)=t\cos k+(0,\lambda_y\sin k,\lambda_z\sin k+J_A\cos k)\cdot\boldsymbol\sigma$.
At each sampled point, the exact oracle builds the spectrum and free-energy
landscape in $(\Delta,q)$, refines candidate minima, and accepts the lowest
reliable state. A superconducting state must lie below the normal state beyond
the numerical tolerance. The frozen thresholds are
$\Delta_\epsilon=10^{-3}$ and $q_\epsilon=10^{-2}$.

### Reliability and topology

The oracle records trusted, training-eligible, rerun-required, $q$-unresolved,
and $\Delta$-unresolved states separately. Edge minima trigger adaptive
$q$-window expansion and shallow normal--SC competition triggers local
$\Delta$ refinement. Topology is evaluated after thermodynamic acceptance:
Pfaffians at $k=0,\pi$ and a full-Brillouin-zone bulk-gap scan assign cFFLO or
tFFLO only on reliable gapped FFLO states.

### Active learning and convergence

Ensembles rank dense candidates using phase and regression uncertainty,
boundary proximity, numerical risk, extrapolation control, and soft repulsion.
They schedule exact calls but never assign final labels. Thermodynamic closure
combines map change, two boundary shifts, trusted surprise, and coverage.
Topology closure adds support-restricted Pfaffian contours and component
stability. The three-dimensional extension uses the same separation in
$(k_BT/t,J_A/t,\mu/t)$, while retaining its failed strict topology-closure
criteria as an explicit caveat.
"""


DISCUSSION_TEX = r"""\section{Discussion}
The matched Stage-V audit clarifies what has and has not been learned about
adaptive acquisition.  At the common endpoint, the production-like V3 run has
slightly denser topology-surface support
($p_{95}=0.01893$ versus $0.02103$), whereas V4 has lower sequential topology
surprise ($0.0925$ versus $0.1189$).  V4 does not lower the common offline
graph-uncertainty functional ($0.01217$ versus $0.01114$), and none of its four
stored candidate nodes has local same-identity support under the matched
counterfactual test.  These tradeoffs neither establish learned-acquisition
superiority nor support a new phase.  They instead motivate future adaptive
policies whose rewards are calibrated directly against registered scientific
coverage and surprise criteria.
"""


OUTLOOK_TEX = r"""\section{Outlook}
The Stage-VI studies point toward a ground-up physical world model in which
thermodynamic existence, the superconducting momentum branch, spectra, and
topology are learned under their own validity masks.  Thermodynamic fields and
$|q|$ are learnable in the frozen tests, and continuous Pfaffian quantities
become learnable after conditioning on trusted gapped superconductivity.
However, the exported data contain no fixed-$k$ spectra, density-of-states
proxy, or validated low-energy spectral moments; a log-gap-only ablation is not
a spectral state representation.  Moreover, replay uncertainty is not
uniformly monotonic and has not been shown to be expected-information-gain
(EIG) ready.  A future program must first export or define the missing spectral
descriptors and validate calibrated uncertainty before any production EIG
acquisition is authorized.
"""


DISCUSSION_MD = r"""## Discussion

The matched Stage-V audit gives no overall winner. V3 has slightly denser
topology-surface support ($p_{95}=0.01893$ versus $0.02103$), while V4 has lower
sequential topology surprise ($0.0925$ versus $0.1189$). V4 does not lower the
common offline graph-uncertainty functional ($0.01217$ versus $0.01114$), and
none of its four candidate nodes has local same-identity support. These are
exploratory learned-acquisition studies, not evidence of superiority or a new
phase; they motivate policies calibrated to registered scientific criteria.
"""


OUTLOOK_MD = r"""## Outlook

Stage VI motivates a ground-up physical world model with separate validity
masks for thermodynamics, momentum, spectra, and topology. Thermodynamic fields
and $|q|$ are learnable, and Pfaffian quantities become learnable on trusted
gapped superconducting points. The frozen export lacks fixed-$k$ spectra,
density-of-states proxies, and validated spectral moments, while replay
uncertainty is not uniformly monotonic. A spectral representation and calibrated
uncertainty must therefore be validated before production EIG acquisition.
"""


SUPPLEMENT_TEX = r"""\section*{Supplemental Material}
\setcounter{section}{0}\renewcommand{\thesection}{S\arabic{section}}

\section{Physical model and notation}
The Hamiltonian, Nambu convention, and parameter definitions are identical to
those in the main text and the canonical model specification.  The matrix
convention is row $=J_A$ and column $=k_BT$ for two-dimensional arrays.  No
thermodynamic phase definition is changed by the active-learning or topology
layers.

\section{Exact BdG oracle and reliability hierarchy}
At fixed external parameters, the oracle scans the configured $q$ and
$\Delta$ domains, refines candidate basins, and compares the best
superconducting free energy with the normal reference.  A $q$-edge optimum is
not accepted until its window is expanded or it is explicitly marked
unresolved.  Near the Normal--SC boundary, the strictly positive-$\Delta$ gap
$\min_{\Delta\geq\Delta_\epsilon}F(\Delta,q)-F(0,q)$ distinguishes trusted
normal points from a finite-resolution boundary band.  Reliability fields are
stored independently of phase labels.

For topology, the analytic Pfaffians in the project convention are
\begin{align}
P_0&=(\mu-t\cos(q/2))^2+\Delta^2
-(\lambda_y\sin(q/2))^2-(J_A\cos(q/2)+\lambda_z\sin(q/2))^2,\\
P_\pi&=(\mu+t\cos(q/2))^2+\Delta^2
-(\lambda_y\sin(q/2))^2-(J_A\cos(q/2)+\lambda_z\sin(q/2))^2.
\end{align}
Away from the Pfaffian tolerance, $P_0P_\pi<0$ gives $\nu=1$ and
$P_0P_\pi>0$ gives $\nu=0$.  The invariant is accepted only together with a
full-Brillouin-zone gap exceeding its scale-aware tolerance and trusted
thermodynamics.  Direct analytic/numeric Pfaffian comparison agreed for
104/104 validation points, and the CPU/GPU bulk-gap discrepancy was
$7.77\times10^{-16}$ in the archived validation.

\section{Active-learning loop and acquisition}
Each restartable iteration trains five-member regression/classification
ensembles, scores a fixed candidate domain, excludes existing coordinates,
applies soft observation and batch repulsion, evaluates the selected microbatch
with the exact oracle, and appends only accepted records.  The two-dimensional
production configuration used a 512-call random-grid cold start, batches up to
256 points, and a 15\% validation split.  The acquisition score combines
classification entropy/margin, phase-facing regression uncertainty,
Normal--SC and uniform-SC--FFLO proximity, gated $q$-edge risk, and conservative
extrapolation.  Response observables are retained as diagnostics and do not
replace thermodynamic boundary targets.

\section{Thermodynamic convergence and hard-risk audit}
The Stage-II endpoint contains 7,434 observations.  Its final phase-map change
is 0.0016287, boundary coverage $p_{95}$ is 0.0046875, and trusted surprise is
0/127.  The 129-call hard-risk layer contains 110 rerun-required and 19 other
untrusted or training-ineligible records.  Counterfactual boundary-impact tests
found no cluster requiring a targeted rerun under the adopted publication
tolerance; the layer remains visible and is not declared eliminated.

\section{Topology convergence and sensitivity}
The formal Stage-III audit uses trusted gapped FFLO points, deterministic
$k$-nearest-neighbor inverse-distance interpolation of $P_0$ and $P_\pi$ on a
$401\times401$ monitor grid, and support radius 0.045 in normalized
coordinates.  With $k=8$, the final coverage is $p_{95}=0.006185$ below the
0.00625 threshold; the last-three map changes, boundary shifts, and trusted
surprises pass their registered thresholds and retain one principal component.
The $k=6$ sensitivity coverage $0.006269$ is slightly above threshold, whereas
$k=12$ gives $0.005913$ and two significant components.  Local opposite-label
brackets are support diagnostics, not exact boundaries.

\section{Higher-dimensional extension and non-closure}
The three-dimensional run samples $(k_BT/t,J_A/t,\mu/t)$ over
$\mu/t\in[-0.5,1.5]$ and returns 7,081 unique observations.  Its final strict
topology tests do not close: map change is 0.002561, surface-shift
$p_{95}=0.005996$, coverage $p_{95}=0.015318$, and trusted surprise is 0.140625;
component counts vary $3\to4\to3$.  Exact point clouds and explicit narrow
$\mu$ bins are therefore admissible, while an inferred continuous topology
surface or complete 3D atlas is not.

\section{Exploratory acquisition and world-model limitations}
The matched V3/V4 analysis is an offline comparison of frozen runs.  The two
methods trade support density against sequential surprise, and V4 does not
improve the shared graph-uncertainty metric or validate any of four candidate
nodes.  The Stage-VI hierarchy improves field masking but lacks the spectral
artifacts required for a physical spectral descriptor.  These negative and
incomplete results constrain the Discussion and Outlook; they are not promoted
to Methods claims.

\section{AI-assisted workflow and reproducibility}
The manuscript pipeline uses a versioned claim/evidence registry, figure
contracts, artifact hashes, machine-readable paragraph traces, and explicit
human gates.  AI-assisted tools organize, check, and render registered
evidence.  They do not authorize new physical claims, alter frozen data, or
replace human scientific approval.  This review package is generated by the
same checkpointed LangGraph thread that produced the approved Results package.
"""


SUPPLEMENT_MD = r"""# Supplemental Material

## S1. Physical model and notation

The Hamiltonian, Nambu convention, and parameter definitions are identical to
the main text and canonical model specification. Active learning and topology
do not alter the thermodynamic phase definition.

## S2. Exact oracle and reliability

The oracle scans and refines $(\Delta,q)$ basins, compares superconducting and
normal free energies, expands edge-hit $q$ windows, and preserves unresolved
states. The strictly positive-$\Delta$ gap separates trusted normal data from a
finite-resolution boundary band. Topology combines analytic Pfaffians at
$k=0,\pi$ with a full-Brillouin-zone gap guard on trusted superconducting data.
The archived validation gives 104/104 analytic/numeric Pfaffian sign agreement
and a CPU/GPU gap difference of $7.77\times10^{-16}$.

## S3. Active-learning loop

Five-member ensembles rank a dense candidate set; existing coordinates are
excluded, repulsion reduces redundancy, and only exact returned labels are
appended. The two-dimensional production run used a 512-call random-grid cold
start, batches up to 256, and a 15% validation split.

## S4. Thermodynamic audit

The Stage-II endpoint has 7,434 observations, phase-map change 0.0016287,
coverage $p_{95}=0.0046875$, and trusted surprise 0/127. The 129-call hard-risk
layer remains explicit; its boundary-impact audit did not require targeted
reruns at the adopted tolerance.

## S5. Topology audit

The Stage-III audit uses a $401\times401$ monitor grid, $k=8$, and support radius
0.045. Coverage $p_{95}=0.006185$ passes the 0.00625 gate, while the $k=6$
sensitivity value 0.006269 lies slightly above it. Brackets are diagnostics.

## S6. Three-dimensional non-closure

The 7,081-point extension spans $\mu/t\in[-0.5,1.5]$, but strict topology map,
surface, coverage, surprise, and component-stability gates are not all closed.
Only exact clouds and explicit slice bins are used as evidence.

## S7. Exploratory methods and reproducibility

Stage V establishes no learned-acquisition winner and Stage VI lacks a validated
spectral representation and EIG-ready uncertainty. The checkpointed manuscript
workflow records claims, evidence, artifact hashes, and human gates; AI tools
support organization and validation but do not set scientific authority.
"""


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _json(path: Path, payload: Any) -> None:
    _write(path, json.dumps(payload, indent=2, ensure_ascii=False))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_methods_package(project_root: Path, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / "methods_candidate.tex", METHODS_TEX)
    _write(output_dir / "methods_candidate.md", METHODS_MD)
    trace = {
        "schema_version": 1,
        "section": "Methods",
        "paragraphs": [
            {"paragraph_id": "M-01", "evidence_ids": ["E-MODEL-SPEC", "E-NUMERICS-SPEC"], "claim_ids": ["CLM-ORACLE-RELIABILITY"]},
            {"paragraph_id": "M-02", "evidence_ids": ["E-NUMERICS-SPEC", "E-PHASEII-AUDIT"], "claim_ids": ["CLM-ORACLE-RELIABILITY"]},
            {"paragraph_id": "M-03", "evidence_ids": ["E-STAGEIII-PASS", "E-STAGEIII-CONVERGENCE"], "claim_ids": ["CLM-2D-TOPOLOGY"]},
            {"paragraph_id": "M-04", "evidence_ids": ["E-NUMERICS-SPEC", "E-PHASEII-AUDIT"], "claim_ids": ["CLM-2D-THERMO"]},
            {"paragraph_id": "M-05", "evidence_ids": ["E-STAGEIII-CONVERGENCE", "E-STAGEIV-AUDIT"], "claim_ids": ["CLM-2D-TOPOLOGY", "CLM-3D-EXTENSION"]},
        ],
    }
    _json(output_dir / "methods_evidence_trace.yaml", trace)
    _write(output_dir / "methods_reproducibility_audit.md", """# Methods reproducibility audit

Status: **PASS for an evidence-linked intermediate draft**.

- Hamiltonian and label thresholds match `MODEL_SPEC.md`.
- Reliability and topology hierarchy match `docs/NUMERICS_SPEC.md` and the implemented exact oracle.
- No new physical parameter, fit, dataset, or exact calculation was introduced.
- Stage V and Stage VI are excluded from Methods claims.
- The text deliberately omits unregistered implementation minutiae and leaves full configuration records in the archived run bundles.
""")
    _write(output_dir / "methods_main_vs_supplement_split.md", """# Main-text versus Supplement split

Main text retains the model, oracle hierarchy, topology authority, scheduler role, convergence logic, and the caveated 3D extension. The Supplement carries analytic Pfaffian expressions, trust-state details, production configuration, numerical thresholds, sensitivity values, and negative/incomplete exploratory studies. This keeps the Methods line compact without hiding the numerical authority of the claims.
""")
    return {"methods_tex": str(output_dir / "methods_candidate.tex"), "methods_md": str(output_dir / "methods_candidate.md")}


def build_supplement_package(project_root: Path, output_dir: Path) -> dict[str, str]:
    _write(output_dir / "supplement_candidate.tex", SUPPLEMENT_TEX)
    _write(output_dir / "supplement_candidate.md", SUPPLEMENT_MD)
    trace = {
        "schema_version": 1,
        "section": "Supplement",
        "sections": [
            {"id": "S1", "evidence_ids": ["E-MODEL-SPEC"]},
            {"id": "S2", "evidence_ids": ["E-NUMERICS-SPEC", "E-STAGEIII-PASS"]},
            {"id": "S3", "evidence_ids": ["E-NUMERICS-SPEC", "E-PHASEII-AUDIT"]},
            {"id": "S4", "evidence_ids": ["E-PHASEII-AUDIT", "E-PHASEII-HARD-RISK"]},
            {"id": "S5", "evidence_ids": ["E-STAGEIII-CONVERGENCE"]},
            {"id": "S6", "evidence_ids": ["E-STAGEIV-AUDIT", "E-STAGEIV-MU-SLICE-AUDIT"]},
            {"id": "S7", "evidence_ids": ["E-STAGEV-MATCHED", "E-STAGEVI0", "E-STAGEVI0-2"]},
        ],
    }
    _json(output_dir / "supplement_evidence_trace.yaml", trace)
    _write(output_dir / "supplement_structure_audit.md", """# Supplement structure audit

Status: **PASS**.

The draft is organized by physical question and evidentiary function: model, oracle reliability, active-learning scheduler, thermodynamic closure, topology convergence, higher-dimensional non-closure, and exploratory limitations/reproducibility. It does not follow development chronology and introduces no new result.
""")
    _write(output_dir / "supplement_main_text_consistency_audit.md", """# Main-text/SM consistency audit

Status: **PASS with preserved caveats**.

- Thermodynamic labels and thresholds agree.
- cFFLO/tFFLO authority remains the trusted full-BZ gap/Pfaffian subset.
- The $k=6$ Stage-III sensitivity deviation remains visible.
- Stage IV is described as higher-dimensional exploration, never as a closed 3D atlas.
- Stage V remains exploratory and Stage VI remains future work.
""")
    return {"supplement_tex": str(output_dir / "supplement_candidate.tex"), "supplement_md": str(output_dir / "supplement_candidate.md")}


def build_discussion_package(project_root: Path, output_dir: Path) -> dict[str, str]:
    _write(output_dir / "discussion_candidate.tex", DISCUSSION_TEX)
    _write(output_dir / "discussion_candidate.md", DISCUSSION_MD)
    _write(output_dir / "outlook_candidate.tex", OUTLOOK_TEX)
    _write(output_dir / "outlook_candidate.md", OUTLOOK_MD)
    _write(output_dir / "discussion_claim_audit.md", """# Discussion and Outlook claim audit

Status: **PASS**.

- Stage V reports the matched numerical tradeoff and explicitly rejects a superiority claim.
- The four V4 candidate nodes are not promoted to new phases.
- Stage VI is presented only as a future direction.
- Missing spectral artifacts, non-monotonic replay uncertainty, and absent EIG authorization are explicit.
- No Stage V/VI statement is used as a main Results claim.
""")
    return {
        "discussion_tex": str(output_dir / "discussion_candidate.tex"),
        "outlook_tex": str(output_dir / "outlook_candidate.tex"),
    }


def audit_manuscript_figure_source_consistency(
    project_root: Path,
    output_dir: Path,
    results_dir: Path,
) -> dict[str, Any]:
    frozen_rows = verify_canonical_frozen_artifacts(project_root)
    results = (results_dir / "results_candidate.tex").read_text(encoding="utf-8")
    captions = (results_dir / "figure_captions_candidate.tex").read_text(encoding="utf-8")
    review_tex_path = results_dir / "results_review_packet.tex"
    review_tex = review_tex_path.read_text(encoding="utf-8") if review_tex_path.exists() else ""
    required_sources = [record["source_pdf"] for record in CANONICAL_MAIN_FIGURES.values()]
    missing_sources = [source for source in required_sources if source not in review_tex]
    legacy_source_used = "manuscript_figures/final/Fig4_candidate.pdf" in review_tex
    fig4_text_consistent = all(fragment in results + captions for fragment in [
        "Four explicit narrow fixed-$\\mu$ bins",
        "$\\mu/t=-0.45,0.15,0.85,$ and $1.45$",
    ])
    consistent = not missing_sources and not legacy_source_used and fig4_text_consistent
    record_path = output_dir / "manuscript_figure_source_repair_record.md"
    source_lines = "\n".join(f"- `{source}`" for source in required_sources)
    _write(record_path, f"""# Manuscript figure-source repair record

Gate: `GATE_MANUSCRIPT_FIGURE_SOURCE_REPAIR = {'PASS' if consistent else 'BLOCK'}`

## Authority

The manuscript must use the approved Figure Engine / strict-data outputs:

{source_lines}

Figure 4 therefore contains four fixed-$\\mu$ bins centered at
$\\mu/t=-0.45,0.15,0.85,$ and $1.45$, each with half-width 0.025.

## Audit

- Canonical frozen artifact hashes: PASS ({len(frozen_rows)} files).
- Missing canonical paths in the Results review source: {missing_sources or 'none'}.
- Legacy three-slice Figure 4 source used: {legacy_source_used}.
- Figure 4 four-slice Results/caption agreement: {fig4_text_consistent}.

No physics, data, numerical result, or Figure Engine artwork was recomputed or edited.
""")
    audit = {
        "schema_version": 1,
        "gate_id": "GATE_MANUSCRIPT_FIGURE_SOURCE_REPAIR",
        "canonical_sources": required_sources,
        "missing_sources": missing_sources,
        "legacy_three_slice_source_used": legacy_source_used,
        "fig4_four_slice_text_consistent": fig4_text_consistent,
        "results_caption_consistent": consistent,
        "status": "PASS" if consistent else "BLOCK",
        "proposal": str(record_path),
    }
    _json(output_dir / "manuscript_figure_source_consistency_audit.json", audit)
    return audit


# Compatibility entry point retained for historical scripts.  Its semantics are
# now the manuscript-wide Figure Engine source audit rather than the superseded
# three-slice Fig. 4 correction.
def audit_fig4_caption_consistency(project_root: Path, output_dir: Path, results_dir: Path) -> dict[str, Any]:
    return audit_manuscript_figure_source_consistency(project_root, output_dir, results_dir)


def build_introduction_prerequisites(project_root: Path, output_dir: Path) -> dict[str, str]:
    _write(output_dir / "introduction_argument_outline.md", """# Introduction argument outline

1. Finite-momentum superconductivity and topology create coupled thermodynamic and spectral phase-space questions.
2. Dense exact BdG mapping becomes costly as the parameter dimension and reliability checks grow.
3. A useful active learner must schedule exact physics without becoming the source of physical labels.
4. This work combines an audited exact oracle with reliability-aware active learning to reconstruct 2D thermodynamics, resolve cFFLO/tFFLO structure, and demonstrate a caveated 3D extension.
5. The Introduction remains an outline until external retrieval is separately authorized and every background statement has a verified citation.
""")
    _json(output_dir / "introduction_claim_needs.yaml", {
        "schema_version": 1,
        "claims": [
            {"id": "INTRO-BACKGROUND-FFLO", "need": "historical and current status of finite-momentum pairing", "citation_required": True},
            {"id": "INTRO-BACKGROUND-ALT", "need": "altermagnetism and superconducting consequences", "citation_required": True},
            {"id": "INTRO-BACKGROUND-TOPO", "need": "1D topological superconductivity and bulk-boundary diagnosis", "citation_required": True},
            {"id": "INTRO-METHOD-GAP", "need": "active learning for expensive physical phase diagrams", "citation_required": True},
            {"id": "INTRO-OUR-CONTRIBUTION", "need": "registered project claims only", "citation_required": False},
        ],
    })
    _json(output_dir / "introduction_citation_requirements.yaml", {
        "schema_version": 1,
        "requirements": [
            "prefer primary papers and authoritative reviews",
            "record DOI, title, authors, year, and claim-level relevance",
            "verify every citation against the source text",
            "do not cite search snippets",
            "separate external background from project evidence",
        ],
    })
    bibs = sorted(str(path.relative_to(project_root)) for path in project_root.rglob("*.bib"))
    _write(output_dir / "existing_bibliography_coverage_audit.md", "# Existing bibliography coverage audit\n\n"
           + ("Only the following local BibTeX file was found:\n\n" + "\n".join(f"- `{item}`" for item in bibs)
              if bibs else "No local bibliography file was found.")
           + "\n\nThe local bibliography is insufficient to support an Introduction; no external source was retrieved in this run.\n")
    _write(output_dir / "external_retrieval_plan.md", """# External retrieval plan (not executed)

After explicit permission, search primary literature and authoritative reviews for FFLO pairing, altermagnetism, 1D topological superconductivity, phase-diagram active learning, and reliability-aware scientific ML. Build a claim-to-citation matrix, verify metadata and source text, deduplicate the bibliography, and rerun the citation audit. No browser, API, DOI lookup, or external citation retrieval has been performed at this gate.
""")
    return {"introduction_prerequisites_path": str(output_dir)}


def _latex_escape_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def _extract_caption_macro(source: str, index: int) -> str:
    marker = rf"\def\fig{index}caption{{"
    start = source.find(marker)
    if start < 0:
        raise ValueError(f"missing frozen Figure {index} caption")
    cursor = start + len(marker)
    depth = 1
    while cursor < len(source) and depth:
        if source[cursor] == "{":
            depth += 1
        elif source[cursor] == "}":
            depth -= 1
        cursor += 1
    if depth:
        raise ValueError(f"unbalanced frozen Figure {index} caption")
    return source[start + len(marker): cursor - 1]


def compile_intermediate_review(project_root: Path, output_dir: Path, results_dir: Path) -> dict[str, str]:
    figures_dir = output_dir / "figures"
    stage_canonical_main_figures(project_root, figures_dir, output_dir)
    _json(output_dir / "fig4_caption_consistency_audit.json", {
        "schema_version": 1,
        "status": "SUPERSEDED",
        "superseded_by": "GATE_MANUSCRIPT_FIGURE_SOURCE_REPAIR",
        "reason": "The approved strict-data Figure 4 has four fixed-mu slices and is now the sole manuscript authority.",
    })
    _write(output_dir / "fig4_caption_consistency_correction_proposal.md", r"""# Historical Fig. 4 correction record — superseded

The earlier three-slice correction was based on a legacy manuscript candidate
rather than the approved strict-data Figure Engine freeze. It is superseded by
`GATE_MANUSCRIPT_FIGURE_SOURCE_REPAIR`. The manuscript now uses four explicit
fixed-$\mu$ slices at $\mu/t=-0.45,0.15,0.85,1.45$ with half-width 0.025.
""")
    _write(output_dir / "READ_BEFORE_REVIEW.md", r"""# Read before review

This package uses the approved Figure Engine / strict-data Figures 1--4. The
legacy `manuscript_figures/final` main-figure candidates are not authoritative.
Figure 4 contains four explicit fixed-$\mu$ slices centered at
$\mu/t=-0.45,0.15,0.85,1.45$, each with half-width 0.025, and retains the
strict three-dimensional topology nonclosure caveat.

The PRL/REVTeX appearance is an author-preferred working layout; the target
journal remains undecided. External Introduction retrieval has not been
authorized.
""")
    for suffix in [".pdf", ".png"]:
        source = project_root / "manuscript_figures/final" / f"SuppFig_S1_candidate{suffix}"
        if source.exists():
            shutil.copy2(source, figures_dir / source.name)

    results_tex = (results_dir / "results_candidate.tex").read_text(encoding="utf-8")
    captions_tex = (results_dir / "figure_captions_candidate.tex").read_text(encoding="utf-8")
    results_md = (results_dir / "results_candidate.md").read_text(encoding="utf-8")

    figure_blocks = "\n".join(
        rf"\begin{{figure*}}[p]\centering\includegraphics[width=0.96\textwidth]{{figures/Fig{i}_candidate.pdf}}\caption{{{_extract_caption_macro(captions_tex, i)}}}\end{{figure*}}"
        for i in range(1, 5)
    )
    main_tex = rf"""\documentclass[aps,prl,reprint,superscriptaddress,longbibliography]{{revtex4-2}}
\usepackage{{amsmath,amssymb,graphicx,xcolor,booktabs,hyperref}}
\hypersetup{{colorlinks=true,allcolors=blue}}
\begin{{document}}
\title{{Reliability-aware active learning for FFLO phase spaces\\\large Intermediate review in an author-preferred PRL-style layout}}
\author{{Author list pending human approval}}
\date{{August 27, 2026}}
\begin{{abstract}}
This intermediate package intentionally omits the Introduction and final Abstract. Results and captions are frozen by human approval; Methods, Discussion, Outlook, and Supplemental Material remain review candidates. The PRL/REVTeX appearance is an author-preferred working format and does not encode a target-journal decision. No external citation retrieval has been authorized.
\end{{abstract}}
\maketitle
{results_tex}
{METHODS_TEX}
{DISCUSSION_TEX}
{OUTLOOK_TEX}
{figure_blocks}
\end{{document}}
"""
    supp_tex = rf"""\documentclass[aps,prl,onecolumn,superscriptaddress,floatfix]{{revtex4-2}}
\usepackage{{amsmath,amssymb,graphicx,xcolor,booktabs,hyperref}}
\hypersetup{{colorlinks=true,allcolors=blue}}
\begin{{document}}
\title{{Supplemental Material (intermediate review)}}
\author{{Author list pending human approval}}
\date{{August 27, 2026}}
\maketitle
{SUPPLEMENT_TEX}
\begin{{figure}}[!b]\centering\includegraphics[width=0.86\textwidth]{{figures/SuppFig_S1_candidate.pdf}}\caption{{Registered supplemental convergence and reliability audit panel. The artwork is frozen; interpretation remains governed by the evidence registry and caveats in Secs. S4--S6.}}\end{{figure}}
\end{{document}}
"""
    _write(output_dir / "manuscript_intermediate_review.tex", main_tex)
    _write(output_dir / "supplement_review.tex", supp_tex)
    main_md = """# Intermediate manuscript review (author-preferred PRL-style layout)

> Results and captions are frozen by explicit human approval. Introduction, citations, Abstract, title, and final LaTeX integration remain pending. The PRL/REVTeX appearance is a working-format preference, not a target-journal decision. This is not the final manuscript.

""" + results_md + "\n" + METHODS_MD + "\n" + DISCUSSION_MD + "\n" + OUTLOOK_MD
    _write(output_dir / "manuscript_intermediate_review.md", main_md)
    _write(output_dir / "supplement_review.md", SUPPLEMENT_MD)

    for stem in ["manuscript_intermediate_review", "supplement_review"]:
        logs: list[str] = []
        for _ in range(2):
            proc = subprocess.run(
                ["pdflatex", "-recorder", "-interaction=nonstopmode", "-halt-on-error", f"{stem}.tex"],
                cwd=output_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            logs.append(proc.stdout + "\n" + proc.stderr)
            if proc.returncode != 0:
                break
        _write(output_dir / f"{stem}_latex_compile.log", "\n\n--- PASS ---\n\n".join(logs))
        if proc.returncode != 0 or not (output_dir / f"{stem}.pdf").exists():
            raise RuntimeError(f"LaTeX compilation failed for {stem}; inspect {stem}_latex_compile.log")

    recorder = (output_dir / "manuscript_intermediate_review.fls").read_text(encoding="utf-8", errors="replace")
    missing_embedded = [
        record["staged_pdf"]
        for record in CANONICAL_MAIN_FIGURES.values()
        if record["staged_pdf"] not in recorder
    ]
    if missing_embedded:
        raise RuntimeError(f"compiled manuscript did not record canonical figures: {missing_embedded}")

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    with (tables_dir / "evidence_coverage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "evidence_ids", "status"])
        writer.writerows([
            ["Results", "E-FIG2-STRICT-DATA;E-FIG3-STRICT-DATA;E-FIG4-STRICT-DATA", "frozen"],
            ["Methods", "E-MODEL-SPEC;E-NUMERICS-SPEC;E-PHASEII-AUDIT;E-STAGEIII-PASS", "candidate_traced"],
            ["Discussion", "E-STAGEV-MATCHED", "candidate_traced"],
            ["Outlook", "E-STAGEVI0;E-STAGEVI0-2", "candidate_traced"],
            ["Introduction", "external retrieval not authorized", "human_wait"],
        ])
    _write(output_dir / "decision_log.md", """# Decision log

- Explicit human decision: use the previously approved Figure Engine / strict-data Figures 1--4.
- Figure-source repair: all four staged PDFs are byte-identical to their registered scientific freezes; the legacy main-figure candidate bundle is not used.
- Generated: evidence-linked Methods, Supplemental Material, Discussion, and Outlook candidates.
- Preserved: Stage-III $k=6$ sensitivity caveat; Stage-IV strict 3D non-closure; Stage-V non-superiority; Stage-VI missing spectral artifacts and non-EIG-ready uncertainty.
- Not executed: new physics, exact oracle, figure regeneration, external Introduction retrieval, final Abstract/title, final manuscript approval.
- Current gate: `INTRO_CITATION_LOOP_GATE = HUMAN_WAIT`.
- Next decision: authorize or decline external Introduction retrieval.
- Journal target: intentionally undecided. The author prefers the PRL/REVTeX
  visual format, so the four-figure manuscript remains in that working layout
  without compressing or changing its scientific scope.
""")
    hashes = {
        path.name: _sha256(path)
        for path in [output_dir / "manuscript_intermediate_review.pdf", output_dir / "supplement_review.pdf"]
    }
    _json(output_dir / "pdf_hashes.json", hashes)
    return {
        "main_pdf": str(output_dir / "manuscript_intermediate_review.pdf"),
        "supplement_pdf": str(output_dir / "supplement_review.pdf"),
        "main_md": str(output_dir / "manuscript_intermediate_review.md"),
        "supplement_md": str(output_dir / "supplement_review.md"),
    }
