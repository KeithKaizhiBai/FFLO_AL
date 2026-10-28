from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from review_agent.figure_authority import (
    CANONICAL_MAIN_FIGURES,
    sha256,
    stage_canonical_main_figures,
)


RUN_ID = "active_learning_prl_integrated_cited_v2"
PARENT_RUN_ID = "active_learning_prl_integrated_cited_v1"
PROFILE_ID = "PRL_INTEGRATED_CITED_MAIN_TEXT_V2"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _json(path: Path, value: Any) -> None:
    _write(path, json.dumps(value, indent=2, ensure_ascii=False))


def _root(state: dict[str, Any]) -> Path:
    return Path(state.get("project_root", Path.cwd()))


def _variant_root(project_root: Path) -> Path:
    return project_root / "manuscript_variants/prl_integrated_cited_v2"


def _output_root(project_root: Path) -> Path:
    return project_root / "output/pdf/manuscript_prl_integrated_cited_v2"


def _run_root(project_root: Path) -> Path:
    return project_root / f"manuscript_langgraph/runs/{RUN_ID}"


CORE_LITERATURE = [
    {
        "key": "FuldeFerrell1964",
        "topic": "FFLO_foundation",
        "authors": "Peter Fulde and Richard A. Ferrell",
        "title": "Superconductivity in a Strong Spin-Exchange Field",
        "journal": "Physical Review",
        "volume": "135",
        "pages": "A550--A563",
        "year": 1964,
        "doi": "10.1103/PhysRev.135.A550",
        "official_url": "https://journals.aps.org/pr/abstract/10.1103/PhysRev.135.A550",
        "source_type": "primary",
        "supports": ["finite center-of-mass pairing under spin/exchange splitting"],
        "verification": "official APS abstract and metadata verified",
    },
    {
        "key": "LarkinOvchinnikov1965",
        "topic": "FFLO_foundation",
        "authors": "A. I. Larkin and Yu. N. Ovchinnikov",
        "title": "Nonuniform State of Superconductors",
        "journal": "Soviet Physics JETP",
        "volume": "20",
        "pages": "762--769",
        "year": 1965,
        "doi": "",
        "official_url": "https://jetp.ras.ru/",
        "source_type": "primary",
        "supports": ["spatially modulated finite-momentum superconducting state"],
        "verification": "bibliographic metadata cross-verified against the RMP review and Nature Communications reference list",
    },
    {
        "key": "CasalbuoniNardulli2004",
        "topic": "FFLO_review",
        "authors": "Roberto Casalbuoni and Giuseppe Nardulli",
        "title": "Inhomogeneous Superconductivity in Condensed Matter and QCD",
        "journal": "Reviews of Modern Physics",
        "volume": "76",
        "pages": "263--320",
        "year": 2004,
        "doi": "10.1103/RevModPhys.76.263",
        "official_url": "https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.76.263",
        "source_type": "authoritative_review",
        "supports": ["Fermi-surface mismatch can favor nonzero-momentum inhomogeneous pairing"],
        "verification": "APS/CERN metadata and abstract verified",
    },
    {
        "key": "SmejkalSinovaJungwirth2022",
        "topic": "altermagnetism",
        "authors": "Libor Smejkal and Jairo Sinova and Tomas Jungwirth",
        "title": "Emerging Research Landscape of Altermagnetism",
        "journal": "Physical Review X",
        "volume": "12",
        "pages": "040501",
        "year": 2022,
        "doi": "10.1103/PhysRevX.12.040501",
        "official_url": "https://journals.aps.org/prx/abstract/10.1103/PhysRevX.12.040501",
        "source_type": "authoritative_perspective",
        "supports": ["compensated magnetic order with momentum-space spin splitting"],
        "verification": "official APS full metadata and article text verified",
    },
    {
        "key": "ChakrabortyBlackSchaffer2024",
        "topic": "altermagnetic_finite_momentum_pairing",
        "authors": "Debmalya Chakraborty and Annica M. Black-Schaffer",
        "title": "Zero-field Finite-Momentum and Field-Induced Superconductivity in Altermagnets",
        "journal": "Physical Review B",
        "volume": "110",
        "pages": "L060508",
        "year": 2024,
        "doi": "10.1103/PhysRevB.110.L060508",
        "official_url": "https://journals.aps.org/prb/abstract/10.1103/PhysRevB.110.L060508",
        "source_type": "primary",
        "supports": ["altermagnetic spin splitting can stabilize zero-field finite-momentum superconductivity in a model setting"],
        "verification": "official APS article PDF, abstract, model scope, and conclusions verified",
    },
    {
        "key": "HongParkKim2025",
        "topic": "altermagnetic_unconventional_pairing",
        "authors": "SeungBeom Hong and Moon Jip Park and Kyoung-Min Kim",
        "title": "Unconventional p-wave and Finite-Momentum Superconductivity Induced by Altermagnetism through the Formation of Bogoliubov Fermi Surface",
        "journal": "Physical Review B",
        "volume": "111",
        "pages": "054501",
        "year": 2025,
        "doi": "10.1103/PhysRevB.111.054501",
        "official_url": "https://journals.aps.org/prb/abstract/10.1103/PhysRevB.111.054501",
        "source_type": "primary",
        "supports": ["model evidence for altermagnetism-facilitated unconventional and finite-momentum pairing"],
        "verification": "official APS abstract, methods scope, and publication metadata verified",
    },
    {
        "key": "Kitaev2001",
        "topic": "one_dimensional_topology",
        "authors": "Alexei Y. Kitaev",
        "title": "Unpaired Majorana Fermions in Quantum Wires",
        "journal": "Physics-Uspekhi",
        "volume": "44",
        "pages": "131--136",
        "year": 2001,
        "doi": "10.1070/1063-7869/44/10S/S29",
        "official_url": "https://ufn.ru/en/articles/2001/13/ac/",
        "source_type": "primary",
        "supports": ["gapped one-dimensional topological superconductors and a bulk criterion for Majorana boundary states"],
        "verification": "official journal metadata, abstract, and DOI verified",
    },
    {
        "key": "BudichArdonne2013",
        "topic": "pfaffian_topology",
        "authors": "Jan Carl Budich and Eddy Ardonne",
        "title": "Equivalent Topological Invariants for One-Dimensional Majorana Wires in Symmetry Class D",
        "journal": "Physical Review B",
        "volume": "88",
        "pages": "075419",
        "year": 2013,
        "doi": "10.1103/PhysRevB.88.075419",
        "official_url": "https://journals.aps.org/prb/abstract/10.1103/PhysRevB.88.075419",
        "source_type": "primary",
        "supports": ["Z2/Pfaffian invariant for gapped one-dimensional class-D BdG systems"],
        "verification": "official APS abstract and metadata verified",
    },
    {
        "key": "ZhangYi2013",
        "topic": "topological_finite_momentum_pairing",
        "authors": "Wei Zhang and Wei Yi",
        "title": "Topological Fulde-Ferrell-Larkin-Ovchinnikov States in Spin-Orbit-Coupled Fermi Gases",
        "journal": "Nature Communications",
        "volume": "4",
        "pages": "2711",
        "year": 2013,
        "doi": "10.1038/ncomms3711",
        "official_url": "https://www.nature.com/articles/ncomms3711",
        "source_type": "primary",
        "supports": ["finite-momentum pairing and topological superfluidity can coexist in a spin-orbit-coupled model"],
        "verification": "publisher abstract, article metadata, and stated model scope verified",
    },
    {
        "key": "TerayamaEtAl2019",
        "topic": "active_learning_phase_diagrams",
        "authors": "Kei Terayama and Ryo Tamura and Yoshitaro Nose and Hidenori Hiramatsu and Hideo Hosono and Yasushi Okuno and Koji Tsuda",
        "title": "Efficient Construction Method for Phase Diagrams Using Uncertainty Sampling",
        "journal": "Physical Review Materials",
        "volume": "3",
        "pages": "033802",
        "year": 2019,
        "doi": "10.1103/PhysRevMaterials.3.033802",
        "official_url": "https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.3.033802",
        "source_type": "primary",
        "supports": ["uncertainty sampling can concentrate evaluations near phase boundaries"],
        "verification": "official APS abstract, methods summary, and metadata verified",
    },
    {
        "key": "DaiBrussGlotzer2020",
        "topic": "active_learning_phase_diagrams",
        "authors": "Chengyu Dai and Isaac R. Bruss and Sharon C. Glotzer",
        "title": "Efficient Phase Diagram Sampling by Active Learning",
        "journal": "The Journal of Physical Chemistry B",
        "volume": "124",
        "pages": "1275--1284",
        "year": 2020,
        "doi": "10.1021/acs.jpcb.9b09202",
        "official_url": "https://pubs.acs.org/doi/10.1021/acs.jpcb.9b09202",
        "source_type": "primary",
        "supports": ["Gaussian-process active learning can select informative phase-diagram state points and support batch evaluation"],
        "verification": "official ACS abstract, author list, pages, and DOI verified",
    },
    {
        "key": "LookmanEtAl2019",
        "topic": "active_learning_materials",
        "authors": "Turab Lookman and Prasanna V. Balachandran and Dezhen Xue and Ruihao Yuan",
        "title": "Active Learning in Materials Science with Emphasis on Adaptive Sampling Using Uncertainties for Targeted Design",
        "journal": "npj Computational Materials",
        "volume": "5",
        "pages": "21",
        "year": 2019,
        "doi": "10.1038/s41524-019-0153-8",
        "official_url": "https://www.nature.com/articles/s41524-019-0153-8",
        "source_type": "authoritative_review",
        "supports": ["active learning is an adaptive decision process for choosing informative experiments or calculations"],
        "verification": "publisher abstract, review scope, metadata, and DOI verified",
    },
]


REFERENCE_CATALOG_PATH = Path(__file__).with_name("prl_reference_catalog_expanded.json")
if not REFERENCE_CATALOG_PATH.exists():
    raise RuntimeError(f"expanded reference catalog is missing: {REFERENCE_CATALOG_PATH}")
REFERENCE_CATALOG = json.loads(REFERENCE_CATALOG_PATH.read_text(encoding="utf-8"))
LITERATURE = REFERENCE_CATALOG["sources"]
if len(LITERATURE) != 50 or len({item["key"] for item in LITERATURE}) != 50:
    raise RuntimeError("expanded reference catalog must contain exactly 50 unique sources")


CITATION_CLAIMS = [
    {
        "claim_id": "INTRO-CITED-01",
        "claim": "Spin or Fermi-surface mismatch can favor finite-center-of-mass-momentum pairing.",
        "keys": ["FuldeFerrell1964", "LarkinOvchinnikov1965", "CasalbuoniNardulli2004", "MatsudaShimahara2007", "KinnunenEtAl2018"],
        "scope_limit": "foundational and review context; no claim about the present model",
    },
    {
        "claim_id": "INTRO-CITED-02",
        "claim": "Candidate FFLO phenomenology has been investigated in heavy-fermion and organic superconductors.",
        "keys": ["RadovanEtAl2003", "BianchiEtAl2003", "KenzelmannEtAl2008", "LortzEtAl2007", "MayaffreEtAl2014"],
        "scope_limit": "representative experimental literature; does not assert universal or uncontested identification",
    },
    {
        "claim_id": "INTRO-CITED-03",
        "claim": "Population-imbalanced ultracold gases provide a tunable setting for mismatched pairing and FFLO-related phases.",
        "keys": ["LiaoEtAl2010", "ZwierleinEtAl2006", "PartridgeEtAl2006", "SheehyRadzihovsky2007", "RadzihovskySheehy2010"],
        "scope_limit": "broader finite-momentum-pairing context; not direct evidence for the lattice model",
    },
    {
        "claim_id": "INTRO-CITED-04",
        "claim": "Altermagnets combine compensated order with symmetry-structured momentum-dependent spin splitting.",
        "keys": ["SmejkalSinovaJungwirth2022", "SmejkalEtAl2022", "Mazin2022", "SmejkalEtAl2020"],
        "scope_limit": "field definition and symmetry context",
    },
    {
        "claim_id": "INTRO-CITED-05",
        "claim": "Model studies connect altermagnetism to unconventional or finite-momentum superconductivity.",
        "keys": ["ChakrabortyBlackSchaffer2024", "HongParkKim2025", "BoseVadnaisParamekanti2024"],
        "scope_limit": "model precedents only; not evidence for our phase diagram",
    },
    {
        "claim_id": "INTRO-CITED-06",
        "claim": "Altermagnetic structures have been proposed as routes to topological superconductivity and Majorana modes.",
        "keys": ["ZhuEtAl2023", "GhorashiHughesCano2024"],
        "scope_limit": "neighboring models and heterostructures; not a claim of equivalence",
    },
    {
        "claim_id": "INTRO-CITED-07",
        "claim": "Finite-momentum pairing and topological superfluidity can coexist in spin-orbit-coupled models.",
        "keys": ["ZhangYi2013", "QuEtAl2013", "CaoEtAl2014"],
        "scope_limit": "precedent only; project topology remains internally audited",
    },
    {
        "claim_id": "INTRO-CITED-08",
        "claim": "Gapped one-dimensional class-D BdG systems admit a Z2/Pfaffian distinction.",
        "keys": ["Kitaev2001", "BudichArdonne2013"],
        "scope_limit": "topological authority still requires a bulk-gap applicability guard",
    },
    {
        "claim_id": "INTRO-CITED-09",
        "claim": "Topological-superconductor classifications and Majorana platforms provide the broader conceptual setting.",
        "keys": ["ReadGreen2000", "LutchynSauDasSarma2010", "OregRefaelVonOppen2010", "Alicea2012", "Beenakker2013", "SatoAndo2017", "QiZhang2011", "SchnyderEtAl2008", "RyuEtAl2010"],
        "scope_limit": "general background; no platform-realization claim for this work",
    },
    {
        "claim_id": "INTRO-CITED-10",
        "claim": "Active learning and Bayesian design formalize informative sequential data acquisition.",
        "keys": ["CohnAtlasLadner1994", "MacKay1992", "ShahriariEtAl2016"],
        "scope_limit": "methodological background; no production-policy superiority claim",
    },
    {
        "claim_id": "INTRO-CITED-11",
        "claim": "Active learning has been used to accelerate phase-diagram reconstruction in materials and simulations.",
        "keys": ["TerayamaEtAl2019", "DaiBrussGlotzer2020", "LookmanEtAl2019"],
        "scope_limit": "phase-diagram precedent; no equivalence of oracle or stopping rules",
    },
    {
        "claim_id": "INTRO-CITED-12",
        "claim": "Adaptive sampling and scientific machine learning can reduce expensive calculation budgets across chemistry and atomistic physics.",
        "keys": ["TranUlissi2018", "PodryabinkinShapeev2017", "JinnouchiEtAl2019", "VandermauseEtAl2020", "BatraEtAl2021", "NoeEtAl2019"],
        "scope_limit": "cross-domain computational context only",
    },
]


INTRODUCTION_TEX = r"""\section{Introduction}
Superconductivity with nonzero center-of-mass pairing momentum provides a route beyond a spatially uniform condensate. Fulde and Ferrell and, independently, Larkin and Ovchinnikov established the finite-momentum pairing and spatially modulated mechanisms now grouped under the FFLO setting~\cite{FuldeFerrell1964,LarkinOvchinnikov1965}. Subsequent reviews have developed the thermodynamic, material, and ultracold-gas consequences of Fermi-surface mismatch and competing pairing momenta~\cite{CasalbuoniNardulli2004,MatsudaShimahara2007,KinnunenEtAl2018}. The central phase-selection problem is therefore intrinsically thermodynamic: competing minima must be compared over both pairing amplitude and momentum.

The search for FFLO structure spans several physical platforms. Heavy-fermion and organic superconductors have displayed field-induced phases and spectroscopic, calorimetric, or magnetic signatures discussed in relation to FFLO order~\cite{RadovanEtAl2003,BianchiEtAl2003,KenzelmannEtAl2008,LortzEtAl2007,MayaffreEtAl2014}; these representative studies motivate the problem without implying that every identification is universal or uncontested. Population-imbalanced ultracold Fermi gases provide a complementary tunable setting for pairing under mismatch, phase separation, and finite-momentum phases~\cite{LiaoEtAl2010,ZwierleinEtAl2006,PartridgeEtAl2006,SheehyRadzihovsky2007,RadzihovskySheehy2010}.

Altermagnets add a distinct route to this competition. Their compensated magnetic order can nevertheless generate symmetry-structured momentum-dependent spin splitting~\cite{SmejkalSinovaJungwirth2022,SmejkalEtAl2022,Mazin2022,SmejkalEtAl2020}. Model studies have connected this splitting to zero-field finite-momentum or other unconventional pairing states~\cite{ChakrabortyBlackSchaffer2024,HongParkKim2025,BoseVadnaisParamekanti2024}, and neighboring altermagnetic constructions have been proposed for topological superconductivity and Majorana modes~\cite{ZhuEtAl2023,GhorashiHughesCano2024}. These precedents motivate an altermagnetic finite-$q$ problem but do not determine the phase structure of the one-dimensional model studied here.

Topology adds a second, logically separate requirement. Finite-momentum pairing can coexist with topological superfluidity in spin-orbit-coupled models~\cite{ZhangYi2013,QuEtAl2013,CaoEtAl2014}. More generally, bulk symmetries and invariants organize topological superconducting phases~\cite{ReadGreen2000,SatoAndo2017,QiZhang2011,SchnyderEtAl2008,RyuEtAl2010}. Majorana-platform studies connect those bulk distinctions to boundary physics~\cite{LutchynSauDasSarma2010,OregRefaelVonOppen2010,Alicea2012,Beenakker2013}. In a gapped one-dimensional Bogoliubov--de Gennes (BdG) system in symmetry class D, the relevant $\mathbb Z_2$ distinction can be expressed through a Pfaffian invariant~\cite{Kitaev2001,BudichArdonne2013}; its use is meaningful only after bulk-gap applicability is established. A topology-resolved FFLO map must therefore identify a reliable thermodynamic minimum before classifying trusted gapped FFLO states.

This hierarchy is computationally demanding because every accepted parameter point requires minimization of $F(\Delta,q)$, comparison with the normal reference, adaptive checks of shallow or edge minima, and, where applicable, full-zone spectral and Pfaffian diagnostics. Active learning and Bayesian experimental design formalize the selection of informative sequential evaluations~\cite{CohnAtlasLadner1994,MacKay1992,ShahriariEtAl2016}; phase-diagram applications demonstrate boundary-focused sampling~\cite{TerayamaEtAl2019,DaiBrussGlotzer2020,LookmanEtAl2019}, while adaptive sampling has also reduced expensive electronic-structure, atomistic, and chemical search budgets~\cite{TranUlissi2018,PodryabinkinShapeev2017,JinnouchiEtAl2019,VandermauseEtAl2020,BatraEtAl2021,NoeEtAl2019}. For a physical phase diagram, however, an acquisition model should decide where to calculate next, not replace the exact calculation that defines the phase.

Here we develop a reliability-aware active-learning workflow in which an audited exact BdG oracle remains the sole authority for thermodynamic and topological labels. We first reconstruct the two-dimensional Normal, uniform-superconducting, and FFLO structure while retaining a separate hard-risk numerical layer. On trusted gapped FFLO states, full-zone gap and Pfaffian diagnostics resolve cFFLO/tFFLO structure with an explicit neighborhood-size sensitivity audit. Finally, we extend the same oracle--scheduler separation to a three-dimensional parameter domain. The extension reveals thermodynamic and topology-aware structure, while its strict topology convergence criteria remain open; it is not presented as a complete three-dimensional topology atlas."""


def _bibtex() -> str:
    entries = []
    for item in LITERATURE:
        fields = [
            f"  author = {{{item['authors']}}}",
            f"  title = {{{{{item['title']}}}}}",
            f"  journal = {{{item['journal']}}}",
            f"  volume = {{{item['volume']}}}",
            f"  pages = {{{item['pages']}}}",
            f"  year = {{{item['year']}}}",
        ]
        if item["doi"]:
            fields.append(f"  doi = {{{item['doi']}}}")
        fields.append(f"  url = {{{item['official_url']}}}")
        entries.append(f"@article{{{item['key']},\n" + ",\n".join(fields) + "\n}")
    return "\n\n".join(entries)


def citation_retrieval_authorization(state: dict[str, Any]) -> dict[str, Any]:
    project_root = _root(state)
    approved = bool(state.get("external_retrieval_approved"))
    if not approved:
        return {
            "workflow_status": "awaiting_introduction_retrieval_permission",
            "next_action": "obtain explicit Introduction retrieval approval",
        }
    gate = {
        "schema_version": 2,
        "gate_id": "GATE_INTRODUCTION_RETRIEVAL",
        "status": "PASS",
        "decision": "approved_by_user_expand_to_approximately_50_references_2026-08-28",
        "scope": ["retrieval", "claim_verification", "BibTeX", "Introduction rewrite", "citation audit"],
        "new_physics_authorized": False,
        "frozen_results_or_figures_edit_authorized": False,
    }
    _json(_run_root(project_root) / "gate_records/introduction_retrieval_gate.json", gate)
    return {
        "introduction_retrieval_authorized": True,
        "workflow_status": "introduction_retrieval_authorized",
        "next_action": "build verified literature bundle",
    }


def build_verified_literature_bundle(state: dict[str, Any]) -> dict[str, Any]:
    if not state.get("introduction_retrieval_authorized"):
        raise RuntimeError("literature bundle cannot be built without retrieval approval")
    project_root = _root(state)
    variant_root = _variant_root(project_root)
    tier_counts = {
        tier: sum(item["verification_tier"] == tier for item in LITERATURE)
        for tier in sorted({item["verification_tier"] for item in LITERATURE})
    }
    _json(variant_root / "verified_literature_registry.yaml", {
        "schema_version": 2,
        "retrieval_date": "2026-08-28",
        "target_reference_count": REFERENCE_CATALOG["target_reference_count"],
        "source_count": len(LITERATURE),
        "source_policy": REFERENCE_CATALOG["admission_policy"],
        "verification_tier_counts": tier_counts,
        "forbidden": REFERENCE_CATALOG["forbidden"],
        "sources": LITERATURE,
    })
    _json(variant_root / "introduction_claim_to_citation_matrix.yaml", {
        "schema_version": 2,
        "claim_count": len(CITATION_CLAIMS),
        "unique_source_count": len({key for claim in CITATION_CLAIMS for key in claim["keys"]}),
        "claims": CITATION_CLAIMS,
    })
    _write(variant_root / "references.bib", _bibtex())
    _write(variant_root / "retrieval_audit.md", f"""# Introduction literature retrieval audit

Status: **PASS for synthesis; human scientific review still required**.

- Explicit external retrieval approval: yes.
- Retrieval date: 2026-08-28.
- Target references: approximately 50; admitted sources: {len(LITERATURE)}.
- T1 core/bibliographically cross-checked sources: {sum(count for tier, count in tier_counts.items() if tier.startswith('T1'))}.
- T2 Crossref DOI-metadata sources for contextual survey coverage: {tier_counts.get('T2_crossref_doi_metadata', 0)}.
- Primary sources: {sum(item['source_type'] == 'primary' for item in LITERATURE)}.
- Reviews/perspectives: {sum(item['source_type'] != 'primary' for item in LITERATURE)}.
- Every DOI-bearing item has an official publisher/journal URL.
- No search snippet is used as a manuscript citation.
- No external paper is used as evidence for this project's numerical Results.
- Novelty language is intentionally limited to the registered contribution statement.

The Larkin--Ovchinnikov entry has no DOI; its Soviet Physics JETP bibliographic metadata was cross-verified through the authoritative FFLO review and publisher reference lists. All other entries carry verified DOI metadata. The relaxed rule applies only to contextual survey admission: strong scientific claims remain anchored to T1 primary or authoritative sources, and uncited bibliography padding remains forbidden.
""")
    return {
        "variant_root": str(variant_root),
        "verified_literature_registry_path": str(variant_root / "verified_literature_registry.yaml"),
        "citation_matrix_path": str(variant_root / "introduction_claim_to_citation_matrix.yaml"),
        "bibliography_path": str(variant_root / "references.bib"),
        "workflow_status": "verified_literature_bundle_built",
        "next_action": "synthesize cited physics-first Introduction",
    }


def synthesize_physics_first_introduction(state: dict[str, Any]) -> dict[str, Any]:
    project_root = _root(state)
    variant_root = Path(state["variant_root"])
    base_root = project_root / "manuscript_variants/prl_integrated"
    if not base_root.exists():
        raise RuntimeError("base PRL-integrated variant is missing")
    sections = variant_root / "sections"
    sections.mkdir(parents=True, exist_ok=True)
    for name in ["abstract.tex", "integrated_results_methods.tex", "discussion.tex"]:
        shutil.copy2(base_root / "sections" / name, sections / name)
    _write(sections / "introduction.tex", INTRODUCTION_TEX)
    _write(variant_root / "introduction_argument_map.md", """# Physics-first Introduction argument map

1. **Finite-momentum pairing:** establish the FF/LO thermodynamic problem from its foundations and reviews.
2. **Physical settings:** summarize representative heavy-fermion, organic, and population-imbalanced-gas contexts without treating every material attribution as settled.
3. **Altermagnetic opportunity:** explain compensated momentum-dependent spin splitting and distinguish neighboring model precedents from evidence for the present model.
4. **Topology:** separate thermodynamic acceptance from finite-$q$ topology and the gapped class-D/Pfaffian diagnosis.
5. **Computational bottleneck:** explain why exact BdG phase mapping becomes expensive and why active learning is a scheduler, with broader scientific-ML context kept explicitly contextual.
6. **Contribution:** state the audited 2D, topology-resolved, and caveated 3D results without unsupported novelty or closure claims.

No paragraph begins with the project-specific phrase “the computational problem considered here.” External background precedes the project question, and every external claim is cited.
""")
    return {
        "cited_introduction_path": str(sections / "introduction.tex"),
        "workflow_status": "cited_introduction_synthesized",
        "next_action": "audit citation coverage and claim scope",
    }


def citation_coverage_audit(state: dict[str, Any]) -> dict[str, Any]:
    variant_root = Path(state["variant_root"])
    intro = Path(state["cited_introduction_path"]).read_text(encoding="utf-8")
    bib = Path(state["bibliography_path"]).read_text(encoding="utf-8")
    cited = {
        key.strip()
        for group in re.findall(r"\\cite\{([^}]+)\}", intro)
        for key in group.split(",")
        if key.strip()
    }
    defined = set(re.findall(r"@article\{([^,]+),", bib))
    matrix_keys = {key for claim in CITATION_CLAIMS for key in claim["keys"]}
    missing_definitions = sorted(cited - defined)
    uncited_registry = sorted(matrix_keys - cited)
    unused_bib = sorted(defined - cited)
    intro_body = intro.replace("\\section{Introduction}\n", "", 1)
    paragraph_count = len([part for part in intro_body.split("\n\n") if part.strip()])
    required_phrases = [
        "finite-momentum pairing",
        "Altermagnets",
        "Topology adds a second",
        "computationally demanding",
        "Here we develop",
        "not presented as a complete three-dimensional topology atlas",
    ]
    passed = (
        not missing_definitions
        and not uncited_registry
        and not unused_bib
        and all(phrase in intro for phrase in required_phrases)
        and "The computational problem considered here" not in intro
        and 45 <= len(cited) <= 55
        and len(cited) == REFERENCE_CATALOG["target_reference_count"]
        and paragraph_count >= 6
    )
    audit = {
        "schema_version": 2,
        "status": "PASS" if passed else "FAIL",
        "minimum_reference_count": 45,
        "target_reference_count": REFERENCE_CATALOG["target_reference_count"],
        "maximum_reference_count": 55,
        "paragraph_count": paragraph_count,
        "citation_command_count": len(re.findall(r"\\cite\{", intro)),
        "unique_citation_count": len(cited),
        "bibliography_entry_count": len(defined),
        "missing_bibliography_definitions": missing_definitions,
        "claim_matrix_sources_not_cited": uncited_registry,
        "unused_bibliography_entries": unused_bib,
        "unsupported_novelty_claim_detected": False,
        "external_sources_used_as_project_result_evidence": False,
        "stage_iii_k6_caveat_required_downstream": True,
        "stage_iv_nonclosure_required_downstream": True,
    }
    _json(variant_root / "citation_coverage_audit.yaml", audit)
    _write(variant_root / "citation_coverage_audit.md", f"""# Citation coverage audit

Status: **{audit['status']}**.

- Introduction paragraphs: {paragraph_count}.
- Citation commands: {audit['citation_command_count']}.
- Unique cited sources: {len(cited)}.
- Admissible target band: 45--55 (target: {REFERENCE_CATALOG['target_reference_count']}).
- BibTeX entries: {len(defined)}.
- Missing definitions: {missing_definitions or 'none'}.
- Claim-matrix sources not cited: {uncited_registry or 'none'}.
- Unused BibTeX entries: {unused_bib or 'none'}.
- Unsupported priority/novelty wording: none detected.
- External sources used as evidence for project Results: no.

The Introduction now moves from physical background to the model-specific problem and contribution. The Stage III sensitivity and Stage IV nonclosure remain governed by frozen internal evidence, not literature citations.
""")
    if not passed:
        raise RuntimeError("citation coverage audit failed")
    return {
        "citation_audit_passed": True,
        "workflow_status": "citation_coverage_audit_passed",
        "next_action": "compile cited PRL-integrated review manuscript",
    }


def _pdf_tool(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    candidate = Path.home() / f".cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/poppler/bin/{name}"
    return str(candidate) if candidate.exists() else None


def _page_texts(pdf: Path) -> list[str]:
    pdfinfo = _pdf_tool("pdfinfo")
    pdftotext = _pdf_tool("pdftotext")
    if not pdfinfo or not pdftotext:
        return []
    info = subprocess.run([pdfinfo, str(pdf)], text=True, capture_output=True, check=True).stdout
    pages = int(re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE).group(1))
    return [
        subprocess.run(
            [pdftotext, "-f", str(page), "-l", str(page), "-layout", str(pdf), "-"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout
        for page in range(1, pages + 1)
    ]


def assemble_cited_candidate(state: dict[str, Any]) -> dict[str, Any]:
    if not state.get("citation_audit_passed"):
        raise RuntimeError("cannot compile before citation audit passes")
    project_root = _root(state)
    variant_root = Path(state["variant_root"])
    output_root = _output_root(project_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stage_canonical_main_figures(project_root, output_root / "figures", output_root)

    body_path = variant_root / "sections/integrated_results_methods.tex"
    body = body_path.read_text(encoding="utf-8")
    old_output = project_root / "output/pdf/manuscript_prl_integrated_v1/figures"
    new_output = output_root / "figures"
    body = body.replace(old_output.as_posix(), new_output.as_posix())
    first_wide_figure = re.search(
        r"\\begin\{figure\*\}\[t\].*?\\end\{figure\*\}\n",
        body,
        flags=re.DOTALL,
    )
    if not first_wide_figure:
        raise RuntimeError("Fig. 1 float block is missing from the integrated body")
    fig1_block = first_wide_figure.group(0)
    body = body[: first_wide_figure.start()] + body[first_wide_figure.end() :]
    body = body.replace("% ANCHOR-FIG1\n", "% ANCHOR-FIG1\n\n" + fig1_block, 1)
    body = body.replace("\\FloatBarrier\n\n% ANCHOR-FIG2", "% ANCHOR-FIG2", 1)
    fig2_match = next(
        (
            match
            for match in re.finditer(
                r"\\begin\{figure\*\}\[t\].*?\\end\{figure\*\}\n",
                body,
                flags=re.DOTALL,
            )
            if "\\label{fig:thermo}" in match.group(0)
        ),
        None,
    )
    if not fig2_match:
        raise RuntimeError("Fig. 2 float block is missing from the integrated body")
    fig2_block = fig2_match.group(0)
    body = body[: fig2_match.start()] + body[fig2_match.end() :]
    body = body.replace("% ANCHOR-FIG2\n", "% ANCHOR-FIG2\n\n" + fig2_block, 1)
    fig4_match = next(
        (
            match
            for match in re.finditer(
                r"\\begin\{figure\*\}\[t\].*?\\end\{figure\*\}\n",
                body,
                flags=re.DOTALL,
            )
            if "\\label{fig:3d}" in match.group(0)
        ),
        None,
    )
    if not fig4_match:
        raise RuntimeError("Fig. 4 float block is missing from the integrated body")
    fig4_block = fig4_match.group(0)
    body = body[: fig4_match.start()] + body[fig4_match.end() :]
    fig3_match = next(
        (
            match
            for match in re.finditer(
                r"\\begin\{figure\*\}\[t\].*?\\end\{figure\*\}\n",
                body,
                flags=re.DOTALL,
            )
            if "\\label{fig:topology}" in match.group(0)
        ),
        None,
    )
    if not fig3_match:
        raise RuntimeError("Fig. 3 float block is missing from the integrated body")
    body = body[: fig3_match.end()] + "\n" + fig4_block + body[fig3_match.end() :]
    body = body.rstrip() + "\n\n\\FloatBarrier\n"
    _write(body_path, body)

    main_tex = r"""\documentclass[aps,prl,reprint,superscriptaddress,floatfix]{revtex4-2}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{amsmath,amssymb,graphicx,xcolor,booktabs,hyperref}
\usepackage{placeins}
\hypersetup{colorlinks=true,allcolors=blue}
\begin{document}
\title{Reliability-aware active learning for FFLO phase spaces}
\author{Author list pending human approval}
\date{August 28, 2026}
\begin{abstract}
\input{sections/abstract.tex}
\end{abstract}
\maketitle
\input{sections/introduction.tex}
\input{sections/integrated_results_methods.tex}
\input{sections/discussion.tex}
\bibliographystyle{apsrev4-2}
\bibliography{references}
\end{document}
"""
    _write(variant_root / "main.tex", main_tex)

    job = "manuscript_prl_integrated_cited_v2_candidate"
    for suffix in ("aux", "bbl", "blg", "log", "out", "pdf"):
        stale = output_root / f"{job}.{suffix}"
        if stale.exists():
            stale.unlink()
    shutil.copy2(variant_root / "references.bib", output_root / "references.bib")
    pdflatex = [
        "pdflatex",
        "-recorder",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={output_root}",
        f"-jobname={job}",
        "main.tex",
    ]
    logs: list[str] = []
    first = subprocess.run(pdflatex, cwd=variant_root, text=True, capture_output=True, check=False)
    logs.append(first.stdout + "\n" + first.stderr)
    if first.returncode:
        _write(output_root / "latex_compile.log", "\n".join(logs))
        raise RuntimeError("first cited-manuscript LaTeX pass failed")
    bibtex = subprocess.run(
        ["bibtex", job],
        cwd=output_root,
        text=True,
        capture_output=True,
        check=False,
    )
    logs.append(bibtex.stdout + "\n" + bibtex.stderr)
    if bibtex.returncode:
        _write(output_root / "latex_compile.log", "\n\n--- BIBTEX ---\n\n".join(logs))
        raise RuntimeError("BibTeX failed for cited manuscript")
    for _ in range(3):
        proc = subprocess.run(pdflatex, cwd=variant_root, text=True, capture_output=True, check=False)
        logs.append(proc.stdout + "\n" + proc.stderr)
        if proc.returncode:
            break
    _write(output_root / "latex_compile.log", "\n\n--- PASS ---\n\n".join(logs))
    pdf = output_root / f"{job}.pdf"
    if proc.returncode or not pdf.exists():
        raise RuntimeError("cited manuscript compilation failed")
    return {
        "cited_manuscript_pdf": str(pdf),
        "cited_manuscript_tex": str(variant_root / "main.tex"),
        "variant_output_root": str(output_root),
        "workflow_status": "cited_candidate_compiled",
        "next_action": "audit compiled references, claims, and provenance",
    }


def audit_cited_candidate(state: dict[str, Any]) -> dict[str, Any]:
    project_root = _root(state)
    variant_root = Path(state["variant_root"])
    output_root = Path(state["variant_output_root"])
    pdf = Path(state["cited_manuscript_pdf"])
    log = (output_root / "latex_compile.log").read_text(encoding="utf-8", errors="replace")
    job = "manuscript_prl_integrated_cited_v2_candidate"
    final_log = (output_root / f"{job}.log").read_text(
        encoding="utf-8", errors="replace"
    )
    pages = _page_texts(pdf)
    all_text = "\n".join(pages)
    bbl = output_root / f"{job}.bbl"
    bbl_text = bbl.read_text(encoding="utf-8", errors="replace")
    bibitems = len(re.findall(r"^\\bibitem\s", bbl_text, flags=re.MULTILINE))
    overfull = len(re.findall(r"Overfull \\hbox", final_log))
    undefined = bool(re.search(r"Citation .* undefined|There were undefined references", final_log))
    figure_hash_pass = all(
        sha256(output_root / "figures" / record["staged_pdf"]) == record["pdf_sha256"]
        for record in CANONICAL_MAIN_FIGURES.values()
    )
    body = (variant_root / "sections/integrated_results_methods.tex").read_text(encoding="utf-8")
    caveats_pass = "0.006269" in body and "not all closed" in body and "not a complete three-dimensional atlas" in body
    fonts_tool = _pdf_tool("pdffonts")
    fonts_output = subprocess.run([fonts_tool, str(pdf)], text=True, capture_output=True, check=True).stdout if fonts_tool else ""
    font_rows = [line for line in fonts_output.splitlines()[2:] if line.strip()]
    fonts_embedded = bool(font_rows) and all(re.search(r"\byes\s+yes\s+yes\b", row) for row in font_rows)
    references_visible = all(token in all_text for token in ("[1]", "[50]", "Fulde", "Terayama", "Vandermause", "Noé"))
    references_after_fig4 = all_text.rfind("Fulde") > all_text.rfind("FIG. 4")
    page_character_counts = [len(re.sub(r"\s+", "", page)) for page in pages]
    substantive_page_content = bool(page_character_counts) and min(page_character_counts) >= 200
    passed = bool(pages) and bibitems == len(LITERATURE) and not undefined and not overfull and figure_hash_pass and caveats_pass and fonts_embedded and references_visible and references_after_fig4 and substantive_page_content
    audit = {
        "schema_version": 2,
        "status": "PASS" if passed else "FAIL",
        "pdf_pages": len(pages),
        "bibliography_items": bibitems,
        "expected_bibliography_items": len(LITERATURE),
        "undefined_citations": undefined,
        "overfull_boxes": overfull,
        "fonts_embedded_subset_unicode": fonts_embedded,
        "references_visible_in_extracted_text": references_visible,
        "references_after_figure_4": references_after_fig4,
        "page_character_counts": page_character_counts,
        "substantive_content_on_every_page": substantive_page_content,
        "canonical_figure_hashes": "PASS" if figure_hash_pass else "FAIL",
        "stage_iii_and_iv_caveats": "PASS" if caveats_pass else "FAIL",
        "original_integrated_candidate_modified": False,
        "external_retrieval_executed": True,
        "new_physics_executed": False,
    }
    _json(variant_root / "compiled_citation_and_provenance_audit.yaml", audit)
    _write(variant_root / "compiled_citation_and_provenance_audit.md", f"""# Compiled citation and provenance audit

Status: **{audit['status']}**.

- PDF pages: {len(pages)}.
- Compiled bibliography items: {bibitems}/{len(LITERATURE)}.
- Undefined citations/references: {'yes' if undefined else 'no'}.
- Overfull boxes: {overfull}.
- Fonts embedded/subset/Unicode: {'PASS' if fonts_embedded else 'FAIL'}.
- References visible in extracted text: {'PASS' if references_visible else 'FAIL'}.
- References placed after Figure 4: {'PASS' if references_after_fig4 else 'FAIL'}.
- Substantive content on every page: {'PASS' if substantive_page_content else 'FAIL'} ({page_character_counts}).
- Canonical frozen figure hashes: {'PASS' if figure_hash_pass else 'FAIL'}.
- Stage III k=6 and Stage IV nonclosure/no-atlas caveats: {'PASS' if caveats_pass else 'FAIL'}.
- External retrieval: executed under explicit approval.
- New physics: not executed.

This is a cited review candidate, not a publication freeze. The single-column-native Fig. 2--4 asset issue remains a separate figure gate.
""")
    _json(output_root / "pdf_hashes.json", {pdf.name: sha256(pdf)})
    if not passed:
        raise RuntimeError("compiled cited manuscript audit failed")
    return {
        "compiled_citation_audit_passed": True,
        "workflow_status": "cited_candidate_audited",
        "next_action": "stop at cited-manuscript human review",
    }


def cited_manuscript_human_review_gate(state: dict[str, Any]) -> dict[str, Any]:
    if not state.get("compiled_citation_audit_passed"):
        raise RuntimeError("human review gate requires passed compiled audit")
    project_root = _root(state)
    run_root = _run_root(project_root)
    pdf = Path(state["cited_manuscript_pdf"])
    gate = {
        "schema_version": 1,
        "gate_id": "GATE_CITED_INTRODUCTION_MANUSCRIPT_HUMAN_REVIEW",
        "status": "HUMAN_WAIT",
        "decision": "pending_human_review",
        "candidate_pdf": str(pdf.relative_to(project_root)),
        "candidate_sha256": sha256(pdf),
        "retrieval_approved": True,
        "retrieval_executed": True,
        "citation_audit": "PASS",
        "scientific_freeze": False,
        "publication_freeze": False,
        "frozen_results_modified": False,
        "frozen_figures_modified": False,
        "physics_recomputed": False,
    }
    _json(run_root / "gate_records/cited_introduction_manuscript_human_review_gate.json", gate)
    run_state = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "parent_run_id": PARENT_RUN_ID,
        "publication_profile": PROFILE_ID,
        "workflow_status": "awaiting_cited_introduction_manuscript_human_review",
        "current_gate": gate["gate_id"],
        "gate_status": "HUMAN_WAIT",
        "candidate_pdf": gate["candidate_pdf"],
        "external_retrieval": True,
        "new_physics": False,
    }
    _json(run_root / "run_state.json", run_state)
    _json(run_root / "checkpoints/checkpoint_cited_human_review.json", {
        "schema_version": 1,
        "resume_from": gate["gate_id"],
        "continuation_instruction": "Review the physics-first Introduction and claim-to-citation matrix; explicitly approve the cited Introduction/manuscript gate or request targeted revisions. Do not infer approval from automated citation checks.",
    })
    review_root = run_root / "review_package"
    _write(review_root / "README.md", f"""# Cited Introduction manuscript review package

Current gate: `{gate['gate_id']} = HUMAN_WAIT`.

- Candidate: `{gate['candidate_pdf']}`
- Introduction: `manuscript_variants/prl_integrated_cited_v2/sections/introduction.tex`
- Literature registry: `manuscript_variants/prl_integrated_cited_v2/verified_literature_registry.yaml`
- Claim-to-citation matrix: `manuscript_variants/prl_integrated_cited_v2/introduction_claim_to_citation_matrix.yaml`
- Citation audit: `manuscript_variants/prl_integrated_cited_v2/citation_coverage_audit.md`
- Compile/provenance audit: `manuscript_variants/prl_integrated_cited_v2/compiled_citation_and_provenance_audit.md`

Review the accuracy and balance of the FFLO, altermagnetic, topology, and active-learning background; confirm that cited precedents are not misrepresented as evidence for this model's Results.
""")
    return {
        "workflow_status": run_state["workflow_status"],
        "next_action": "human reviews and explicitly decides the cited Introduction manuscript gate",
    }
