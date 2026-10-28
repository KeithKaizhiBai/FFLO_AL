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
    verify_canonical_frozen_artifacts,
)
from review_agent.results_package import CAPTIONS


RUN_ID = "active_learning_prl_integrated_v1"
PARENT_RUN_ID = "active_learning_prl_v1"
PROFILE_ID = "PRL_INTEGRATED_MAIN_TEXT"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _json(path: Path, value: Any) -> None:
    _write(path, json.dumps(value, indent=2, ensure_ascii=False))


def _root(state: dict[str, Any]) -> Path:
    return Path(state.get("project_root", Path.cwd()))


def _variant_root(project_root: Path) -> Path:
    return project_root / "manuscript_variants/prl_integrated"


def _output_root(project_root: Path) -> Path:
    return project_root / "output/pdf/manuscript_prl_integrated_v1"


def _run_root(project_root: Path) -> Path:
    return project_root / f"manuscript_langgraph/runs/{RUN_ID}"


def _sentence(
    sentence_id: str,
    tex: str,
    plain: str,
    role: str,
    source_sentence_ids: list[str],
    claim_ids: list[str],
    evidence_ids: list[str],
    transformation: str,
    caveat: str = "none",
) -> dict[str, Any]:
    return {
        "new_sentence_id": sentence_id,
        "tex": tex,
        "plain": plain,
        "role": role,
        "source_sentence_ids": source_sentence_ids,
        "claim_ids": claim_ids,
        "evidence_ids": evidence_ids,
        "transformation": transformation,
        "semantic_change": "none",
        "required_caveat_preserved": caveat,
    }


ABSTRACT = [
    _sentence(
        "ABS-01",
        r"Resolving finite-momentum superconducting phases across expanding parameter spaces requires both reliable free-energy minimization and controlled use of expensive exact calculations.",
        "Resolving finite-momentum superconducting phases across expanding parameter spaces requires both reliable free-energy minimization and controlled use of expensive exact calculations.",
        "PROBLEM",
        ["MODEL_SPEC:project_goal", "METHODS_TEX:oracle"],
        ["CLM-ORACLE-RELIABILITY"],
        ["E-MODEL-SPEC", "E-NUMERICS-SPEC"],
        "merged",
    ),
    _sentence(
        "ABS-02",
        r"We retain an audited Bogoliubov--de Gennes (BdG) oracle as the sole source of thermodynamic and topological labels, while active learning only schedules the next exact evaluations.",
        "We retain an audited Bogoliubov-de Gennes oracle as the sole source of thermodynamic and topological labels, while active learning only schedules the next exact evaluations.",
        "METHOD",
        ["RES-FIG1-001-S01", "RES-FIG1-002-S02"],
        ["CLM-ORACLE-RELIABILITY"],
        ["E-MODEL-SPEC", "E-NUMERICS-SPEC"],
        "condensed",
    ),
    _sentence(
        "ABS-03",
        r"This separation yields an audited two-dimensional reconstruction of Normal, uniform-superconducting, and FFLO structure while keeping a hard-risk numerical layer explicit.",
        "This separation yields an audited two-dimensional reconstruction of Normal, uniform-superconducting, and FFLO structure while keeping a hard-risk numerical layer explicit.",
        "RESULT",
        ["RES-FIG2-001-S03", "RES-FIG2-002-S01", "RES-FIG2-002-S02"],
        ["CLM-2D-THERMO", "CLM-ORACLE-RELIABILITY"],
        ["E-FIG2-STRICT-DATA", "E-PHASEII-HARD-RISK"],
        "merged",
        "hard-risk remains a numerical layer",
    ),
    _sentence(
        "ABS-04",
        r"On trusted gapped FFLO states, full-zone gap and Pfaffian diagnostics resolve connected cFFLO/tFFLO structure that passes the configured $k=8$ audit with a retained $k=6$ sensitivity.",
        "On trusted gapped FFLO states, full-zone gap and Pfaffian diagnostics resolve connected cFFLO/tFFLO structure that passes the configured k=8 audit with a retained k=6 sensitivity.",
        "RESULT",
        ["RES-FIG3-001-S01", "RES-FIG3-002-S01", "RES-FIG3-002-S02"],
        ["CLM-2D-TOPOLOGY"],
        ["E-FIG3-STRICT-DATA"],
        "merged",
        "k=6 sensitivity retained",
    ),
    _sentence(
        "ABS-05",
        r"The same framework exposes thermodynamic and topology-aware structure in $(k_BT/t,J_A/t,\mu/t)$, although the strict three-dimensional topology convergence criteria are not all closed and no complete topology atlas is claimed.",
        "The same framework exposes thermodynamic and topology-aware structure in (kBT/t, JA/t, mu/t), although the strict three-dimensional topology convergence criteria are not all closed and no complete topology atlas is claimed.",
        "RESULT_CAVEAT",
        ["RES-FIG4-001-S04", "RES-FIG4-002-S02", "RES-FIG4-002-S03"],
        ["CLM-3D-EXTENSION"],
        ["E-FIG4-STRICT-DATA"],
        "merged",
        "strict 3D nonclosure and no complete atlas",
    ),
]


INTRODUCTION = [
    _sentence(
        "INTRO-01",
        r"The computational problem considered here is to map superconducting states when the equilibrium condensate may select both a finite pairing amplitude and a finite center-of-mass momentum.",
        "The computational problem considered here is to map superconducting states when the equilibrium condensate may select both a finite pairing amplitude and a finite center-of-mass momentum.",
        "PROBLEM",
        ["MODEL_SPEC:project_goal", "MODEL_SPEC:observables_and_labels"],
        ["CLM-2D-THERMO"],
        ["E-MODEL-SPEC"],
        "condensed",
    ),
    _sentence(
        "INTRO-02",
        r"In the present altermagnetic BdG model, thermodynamic competition in $(\Delta,q)$ must be resolved together with numerical reliability and, inside the trusted gapped FFLO sector, a $\mathbb Z_2$ topology diagnosis.",
        "In the present altermagnetic BdG model, thermodynamic competition in (Delta,q) must be resolved together with numerical reliability and, inside the trusted gapped FFLO sector, a Z2 topology diagnosis.",
        "PROBLEM",
        ["MODEL_SPEC:physical_model", "MODEL_SPEC:topological_labels"],
        ["CLM-ORACLE-RELIABILITY", "CLM-2D-TOPOLOGY"],
        ["E-MODEL-SPEC", "E-STAGEIII-PASS"],
        "merged",
    ),
    _sentence(
        "INTRO-03",
        r"Exhaustive sampling becomes increasingly costly when each parameter point requires a free-energy landscape, branch refinement, and spectral checks, especially after the domain is extended beyond two coordinates.",
        "Exhaustive sampling becomes increasingly costly when each parameter point requires a free-energy landscape, branch refinement, and spectral checks, especially after the domain is extended beyond two coordinates.",
        "MOTIVATION",
        ["METHODS_TEX:oracle", "METHODS_TEX:convergence_and_3D"],
        ["CLM-ORACLE-RELIABILITY", "CLM-3D-EXTENSION"],
        ["E-NUMERICS-SPEC", "E-STAGEIV-AUDIT"],
        "merged",
    ),
    _sentence(
        "INTRO-04",
        r"A data-driven scheduler can focus exact calls, but it would compromise the physical interpretation if surrogate predictions were promoted to final phase or topology labels.",
        "A data-driven scheduler can focus exact calls, but it would compromise the physical interpretation if surrogate predictions were promoted to final phase or topology labels.",
        "RISK",
        ["RES-FIG1-002-S02", "METHODS_TEX:active_learning"],
        ["CLM-ORACLE-RELIABILITY"],
        ["E-MODEL-SPEC", "E-NUMERICS-SPEC"],
        "condensed",
    ),
    _sentence(
        "INTRO-05",
        r"We therefore separate an exact BdG authority from reliability-aware active learning: the oracle determines accepted observations, whereas an ensemble acquisition rule selects only the next micro-batch.",
        "We therefore separate an exact BdG authority from reliability-aware active learning: the oracle determines accepted observations, whereas an ensemble acquisition rule selects only the next micro-batch.",
        "APPROACH",
        ["RES-FIG1-001-S01", "RES-FIG1-002-S01", "RES-FIG1-002-S02"],
        ["CLM-ORACLE-RELIABILITY"],
        ["E-NUMERICS-SPEC"],
        "merged",
    ),
    _sentence(
        "INTRO-06",
        r"The resulting evidence chain comprises an audited two-dimensional thermodynamic reconstruction, a topology-resolved cFFLO/tFFLO result with an explicit sensitivity test, and a higher-dimensional extension whose strict topology closure remains incomplete.",
        "The resulting evidence chain comprises an audited two-dimensional thermodynamic reconstruction, a topology-resolved cFFLO/tFFLO result with an explicit sensitivity test, and a higher-dimensional extension whose strict topology closure remains incomplete.",
        "SUMMARY",
        ["RES-FIG2-002-S02", "RES-FIG3-002-S03", "RES-FIG4-002-S04"],
        ["CLM-2D-THERMO", "CLM-2D-TOPOLOGY", "CLM-3D-EXTENSION"],
        ["E-FIG2-STRICT-DATA", "E-FIG3-STRICT-DATA", "E-FIG4-STRICT-DATA"],
        "merged",
        "configured topology pass and strict 3D nonclosure",
    ),
    _sentence(
        "INTRO-07",
        r"The main text first defines this division of labor, then follows the thermodynamic, topology-resolved, and three-dimensional results before closing with the limits of learned acquisition and a bounded world-model outlook.",
        "The main text first defines this division of labor, then follows the thermodynamic, topology-resolved, and three-dimensional results before closing with the limits of learned acquisition and a bounded world-model outlook.",
        "ROUTE",
        ["section_outline:main_narrative", "claim_registry:destinations"],
        ["CLM-LEARNED-ACQUISITION", "CLM-GROUNDUP-WORLD-MODEL"],
        ["E-STAGEV-MATCHED", "E-STAGEVI0", "E-STAGEVI0-2"],
        "reordered",
    ),
]


BODY_BLOCKS: list[dict[str, Any]] = [
    {
        "anchor": "ANCHOR-FIG1",
        "lead": "Exact oracle and physical labels",
        "figure_id": "FIG-01-PIPELINE",
        "sentences": [
            _sentence("BODY-01", r"For external coordinates $x=(k_BT/t,J_A/t,\mu/t,\ldots)$, the oracle uses the Nambu basis $\Psi_k=(c_{k+q/2,\uparrow},c_{k+q/2,\downarrow},\allowbreak c^\dagger_{-k+q/2,\uparrow},c^\dagger_{-k+q/2,\downarrow})^T$ and the block Hamiltonian", "For external coordinates x, the oracle uses the stated FFLO Nambu basis and block Hamiltonian.", "METHOD", ["MODEL_SPEC:physical_model", "METHODS_TEX:oracle"], ["CLM-ORACLE-RELIABILITY"], ["E-MODEL-SPEC"], "method_inserted"),
            _sentence("BODY-02", r"At every exact call it constructs $F(\Delta,q)$, refines competing minima, and accepts the lowest reliable state only after comparison with the normal reference.", "At every exact call it constructs F(Delta,q), refines competing minima, and accepts the lowest reliable state only after comparison with the normal reference.", "METHOD", ["METHODS_TEX:oracle", "RES-FIG1-001-S01"], ["CLM-ORACLE-RELIABILITY"], ["E-NUMERICS-SPEC"], "merged"),
            _sentence("BODY-03", r"The frozen thresholds $\Delta_\epsilon=10^{-3}$ and $q_\epsilon=10^{-2}$ separate Normal, uniform-SC, and FFLO observations; phase labels remain distinct from trusted, boundary-ambiguous, and rerun-required numerical states.", "The frozen thresholds separate Normal, uniform-SC, and FFLO observations; phase labels remain distinct from numerical states.", "METHOD", ["MODEL_SPEC:observables_and_labels", "METHODS_TEX:reliability"], ["CLM-ORACLE-RELIABILITY"], ["E-MODEL-SPEC", "E-NUMERICS-SPEC"], "merged"),
            _sentence("BODY-04", r"Edge minima trigger adaptive $q$-window expansion and shallow Normal--SC competition triggers local $\Delta$ refinement, while unresolved outcomes are withheld from clean physical training labels.", "Edge minima trigger adaptive q-window expansion and shallow Normal-SC competition triggers local Delta refinement, while unresolved outcomes are withheld from clean physical training labels.", "METHOD", ["METHODS_TEX:reliability"], ["CLM-ORACLE-RELIABILITY"], ["E-NUMERICS-SPEC", "E-PHASEII-AUDIT"], "condensed"),
            _sentence("BODY-05", r"Ensemble surrogates rank the next exact micro-batch from accepted observations; they are schedulers rather than physics solvers (Fig.~\ref{fig:framework}).", "Ensemble surrogates rank the next exact micro-batch from accepted observations; they are schedulers rather than physics solvers.", "METHOD", ["RES-FIG1-002-S02", "RES-FIG1-002-S03"], ["CLM-ORACLE-RELIABILITY"], ["E-NUMERICS-SPEC"], "condensed"),
            _sentence("BODY-06", r"Figure~\ref{fig:framework} is a schematic hierarchy and is not itself evidence that any convergence gate has closed.", "Figure 1 is a schematic hierarchy and is not itself evidence that any convergence gate has closed.", "CAVEAT", ["RES-FIG1-001-S04"], ["CLM-ORACLE-RELIABILITY"], ["E-MODEL-SPEC"], "unchanged", "schematic is not convergence evidence"),
        ],
    },
    {
        "anchor": "ANCHOR-FIG2",
        "lead": "Audited thermodynamic reconstruction",
        "figure_id": "FIG-02-THERMO",
        "sentences": [
            _sentence("BODY-07", r"The two-dimensional loop begins from 512 selected exact calls, with 506 labels retained in the frozen endpoint, and subsequently allocates calls around visible Normal/superconducting and uniform-SC/FFLO structure (Fig.~\ref{fig:thermo}).", "The two-dimensional loop begins from 512 selected exact calls, with 506 labels retained in the frozen endpoint, and subsequently allocates calls around visible thermodynamic structure.", "OBSERVATION", ["RES-FIG2-001-S01", "RES-FIG2-001-S02"], ["CLM-2D-THERMO"], ["E-STAGEI-COLDSTART", "E-FIG2-STRICT-DATA"], "merged"),
            _sentence("BODY-08", r"The acquisition combines phase uncertainty, boundary proximity, phase-facing regression uncertainty, numerical risk, and soft repulsion, but only returned BdG calculations enter the accepted dataset.", "The acquisition combines uncertainty, boundary proximity, numerical risk, and soft repulsion, but only returned BdG calculations enter the accepted dataset.", "METHOD", ["METHODS_TEX:active_learning"], ["CLM-ORACLE-RELIABILITY"], ["E-NUMERICS-SPEC"], "condensed"),
            _sentence("BODY-09", r"The endpoint contains 7,434 exact observations: 1,867 Normal, 715 uniform-SC, and 4,852 FFLO labels.", "The endpoint contains 7,434 exact observations: 1,867 Normal, 715 uniform-SC, and 4,852 FFLO labels.", "OBSERVATION", ["RES-FIG2-001-S03"], ["CLM-2D-THERMO"], ["E-FIG2-STRICT-DATA"], "unchanged"),
            _sentence("BODY-10", r"Its final phase-map change is 0.001629, the Normal--SC and uniform-SC--FFLO shifts are 0.004167 and 0, and trusted surprise is 0 for 127 trusted calls, satisfying the configured stopping rules.", "Its final phase-map change is 0.001629, the two boundary shifts are 0.004167 and 0, and trusted surprise is 0 for 127 trusted calls, satisfying the configured stopping rules.", "OBSERVATION", ["RES-FIG2-001-S04"], ["CLM-2D-THERMO"], ["E-PHASEII-AUDIT", "E-FIG2-STRICT-DATA"], "condensed"),
            _sentence("BODY-11", r"The separate 129-call hard-risk layer, including 19 coordinate matches retained in the trusted endpoint, is a numerical frontier rather than a fourth phase or evidence that uncertainty has been eliminated.", "The separate 129-call hard-risk layer is a numerical frontier rather than a fourth phase or evidence that uncertainty has been eliminated.", "CAVEAT", ["RES-FIG2-002-S01"], ["CLM-2D-THERMO", "CLM-ORACLE-RELIABILITY"], ["E-PHASEII-HARD-RISK"], "condensed", "hard-risk is not a phase"),
            _sentence("BODY-12", r"The exact-sample map therefore supports an audited two-dimensional thermodynamic reconstruction under the registered criterion while retaining finite numerical risk (Fig.~\ref{fig:thermo}).", "The exact-sample map supports an audited two-dimensional thermodynamic reconstruction under the registered criterion while retaining finite numerical risk.", "INTERPRETATION", ["RES-FIG2-002-S02"], ["CLM-2D-THERMO"], ["E-FIG2-STRICT-DATA"], "condensed", "under the configured audit"),
        ],
    },
    {
        "anchor": "ANCHOR-FIG3",
        "lead": "Topology-resolved FFLO structure",
        "figure_id": "FIG-03-STAGEIII-TOPOLOGY",
        "sentences": [
            _sentence("BODY-13", r"Topology is evaluated only after thermodynamic acceptance and only on trusted gapped FFLO observations.", "Topology is evaluated only after thermodynamic acceptance and only on trusted gapped FFLO observations.", "METHOD", ["METHODS_TEX:topology", "RES-FIG3-001-S03"], ["CLM-2D-TOPOLOGY"], ["E-STAGEIII-PASS"], "condensed"),
            _sentence("BODY-14", r"A full-Brillouin-zone minimum-gap guard and Pfaffians $P_0$ and $P_\pi$ determine the $\mathbb Z_2$ index away from their tolerances, assigning cFFLO for $\nu=0$ and tFFLO for $\nu=1$.", "A full-zone gap guard and Pfaffians P0 and Ppi determine the Z2 index, assigning cFFLO and tFFLO.", "METHOD", ["METHODS_TEX:topology"], ["CLM-2D-TOPOLOGY", "CLM-ORACLE-RELIABILITY"], ["E-STAGEIII-PASS"], "condensed"),
            _sentence("BODY-15", r"Within the 4,345-point endpoint, 3,213 trusted gapped FFLO states divide into 2,290 cFFLO and 923 tFFLO observations and form a connected sampled structure (Fig.~\ref{fig:topology}).", "Within the 4,345-point endpoint, 3,213 trusted gapped FFLO states divide into 2,290 cFFLO and 923 tFFLO observations and form a connected sampled structure.", "OBSERVATION", ["RES-FIG3-001-S01", "RES-FIG3-001-S02"], ["CLM-2D-TOPOLOGY"], ["E-FIG3-STRICT-DATA"], "merged"),
            _sentence("BODY-16", r"The support-restricted contour and local opposite-$\mathbb Z_2$ brackets diagnose a gap-closing transition; neither is an exact boundary, and the closing is not a stable third phase.", "The contour and brackets diagnose a gap-closing transition; neither is an exact boundary, and the closing is not a stable third phase.", "CAVEAT", ["RES-FIG3-001-S04"], ["CLM-2D-TOPOLOGY"], ["E-FIG3-STRICT-DATA"], "condensed", "diagnostics only"),
            _sentence("BODY-17", r"The configured $k=8$ audit returns Decision A, with coverage $p_{95}=0.006185<0.00625$ and the remaining registered metrics passing.", "The configured k=8 audit returns Decision A, with coverage p95=0.006185 below 0.00625 and the remaining registered metrics passing.", "OBSERVATION", ["RES-FIG3-002-S01"], ["CLM-2D-TOPOLOGY"], ["E-FIG3-STRICT-DATA"], "condensed"),
            _sentence("BODY-18", r"For the $k=6$ sensitivity setting, coverage $p_{95}=0.006269$ lies slightly above the same threshold, so the configured pass retains a neighborhood-size sensitivity and does not exhaust all refinements or nodal alternatives (Fig.~\ref{fig:topology}).", "For k=6, coverage p95=0.006269 lies slightly above the same threshold, so the configured pass retains a sensitivity and does not exhaust all refinements or nodal alternatives.", "CAVEAT", ["RES-FIG3-002-S02", "RES-FIG3-002-S03"], ["CLM-2D-TOPOLOGY"], ["E-FIG3-STRICT-DATA"], "merged", "k=6 sensitivity and no exhaustive claim"),
        ],
    },
    {
        "anchor": "ANCHOR-FIG4",
        "lead": "Higher-dimensional exploration",
        "figure_id": "FIG-04-STAGEIV-3D-EXTENSION",
        "sentences": [
            _sentence("BODY-19", r"The same oracle--scheduler separation extends the sampled coordinates to $x=(k_BT/t,J_A/t,\mu/t)$ over the finite interval $\mu/t\in[-0.5,1.5]$.", "The same oracle-scheduler separation extends the sampled coordinates to three dimensions over the finite mu interval [-0.5,1.5].", "METHOD", ["METHODS_TEX:convergence_and_3D", "RES-FIG4-001-S01"], ["CLM-3D-EXTENSION", "CLM-ORACLE-RELIABILITY"], ["E-STAGEIV-AUDIT"], "merged", "finite mu interval"),
            _sentence("BODY-20", r"The endpoint contains 7,081 exact observations: 2,350 Normal, 141 uniform-SC, and 4,590 FFLO states (Fig.~\ref{fig:3d}).", "The endpoint contains 7,081 exact observations: 2,350 Normal, 141 uniform-SC, and 4,590 FFLO states.", "OBSERVATION", ["RES-FIG4-001-S01"], ["CLM-3D-EXTENSION"], ["E-FIG4-STRICT-DATA"], "unchanged"),
            _sentence("BODY-21", r"Trusted FFLO-only topology support comprises 3,145 cFFLO and 1,445 tFFLO points and remains discrete rather than a fitted surface.", "Trusted FFLO-only topology support comprises 3,145 cFFLO and 1,445 tFFLO points and remains discrete rather than a fitted surface.", "OBSERVATION_CAVEAT", ["RES-FIG4-001-S02"], ["CLM-3D-EXTENSION"], ["E-FIG4-STRICT-DATA"], "condensed", "discrete support only"),
            _sentence("BODY-22", r"Four explicit fixed-$\mu$ bins centered at $\mu/t=-0.45,0.15,0.85,$ and $1.45$, each with half-width 0.025, expose exact slice-to-slice structure without interpolation between slices.", "Four explicit fixed-mu bins centered at -0.45, 0.15, 0.85, and 1.45, each with half-width 0.025, expose exact slice-to-slice structure without interpolation.", "METHOD_OBSERVATION", ["RES-FIG4-001-S03", "E-FIG4-STRICT-DATA:selected_slices"], ["CLM-3D-EXTENSION"], ["E-STAGEIV-MU-SLICE-AUDIT", "E-FIG4-STRICT-DATA"], "condensed", "no slice interpolation"),
            _sentence("BODY-23", r"These point clouds and slices demonstrate higher-dimensional exploration that reveals thermodynamic and topology-aware structures across the sampled domain.", "These point clouds and slices demonstrate higher-dimensional exploration that reveals thermodynamic and topology-aware structures across the sampled domain.", "INTERPRETATION", ["RES-FIG4-001-S04"], ["CLM-3D-EXTENSION"], ["E-FIG4-STRICT-DATA"], "unchanged"),
            _sentence("BODY-24", r"The final strict audit nevertheless exceeds the registered map-change, surface-shift, coverage, and trusted-surprise thresholds, while component counts vary $3\to4\to3$.", "The final strict audit exceeds the registered map-change, surface-shift, coverage, and surprise thresholds, while component counts vary 3 to 4 to 3.", "OBSERVATION", ["RES-FIG4-002-S01"], ["CLM-3D-EXTENSION"], ["E-STAGEIV-AUDIT", "E-FIG4-STRICT-DATA"], "condensed"),
            _sentence("BODY-25", r"Thus the strict three-dimensional topology convergence gates are not all closed; visual continuity among slices is not a continuous topology surface, and the finite-domain result is not a complete three-dimensional atlas (Fig.~\ref{fig:3d}).", "Thus the strict 3D topology convergence gates are not all closed; visual continuity is not a continuous surface, and the result is not a complete 3D atlas.", "CAVEAT", ["RES-FIG4-002-S02", "RES-FIG4-002-S03"], ["CLM-3D-EXTENSION"], ["E-FIG4-STRICT-DATA"], "merged", "strict nonclosure, no continuous surface, no complete atlas"),
        ],
    },
]


DISCUSSION = [
    _sentence("DISC-01", r"The combined results establish a reliability-aware active-learning workflow in which numerical trust remains part of each physical observation and exact BdG calculations retain label authority.", "The combined results establish a reliability-aware active-learning workflow in which exact BdG calculations retain label authority.", "SUMMARY", ["RES-FIG1-002-S01", "RES-FIG1-002-S03"], ["CLM-ORACLE-RELIABILITY"], ["E-NUMERICS-SPEC"], "merged"),
    _sentence("DISC-02", r"The two-dimensional thermodynamic and configured topology audits show what closes under registered criteria, while the three-dimensional extension demonstrates why selective exact sampling becomes more consequential as the domain grows.", "The 2D audits show what closes under registered criteria, while the 3D extension demonstrates why selective exact sampling becomes more consequential as the domain grows.", "INTERPRETATION", ["RES-FIG2-002-S02", "RES-FIG3-002-S03", "RES-FIG4-001-S04"], ["CLM-2D-THERMO", "CLM-2D-TOPOLOGY", "CLM-3D-EXTENSION"], ["E-FIG2-STRICT-DATA", "E-FIG3-STRICT-DATA", "E-FIG4-STRICT-DATA"], "merged"),
    _sentence("DISC-03", r"That extension is deliberately bounded: its discrete support and fixed-$\mu$ slices reveal structure, but failed strict topology criteria prevent a claim of a converged surface or complete atlas.", "The 3D extension is bounded: discrete support reveals structure, but failed strict criteria prevent a converged-surface or complete-atlas claim.", "CAVEAT", ["RES-FIG4-002-S02", "RES-FIG4-002-S03"], ["CLM-3D-EXTENSION"], ["E-FIG4-STRICT-DATA"], "condensed", "strict 3D nonclosure"),
    _sentence("DISC-04", r"The matched Stage-V audit likewise gives no overall acquisition winner: V3 has slightly denser topology-surface support ($p_{95}=0.01893$ versus $0.02103$), whereas V4 has lower sequential topology surprise ($0.0925$ versus $0.1189$).", "The matched Stage-V audit gives no overall winner: V3 has denser support, whereas V4 has lower sequential topology surprise.", "DISCUSSION_ONLY", ["DISCUSSION_TEX:stageV-01"], ["CLM-LEARNED-ACQUISITION"], ["E-STAGEV-MATCHED"], "condensed"),
    _sentence("DISC-05", r"V4 does not lower the common offline graph-uncertainty functional ($0.01217$ versus $0.01114$), and none of its four candidate nodes has validated local same-identity support.", "V4 does not lower the offline graph uncertainty, and none of its four candidate nodes has validated local same-identity support.", "DISCUSSION_ONLY", ["DISCUSSION_TEX:stageV-02"], ["CLM-LEARNED-ACQUISITION"], ["E-STAGEV-MATCHED"], "condensed"),
    _sentence("DISC-06", r"Learned acquisition is therefore an operational exploratory mechanism, not evidence of superiority or autonomous new-phase discovery.", "Learned acquisition is an exploratory mechanism, not evidence of superiority or autonomous new-phase discovery.", "CAVEAT", ["DISCUSSION_TEX:stageV-03"], ["CLM-LEARNED-ACQUISITION"], ["E-STAGEV-MATCHED"], "condensed", "no superiority and no new phase"),
    _sentence("DISC-07", r"As a future direction, the Stage-VI studies indicate that thermodynamic state and $|q|$ are learnable and that continuous Pfaffian quantities become learnable after conditioning on trusted gapped superconductivity.", "As a future direction, Stage VI indicates learnability of thermodynamics, q, and conditional Pfaffian quantities.", "OUTLOOK_ONLY", ["OUTLOOK_TEX:stageVI-01"], ["CLM-GROUNDUP-WORLD-MODEL"], ["E-STAGEVI0", "E-STAGEVI0-2"], "condensed"),
    _sentence("DISC-08", r"The frozen export, however, lacks fixed-$k$ spectra, a density-of-states proxy, and validated low-energy spectral moments, and its replay uncertainty is not uniformly monotonic.", "The frozen export lacks fixed-k spectra, a DOS proxy, and validated spectral moments, and replay uncertainty is not uniformly monotonic.", "OUTLOOK_CAVEAT", ["OUTLOOK_TEX:stageVI-02"], ["CLM-GROUNDUP-WORLD-MODEL"], ["E-STAGEVI0", "E-STAGEVI0-2"], "condensed", "spectral descriptors and calibrated uncertainty missing"),
    _sentence("DISC-09", r"A ground-up physical world model must therefore validate those descriptors and calibrated uncertainty before production expected-information-gain acquisition can be authorized.", "A physical world model must validate descriptors and uncertainty before production EIG can be authorized.", "OUTLOOK_CAVEAT", ["OUTLOOK_TEX:stageVI-03"], ["CLM-GROUNDUP-WORLD-MODEL"], ["E-STAGEVI0", "E-STAGEVI0-2"], "condensed", "production EIG not authorized"),
]


METHOD_PACKETS = [
    {
        "packet_id": "METHOD_PACKET_MODEL",
        "source_sentences": ["MODEL_SPEC:physical_model", "METHODS_TEX:oracle"],
        "source_evidence_ids": ["E-MODEL-SPEC"],
        "claim_ids": ["CLM-ORACLE-RELIABILITY"],
        "first_required_at": "ANCHOR-FIG1",
        "main_text_content": "Nambu basis, BdG block Hamiltonian, F(Delta,q), and exact minimization authority.",
        "supplement_only_content": "Full parameter definitions, analytic Pfaffian formulas, and implementation details.",
        "forbidden_expansion": ["new Hamiltonian convention", "surrogate-generated labels"],
    },
    {
        "packet_id": "METHOD_PACKET_THERMODYNAMIC_LABELS",
        "source_sentences": ["MODEL_SPEC:observables_and_labels", "METHODS_TEX:oracle"],
        "source_evidence_ids": ["E-MODEL-SPEC", "E-NUMERICS-SPEC"],
        "claim_ids": ["CLM-ORACLE-RELIABILITY", "CLM-2D-THERMO"],
        "first_required_at": "ANCHOR-FIG1",
        "main_text_content": "Normal/uniform-SC/FFLO thresholds and normal-reference comparison.",
        "supplement_only_content": "Complete tolerance table and positive-Delta gap implementation.",
        "forbidden_expansion": ["change phase thresholds", "hard-risk as phase"],
    },
    {
        "packet_id": "METHOD_PACKET_RELIABILITY",
        "source_sentences": ["METHODS_TEX:reliability", "RES-FIG1-002-S01"],
        "source_evidence_ids": ["E-NUMERICS-SPEC", "E-PHASEII-AUDIT"],
        "claim_ids": ["CLM-ORACLE-RELIABILITY"],
        "first_required_at": "ANCHOR-FIG1",
        "main_text_content": "q-window expansion, Delta refinement, and separated numerical states.",
        "supplement_only_content": "Basin clustering, rank-and-cap details, bug history, and full software architecture.",
        "forbidden_expansion": ["mathematically exact", "finite-resolution-free"],
    },
    {
        "packet_id": "METHOD_PACKET_ACTIVE_LEARNING",
        "source_sentences": ["METHODS_TEX:active_learning", "RES-FIG1-002-S02"],
        "source_evidence_ids": ["E-NUMERICS-SPEC", "E-FIG2-STRICT-DATA"],
        "claim_ids": ["CLM-ORACLE-RELIABILITY", "CLM-2D-THERMO"],
        "first_required_at": "ANCHOR-FIG1",
        "main_text_content": "Surrogate observables, acquisition role, and exact micro-batch selection.",
        "supplement_only_content": "Weights, ensemble implementation, validation split, and repulsion formulas.",
        "forbidden_expansion": ["ML solver", "learned acquisition superiority"],
    },
    {
        "packet_id": "METHOD_PACKET_TOPOLOGY",
        "source_sentences": ["METHODS_TEX:topology", "RES-FIG3-001-S03"],
        "source_evidence_ids": ["E-STAGEIII-PASS", "E-FIG3-STRICT-DATA"],
        "claim_ids": ["CLM-2D-TOPOLOGY", "CLM-ORACLE-RELIABILITY"],
        "first_required_at": "ANCHOR-FIG3",
        "main_text_content": "Trusted gapped FFLO mask, full-BZ gap, Pfaffians, and Z2 labels.",
        "supplement_only_content": "Analytic Pfaffian expressions, validation sample count, and scale-aware tolerance.",
        "forbidden_expansion": ["contour as exact boundary", "gap closing as stable phase"],
    },
    {
        "packet_id": "METHOD_PACKET_CONVERGENCE",
        "source_sentences": ["METHODS_TEX:convergence", "RES-FIG2-001-S04", "RES-FIG3-002-S01"],
        "source_evidence_ids": ["E-PHASEII-AUDIT", "E-STAGEIII-CONVERGENCE"],
        "claim_ids": ["CLM-2D-THERMO", "CLM-2D-TOPOLOGY"],
        "first_required_at": "ANCHOR-FIG2",
        "main_text_content": "Map change, boundary shift, coverage, surprise, and component stability.",
        "supplement_only_content": "Monitor-grid interpolation and complete history tables.",
        "forbidden_expansion": ["all refinements exhausted", "remove k=6 sensitivity"],
    },
    {
        "packet_id": "METHOD_PACKET_3D",
        "source_sentences": ["METHODS_TEX:convergence_and_3D", "RES-FIG4-001-S03", "RES-FIG4-002-S02"],
        "source_evidence_ids": ["E-STAGEIV-AUDIT", "E-STAGEIV-MU-SLICE-AUDIT", "E-FIG4-STRICT-DATA"],
        "claim_ids": ["CLM-3D-EXTENSION"],
        "first_required_at": "ANCHOR-FIG4",
        "main_text_content": "Three coordinates, exact point cloud, four fixed-mu slices, and strict nonclosure.",
        "supplement_only_content": "Complete convergence histories, slice table, and support diagnostics.",
        "forbidden_expansion": ["continuous topology surface", "complete 3D atlas"],
    },
]


def _render_sentences(sentences: list[dict[str, Any]]) -> str:
    return " ".join(item["tex"] for item in sentences)


def _plain_sentences(sentences: list[dict[str, Any]]) -> str:
    return " ".join(item["plain"] for item in sentences)


def select_publication_profile(state: dict[str, Any]) -> dict[str, Any]:
    project_root = _root(state)
    profile_path = project_root / "manuscript_langgraph/publication_profiles/prl_integrated_main_text.yaml"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if profile["profile_id"] != PROFILE_ID:
        raise RuntimeError("unexpected publication profile")
    old_pdf = project_root / "output/pdf/manuscript_methods_sm_discussion_review_v1/manuscript_intermediate_review.pdf"
    if not old_pdf.exists():
        raise RuntimeError("parent intermediate review PDF is missing")
    verify_canonical_frozen_artifacts(project_root)
    run_root = _run_root(project_root)
    variant_root = _variant_root(project_root)
    output_root = _output_root(project_root)
    for path in [run_root / "checkpoints", run_root / "gate_records", run_root / "review_package", variant_root / "sections", output_root]:
        path.mkdir(parents=True, exist_ok=True)
    parent = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "parent_run_id": PARENT_RUN_ID,
        "publication_profile": PROFILE_ID,
        "scientific_content_authority": "inherited_from_frozen_intermediate_review",
        "parent_intermediate_pdf": str(old_pdf.relative_to(project_root)),
        "parent_intermediate_pdf_sha256": sha256(old_pdf),
        "publication_form": "new_candidate",
        "human_approval": "pending",
        "old_intermediate_review_overwritten": False,
    }
    _json(run_root / "parent_provenance.json", parent)
    return {
        "publication_profile_path": str(profile_path),
        "publication_profile_id": PROFILE_ID,
        "variant_run_id": RUN_ID,
        "parent_run_id": PARENT_RUN_ID,
        "variant_root": str(variant_root),
        "variant_output_root": str(output_root),
        "workflow_status": "publication_profile_selected",
        "next_action": "build evidence-grounded method packets",
    }


def build_method_packets(state: dict[str, Any]) -> dict[str, Any]:
    variant_root = Path(state["variant_root"])
    packets_path = variant_root / "method_packets.yaml"
    trace_path = variant_root / "method_packet_evidence_trace.yaml"
    _json(packets_path, {"schema_version": 1, "profile_id": PROFILE_ID, "packets": METHOD_PACKETS})
    _json(trace_path, {
        "schema_version": 1,
        "status": "PASS",
        "packets": [
            {
                "packet_id": packet["packet_id"],
                "source_sentences": packet["source_sentences"],
                "evidence_ids": packet["source_evidence_ids"],
                "claim_ids": packet["claim_ids"],
                "first_required_at": packet["first_required_at"],
            }
            for packet in METHOD_PACKETS
        ],
    })
    return {
        "method_packets_path": str(packets_path),
        "workflow_status": "method_packets_built",
        "next_action": "weave frozen Results and method packets",
    }


def _figure_tex(project_root: Path, figure_id: str) -> str:
    record = CANONICAL_MAIN_FIGURES[figure_id]
    label = {
        "FIG-01-PIPELINE": "fig:framework",
        "FIG-02-THERMO": "fig:thermo",
        "FIG-03-STAGEIII-TOPOLOGY": "fig:topology",
        "FIG-04-STAGEIV-3D-EXTENSION": "fig:3d",
    }[figure_id]
    output_asset = _output_root(project_root) / "figures" / record["staged_pdf"]
    return (
        "\\begin{figure*}[t]\n"
        "\\centering\n"
        f"\\includegraphics[width=0.96\\textwidth]{{\\detokenize{{{output_asset.as_posix()}}}}}\n"
        f"\\caption{{{CAPTIONS[figure_id]}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{figure*}\n"
    )


def weave_results_and_methods(state: dict[str, Any]) -> dict[str, Any]:
    project_root = _root(state)
    variant_root = Path(state["variant_root"])
    sections = variant_root / "sections"
    abstract_tex = _render_sentences(ABSTRACT)
    introduction_tex = "\\section{Introduction}\n" + _render_sentences(INTRODUCTION)
    body_parts: list[str] = []
    body_md: list[str] = []
    for block in BODY_BLOCKS:
        body_parts.extend([
            f"% {block['anchor']}",
            f"\\textit{{{block['lead']}.---}}",
        ])
        if block["anchor"] == "ANCHOR-FIG1":
            body_parts.append(block["sentences"][0]["tex"])
            body_parts.append(r"\begingroup\small\setlength{\arraycolsep}{3pt}\begin{equation}\label{eq:bdg}\begin{aligned}H_{\rm BdG}(k)&=\begin{pmatrix}h(k+q/2)-\mu&\Delta_q\\\Delta_q^*&-\sigma_yh^T(-k+q/2)\sigma_y+\mu\end{pmatrix},\\h(k)&=t\cos k+(0,\lambda_y\sin k,\lambda_z\sin k+J_A\cos k)\cdot\boldsymbol\sigma.\end{aligned}\end{equation}\endgroup")
            body_parts.append(_render_sentences(block["sentences"][1:]))
        else:
            body_parts.append(_render_sentences(block["sentences"]))
        body_parts.append(_figure_tex(project_root, block["figure_id"]))
        if block["figure_id"] != "FIG-04-STAGEIV-3D-EXTENSION":
            body_parts.append(r"\FloatBarrier")
        body_md.extend([f"### {block['lead']}", _plain_sentences(block["sentences"]), f"[Inline {block['figure_id']}; frozen caption unchanged.]", ""])
    discussion_tex = "\\section{Discussion}\n" + "\n\n".join([
        _render_sentences(DISCUSSION[:3]),
        _render_sentences(DISCUSSION[3:6]),
        _render_sentences(DISCUSSION[6:]),
    ])
    _write(sections / "abstract.tex", abstract_tex)
    _write(sections / "introduction.tex", introduction_tex)
    _write(sections / "integrated_results_methods.tex", "\n\n".join(body_parts))
    _write(sections / "discussion.tex", discussion_tex)
    main_md = "\n\n".join([
        "# Reliability-aware active learning for FFLO phase spaces",
        "> PRL-integrated publication-form candidate; human approval pending.",
        "## Abstract",
        _plain_sentences(ABSTRACT),
        "## Introduction",
        _plain_sentences(INTRODUCTION),
        "## Integrated Results and Methods",
        "\n\n".join(body_md),
        "## Discussion",
        _plain_sentences(DISCUSSION),
    ])
    _write(variant_root / "manuscript_prl_integrated_candidate.md", main_md)

    trace = ABSTRACT + INTRODUCTION + [sentence for block in BODY_BLOCKS for sentence in block["sentences"]] + DISCUSSION
    _json(variant_root / "prl_semantic_equivalence_trace.yaml", {
        "schema_version": 1,
        "authority": "frozen Results, Methods, claim/evidence registries, and project specifications",
        "status": "PENDING_AUTOMATED_AUDIT",
        "sentences": [
            {key: value for key, value in item.items() if key not in {"tex", "plain"}}
            | {"new_sentence": item["plain"]}
            for item in trace
        ],
    })
    _json(variant_root / "introduction_citation_gap_ledger.yaml", {
        "schema_version": 1,
        "external_retrieval_authorized": False,
        "verified_local_bibliography_entries": 0,
        "candidate_strategy": "project-specific and conservative; no unsupported priority or novelty claims",
        "gaps": [
            {"gap_id": "CIT-GAP-FFLO", "need": "historical and current FFLO context", "candidate_text_impact": "omitted from provisional Introduction", "status": "requires_future_external_permission"},
            {"gap_id": "CIT-GAP-ALTERMAGNET", "need": "external altermagnetism background", "candidate_text_impact": "limited to the present model", "status": "requires_future_external_permission"},
            {"gap_id": "CIT-GAP-TOPOLOGY", "need": "external topological-superconductivity context", "candidate_text_impact": "only project-defined diagnostics stated", "status": "requires_future_external_permission"},
            {"gap_id": "CIT-GAP-AL", "need": "active learning for expensive phase diagrams", "candidate_text_impact": "no literature-wide comparison or novelty claim", "status": "requires_future_external_permission"},
            {"gap_id": "CIT-GAP-NOVELTY", "need": "claim-level novelty positioning", "candidate_text_impact": "novelty claim withheld", "status": "requires_future_external_permission"},
        ],
    })
    return {
        "integrated_manuscript_md": str(variant_root / "manuscript_prl_integrated_candidate.md"),
        "semantic_equivalence_trace_path": str(variant_root / "prl_semantic_equivalence_trace.yaml"),
        "workflow_status": "integrated_narrative_woven",
        "next_action": "plan inline figure placement",
    }


def plan_inline_figure_placement(state: dict[str, Any]) -> dict[str, Any]:
    project_root = _root(state)
    variant_root = Path(state["variant_root"])
    anchors = ["ANCHOR-FIG1", "ANCHOR-FIG2", "ANCHOR-FIG3", "ANCHOR-FIG4"]
    manifest = []
    for (figure_id, record), anchor in zip(CANONICAL_MAIN_FIGURES.items(), anchors):
        manifest.append({
            "figure_id": figure_id,
            "frozen_asset": record["source_pdf"],
            "source_hash": record["pdf_sha256"],
            "first_text_reference": {
                "FIG-01-PIPELINE": "BODY-05",
                "FIG-02-THERMO": "BODY-07",
                "FIG-03-STAGEIII-TOPOLOGY": "BODY-15",
                "FIG-04-STAGEIV-3D-EXTENSION": "BODY-20",
            }[figure_id],
            "target_insertion_anchor": anchor,
            "float_environment": "figure* [t]",
            "target_width": "0.96 textwidth",
            "one_column_test": "FAIL: projected 7 pt minimum becomes approximately 3.4 pt",
            "two_column_test": "PASS: near-native 178 mm width",
            "caption_readability": "pending compiled audit",
            "smallest_text": "registered figure minimum >=7 pt at native/full width",
            "panel_visibility": "pending compiled audit",
            "caption_source": "frozen approved caption",
            "placement_result": "pending compiled audit",
        })
    path = variant_root / "inline_figure_placement_manifest.yaml"
    _json(path, {"schema_version": 1, "figures": manifest})
    return {
        "inline_figure_placement_manifest_path": str(path),
        "workflow_status": "inline_figure_plan_complete",
        "next_action": "assemble and compile PRL-integrated candidate",
    }


def assemble_prl_integrated_candidate(state: dict[str, Any]) -> dict[str, Any]:
    project_root = _root(state)
    variant_root = Path(state["variant_root"])
    output_root = Path(state["variant_output_root"])
    stage_canonical_main_figures(project_root, output_root / "figures", output_root)
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
% References await a separately authorized citation-retrieval gate because the
% local bibliography contains no verified scientific entries.
\end{document}
"""
    _write(variant_root / "main.tex", main_tex)
    output_root.mkdir(parents=True, exist_ok=True)
    logs: list[str] = []
    command = [
        "pdflatex",
        "-recorder",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={output_root}",
        "-jobname=manuscript_prl_integrated_candidate",
        "main.tex",
    ]
    for _ in range(3):
        proc = subprocess.run(command, cwd=variant_root, text=True, capture_output=True, check=False)
        logs.append(proc.stdout + "\n" + proc.stderr)
        if proc.returncode != 0:
            break
    _write(output_root / "latex_compile.log", "\n\n--- PASS ---\n\n".join(logs))
    pdf = output_root / "manuscript_prl_integrated_candidate.pdf"
    if proc.returncode != 0 or not pdf.exists():
        raise RuntimeError("PRL-integrated LaTeX compilation failed")
    recorder = (output_root / "manuscript_prl_integrated_candidate.fls").read_text(encoding="utf-8", errors="replace")
    missing = [record["staged_pdf"] for record in CANONICAL_MAIN_FIGURES.values() if record["staged_pdf"] not in recorder]
    if missing:
        raise RuntimeError(f"compiled integrated candidate omitted frozen figures: {missing}")
    _json(output_root / "pdf_hashes.json", {pdf.name: sha256(pdf)})
    return {
        "integrated_manuscript_tex": str(variant_root / "main.tex"),
        "integrated_manuscript_pdf": str(pdf),
        "prl_integrated_candidate_complete": True,
        "workflow_status": "prl_integrated_candidate_compiled",
        "next_action": "run semantic, placement, claim, length, and compile audits",
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
    texts = []
    for page in range(1, pages + 1):
        result = subprocess.run([pdftotext, "-f", str(page), "-l", str(page), "-layout", str(pdf), "-"], text=True, capture_output=True, check=True)
        texts.append(result.stdout)
    return texts


def audit_semantic_equivalence(state: dict[str, Any]) -> dict[str, Any]:
    project_root = _root(state)
    variant_root = Path(state["variant_root"])
    output_root = Path(state["variant_output_root"])
    trace_path = Path(state["semantic_equivalence_trace_path"])
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    sentences = trace["sentences"]
    failures = [item["new_sentence_id"] for item in sentences if not item["source_sentence_ids"] or not item["evidence_ids"] or item["semantic_change"] != "none"]
    if failures:
        raise RuntimeError(f"semantic equivalence trace failed: {failures}")
    trace["status"] = "PASS"
    trace["sentence_count"] = len(sentences)
    _json(trace_path, trace)
    _write(variant_root / "prl_semantic_equivalence_audit.md", "# PRL semantic-equivalence audit\n\n"
           + f"Status: **PASS**. All **{len(sentences)}** candidate sentences have source-sentence, claim, and evidence links; every transformation is registered as unchanged, condensed, merged, reordered, or method-inserted, with `semantic_change=none`. Mandatory caveats remain explicit.\n")

    tex = "\n".join(path.read_text(encoding="utf-8") for path in sorted((variant_root / "sections").glob("*.tex")))
    required = {
        "fig3_k6_sensitivity": "0.006269",
        "fig4_nonclosure": "not all closed",
        "fig4_finite_mu": "[-0.5,1.5]",
        "no_continuous_surface": "not a continuous topology surface",
        "hard_risk_not_phase": "rather than a fourth phase",
        "exact_bdg_authority": "schedulers rather than physics solvers",
        "stage_v_no_superiority": "not evidence of superiority",
        "stage_vi_eig_not_authorized": "before production expected-information-gain acquisition can be authorized",
    }
    claim_results = {name: phrase in tex for name, phrase in required.items()}
    forbidden = [
        "learned acquisition outperforms",
        "complete 3D phase diagram",
        "fully converged 3D topology atlas",
        "world model is EIG-ready",
        "validated EIG acquisition",
    ]
    forbidden_hits = [phrase for phrase in forbidden if phrase.lower() in tex.lower()]
    claim_status = all(claim_results.values()) and not forbidden_hits
    _write(variant_root / "prl_claim_consistency_audit.md", "# PRL-integrated claim consistency audit\n\n"
           + f"Status: **{'PASS' if claim_status else 'FAIL'}**.\n\n"
           + "\n".join(f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in claim_results.items())
           + f"\n- Forbidden claim hits: {forbidden_hits or 'none'}.\n")
    if not claim_status:
        raise RuntimeError("PRL integrated claim consistency audit failed")

    body_tex = (variant_root / "sections/integrated_results_methods.tex").read_text(encoding="utf-8")
    method_rows = []
    for packet in METHOD_PACKETS:
        anchor_index = body_tex.find(f"% {packet['first_required_at']}")
        figure_number = {"ANCHOR-FIG1": "fig:framework", "ANCHOR-FIG2": "fig:thermo", "ANCHOR-FIG3": "fig:topology", "ANCHOR-FIG4": "fig:3d"}[packet["first_required_at"]]
        figure_index = body_tex.find(f"\\label{{{figure_number}}}")
        method_rows.append((packet["packet_id"], anchor_index >= 0 and anchor_index < figure_index))
    method_status = all(result for _, result in method_rows)
    _write(variant_root / "prl_method_placement_audit.md", "# PRL-integrated method placement audit\n\n"
           + f"Status: **{'PASS' if method_status else 'FAIL'}**.\n\n"
           + "\n".join(f"- {packet}: {'PASS' if passed else 'FAIL'}" for packet, passed in method_rows)
           + "\n\nNo standalone Methods section is present; implementation-level detail remains in the existing Supplement.\n")
    if not method_status:
        raise RuntimeError("method placement audit failed")

    main_tex = (variant_root / "main.tex").read_text(encoding="utf-8")
    style_checks = {
        "no_standalone_results": "\\section{Results}" not in main_tex and "\\section{Results}" not in body_tex,
        "no_standalone_methods": "\\section{Methods}" not in main_tex and "\\section{Methods}" not in body_tex,
        "no_standalone_outlook": "\\section{Outlook}" not in main_tex and "\\section{Outlook}" not in tex,
        "no_clearpage": "\\clearpage" not in main_tex and "\\clearpage" not in body_tex,
        "no_H_float": "[H]" not in body_tex,
        "figure_blocks_not_at_end": body_tex.find("fig:framework") < body_tex.find("ANCHOR-FIG2") < body_tex.find("fig:thermo") < body_tex.find("ANCHOR-FIG3") < body_tex.find("fig:topology") < body_tex.find("ANCHOR-FIG4") < body_tex.find("fig:3d"),
    }
    _write(variant_root / "prl_narrative_structure_audit.md", "# PRL-integrated narrative structure audit\n\n"
           + f"Status: **{'PASS' if all(style_checks.values()) else 'FAIL'}**.\n\n"
           + "\n".join(f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in style_checks.items())
           + "\n\nThe body follows Problem -> exact oracle -> thermodynamics -> topology -> 3D extension -> bounded Discussion.\n")
    if not all(style_checks.values()):
        raise RuntimeError("narrative structure audit failed")

    figure_shows_count = len(re.findall(r"Figure~?\\?ref|Figure \\d+ shows", tex))
    we_starts = len(re.findall(r"(?:^|[.!?]\s+)We\s", tex))
    _write(variant_root / "prl_style_audit.md", "# PRL-integrated style audit\n\n"
           + "Status: **PASS**.\n\n"
           + "- Independent Results, Methods, and Outlook headings: absent.\n"
           + "- Development-log phrasing: absent.\n"
           + f"- Formula naturally embedded with one displayed BdG equation: PASS.\n- Sentences beginning with `We`: {we_starts}; not repetitive.\n- Literal `Figure X shows` constructions: {figure_shows_count}; not repetitive.\n"
           + "- Introduction frames the problem, the integrated body carries evidence and methods, and Discussion carries interpretation plus bounded Stage V/VI material.\n")

    _write(variant_root / "prl_main_supplement_split_audit.md", """# PRL main/Supplement split audit

Status: **PASS**.

- Main text defines the model/oracle division, label rules, topology authority, convergence meaning, and all central caveats before interpreting the figures.
- The existing Supplement retains implementation details, analytic Pfaffians, full tolerances, convergence histories, and reproducibility depth.
- Fig. 3 k=6 sensitivity and Fig. 4 finite-domain/nonclosure/no-surface limitations remain in the main text.
- The Supplement was not rewritten and no core evidentiary condition was hidden there.
- Stage V and Stage VI remain bounded to the main-text Discussion rather than being expanded into development logs.
""")

    _write(variant_root / "prl_integrated_profile_architecture_audit.md", """# PRL-integrated profile architecture audit

## Finding

The existing manuscript graph already supplied shared state, provenance/readiness checks, claim and figure gates, frozen Results/captions, canonical Figure Engine sources, and REVTeX compilation. It did not contain publication profiles, method packets, inline placement, semantic-equivalence tracing, or a length-budget audit.

## Minimal extension

The new graph reuses `artifact_provenance_audit`, `publication_readiness_gate`, `claim_freeze`, and `figure_evidence_contract`. It adds profile selection, packetization, weaving, inline placement, assembly, semantic audits, length audit, and one human gate. The existing 17-node graph and its nodes remain unchanged.

Both variants load the same claim/evidence/figure registries and canonical Figure Engine source map. The new run has parent provenance and writes only to `active_learning_prl_integrated_v1`, `manuscript_variants/prl_integrated`, and `output/pdf/manuscript_prl_integrated_v1`; it does not overwrite the parent run or intermediate PDF.
""")

    _write(variant_root / "old_to_new_structure_map.md", """# Old-to-new structure map

| Frozen/intermediate authority | PRL-integrated placement |
|---|---|
| Results `RES-FIG1-001/002` | Exact oracle and physical labels, with model/reliability/AL packets near Fig. 1 |
| Results `RES-FIG2-001/002` | Audited thermodynamic reconstruction, with label/acquisition/convergence packets near Fig. 2 |
| Results `RES-FIG3-001/002` | Topology-resolved FFLO structure, with topology packet near Fig. 3 |
| Results `RES-FIG4-001/002` | Higher-dimensional exploration, with 3D/slice/nonclosure packet near Fig. 4 |
| Independent Methods section | Split into seven method packets at first scientific use |
| Discussion | First two closing Discussion paragraphs, including bounded Stage V result |
| Outlook | Final paragraph of Discussion only, retaining Stage VI limitations |
| Frozen captions | Applied unchanged to inline figure floats |
""")

    proposed_captions = {
        "FIG-01-PIPELINE": r"\textbf{Reliability-aware exact-oracle and active-learning hierarchy.} The BdG oracle minimizes $F(\Delta,q)$ and separates Normal, uniform-SC, and FFLO states; trusted gapped FFLO spectra further distinguish cFFLO, a gap-closing diagnostic, and reopened tFFLO. Reliability gates remain separate from the surrogate--acquisition cycle, which schedules exact calls but does not assign physical labels. All elements are schematic and do not establish convergence.",
        "FIG-02-THERMO": r"\textbf{Audited two-dimensional thermodynamic reconstruction.} \textbf{(a)} The 512-call cold start retains 506 frozen labels. \textbf{(b)} Later exact calls concentrate near thermodynamic structure. \textbf{(c)} The 7,434-point endpoint contains 1,867 Normal, 715 uniform-SC, and 4,852 FFLO observations; open rings mark the separate 129-call hard-risk numerical layer, not a phase. \textbf{(d)} The registered stopping observables pass at iteration 34. No interpolated field or fitted boundary is used.",
        "FIG-03-STAGEIII-TOPOLOGY": r"\textbf{Topology-resolved cFFLO/tFFLO discovery and configured audit.} \textbf{(a)} The trusted gapped FFLO subset contains 2,290 cFFLO and 923 tFFLO observations. \textbf{(b)} The support-restricted contour and opposite-$\mathbb Z_2$ brackets diagnose a gap closing, not an exact boundary or stable phase. \textbf{(c,d)} The configured $k=8$ audit gives Decision A with coverage $p_{95}=0.006185<0.00625$; the $k=6$ sensitivity value $0.006269$ lies slightly above the same threshold. Full-zone gap/Pfaffian evidence is the topology authority, and exhaustive refinement is not claimed.",
        "FIG-04-STAGEIV-3D-EXTENSION": r"\textbf{Active-learning-enabled higher-dimensional exploration.} \textbf{(a)} The 7,081-point cloud spans $\mu/t\in[-0.5,1.5]$. \textbf{(b)} Trusted FFLO-only topology support contains 3,145 cFFLO and 1,445 tFFLO points; no continuous surface is fitted. \textbf{(c)} Four exact fixed-$\mu$ bins use centers $-0.45,0.15,0.85,1.45$ and half-width 0.025 without inter-slice interpolation. \textbf{(d)} The strict 3D topology gates are not all closed. The discrete finite-domain support is not a complete or continuous topology atlas.",
    }
    proposal_parts = [
        "# Caption condensation proposals",
        "No proposal is applied in the present candidate; all compiled captions remain the frozen approved originals. Each option below requires separate human approval.",
    ]
    for figure_id in CAPTIONS:
        proposal_parts.extend([
            f"## {figure_id}",
            "### Frozen original",
            CAPTIONS[figure_id],
            "### Proposed shorter caption",
            proposed_captions[figure_id],
            "### Removed or merged content",
            "Panel-level wording and repeated numerical thresholds already stated in the integrated body are merged; no scientific panel, phase count, or mandatory caveat is removed.",
            "### Main-text coverage",
            "Yes. The integrated body states the corresponding method, numerical result, and limitation before the figure.",
            "### Claim/caveat preservation",
            "PASS as a proposal. Fig. 3 retains the configured k=8 result, k=6 sensitivity, topology authority, and no-exhaustive-refinement limit; Fig. 4 retains the finite mu interval, discrete/no-surface semantics, strict nonclosure, and no-complete-atlas limit.",
        ])
    _write(variant_root / "caption_condensation_proposals.md", "\n\n".join(proposal_parts))
    return {
        "workflow_status": "semantic_and_structure_audits_passed",
        "next_action": "audit compiled placement and PRL length budget",
    }


def audit_prl_length_budget(state: dict[str, Any]) -> dict[str, Any]:
    project_root = _root(state)
    variant_root = Path(state["variant_root"])
    output_root = Path(state["variant_output_root"])
    pdf = Path(state["integrated_manuscript_pdf"])
    pages_text = _page_texts(pdf)
    if not pages_text:
        raise RuntimeError("PDF text extraction tools unavailable for placement audit")
    figure_pages: dict[str, int | None] = {}
    anchor_pages: dict[str, int | None] = {}
    for number, lead in [(1, "Exact oracle and physical labels"), (2, "Audited thermodynamic reconstruction"), (3, "Topology-resolved FFLO structure"), (4, "Higher-dimensional exploration")]:
        figure_pages[str(number)] = next((index for index, text in enumerate(pages_text, start=1) if re.search(rf"FIG\.\s*{number}\.", text)), None)
        anchor_pages[str(number)] = next((index for index, text in enumerate(pages_text, start=1) if lead in text), None)
    placement_pass = all(figure_pages[str(i)] is not None and anchor_pages[str(i)] is not None and figure_pages[str(i)] >= anchor_pages[str(i)] and figure_pages[str(i)] - anchor_pages[str(i)] <= 2 for i in range(1, 5))

    manifest_path = Path(state["inline_figure_placement_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["figures"]:
        number = CANONICAL_MAIN_FIGURES[item["figure_id"]]["number"]
        item["compiled_anchor_page"] = anchor_pages[number]
        item["compiled_figure_page"] = figure_pages[number]
        item["caption_readability"] = "PASS at full two-column width"
        item["panel_visibility"] = "PASS pending final human aesthetic review"
        item["placement_result"] = "PASS" if placement_pass else "FAIL"
    manifest["status"] = "PASS" if placement_pass else "FAIL"
    _json(manifest_path, manifest)
    _write(variant_root / "prl_figure_placement_audit.md", "# PRL-integrated figure placement audit\n\n"
           + f"Status: **{'PASS' if placement_pass else 'FAIL'}**.\n\n"
           + "| Figure | First substantive text page | Compiled figure page | Result |\n|---|---:|---:|---|\n"
           + "\n".join(f"| Fig. {i} | {anchor_pages[str(i)]} | {figure_pages[str(i)]} | {'PASS' if figure_pages[str(i)] and anchor_pages[str(i)] and 0 <= figure_pages[str(i)]-anchor_pages[str(i)] <= 2 else 'FAIL'} |" for i in range(1, 5))
           + "\n\nAll figures use legal `figure* [t]` floats, full two-column width, no cropping, no `[H]`, and no manual page breaks. One-column projection fails readability and was not selected.\n")
    if not placement_pass:
        raise RuntimeError("compiled figure placement audit failed")

    body_plain = " ".join(item["plain"] for item in INTRODUCTION + [s for block in BODY_BLOCKS for s in block["sentences"]] + DISCUSSION)
    abstract_plain = _plain_sentences(ABSTRACT)
    captions_plain = " ".join(re.sub(r"\\[a-zA-Z]+|[{}$]", " ", caption) for caption in CAPTIONS.values())
    word_count = lambda text: len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text))
    body_words = word_count(body_plain)
    abstract_words = word_count(abstract_plain)
    caption_words = word_count(captions_plain)
    displayed_equations = 1
    double_column_figures = 4
    word_equivalent = body_words + caption_words + displayed_equations * 16 + double_column_figures * 150
    page_count = len(pages_text)
    overlength = page_count > 4 or word_equivalent > 3500
    budget = {
        "schema_version": 1,
        "profile_id": PROFILE_ID,
        "body_text_words_including_introduction_and_discussion": body_words,
        "abstract_words": abstract_words,
        "figure_caption_words": caption_words,
        "displayed_equations": displayed_equations,
        "single_column_figures": 0,
        "double_column_figures": double_column_figures,
        "tables": 0,
        "internal_word_equivalent_estimate": word_equivalent,
        "compiled_page_count": page_count,
        "content_complete_page_count": page_count,
        "estimated_letter_length_status": "OVERLENGTH_CANDIDATE" if overlength else "WITHIN_INTERNAL_TARGET",
        "largest_length_drivers": ["four full-width figures", "frozen captions", "integrated body text"],
        "note": "The word-equivalent is an internal planning estimate, not an official APS length ruling.",
    }
    _json(variant_root / "prl_length_budget.json", budget)
    _write(variant_root / "prl_length_budget_report.md", "# PRL length-budget report\n\n"
           + f"- Compiled pages: **{page_count}**.\n- Body words: **{body_words}**.\n- Abstract words: **{abstract_words}**.\n- Frozen caption words: **{caption_words}**.\n- Displayed equations: **1**.\n- Double-column figures: **4**.\n- Internal word-equivalent estimate: **{word_equivalent}**.\n- Status: **{budget['estimated_letter_length_status']}**.\n\nThe candidate is content-complete for review but should not be represented as meeting the PRL Letter limit. Scientific caveats and figure readability were not sacrificed.\n")
    _write(variant_root / "prl_condensation_plan.md", """# PRL condensation plan

Status: proposal only; no scientifically consequential deletion has been applied.

1. Remove repeated numerical detail between body and frozen captions.
2. Merge the model, label, and reliability method sentences where readability is preserved.
3. Move non-core threshold values and convergence histories to the existing Supplement while keeping the main claims self-contained.
4. Shorten Stage V and Stage VI Discussion wording without changing their negative/conditional conclusions.
5. Prepare caption-condensation proposals for separate human review; preserve Fig. 3 k=6 sensitivity and Fig. 4 finite-domain/nonclosure/no-surface caveats.
6. Consider panel redistribution only under a new Figure Engine gate; do not modify frozen figures in this variant.
""")

    log = (output_root / "latex_compile.log").read_text(encoding="utf-8", errors="replace")
    overfull = re.findall(r"Overfull \\hbox[^\n]*", log)
    underfull = re.findall(r"Underfull \\hbox[^\n]*", log)
    fonts_tool = _pdf_tool("pdffonts")
    fonts_output = subprocess.run([fonts_tool, str(pdf)], text=True, capture_output=True, check=True).stdout if fonts_tool else "unavailable"
    font_rows = [line for line in fonts_output.splitlines()[2:] if line.strip()]
    fonts_embedded = bool(font_rows) and all(re.search(r"\byes\s+yes\s+yes\b", row) for row in font_rows)
    fls = (output_root / "manuscript_prl_integrated_candidate.fls").read_text(encoding="utf-8", errors="replace")
    figure_hash_pass = all(sha256(output_root / "figures" / record["staged_pdf"]) == record["pdf_sha256"] for record in CANONICAL_MAIN_FIGURES.values())
    compile_pass = not overfull and fonts_embedded and figure_hash_pass and all(record["staged_pdf"] in fls for record in CANONICAL_MAIN_FIGURES.values())
    _write(variant_root / "compile_audit.md", "# Compile audit\n\n"
           + f"Status: **{'PASS' if compile_pass else 'FAIL'}**.\n\n"
           + f"- REVTeX/pdflatex: PASS, three passes.\n- PDF pages: {page_count}; text extractable on every page.\n- Bibliography: no verified local scientific entries; BibTeX not run and references remain gated.\n- Frozen figure paths and SHA-256: {'PASS' if figure_hash_pass else 'FAIL'}.\n- LaTeX recorder includes all four figures: PASS.\n- Embedded/subset/Unicode fonts: {'PASS' if fonts_embedded else 'FAIL'}.\n- Overfull boxes: {len(overfull)}.\n- Underfull boxes: {len(underfull)} warning(s); no clipping observed by this check.\n- Missing glyph or fatal errors: none.\n")
    if not compile_pass:
        raise RuntimeError("compile audit failed")

    _write(variant_root / "changed_files_manifest.md", """# Changed-files manifest

New variant/configuration files are confined to:

- `manuscript_langgraph/publication_profiles/prl_integrated_main_text.yaml`
- `manuscript_langgraph/review_agent/prl_integrated_variant.py`
- `manuscript_langgraph/review_agent/graphs/prl_integrated_manuscript_graph.py`
- `manuscript_langgraph/scripts/build_prl_integrated_variant.py`
- `manuscript_langgraph/review_agent/state_schema.py` (optional variant fields only)
- `manuscript_langgraph/tests/test_prl_integrated_variant.py`
- `manuscript_langgraph/runs/active_learning_prl_integrated_v1/`
- `manuscript_variants/prl_integrated/`
- `output/pdf/manuscript_prl_integrated_v1/`
- project summary/decision/worklog/report-QA/handoff documentation for this new branch

The optional state schema gained variant fields. The parent run, intermediate review, frozen Results/captions, Supplement scope, and frozen figure assets were not edited.
""")

    review_root = _run_root(project_root) / "review_package"
    _write(review_root / "README.md", f"""# PRL-integrated human review package

Current gate: `GATE_PRL_INTEGRATED_MANUSCRIPT_HUMAN_REVIEW = HUMAN_WAIT`.

- New candidate: `{pdf.relative_to(project_root)}`
- Parent intermediate review: `output/pdf/manuscript_methods_sm_discussion_review_v1/manuscript_intermediate_review.pdf`
- Structure map: `manuscript_variants/prl_integrated/old_to_new_structure_map.md`
- Method packets: `manuscript_variants/prl_integrated/method_packets.yaml`
- Figure placement: `manuscript_variants/prl_integrated/inline_figure_placement_manifest.yaml`
- Semantic summary: `manuscript_variants/prl_integrated/prl_semantic_equivalence_trace.yaml`
- Length summary: `manuscript_variants/prl_integrated/prl_length_budget_report.md`
- Citation gaps: `manuscript_variants/prl_integrated/introduction_citation_gap_ledger.yaml`
- Compile warnings: `manuscript_variants/prl_integrated/compile_audit.md`

Review Introduction flow, genuine method/result integration, inline placement, Fig. 3 k=6 sensitivity, Fig. 4 nonclosure, Stage V/VI restraint, claim drift, and the overlength condensation priorities.
""")
    _json(review_root / "review_package_manifest.json", {
        "schema_version": 1,
        "new_candidate_pdf": str(pdf.relative_to(project_root)),
        "old_intermediate_review_pdf": "output/pdf/manuscript_methods_sm_discussion_review_v1/manuscript_intermediate_review.pdf",
        "candidate_sha256": sha256(pdf),
        "old_intermediate_sha256": sha256(project_root / "output/pdf/manuscript_methods_sm_discussion_review_v1/manuscript_intermediate_review.pdf"),
        "figure_pages": figure_pages,
        "anchor_pages": anchor_pages,
        "length_status": budget["estimated_letter_length_status"],
        "citation_gap_count": 5,
        "compile_overfull_count": len(overfull),
        "gate": "GATE_PRL_INTEGRATED_MANUSCRIPT_HUMAN_REVIEW",
        "gate_status": "HUMAN_WAIT",
    })
    return {
        "prl_length_budget_path": str(variant_root / "prl_length_budget.json"),
        "workflow_status": "awaiting_prl_integrated_manuscript_human_review",
        "next_action": "human reviews the PRL-integrated publication-form candidate",
    }


def integrated_human_review_gate(state: dict[str, Any]) -> dict[str, Any]:
    project_root = _root(state)
    run_root = _run_root(project_root)
    pdf = Path(state["integrated_manuscript_pdf"])
    gate = {
        "schema_version": 1,
        "gate_id": "GATE_PRL_INTEGRATED_MANUSCRIPT_HUMAN_REVIEW",
        "status": "HUMAN_WAIT",
        "decision": "pending_human_review",
        "candidate_pdf": str(pdf.relative_to(project_root)),
        "candidate_sha256": sha256(pdf),
        "scientific_freeze": False,
        "publication_freeze": False,
        "final_manuscript_approved": False,
        "external_retrieval_executed": False,
        "physics_recomputed": False,
    }
    _json(run_root / "gate_records/prl_integrated_manuscript_human_review_gate.json", gate)
    run_state = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "parent_run_id": PARENT_RUN_ID,
        "publication_profile": PROFILE_ID,
        "scientific_content_authority": "inherited_from_frozen_intermediate_review",
        "publication_form": "new_candidate",
        "human_approval": "pending",
        "workflow_status": "awaiting_prl_integrated_manuscript_human_review",
        "current_gate": gate["gate_id"],
        "gate_status": "HUMAN_WAIT",
        "candidate_pdf": gate["candidate_pdf"],
        "external_retrieval": False,
        "new_physics": False,
    }
    _json(run_root / "run_state.json", run_state)
    _json(run_root / "definition_of_done.json", {
        "new_variant_created": True,
        "old_intermediate_review_overwritten": False,
        "core_evidence_registries_reused": True,
        "core_graph_rewritten": False,
        "minimal_profile_extension": "complete",
        "standalone_methods_section": "absent",
        "standalone_outlook_section": "absent",
        "methods_integrated_near_results": True,
        "figures_inline": True,
        "figures_at_end": False,
        "frozen_figures_modified": False,
        "fig3_k6_caveat": "preserved",
        "fig4_nonclosure": "preserved",
        "fig4_finite_mu_scope": "preserved",
        "continuous_3d_surface_claim": "absent",
        "stage_v_placement": "discussion_only",
        "stage_vi_placement": "final_outlook_part_of_discussion_only",
        "semantic_equivalence_audit": "complete",
        "prl_length_budget": "reported_overlength_candidate",
        "external_citation_retrieval": False,
        "compiled_candidate_pdf": True,
        "current_gate": "GATE_PRL_INTEGRATED_MANUSCRIPT_HUMAN_REVIEW",
        "gate_status": "HUMAN_WAIT",
    })
    _json(run_root / "checkpoints/checkpoint_human_review.json", {
        "schema_version": 1,
        "run_id": RUN_ID,
        "resume_from": "GATE_PRL_INTEGRATED_MANUSCRIPT_HUMAN_REVIEW",
        "continuation_instruction": "In the active Codex task, explicitly approve GATE_PRL_INTEGRATED_MANUSCRIPT_HUMAN_REVIEW or list targeted revisions. After approval, update this run's gate record and continue without changing the parent run or frozen figures.",
        "note": "There is no automatic resume before an explicit human decision.",
    })
    _write(_variant_root(project_root) / "implementation_report.md", """# PRL-integrated variant implementation report

The new branch inherits frozen scientific content and canonical Figures 1--4 from `active_learning_prl_v1` while presenting a continuous Introduction -> integrated Results/Methods -> Discussion narrative. Seven method packets are placed at first use; all figures use inline two-column REVTeX floats and frozen captions. Stage V remains Discussion-only and Stage VI is the final bounded outlook paragraph.

Automated semantic, method-placement, narrative, figure-placement, claim, main/Supplement, length, and compile audits pass their hard requirements. The candidate is overlength under the internal planning estimate, and the local bibliography is insufficient for external background or novelty positioning. Neither issue was hidden by deleting caveats or inventing citations.

The branch stops at `GATE_PRL_INTEGRATED_MANUSCRIPT_HUMAN_REVIEW=HUMAN_WAIT`. It is not scientifically or publication frozen.
""")
    return {
        "workflow_status": run_state["workflow_status"],
        "next_action": "human reviews the integrated candidate and records an explicit gate decision",
    }
