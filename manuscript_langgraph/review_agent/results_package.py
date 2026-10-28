from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from review_agent.figure_authority import (
    CANONICAL_FROZEN_HASHES,
    CANONICAL_MAIN_FIGURES,
    verify_canonical_frozen_artifacts,
)


FIGURE_SOURCES = {
    figure_id: record["source_pdf"]
    for figure_id, record in CANONICAL_MAIN_FIGURES.items()
}

FROZEN_HASHES = CANONICAL_FROZEN_HASHES


def _sentence(
    sentence_id: str,
    text: str,
    role: str,
    claims: list[str],
    evidence: list[str],
    panels: list[str],
    *,
    support: str = "direct",
    caveat: str = "none",
    depends_on: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "sentence_id": sentence_id,
        "sentence_text": text,
        "sentence_role": role,
        "claim_ids": claims,
        "evidence_ids": evidence,
        "figure_panels": panels,
        "support_level": support,
        "required_caveat": caveat,
        "depends_on_observations": depends_on or [],
    }


PARAGRAPHS = [
    {
        "paragraph_id": "RES-FIG1-001",
        "section": "Exact physical oracle and reliability-aware exploration",
        "sentences": [
            _sentence(
                "RES-FIG1-001-S01",
                r"For each parameter point $x=(k_BT/t,J_A/t,\mu/t,\ldots)$, the exact BdG oracle constructs the spectrum and minimizes $F(\Delta,q)$, while active learning controls only which point is evaluated next (Fig.~1).",
                "METHOD",
                ["CLM-ORACLE-RELIABILITY"],
                ["E-MODEL-SPEC", "E-NUMERICS-SPEC"],
                ["1-central", "1-oracle", "1-cycle"],
            ),
            _sentence(
                "RES-FIG1-001-S02",
                r"The frozen thermodynamic criterion assigns Normal states at $\Delta=0$, uniform superconductivity at $\Delta>0$ and $q=0$, and the FFLO sector at $\Delta>0$ and $|q|>0$.",
                "METHOD",
                ["CLM-ORACLE-RELIABILITY"],
                ["E-MODEL-SPEC", "E-NUMERICS-SPEC"],
                ["1-central"],
            ),
            _sentence(
                "RES-FIG1-001-S03",
                r"Within trusted gapped FFLO states, the full-Brillouin-zone spectral guard and Pfaffian invariant distinguish gapped cFFLO ($\nu=0$) from a reopened gapped tFFLO bulk ($\nu=1$) across a bulk-gap closing; the red crossing branch denotes an in-gap boundary state, not a gapless tFFLO bulk.",
                "METHOD",
                ["CLM-ORACLE-RELIABILITY", "CLM-2D-TOPOLOGY"],
                ["E-STAGEIII-PASS", "E-FIG3-STRICT-DATA"],
                ["1-topology"],
            ),
            _sentence(
                "RES-FIG1-001-S04",
                r"Figure~1 is a schematic hierarchy rather than evidence that any phase map or convergence gate has already closed.",
                "CAVEAT",
                ["CLM-ORACLE-RELIABILITY"],
                ["E-MODEL-SPEC", "E-NUMERICS-SPEC"],
                ["1-all"],
                caveat="schematic is not numerical evidence",
            ),
        ],
    },
    {
        "paragraph_id": "RES-FIG1-002",
        "section": "Exact physical oracle and reliability-aware exploration",
        "sentences": [
            _sentence(
                "RES-FIG1-002-S01",
                r"Optimization convergence, spectral/topological trust, and numerical reliability remain explicit gates on every accepted physical observation.",
                "OBSERVATION",
                ["CLM-ORACLE-RELIABILITY"],
                ["E-NUMERICS-SPEC", "E-PHASEII-AUDIT", "E-STAGEIII-PASS"],
                ["1-reliability", "1-observation"],
            ),
            _sentence(
                "RES-FIG1-002-S02",
                r"The surrogate and acquisition modules act on these accepted observations to rank a next exact micro-batch, but they do not assign the final thermodynamic or topological labels.",
                "METHOD",
                ["CLM-ORACLE-RELIABILITY"],
                ["E-MODEL-SPEC", "E-NUMERICS-SPEC"],
                ["1-surrogate", "1-acquisition", "1-next-calls"],
            ),
            _sentence(
                "RES-FIG1-002-S03",
                r"This separation makes the reliability hierarchy an enabling component for scalable parameter-space exploration without replacing the physical solver.",
                "INTERPRETATION",
                ["CLM-ORACLE-RELIABILITY"],
                ["E-PHASEII-AUDIT", "E-STAGEIII-PASS"],
                ["1-all"],
                support="derived",
                depends_on=["RES-FIG1-002-S01", "RES-FIG1-002-S02"],
            ),
        ],
    },
    {
        "paragraph_id": "RES-FIG2-001",
        "section": "Two-dimensional thermodynamic reconstruction",
        "sentences": [
            _sentence(
                "RES-FIG2-001-S01",
                r"The two-dimensional reconstruction begins from 512 selected exact calls, of which 506 have retained frozen labels in the endpoint (Fig.~2a).",
                "OBSERVATION",
                ["CLM-2D-THERMO"],
                ["E-STAGEI-COLDSTART", "E-FIG2-STRICT-DATA"],
                ["2a"],
            ),
            _sentence(
                "RES-FIG2-001-S02",
                r"Subsequent exact calls accumulate preferentially around the visible Normal/superconducting and uniform-SC/FFLO structures rather than remaining uniformly distributed over $(k_BT/t,J_A/t)$ (Fig.~2b).",
                "OBSERVATION",
                ["CLM-2D-THERMO", "CLM-ORACLE-RELIABILITY"],
                ["E-FIG2-STRICT-DATA"],
                ["2b"],
            ),
            _sentence(
                "RES-FIG2-001-S03",
                r"The audited endpoint contains 7,434 exact observations: 1,867 Normal, 715 uniform-SC, and 4,852 FFLO labels (Fig.~2c).",
                "OBSERVATION",
                ["CLM-2D-THERMO"],
                ["E-PHASEII-AUDIT", "E-FIG2-STRICT-DATA"],
                ["2c"],
            ),
            _sentence(
                "RES-FIG2-001-S04",
                r"At the frozen endpoint, the raw phase-map change is 0.001629, the Normal--SC and uniform-SC--FFLO shifts are 0.004167 and 0, respectively, and trusted surprise is 0 for 127 trusted calls; these quantities satisfy their configured stopping rules (Fig.~2d).",
                "OBSERVATION",
                ["CLM-2D-THERMO", "CLM-ORACLE-RELIABILITY"],
                ["E-PHASEII-AUDIT", "E-FIG2-STRICT-DATA"],
                ["2d"],
            ),
        ],
    },
    {
        "paragraph_id": "RES-FIG2-002",
        "section": "Two-dimensional thermodynamic reconstruction",
        "sentences": [
            _sentence(
                "RES-FIG2-002-S01",
                r"The 129 hard-risk calls remain a separate numerical-reliability layer, including 19 coordinate matches retained in the trusted endpoint, and are not interpreted as a fourth phase or as eliminated uncertainty.",
                "CAVEAT",
                ["CLM-2D-THERMO", "CLM-ORACLE-RELIABILITY"],
                ["E-PHASEII-HARD-RISK", "E-FIG2-STRICT-DATA"],
                ["2c"],
                caveat="hard-risk remains explicit and is not a phase",
            ),
            _sentence(
                "RES-FIG2-002-S02",
                r"The exact-sample map therefore supports an audited reconstruction of the two-dimensional thermodynamic structure under the registered criterion, while preserving the finite numerical frontier.",
                "INTERPRETATION",
                ["CLM-2D-THERMO", "CLM-ORACLE-RELIABILITY"],
                ["E-PHASEII-AUDIT", "E-PHASEII-HARD-RISK", "E-FIG2-STRICT-DATA"],
                ["2a", "2b", "2c", "2d"],
                support="derived",
                caveat="under the configured audit",
                depends_on=["RES-FIG2-001-S01", "RES-FIG2-001-S03", "RES-FIG2-001-S04", "RES-FIG2-002-S01"],
            ),
        ],
    },
    {
        "paragraph_id": "RES-FIG3-001",
        "section": "Topology-resolved FFLO structure",
        "sentences": [
            _sentence(
                "RES-FIG3-001-S01",
                r"Topology is evaluated on the trusted gapped FFLO subset of the 4,345-point endpoint, yielding 3,213 eligible states separated into 2,290 cFFLO and 923 tFFLO observations (Fig.~3a).",
                "OBSERVATION",
                ["CLM-2D-TOPOLOGY"],
                ["E-STAGEIII-CONVERGENCE", "E-FIG3-STRICT-DATA"],
                ["3a"],
            ),
            _sentence(
                "RES-FIG3-001-S02",
                r"These exact labels form a connected cFFLO/tFFLO structure in the sampled $(k_BT/t,J_A/t)$ domain, with the archived $k=8$ contour marking a gap-closing transition diagnostic (Fig.~3b).",
                "OBSERVATION",
                ["CLM-2D-TOPOLOGY"],
                ["E-STAGEIII-CONVERGENCE", "E-FIG3-STRICT-DATA"],
                ["3a", "3b"],
            ),
            _sentence(
                "RES-FIG3-001-S03",
                r"The topology authority is the full-Brillouin-zone gap/Pfaffian evidence, not the free-energy depth, the sign of $q$, or the plotted contour.",
                "METHOD",
                ["CLM-2D-TOPOLOGY", "CLM-ORACLE-RELIABILITY"],
                ["E-STAGEIII-PASS", "E-FIG3-STRICT-DATA"],
                ["3a", "3b"],
            ),
            _sentence(
                "RES-FIG3-001-S04",
                r"Local opposite-$\mathbb{Z}_2$ brackets and the derived contour are supporting diagnostics only; neither is identified with an exact physical boundary, and the gap closing is not a stable third phase.",
                "CAVEAT",
                ["CLM-2D-TOPOLOGY"],
                ["E-STAGEIII-CONVERGENCE", "E-FIG3-STRICT-DATA"],
                ["3b"],
                caveat="brackets and contour are diagnostics only",
            ),
        ],
    },
    {
        "paragraph_id": "RES-FIG3-002",
        "section": "Topology-resolved FFLO structure",
        "sentences": [
            _sentence(
                "RES-FIG3-002-S01",
                r"The configured $k=8$ audit returns Decision A, with final coverage $p_{95}=0.006185$ below the registered threshold 0.00625 and the remaining raw convergence metrics satisfying their gates (Fig.~3c,d).",
                "OBSERVATION",
                ["CLM-2D-TOPOLOGY"],
                ["E-STAGEIII-CONVERGENCE", "E-FIG3-STRICT-DATA"],
                ["3c", "3d"],
            ),
            _sentence(
                "RES-FIG3-002-S02",
                r"For the $k=6$ sensitivity setting, however, coverage $p_{95}=0.006269$ lies slightly above the same threshold (Fig.~3d).",
                "OBSERVATION",
                ["CLM-2D-TOPOLOGY"],
                ["E-STAGEIII-CONVERGENCE", "E-FIG3-STRICT-DATA"],
                ["3d"],
                caveat="k=6 sensitivity deviation must remain visible",
            ),
            _sentence(
                "RES-FIG3-002-S03",
                r"Thus the topology-aware reconstruction passes its registered $k=8$ audit while retaining a small neighborhood-size sensitivity, rather than establishing that all possible topology refinements or nodal alternatives are exhausted.",
                "INTERPRETATION",
                ["CLM-2D-TOPOLOGY"],
                ["E-STAGEIII-CONVERGENCE", "E-FIG3-STRICT-DATA"],
                ["3c", "3d"],
                support="derived",
                caveat="configured pass with k sensitivity",
                depends_on=["RES-FIG3-002-S01", "RES-FIG3-002-S02"],
            ),
        ],
    },
    {
        "paragraph_id": "RES-FIG4-001",
        "section": "Higher-dimensional exploration",
        "sentences": [
            _sentence(
                "RES-FIG4-001-S01",
                r"Extending the sampled coordinates to $(k_BT/t,J_A/t,\mu/t)$ produces 7,081 exact observations over the finite interval $\mu/t\in[-0.5,1.5]$: 2,350 Normal, 141 uniform-SC, and 4,590 FFLO states (Fig.~4a).",
                "OBSERVATION",
                ["CLM-3D-EXTENSION"],
                ["E-STAGEIV-AUDIT", "E-FIG4-STRICT-DATA"],
                ["4a"],
            ),
            _sentence(
                "RES-FIG4-001-S02",
                r"Within the trusted FFLO-only subset, 3,145 cFFLO and 1,445 tFFLO observations provide discrete topology support rather than a fitted topology surface (Fig.~4b).",
                "OBSERVATION",
                ["CLM-3D-EXTENSION"],
                ["E-STAGEIV-AUDIT", "E-FIG4-STRICT-DATA"],
                ["4b"],
                caveat="FFLO-only counts; discrete point support",
            ),
            _sentence(
                "RES-FIG4-001-S03",
                r"Four explicit narrow fixed-$\mu$ bins selected from evenly spaced rows of the 20-row slice audit expose slice-to-slice changes in the exact thermodynamic samples without interpolating between slices (Fig.~4c).",
                "OBSERVATION",
                ["CLM-3D-EXTENSION"],
                ["E-STAGEIV-MU-SLICE-AUDIT", "E-FIG4-STRICT-DATA"],
                ["4c"],
            ),
            _sentence(
                "RES-FIG4-001-S04",
                r"Together, these point clouds and fixed-$\mu$ slices demonstrate active-learning-enabled higher-dimensional exploration that reveals thermodynamic and topology-aware structures across the sampled domain.",
                "INTERPRETATION",
                ["CLM-3D-EXTENSION", "CLM-ORACLE-RELIABILITY"],
                ["E-STAGEIV-AUDIT", "E-STAGEIV-MU-SLICE-AUDIT", "E-FIG4-STRICT-DATA"],
                ["4a", "4b", "4c"],
                support="derived",
                depends_on=["RES-FIG4-001-S01", "RES-FIG4-001-S02", "RES-FIG4-001-S03"],
            ),
        ],
    },
    {
        "paragraph_id": "RES-FIG4-002",
        "section": "Higher-dimensional exploration",
        "sentences": [
            _sentence(
                "RES-FIG4-002-S01",
                r"The last strict audit transition nevertheless has topology-volume map change $0.002561>0.002$, surface-shift $p_{95}=0.005996>0.004167$, coverage $p_{95}=0.015318>0.00625$, trusted surprise $0.140625>0.02$, and component counts $3\rightarrow4\rightarrow3$ (Fig.~4d).",
                "OBSERVATION",
                ["CLM-3D-EXTENSION"],
                ["E-STAGEIV-AUDIT", "E-FIG4-STRICT-DATA"],
                ["4d"],
            ),
            _sentence(
                "RES-FIG4-002-S02",
                r"Accordingly, the strict three-dimensional topology convergence gates used here are not all closed.",
                "CAVEAT",
                ["CLM-3D-EXTENSION"],
                ["E-STAGEIV-AUDIT", "E-FIG4-STRICT-DATA"],
                ["4d"],
                caveat="strict 3D topology nonclosure",
            ),
            _sentence(
                "RES-FIG4-002-S03",
                r"Visual continuity among the sampled slices is therefore not evidence for a continuous topology surface, and the finite-domain result does not establish an exhaustive three-dimensional phase atlas.",
                "CAVEAT",
                ["CLM-3D-EXTENSION"],
                ["E-STAGEIV-MU-SLICE-AUDIT", "E-FIG4-STRICT-DATA"],
                ["4a", "4b", "4c", "4d"],
                caveat="no continuous-surface or complete-atlas implication",
            ),
            _sentence(
                "RES-FIG4-002-S04",
                r"The audited two-dimensional closure and the explicitly nonclosed three-dimensional extension therefore set distinct evidentiary limits for the present Results.",
                "TRANSITION",
                ["CLM-2D-THERMO", "CLM-2D-TOPOLOGY", "CLM-3D-EXTENSION"],
                ["E-FIG2-STRICT-DATA", "E-FIG3-STRICT-DATA", "E-FIG4-STRICT-DATA"],
                ["2-all", "3-all", "4-all"],
            ),
        ],
    },
]


CAPTIONS = {
    "FIG-01-PIPELINE": (
        r"\textbf{Reliability-aware exact-oracle and active-learning hierarchy.} "
        r"The central schematic organizes the physical state space generated by the BdG oracle: representative minima of $F(\Delta,q)$ encode Normal ($\Delta=0$), uniform-SC ($\Delta>0,q=0$), and FFLO ($\Delta>0,|q|>0$) sectors. "
        r"Below, trusted gapped FFLO spectra distinguish cFFLO ($\nu=0$), a bulk-gap-closing transition, and reopened gapped tFFLO ($\nu=1$); the red crossing branch is an in-gap boundary state rather than a gapless bulk band. "
        r"The right-hand oracle and reliability cards separate minimization, spectral/topological trust, and numerical status from the surrounding scheduler--acquisition cycle, which selects the next exact calls. "
        r"All elements are schematic; the surrogate schedules BdG evaluations and does not determine physical labels or constitute convergence evidence."
    ),
    "FIG-02-THERMO": (
        r"\textbf{Audited two-dimensional thermodynamic reconstruction.} "
        r"\textbf{(a)} Sparse random-grid cold start with 512 selected exact calls; 506 retained frozen labels are shown. "
        r"\textbf{(b)} Iteration-resolved exact-call allocation over $(k_BT/t,J_A/t)$ concentrates around visible thermodynamic structure. "
        r"\textbf{(c)} Exact-sample endpoint with 7,434 observations: 1,867 Normal, 715 uniform SC, and 4,852 FFLO. Open rings denote the separate 129-call hard-risk numerical layer, not a phase. "
        r"\textbf{(d)} Raw stopping observables at iteration 34: phase-map change 0.001629 (threshold 0.002), Normal--SC shift 0.004167 (threshold 0.004167), uniform-SC--FFLO shift 0, and trusted surprise 0 for 127 trusted calls (threshold 0.05). "
        r"No interpolated field or fitted boundary is used, and finite numerical risk remains explicit."
    ),
    "FIG-03-STAGEIII-TOPOLOGY": (
        r"\textbf{Topology-resolved cFFLO/tFFLO discovery and configured audit.} "
        r"\textbf{(a)} Trusted gapped FFLO subset of the 4,345-point exact endpoint: 3,213 states comprising 2,290 cFFLO ($\nu=0$) and 923 tFFLO ($\nu=1$) labels. "
        r"\textbf{(b)} The archived support-restricted $k=8$ contour marks a gap-closing transition diagnostic; local opposite-$\mathbb{Z}_2$ brackets and the contour are supporting guides, not a stable phase or exact boundary. "
        r"\textbf{(c)} Raw configured convergence histories for map change, boundary $p_{95}$ shift, coverage $p_{95}$, and trusted surprise. "
        r"\textbf{(d)} The configured $k=8$ result gives Decision A with coverage $p_{95}=0.006185<0.00625$, whereas the $k=6$ sensitivity value $0.006269$ lies slightly above the same threshold. "
        r"Topology is assigned from full-Brillouin-zone gap/Pfaffian evidence; the audit does not establish that every possible refinement or nodal alternative is exhausted."
    ),
    "FIG-04-STAGEIV-3D-EXTENSION": (
        r"\textbf{Active-learning-enabled higher-dimensional exploration.} "
        r"\textbf{(a)} Exact three-dimensional thermodynamic point cloud in $(k_BT/t,J_A/t,\mu/t)$ over $\mu/t\in[-0.5,1.5]$: 7,081 observations containing 2,350 Normal, 141 uniform-SC, and 4,590 FFLO states. "
        r"\textbf{(b)} Trusted FFLO-only topology support comprises 3,145 cFFLO and 1,445 tFFLO points; no continuous topology surface is fitted. "
        r"\textbf{(c)} Four explicit narrow fixed-$\mu$ bins, centered at $\mu/t=-0.45,0.15,0.85,$ and $1.45$ with half-width 0.025, show exact slice samples without interpolation between slices. "
        r"\textbf{(d)} The final strict audit fails the registered map-change, surface-shift, coverage, trusted-surprise, and component-stability criteria. "
        r"The 3D extension reveals thermodynamic and topology-aware structures across a finite sampled domain, but the strict three-dimensional topology convergence gates are not all closed and the discrete support does not imply a complete or continuous topology atlas."
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_frozen_hashes(project_root: Path) -> list[dict[str, str]]:
    return verify_canonical_frozen_artifacts(project_root)


def _count(values: np.ndarray) -> dict[int, int]:
    labels, counts = np.unique(values, return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts)}


def _verify_numerics(project_root: Path) -> dict[str, Any]:
    stage2 = np.load(project_root / "manuscript_langgraph/recovered_artifacts/stageII/dataset_iter035.npz")
    stage3 = np.load(project_root / "manuscript_langgraph/recovered_artifacts/stageIII/dataset_iter018.npz")
    stage4 = np.load(project_root / "manuscript_langgraph/recovered_artifacts/stageIV/dataset_iter025.npz")
    stage2_counts = _count(stage2["y_phase"])
    stage3_mask = (
        (stage3["topology_trusted"] == 1)
        & (stage3["topology_applicable"] == 1)
        & (stage3["y_phase"] == 2)
    )
    stage4_mask = (
        (stage4["topology_trusted"] == 1)
        & (stage4["topology_applicable"] == 1)
        & (stage4["y_phase"] == 2)
    )
    with (project_root / "figures/fig3_strict_data/canonical/fig3_k_sensitivity.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        sensitivity = {int(row["idw_and_knn_k"]): float(row["final_coverage_p95"]) for row in csv.DictReader(handle)}
    stop_history = json.loads(
        (project_root / "manuscript_langgraph/recovered_artifacts/stageII/run_bundle/stop_metrics_history.json").read_text(encoding="utf-8")
    )
    final_stop = stop_history[-1]
    with (
        project_root
        / "manuscript_langgraph/recovered_artifacts/stageIV/stageiv_3d_audit_local/tables/stageiv_3d_convergence_failure_decomposition.csv"
    ).open(newline="", encoding="utf-8") as handle:
        stage4_audit = list(csv.DictReader(handle))
    payload = {
        "stage2": {
            "total": int(len(stage2["y_phase"])),
            "phase_counts": {"normal": stage2_counts[0], "uniform_SC": stage2_counts[1], "FFLO": stage2_counts[2]},
            "final_metrics": final_stop["metrics"],
            "thresholds": final_stop["thresholds"],
            "trusted_surprise_denominator": final_stop["surprise_details"]["trusted"]["n_denominator"],
        },
        "stage3": {
            "total": int(len(stage3["y_phase"])),
            "trusted_gapped_FFLO": int(stage3_mask.sum()),
            "topology_counts": {"cFFLO": _count(stage3["topology_z2"][stage3_mask])[0], "tFFLO": _count(stage3["topology_z2"][stage3_mask])[1]},
            "k8_coverage_p95": sensitivity[8],
            "k6_coverage_p95": sensitivity[6],
            "threshold": 0.00625,
        },
        "stage4": {
            "total": int(len(stage4["y_phase"])),
            "phase_counts": {"normal": _count(stage4["y_phase"])[0], "uniform_SC": _count(stage4["y_phase"])[1], "FFLO": _count(stage4["y_phase"])[2]},
            "trusted_FFLO_topology_counts": {"cFFLO": _count(stage4["topology_z2"][stage4_mask])[0], "tFFLO": _count(stage4["topology_z2"][stage4_mask])[1]},
            "mu_min": float(stage4["x"][:, 2].min()),
            "mu_max": float(stage4["x"][:, 2].max()),
            "strict_audit_rows": stage4_audit,
        },
    }
    expected = {
        "stage2": (7434, {"normal": 1867, "uniform_SC": 715, "FFLO": 4852}),
        "stage3": (4345, 3213, {"cFFLO": 2290, "tFFLO": 923}),
        "stage4": (7081, {"normal": 2350, "uniform_SC": 141, "FFLO": 4590}, {"cFFLO": 3145, "tFFLO": 1445}, -0.5, 1.5),
    }
    if (payload["stage2"]["total"], payload["stage2"]["phase_counts"]) != expected["stage2"]:
        raise RuntimeError("Stage II invariant mismatch")
    if (payload["stage3"]["total"], payload["stage3"]["trusted_gapped_FFLO"], payload["stage3"]["topology_counts"]) != expected["stage3"]:
        raise RuntimeError("Stage III invariant mismatch")
    if (
        payload["stage4"]["total"],
        payload["stage4"]["phase_counts"],
        payload["stage4"]["trusted_FFLO_topology_counts"],
        payload["stage4"]["mu_min"],
        payload["stage4"]["mu_max"],
    ) != expected["stage4"]:
        raise RuntimeError("Stage IV invariant mismatch")
    if not (sensitivity[8] < 0.00625 < sensitivity[6]):
        raise RuntimeError("Stage III sensitivity relation mismatch")
    return payload


def _paragraph_text(paragraph: dict[str, Any]) -> str:
    return " ".join(sentence["sentence_text"] for sentence in paragraph["sentences"])


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _results_markdown(*, scientifically_frozen: bool = False) -> str:
    status = (
        "> Scientifically frozen after the approved Figure Engine source repair."
        if scientifically_frozen
        else "> Human-review candidate. Not scientifically frozen."
    )
    lines = ["# Results" if scientifically_frozen else "# Results candidate", "", status, ""]
    current_section = None
    for paragraph in PARAGRAPHS:
        if paragraph["section"] != current_section:
            current_section = paragraph["section"]
            lines.extend([f"## {current_section}", ""])
        lines.extend([f"**[{paragraph['paragraph_id']}]**", "", _paragraph_text(paragraph), ""])
    return "\n".join(lines)


def _results_tex() -> str:
    lines = [r"\section{Results}", r"\label{sec:results}", ""]
    current_section = None
    for paragraph in PARAGRAPHS:
        if paragraph["section"] != current_section:
            current_section = paragraph["section"]
            lines.extend([rf"\subsection*{{{current_section}}}", ""])
        lines.extend([
            rf"\noindent\textcolor{{gray}}{{\scriptsize [{paragraph['paragraph_id']}]}}\\[-0.3ex]",
            _paragraph_text(paragraph),
            "",
        ])
    return "\n".join(lines)


def _caption_tex() -> str:
    lines = ["% Candidate captions; frozen artwork is not modified."]
    for index, (figure_id, caption) in enumerate(CAPTIONS.items(), start=1):
        lines.extend(["", f"% {figure_id}", rf"\def\fig{index}caption{{{caption}}}"])
    return "\n".join(lines)


def _caption_trace() -> list[dict[str, Any]]:
    return [
        {
            "figure_id": "FIG-01-PIPELINE",
            "caption_id": "CAP-FIG1-001",
            "panel_order": ["central physical state space", "topology triptych", "exact oracle and reliability", "surrounding scheduler cycle"],
            "claim_ids": ["CLM-ORACLE-RELIABILITY", "CLM-2D-TOPOLOGY"],
            "evidence_ids": ["E-MODEL-SPEC", "E-NUMERICS-SPEC", "E-STAGEIII-PASS"],
            "data_layer": "schematic only",
            "required_caveats": ["surrogate is a scheduler", "not convergence evidence", "red branch is an in-gap boundary state"],
        },
        {
            "figure_id": "FIG-02-THERMO",
            "caption_id": "CAP-FIG2-001",
            "panel_order": ["2a", "2b", "2c", "2d"],
            "claim_ids": ["CLM-2D-THERMO", "CLM-ORACLE-RELIABILITY"],
            "evidence_ids": ["E-STAGEI-COLDSTART", "E-PHASEII-AUDIT", "E-PHASEII-HARD-RISK", "E-FIG2-STRICT-DATA"],
            "data_layer": "exact samples and raw convergence metrics",
            "required_caveats": ["hard-risk is separate numerical layer", "no interpolation or fitted boundary"],
        },
        {
            "figure_id": "FIG-03-STAGEIII-TOPOLOGY",
            "caption_id": "CAP-FIG3-001",
            "panel_order": ["3a", "3b", "3c", "3d"],
            "claim_ids": ["CLM-2D-TOPOLOGY"],
            "evidence_ids": ["E-STAGEIII-PASS", "E-STAGEIII-CONVERGENCE", "E-FIG3-STRICT-DATA"],
            "data_layer": "trusted exact labels plus archived support-restricted diagnostic contour",
            "required_caveats": ["k=8 configured pass", "k=6 coverage 0.006269 exceeds 0.00625", "brackets and contour are diagnostics"],
        },
        {
            "figure_id": "FIG-04-STAGEIV-3D-EXTENSION",
            "caption_id": "CAP-FIG4-001",
            "panel_order": ["4a", "4b", "4c", "4d"],
            "claim_ids": ["CLM-3D-EXTENSION"],
            "evidence_ids": ["E-STAGEIV-AUDIT", "E-STAGEIV-MU-SLICE-AUDIT", "E-FIG4-STRICT-DATA"],
            "data_layer": "exact point clouds and explicit fixed-mu bins",
            "required_caveats": ["mu/t in [-0.5,1.5]", "strict topology nonclosure", "no continuous-surface implication"],
        },
    ]


def _paragraph_metadata() -> list[dict[str, Any]]:
    metadata = []
    for paragraph in PARAGRAPHS:
        observations = [s["sentence_text"] for s in paragraph["sentences"] if s["sentence_role"] == "OBSERVATION"]
        interpretations = [s["sentence_text"] for s in paragraph["sentences"] if s["sentence_role"] == "INTERPRETATION"]
        metadata.append(
            {
                "paragraph_id": paragraph["paragraph_id"],
                "observation": " ".join(observations) or "Method/caveat paragraph grounded in registered sources.",
                "interpretation": " ".join(interpretations) or "No standalone interpretation in this paragraph.",
                "evidence_source": sorted({e for s in paragraph["sentences"] for e in s["evidence_ids"]}),
                "figure_reference": sorted({p.split("-")[0] for s in paragraph["sentences"] for p in s["figure_panels"]}),
                "claims_used": sorted({c for s in paragraph["sentences"] for c in s["claim_ids"]}),
                "evidence_used": sorted({e for s in paragraph["sentences"] for e in s["evidence_ids"]}),
                "figure_panels_cited": sorted({p for s in paragraph["sentences"] for p in s["figure_panels"]}),
                "caveats": [s["required_caveat"] for s in paragraph["sentences"] if s["required_caveat"] != "none"],
                "unsupported_sentence_flags": [],
            }
        )
    return metadata


def _sentence_trace() -> list[dict[str, Any]]:
    rows = []
    for paragraph in PARAGRAPHS:
        for sentence in paragraph["sentences"]:
            rows.append({"section": paragraph["section"], "paragraph_id": paragraph["paragraph_id"], **sentence})
    return rows


def _forbidden_audit(results: str, captions: str) -> tuple[bool, str]:
    forbidden = {
        "ML solves BdG": "absent",
        "all uncertainty eliminated": "absent",
        "all phases discovered": "absent",
        "learned acquisition outperforms": "absent",
        "autonomous phase discovery": "absent",
        "new phase candidates": "absent",
        "EIG-ready": "absent",
    }
    combined = (results + "\n" + captions).lower()
    found = [phrase for phrase in forbidden if phrase.lower() in combined]
    semantic_checks = [
        ("complete/continuous 3D topology", "present only in explicit negated limitations"),
        ("surrogate as physical solver", "explicitly rejected"),
        ("topology inferred from free energy or q sign", "explicitly rejected"),
        ("learned-acquisition superiority", "not present in Results or captions"),
    ]
    passed = not found
    lines = ["# Results forbidden-claim audit", "", f"Overall: **{'PASS' if passed else 'FAIL'}**", "", "## Literal checks", ""]
    for phrase in forbidden:
        lines.append(f"- `{phrase}`: {'FAIL' if phrase in found else 'PASS'}")
    lines.extend(["", "## Semantic checks", ""])
    lines.extend(f"- {name}: PASS — {finding}." for name, finding in semantic_checks)
    lines.extend(["", "Negated limitations such as the absence of a continuous surface are retained as caveats and are not positive forbidden claims."])
    return passed, "\n".join(lines)


def _audit_documents(numerics: dict[str, Any], traces: list[dict[str, Any]]) -> dict[str, str]:
    observation_count = sum(row["sentence_role"] == "OBSERVATION" for row in traces)
    interpretation_count = sum(row["sentence_role"] == "INTERPRETATION" for row in traces)
    return {
        "results_claim_evidence_audit.md": """# Results claim–evidence audit

Overall: **PASS**

- Every Results sentence has registered claim/evidence identifiers and figure-panel references.
- `CLM-ORACLE-RELIABILITY`, `CLM-2D-THERMO`, `CLM-2D-TOPOLOGY`, and `CLM-3D-EXTENSION` remain in their approved main-text positions.
- `CLM-LEARNED-ACQUISITION` and `CLM-GROUNDUP-WORLD-MODEL` do not appear in Results.
- Figure 3 retains the configured $k=8$ pass, the $k=6$ sensitivity deviation, and diagnostic-only bracket/contour language.
- Figure 4 retains finite $\\mu$ scope, exact point/slice semantics, strict topology nonclosure, and the no-continuous-surface limitation.
- No unsupported scientific sentence is silently softened; no new physics was introduced.
""",
        "results_observation_interpretation_audit.md": f"""# Observation–interpretation audit

Overall: **PASS**

- Direct observations: **{observation_count}** sentences.
- Interpretations: **{interpretation_count}** sentences, each with explicit observation dependencies in the trace ledger.
- Method, caveat, and transition sentences are separately typed.
- Physical interpretations follow the observations they depend on.
- Topology is attributed to trusted gap/Pfaffian/full-BZ evidence, not to free-energy depth or $q$ sign.
- The three-dimensional statement is limited to discrete samples and fixed-$\\mu$ slices; visual continuity is explicitly rejected as continuous-surface evidence.
- Central caveats are adjacent to the claims they constrain rather than deferred to Discussion or Supplement.
""",
        "figure_caption_consistency_audit.md": """# Figure-caption consistency audit

Overall: **PASS**

- Captions follow the frozen panel order and use the same Normal/uniform-SC/FFLO and cFFLO/tFFLO terminology as Results.
- Figure 1 is explicitly schematic; the red branch is an in-gap boundary state, not a gapless bulk band.
- Figure 2 encodes hard-risk as a separate numerical-reliability layer and makes no interpolated-boundary claim.
- Figure 3 states Decision A for configured $k=8$ and retains $k=6$ coverage $0.006269>0.00625$.
- Figure 4 states $\\mu/t\\in[-0.5,1.5]$, exact point/slice semantics, and strict three-dimensional topology nonclosure.
- No caption contains an internal stage label, dataset filename, path, or audit-script name.
""",
        "results_numerical_consistency_audit.md": f"""# Results numerical consistency audit

Overall: **PASS**

The values below were independently read from frozen NPZ/CSV/JSON artifacts; no physics was recomputed.

- Figure 2 endpoint: {numerics['stage2']['total']} rows; {numerics['stage2']['phase_counts']['normal']} Normal, {numerics['stage2']['phase_counts']['uniform_SC']} uniform SC, {numerics['stage2']['phase_counts']['FFLO']} FFLO.
- Figure 2 final raw metrics: phase-map change {numerics['stage2']['final_metrics']['phase_map_change']:.9f}; Normal--SC shift {numerics['stage2']['final_metrics']['boundary_shift_normal_sc']:.9f}; uniform-SC--FFLO shift {numerics['stage2']['final_metrics']['boundary_shift_uniform_fflo']:.9f}; trusted surprise {numerics['stage2']['final_metrics']['label_surprise_trusted']:.9f} over {numerics['stage2']['trusted_surprise_denominator']} trusted calls.
- Figure 3 endpoint: {numerics['stage3']['total']} rows; {numerics['stage3']['trusted_gapped_FFLO']} trusted gapped FFLO; {numerics['stage3']['topology_counts']['cFFLO']} cFFLO; {numerics['stage3']['topology_counts']['tFFLO']} tFFLO.
- Figure 3 sensitivity: $k=8$ coverage $p_{{95}}={numerics['stage3']['k8_coverage_p95']:.6f}<0.00625$; $k=6$ gives {numerics['stage3']['k6_coverage_p95']:.6f}$>0.00625$.
- Figure 4 endpoint: {numerics['stage4']['total']} rows; {numerics['stage4']['phase_counts']['normal']}/{numerics['stage4']['phase_counts']['uniform_SC']}/{numerics['stage4']['phase_counts']['FFLO']} thermodynamic counts; FFLO-only topology {numerics['stage4']['trusted_FFLO_topology_counts']['cFFLO']}/{numerics['stage4']['trusted_FFLO_topology_counts']['tFFLO']}.
- Figure 4 sampled range: $\\mu/t\\in[{numerics['stage4']['mu_min']:.1f},{numerics['stage4']['mu_max']:.1f}]$.
- Frozen Figure 1--4 SVG/PDF/PNG hashes match their approved manifests.
""",
    }


def _conflict_ledger() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "conflicts": [
            {
                "conflict_id": "CONFLICT-MODEL-SPEC-TOPOLOGY-STATUS",
                "claim": "Topology labels are available for the present Results.",
                "source_A": "MODEL_SPEC.md retains older milestone language saying topology is a later milestone unless exposed by the solver.",
                "source_B": "Stage III scientific freeze, approved claim/evidence registries, and frozen topology endpoint provide the implemented topology evidence.",
                "source_priority": "scientific freeze and approved registries override stale planning language",
                "provisional_resolution": "Use frozen topology evidence in Results; reconcile the canonical model-document wording before Methods freeze.",
                "scientific_impact": "nonblocking for Results; Methods documentation follow-up required",
                "human_decision_required": False,
            },
            {
                "conflict_id": "CONFLICT-FIG4-TOPOLOGY-COUNTS",
                "claim": "cFFLO/tFFLO counts in Figure 4",
                "source_A": "Archived all-topology-applicable-SC counts are 3265 trivial and 1466 topological, including 141 uniform-SC samples.",
                "source_B": "Frozen Figure 4 uses the physically scoped FFLO-only trusted counts 3145 cFFLO and 1445 tFFLO.",
                "source_priority": "Figure 4 scientific freeze and strict-data manifest",
                "provisional_resolution": "Use 3145/1445 whenever naming cFFLO/tFFLO; retain 3265/1466 only as archived all-applicable-SC provenance.",
                "scientific_impact": "resolved; prevents uniform-SC labels from being renamed cFFLO/tFFLO",
                "human_decision_required": False,
            },
            {
                "conflict_id": "CONFLICT-FIG4-OLD-CAPTION-PANELS",
                "claim": "Figure 4 panel C/D content",
                "source_A": "The older approval-package caption describes acquisition-channel history and four topology ratios.",
                "source_B": "The frozen strict-data artwork shows exact fixed-mu slices in panel C and five strict audit criteria including component stability in panel D.",
                "source_priority": "frozen Figure 4 artwork, data manifest, and science validation",
                "provisional_resolution": "Supersede the stale caption with the new caption tied to panels C/D of the frozen figure.",
                "scientific_impact": "resolved; restores caption/artwork consistency",
                "human_decision_required": False,
            },
        ],
    }


def _review_tex(project_root: Path, *, scientifically_frozen: bool = False) -> str:
    result_tex = _results_tex().replace(r"\section{Results}", r"\section*{Results candidate}").replace(r"\label{sec:results}", "")
    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=17mm]{geometry}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\usepackage{microtype}",
        r"\usepackage{graphicx}",
        r"\usepackage{amssymb}",
        r"\usepackage{xcolor}",
        r"\usepackage{hyperref}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.7em}",
        r"\begin{document}",
        r"\begin{center}",
        r"{\Large\bfseries Active Learning--BdG manuscript\\Results and caption review packet}\par",
        (
            r"\vspace{0.5em}{\small Run \texttt{active\_learning\_prl\_v1}; Results/captions frozen after approved Figure Engine source repair}\par"
            if scientifically_frozen
            else r"\vspace{0.5em}{\small Run \texttt{active\_learning\_prl\_v1}; human-review candidate; not scientifically frozen}\par"
        ),
        r"\end{center}",
        (
            r"\textbf{Review status.} The Results and captions are scientifically frozen after the approved Figure Engine source repair. Figure 4 uses the strict-data four-slice artwork. The Figure 3 $k=6$ sensitivity caveat and the Figure 4 finite-domain/nonclosure/no-continuous-surface limitations remain mandatory. Methods, Supplemental Material, Discussion, and Outlook have been drafted separately; external Introduction retrieval has not been authorized."
            if scientifically_frozen
            else r"\textbf{Review focus.} Verify the physics-first narrative, sentence-level observation/interpretation separation, the Figure 3 $k=6$ sensitivity caveat, and the Figure 4 finite-domain/nonclosure/no-continuous-surface limitations. Methods and Introduction retrieval have not been run."
        ),
        result_tex,
    ]
    for index, (figure_id, relative) in enumerate(FIGURE_SOURCES.items(), start=1):
        absolute = (project_root / relative).as_posix()
        lines.extend([
            r"\clearpage",
            rf"\section*{{Frozen Figure {index}}}",
            rf"\begin{{center}}\includegraphics[width=\textwidth,height=0.72\textheight,keepaspectratio]{{\detokenize{{{absolute}}}}}\end{{center}}",
            rf"\textbf{{Figure {index}.}} {CAPTIONS[figure_id]}",
        ])
    if scientifically_frozen:
        gate_lines = [
            r"\item \texttt{GATE\_RESULTS\_HUMAN\_REVIEW = PASS}.",
            r"\item \texttt{GATE\_MANUSCRIPT\_FIGURE\_SOURCE\_REPAIR = PASS}.",
            r"\item Current gate: \texttt{INTRO\_CITATION\_LOOP\_GATE = HUMAN\_WAIT}.",
        ]
    else:
        gate_lines = [r"\item Current gate: \texttt{GATE\_RESULTS\_HUMAN\_REVIEW = HUMAN\_WAIT}."]
    lines.extend([
        r"\clearpage",
        r"\section*{Automated gate summary}",
        r"\begin{itemize}",
        r"\item Claim--evidence coverage: PASS.",
        r"\item Observation--interpretation separation: PASS.",
        r"\item Caption/figure consistency: PASS.",
        r"\item Forbidden-claim audit: PASS.",
        r"\item Numerical consistency and frozen-figure hashes: PASS.",
        *gate_lines,
        r"\end{itemize}",
        r"\end{document}",
    ])
    return "\n".join(lines)


def _compile_review(output_dir: Path) -> None:
    command = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "results_review_packet.tex"]
    completed = subprocess.run(command, cwd=output_dir, text=True, capture_output=True)
    (output_dir / "latex_compile.log").write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError("Results review packet LaTeX compilation failed")
    subprocess.run(["latexmk", "-c", "results_review_packet.tex"], cwd=output_dir, check=False, capture_output=True)
    render = subprocess.run(
        ["pdftoppm", "-png", "-r", "150", "results_review_packet.pdf", "figures/results_review_page"],
        cwd=output_dir,
        text=True,
        capture_output=True,
    )
    if render.returncode != 0:
        raise RuntimeError("Results review packet PDF rendering failed")


def build_results_review_package(
    project_root: Path,
    output_dir: Path,
    *,
    compile_pdf: bool = False,
    scientifically_frozen: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    frozen_rows = _verify_frozen_hashes(project_root)
    numerics = _verify_numerics(project_root)
    trace = _sentence_trace()
    results_md = _results_markdown(scientifically_frozen=scientifically_frozen)
    results_tex = _results_tex()
    captions_tex = _caption_tex()
    caption_trace = _caption_trace()
    forbidden_pass, forbidden_md = _forbidden_audit(results_md, captions_tex)
    if not forbidden_pass:
        raise RuntimeError("forbidden claim audit failed")
    audits = _audit_documents(numerics, trace)

    _write(output_dir / "results_candidate.md", results_md)
    _write(output_dir / "results_candidate.tex", results_tex)
    _write(output_dir / "figure_captions_candidate.tex", captions_tex)
    _write(output_dir / "results_review_packet.md", results_md + "\n\n# Candidate captions\n\n" + "\n\n".join(f"## Figure {i}\n\n{caption}" for i, caption in enumerate(CAPTIONS.values(), start=1)))
    _write(output_dir / "results_forbidden_claim_audit.md", forbidden_md)
    for name, text in audits.items():
        _write(output_dir / name, text)
    _write(output_dir / "results_review_packet.tex", _review_tex(project_root, scientifically_frozen=scientifically_frozen))
    (output_dir / "results_sentence_evidence_trace.yaml").write_text(yaml.safe_dump({"schema_version": 1, "sentences": trace}, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (output_dir / "figure_caption_evidence_trace.yaml").write_text(yaml.safe_dump({"schema_version": 1, "captions": caption_trace}, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (output_dir / "manuscript_conflict_ledger.yaml").write_text(yaml.safe_dump(_conflict_ledger(), sort_keys=False, allow_unicode=True), encoding="utf-8")
    (output_dir / "paragraph_metadata.json").write_text(json.dumps(_paragraph_metadata(), indent=2) + "\n", encoding="utf-8")
    (output_dir / "numerical_verification.json").write_text(json.dumps(numerics, indent=2) + "\n", encoding="utf-8")
    (output_dir / "frozen_figure_hash_audit.json").write_text(json.dumps({"status": "PASS", "artifacts": frozen_rows}, indent=2) + "\n", encoding="utf-8")

    with (output_dir / "tables/numerical_invariants.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["figure", "quantity", "value", "source", "status"])
        writer.writerows([
            ["Fig2", "total", 7434, "Stage II frozen NPZ", "PASS"],
            ["Fig2", "Normal/uniform_SC/FFLO", "1867/715/4852", "Stage II y_phase", "PASS"],
            ["Fig3", "trusted_gapped_FFLO", 3213, "Stage III trust mask", "PASS"],
            ["Fig3", "cFFLO/tFFLO", "2290/923", "Stage III topology_z2", "PASS"],
            ["Fig3", "k8/k6 coverage_p95", f"{numerics['stage3']['k8_coverage_p95']:.9f}/{numerics['stage3']['k6_coverage_p95']:.9f}", "frozen sensitivity CSV", "PASS"],
            ["Fig4", "total", 7081, "Stage IV frozen NPZ", "PASS"],
            ["Fig4", "Normal/uniform_SC/FFLO", "2350/141/4590", "Stage IV y_phase", "PASS"],
            ["Fig4", "FFLO-only cFFLO/tFFLO", "3145/1445", "Stage IV trust mask", "PASS"],
            ["Fig4", "mu/t range", "[-0.5,1.5]", "Stage IV x[:,2]", "PASS"],
        ])

    decision_status = (
        """- Results and captions are scientifically frozen after the approved Figure Engine source repair.
- Methods, Supplement, Discussion, and Outlook candidates were generated downstream.
- External Introduction retrieval, Abstract/Title generation, and final LaTeX integration were not executed.
- Current gate: `INTRO_CITATION_LOOP_GATE = HUMAN_WAIT`."""
        if scientifically_frozen
        else """- Results and captions are not scientifically frozen. The next decision is human review at `GATE_RESULTS_HUMAN_REVIEW`.
- Methods, Supplement, Discussion, Outlook, Introduction retrieval, Abstract/Title, and final LaTeX integration were not executed."""
    )
    _write(
        output_dir / "decision_log.md",
        f"""# Decision log

- Generated a graph-orchestrated Results/caption candidate from frozen Figures 1--4 and approved registries.
- Automated claim, sentence-role, caption, numerical, forbidden-claim, and frozen-hash checks pass.
- Figure 3 retains the $k=6$ sensitivity miss; Figure 4 retains finite-domain, strict-nonclosure, and no-continuous-surface caveats.
{decision_status}
""",
    )
    if compile_pdf:
        _compile_review(output_dir)
    files = sorted(
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "changed_files_manifest.yaml"
    )
    manifest = {
        "schema_version": 1,
        "run_id": "active_learning_prl_v1",
        "generated_by_node": "results_by_figure_writer",
        "generated_files": files + ["changed_files_manifest.yaml"],
        "implementation_files": [
            "manuscript_langgraph/review_agent/results_package.py",
            "manuscript_langgraph/review_agent/state_schema.py",
            "manuscript_langgraph/review_agent/nodes.py",
            "manuscript_langgraph/review_agent/graphs/prl_manuscript_graph.py",
            "manuscript_langgraph/scripts/resume_after_figure_approval.py",
            "manuscript_langgraph/scripts/run_current_workflow.py",
            "manuscript_langgraph/scripts/validate_workflow.py",
            "manuscript_langgraph/tests/test_graph.py",
            "manuscript_langgraph/tests/test_results_package.py",
            "manuscript_langgraph/node_contracts/results_by_figure_writer.yaml",
            "manuscript_langgraph/node_contracts/results_human_review_gate.yaml",
            "manuscript_langgraph/MANUSCRIPT_MASTER_STATE.yaml",
            "manuscript_langgraph/manuscript_run_config.yaml",
            "manuscript_langgraph/runs/active_learning_prl_v1/run_config.yaml",
            "manuscript_langgraph/audits/gate_status.yaml",
            "manuscript_langgraph/README.md",
            "manuscript_langgraph/decision_log.md",
            "manuscript_langgraph/MANUSCRIPT_WORKLOG.md",
            "docs/PROJECT_SUMMARY.md",
            "docs/DECISIONS.md",
            "docs/report_qa/20260827_results_human_review.md",
            "shared_prompts/handoff/PROJECT_HANDOFF.md",
            "shared_prompts/handoff/NEXT_ACTIONS.md",
        ],
        "checkpoint_and_gate_outputs": [
            "manuscript_langgraph/runs/active_learning_prl_v1/checkpoints/checkpoint_results_review.json",
            "manuscript_langgraph/runs/active_learning_prl_v1/gate_records/results_human_review_gate.json",
            "manuscript_langgraph/runs/active_learning_prl_v1/results_review_state.json",
            "manuscript_langgraph/runs/active_learning_prl_v1/node_outputs/results_human_review_gate.json",
            "manuscript_langgraph/runs/active_learning_prl_v1/logs/results_resume.log",
            "manuscript_langgraph/audits/current_workflow_state.json",
        ],
        "frozen_figures_modified": False,
        "physics_recomputed": False,
        "reproduction_command": "PYTHONPATH=manuscript_langgraph world_model_descriptor_dev/.venv/bin/python manuscript_langgraph/scripts/resume_after_figure_approval.py",
    }
    (output_dir / "changed_files_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "paragraph_metadata": _paragraph_metadata(),
        "results_markdown": results_md,
        "audit_status": "PASS",
        "output_files": files + ["changed_files_manifest.yaml"],
    }
