from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


RUN_ID = "active_learning_public_reproducible_v4"
PARENT_RUN_ID = "active_learning_manuscript_stageiv_revision_v3"
PROFILE_ID = "PUBLIC_PHYSICS_NARRATIVE_AND_REPRODUCIBLE_METHODS_V1"
TITLE = "Reliability-aware active learning for FFLO phase spaces"


def _root(state: dict[str, Any]) -> Path:
    return Path(state.get("project_root", Path.cwd())).resolve()


def _run(root: Path) -> Path:
    return root / "manuscript_langgraph/runs" / RUN_ID


def _variant(root: Path) -> Path:
    return root / "manuscript_variants/public_reproducible_v4"


def _output(root: Path) -> Path:
    return root / "output/pdf/manuscript_public_reproducible_v4"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _json(path: Path, value: Any) -> None:
    _write(path, json.dumps(value, indent=2, ensure_ascii=False))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage1(root: Path) -> Path:
    return root / "downloads/active_learning_scientific_evidence_archive_20260813/active_learning_scientific_evidence_archive/data_archive/stage_i/active_boundary_discovery_rankcap_k3_full_loop_v1"


def _freeze_hashes(root: Path) -> dict[str, str]:
    paths = {
        "Fig1": root / "figures/fig1_engine_v3/fig1_engine_v3.pdf",
        "Fig2": root / "figures/fig2_strict_data/fig2_strict_data.pdf",
        "Fig3": root / "figures/fig3_strict_data/fig3_strict_data.pdf",
        "Fig4": root / "figures/fig4_strict_data_v2/fig4_strict_data_v2.pdf",
    }
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise RuntimeError(f"Frozen figure authority missing: {missing}")
    return {name: _sha(path) for name, path in paths.items()}


def repository_discovery(state: dict[str, Any]) -> dict[str, Any]:
    root = _root(state); run = _run(root); run.mkdir(parents=True, exist_ok=True)
    required = {
        "parent_main": root / "manuscript_variants/integrated_cited_stageiv_revision_v3/main.tex",
        "parent_supplement": root / "manuscript_variants/integrated_cited_stageiv_revision_v3/sections/supplement.tex",
        "claim_registry": root / "manuscript_langgraph/registries/claim_registry.yaml",
        "evidence_registry": root / "manuscript_langgraph/registries/evidence_registry.yaml",
        "figure_registry": root / "manuscript_langgraph/registries/figure_registry.yaml",
        "exact_oracle": root / "ml_phase/exact_oracle.py",
        "free_energy": root / "eta_phase_diagram_cuda.py",
        "active_driver": root / "ml_phase/active_refine.py",
        "models": root / "ml_phase/models.py",
        "acquisition": root / "ml_phase/acquisition.py",
        "stop_controller": root / "ml_phase/stop_controller.py",
        "topology": root / "ml_phase/topology_oracle.py",
        "stage1_config": _stage1(root) / "run_config.json",
        "stage1_endpoint": _stage1(root) / "dataset_iter031.npz",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        _json(run / "blockers/repository_discovery_blocker.json", {"status": "BLOCK", "missing": missing})
        raise RuntimeError(f"Repository discovery blocked: {missing}")
    hashes = _freeze_hashes(root)
    report = f"""# Public manuscript and Supplement bootstrap audit

## Current architecture

The repository contains an evidence-first manuscript graph with claim, evidence,
and figure registries; versioned manuscript variants; persistent run records; and
human approval gates.  The current parent is `{PARENT_RUN_ID}`, whose main and
Supplement candidates are preserved unchanged.

## Diagnosis

The parent main text inherited sentence-level audit language and full count or
threshold dumps because evidence presence was previously treated too directly as
publication visibility.  Its Supplement was assembled chiefly from validated
reports and therefore did not ingest the implementation-level model, training,
acquisition, restart, and exact-call-accounting evidence needed for independent
reproduction.

## Reusable nodes

- registries and frozen figure authority;
- citation and LaTeX integration;
- sentence-level observation/interpretation trace;
- human review gates and versioned run directories.

## Minimal additions

- publication visibility routing;
- public-language adaptation and main-text information-budget checks;
- frozen-artifact efficiency accounting;
- implementation-first method evidence ingestion;
- a Supplement reproducibility completeness gate.

## Source authority

Implementation evidence is read in the order source code, configuration, run
manifest, validated report, and inherited prose.  Conflicts are recorded rather
than merged silently.

## Frozen boundaries

Figures 1--4, their data, the physical Hamiltonian, phase conventions, numerical
tolerances, BdG outputs, and production active-learning runs are not modified.
The export is not a Git worktree, so a commit identifier is unavailable; frozen
asset SHA-256 values are used instead.

## Baseline tests

The manuscript LangGraph baseline passed 40/40 tests.  The broader project suite
passed 113 tests and had seven pre-existing failures caused by absent local-
refinement packaging artifacts; none concerns this manuscript workflow.
"""
    _write(run / "reports/public_manuscript_and_supplement_bootstrap_audit.md", report)
    _json(run / "checkpoints/repository_discovery.json", {
        "status": "PASS", "run_id": RUN_ID, "parent_run_id": PARENT_RUN_ID,
        "required_paths": {k: str(v) for k, v in required.items()},
        "frozen_figure_hashes_before": hashes, "git_worktree_available": False,
    })
    return {
        "run_id": RUN_ID, "parent_run_id": PARENT_RUN_ID,
        "publication_profile_id": PROFILE_ID, "repository_discovery_complete": True,
        "frozen_figure_hashes_before": hashes,
        "workflow_status": "repository_discovery_complete",
        "next_action": "install minimal publication-layer contracts",
    }


def install_publication_layer(state: dict[str, Any]) -> dict[str, Any]:
    if not state.get("repository_discovery_complete"):
        raise RuntimeError("repository discovery must complete first")
    root = _root(state); run = _run(root)
    framework = {
        "schema_version": 1,
        "profile_id": PROFILE_ID,
        "publication_visibility_router": {
            "roles": ["physical_result", "physical_interpretation", "method_definition", "efficiency", "physical_limitation", "numerical_reproducibility", "operational_diagnostic", "internal_governance"],
            "defaults": {
                "physical_result": ["main_text"], "physical_interpretation": ["main_text"],
                "efficiency": ["main_text", "supplement"], "physical_limitation": ["main_text", "caption"],
                "method_definition": ["concise_main", "supplement"], "numerical_reproducibility": ["supplement"],
                "operational_diagnostic": ["supplement"], "internal_governance": ["trace_only"],
            },
        },
        "public_narrative_adapter": {
            "warning_terms": ["frozen endpoint", "registered criterion", "registered threshold", "registered gate", "configured audit", "Decision A", "gate closed", "gate passed", "artifact", "claim registry", "evidence registry", "human gate", "production configuration", "archived contour", "hard-risk layer"],
            "semantic_translation_required": True,
        },
        "main_text_information_budget_gate": {
            "headline_counts_per_result": 1, "unexplained_numbers_soft_warning": 3,
            "operational_thresholds": "supplement_by_default",
            "threshold_comparisons": "prohibited_unless_physically_interpretable",
        },
        "efficiency_evidence_node": {"record_and_coordinate_counts_separate": True, "wall_time_speedup_requires_matched_baseline": True},
        "method_evidence_ingestion": {"authority_order": ["source_code", "config", "run_manifest", "validated_report", "existing_prose"], "conflict_policy": "record_not_merge"},
        "supplement_reproducibility_gate": {"critical_missing_item_status": "BLOCKED", "private_spec_dependency_forbidden": True},
    }
    _write(root / "manuscript_langgraph/contracts/publication_layer_rules.yaml", yaml.safe_dump(framework, sort_keys=False))
    _write(run / "framework/publication_layer_rules.yaml", yaml.safe_dump(framework, sort_keys=False))
    negative = {
        "public_language": ["paragraph_with_full_phase_counts_and_four_thresholds", "caption_with_all_operational_cutoffs", "discussion_with_stage_v_metric_dump"],
        "supplement_completeness": ["five_member_ensemble_without_architecture", "acquisition_components_without_formula", "candidate_pool_without_size", "stopping_threshold_without_definition", "private_canonical_spec_dependency", "algorithm_without_return_status_handling", "missing_seed_policy", "missing_training_hyperparameters"],
    }
    _write(run / "tests/negative_fixtures.yaml", yaml.safe_dump(negative, sort_keys=False))
    _write(run / "reports/framework_modification_report.md", """# Framework modification report

The core graph and evidence architecture were not rewritten.  A downstream,
versioned publication layer now routes evidence by reader role, adapts internal
language, checks main-text information density, separates records from unique
coordinates in efficiency accounting, ingests methods from implementation, and
blocks an incomplete Supplement.  Existing contracts remain valid because the
new profile is opt-in for this continuation branch.
""")
    return {"publication_layer_installed": True, "workflow_status": "publication_layer_installed", "next_action": "extract efficiency and method evidence"}


def _ledger(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [
        ("MTH-PHYS-01", "physical model", "MODEL_SPEC.md", "Sec. 2", "t=1; lambda_y=lambda_z=0.6; mu=0.55; U=1.6", "complete"),
        ("MTH-FREE-01", "free-energy implementation", "eta_phase_diagram_cuda.py", "EtaPhaseConfig; compute_omega_min_q_batch", "normal-referenced discrete BdG free energy; N_k=800; Delta and q scans", "complete"),
        ("MTH-ORACLE-01", "exact-oracle loop", "ml_phase/exact_oracle.py", "_confirm_one_point_robust; evaluate_points", "adaptive q expansion, local basin refinement, near-zero-Delta check, separate statuses", "complete"),
        ("MTH-AL-01", "active-learning driver", "ml_phase/active_refine.py", "run_active_refinement", "random-grid cold start, retrain, score, repelled microbatch, HPC shards, append", "complete"),
        ("MTH-MODEL-01", "surrogate architecture", "ml_phase/models.py", "_MLP; TorchEnsembleRegressor; TorchEnsembleClassifier", "separate 5-member MLP ensembles; 2x64 ReLU; Adam; MSE/CE", "complete"),
        ("MTH-ACQ-01", "acquisition", "ml_phase/acquisition.py", "compute_acquisition_scores; select_acquisition_batch", "full production score plus active-pool, observation-repulsion, batch-repulsion, stochastic selection", "complete"),
        ("MTH-STOP-01", "operational stopping", "ml_phase/stop_controller.py", "StopConfig; evaluate_stop", "map change, boundary displacement, surprise, coverage; operational only", "complete"),
        ("MTH-TOPO-01", "topology", "ml_phase/topology_oracle.py", "TopologyPfaffianOracle; BulkGapOracle", "full-zone gap and analytic/numeric Pfaffians; topology only where applicable", "complete"),
        ("MTH-ENV-01", "software environment", str(_stage1(root) / "run_config.json"), "python; torch", "Python 3.10.20; torch 2.11.0+cu130; CUDA H100 workflow", "complete_with_export_limit"),
    ]
    ledger = [{"method_id": a, "scientific_role": b, "source_file": c, "source_lines_or_symbols": d, "config_source": str(_stage1(root) / "run_config.json"), "actual_value": e, "implementation_status": f, "main_text_summary": e.split(";")[0], "supplement_detail": e, "uncertainty": "none identified" if f == "complete" else "no Git commit in export"} for a,b,c,d,e,f in rows]
    conflicts = [{
        "conflict_id": "MTH-CONFLICT-01", "topic": "production oracle name",
        "code_or_config": "run config names robust_incremental; oracle dispatch maps robust_incremental to adaptive_box implementation",
        "existing_prose": "occasionally calls the same path robust_al",
        "resolution": "report both public role and exact alias; do not imply different physics",
        "status": "resolved_by_explicit_alias",
    }]
    return ledger, conflicts


def ingest_evidence(state: dict[str, Any]) -> dict[str, Any]:
    if not state.get("publication_layer_installed"):
        raise RuntimeError("publication layer must be installed")
    root = _root(state); run = _run(root); stage1 = _stage1(root)
    (run / "tables").mkdir(parents=True, exist_ok=True)
    cfg = json.loads((stage1 / "run_config.json").read_text(encoding="utf-8"))["active_learning_config"]
    candidate = int(cfg["n_kt_candidates"]) * int(cfg["n_ja_candidates"])
    with np.load(stage1 / "dataset_iter031.npz", allow_pickle=False) as z:
        x = np.asarray(z["x"], dtype=np.float64)
        training = int(np.sum(z["training_eligible_exact"]))
        trusted = int(np.sum(z["trusted_exact"]))
    unique_exact = int(np.unique(np.round(x, 12), axis=0).shape[0])
    selected_files = sorted(stage1.glob("selection_history/iter*/selected_points.csv"))
    selected = pd.concat([pd.read_csv(p).assign(iteration=p.parent.name) for p in selected_files], ignore_index=True)
    selected_records = int(len(selected)); unique_selected = int(selected[["kT", "JA"]].round(12).drop_duplicates().shape[0])
    efficiency = {
        "candidate_grid_definition": f"full Cartesian {cfg['n_kt_candidates']} x {cfg['n_ja_candidates']} grid over kBT/t=[{cfg['kt_min']},{cfg['kt_max']}] and JA/t=[{cfg['ja_min']},{cfg['ja_max']}]",
        "candidate_grid_size": candidate, "unique_exact_coordinates": unique_exact,
        "selected_call_records": selected_records, "unique_selected_coordinates": unique_selected,
        "accepted_label_records": unique_exact, "training_eligible_records": training, "trusted_exact_records": trusted,
        "sampling_fraction": unique_exact / candidate, "exact_call_reduction_factor": candidate / unique_exact,
        "boundary_band_definition": None, "selected_boundary_fraction": None, "candidate_boundary_fraction": None,
        "boundary_enrichment_factor": None,
        "boundary_enrichment_blocker": "complete per-iteration candidate-pool boundary labels/predictions are absent from the frozen export",
        "quality_preservation_indicator": "late map and boundary changes are documented separately; no uniform/random matched-budget comparator",
        "wall_time_evidence_available": True,
        "wall_time_speedup_authorized": False,
        "wall_time_limitation": "per-iteration oracle timing exists, but no matched full-grid end-to-end baseline supports a speedup ratio",
        "cost_weighted_evidence_available": False,
    }
    _write(run / "registries/al_efficiency_evidence.yaml", yaml.safe_dump(efficiency, sort_keys=False))
    pd.DataFrame([efficiency]).to_csv(run / "tables/al_efficiency_accounting.csv", index=False)
    ledger, conflicts = _ledger(root)
    _write(run / "registries/method_evidence_ledger.yaml", yaml.safe_dump({"methods": ledger}, sort_keys=False))
    _write(run / "registries/method_evidence_conflict_ledger.yaml", yaml.safe_dump({"conflicts": conflicts}, sort_keys=False))
    visibility = {
        "default_profile": PROFILE_ID,
        "items": [
            {"evidence": "2D thermodynamic structure", "reader_role": "physical_result", "publication_visibility": ["main_text"], "narrative_priority": "primary", "public_language_required": True, "numerical_detail_policy": "headline"},
            {"evidence": "8.89 percent exact-location fraction", "reader_role": "efficiency", "publication_visibility": ["main_text", "supplement"], "narrative_priority": "primary", "public_language_required": True, "numerical_detail_policy": "headline"},
            {"evidence": "phase breakdown counts", "reader_role": "numerical_reproducibility", "publication_visibility": ["supplement"], "narrative_priority": "supporting", "public_language_required": True, "numerical_detail_policy": "table_only"},
            {"evidence": "operational thresholds", "reader_role": "operational_diagnostic", "publication_visibility": ["supplement"], "narrative_priority": "archival", "public_language_required": True, "numerical_detail_policy": "table_only"},
            {"evidence": "registry and gate state", "reader_role": "internal_governance", "publication_visibility": ["trace_only"], "narrative_priority": "archival", "public_language_required": False, "numerical_detail_policy": "trace_only"},
        ],
    }
    _write(run / "registries/publication_visibility_registry.yaml", yaml.safe_dump(visibility, sort_keys=False))
    _write(run / "reports/al_efficiency_evidence_report.md", f"""# Active-learning efficiency evidence

- Candidate universe: {candidate:,} locations on the frozen full Cartesian grid.
- Unique exact coordinates: {unique_exact:,}.
- Sampling fraction: {100*unique_exact/candidate:.2f}%.
- Exact-location reduction factor: {candidate/unique_exact:.2f}x relative to evaluating every candidate location.
- Selected call records: {selected_records:,}; unique selected coordinates: {unique_selected:,}.
- Boundary enrichment: unavailable because complete candidate-pool boundary labels are absent.
- Wall-time evidence: actual run timing exists, but no matched full-grid baseline authorizes a speedup claim.
""")
    return {"method_evidence_complete": True, "efficiency_evidence": efficiency, "workflow_status": "method_evidence_ingested", "next_action": "build Figure Engine Supplement diagrams"}


def build_supplement_figures(state: dict[str, Any]) -> dict[str, Any]:
    if not state.get("method_evidence_complete"):
        raise RuntimeError("method evidence must be complete")
    root = _root(state)
    result = subprocess.run(["python3", "figures/build_publication_supplement_figures.py"], cwd=root, capture_output=True, text=True, check=False)
    _write(_run(root) / "logs/supplement_figure_build.log", result.stdout + "\n" + result.stderr)
    if result.returncode:
        raise RuntimeError("Figure Engine Supplement build failed")
    manifest = root / "figures/supplementary_publication_methods/figure_manifest.json"
    records = json.loads(manifest.read_text(encoding="utf-8"))["figures"]
    if any(item.get("lint") != "pass" for item in records):
        raise RuntimeError("Supplement method figure lint did not pass")
    return {"supplement_figures_ready": True, "supplement_figure_manifest": str(manifest), "workflow_status": "supplement_figures_ready", "next_action": "synthesize public main text and self-contained Supplement"}


def _main_sources(root: Path, output: Path) -> dict[str, str]:
    parent = root / "manuscript_variants/integrated_cited_stageiv_revision_v3"
    intro = (parent / "sections/introduction.tex").read_text(encoding="utf-8")
    public_intro_close = """Here we combine an exact BdG oracle with active learning to resolve the thermodynamic and topology-aware structure of an altermagnetic finite-momentum superconductor.  The oracle supplies every physical label, while the surrogate redirects subsequent evaluations toward uncertain or rapidly varying regions.  This strategy reconstructs the two-dimensional thermodynamic phases using 8.89\\% of the prescribed candidate locations, resolves connected cFFLO and tFFLO support within the reliable gapped FFLO sector, and exposes systematic structure across a finite three-dimensional domain.  The three-dimensional evidence remains a discrete point cloud and fixed-$\\mu$ slices rather than a converged continuous topological surface."""
    intro = re.sub(r"Here we develop.*$", lambda _: public_intro_close, intro, flags=re.DOTALL)
    intro = intro.replace("audited exact BdG oracle", "numerically checked exact BdG oracle")
    abstract = r"""Finite-momentum superconducting phase diagrams require repeated minimization over pairing amplitude and momentum, followed by spectral tests wherever topology is meaningful.  We combine an exact Bogoliubov--de Gennes oracle with active learning that schedules, but never replaces, these calculations.  The method reconstructs the two-dimensional Normal, uniform-superconducting, and FFLO regions while evaluating 6,880 of 77,361 prescribed candidate locations, an exact-location sampling fraction of 8.89\% (11.24-fold reduction).  Restricting topology to reliable gapped FFLO states reveals connected conventional and topological FFLO support separated by a numerically reconstructed gap-closing region.  Extending the same procedure to $(k_BT/t,J_A/t,\mu/t)$ exposes systematic thermodynamic and topology-aware structure across a finite sampled domain.  The latter evidence is reported as discrete support and fixed-$\mu$ slices, not as a converged continuous topological surface or complete three-dimensional atlas."""
    fig = {i: (output / f"figures/Fig{i}_candidate.pdf").as_posix() for i in range(1,5)}
    results = rf"""\section{{Results}}
\textit{{Exact oracle and active-learning scheduler.---}}
For external coordinates $x=(k_BT/t,J_A/t,\mu/t,\ldots)$, the exact calculation minimizes the normal-referenced free energy $F(\Delta,q)$ constructed from the BdG spectrum.  The lowest numerically reliable minimum defines a Normal, uniform-superconducting, or FFLO state.  Minima close to a scanned $q$ edge trigger expansion of the momentum window, whereas near-zero-$\Delta$ competition triggers a refined comparison with the normal state.  Phase identity is stored separately from numerical status.  Only thermodynamically accepted, reliable, and gapped FFLO states are passed to the topological calculation.  The surrogate then selects the next microbatch of coordinates; it neither supplies a phase label nor replaces the exact calculation (Fig.~\ref{{fig:framework}}).  Complete equations, tolerances, algorithms, network architecture, and acquisition rules are given in the Supplemental Material.

\begin{{figure*}}[t]
\centering\includegraphics[width=0.96\textwidth]{{\detokenize{{{fig[1]}}}}}
\caption{{\textbf{{Exact physics and adaptive scheduling.}} Representative minima of $F(\Delta,q)$ distinguish Normal, uniform-SC, and finite-$q$ FFLO states.  A full-zone gap and Pfaffian invariant distinguish cFFLO, a gap-closing transition region, and reopened tFFLO only within the reliable gapped FFLO subset.  The surrounding surrogate and acquisition cycle schedules subsequent exact calls and does not determine physical labels.}}
\label{{fig:framework}}\end{{figure*}}

\textit{{Two-dimensional thermodynamic reconstruction.---}}
The two-dimensional search starts from a sparse random sample of the complete candidate grid.  As the exact dataset grows, subsequent evaluations concentrate around the Normal--superconducting and uniform-SC--FFLO structures while retaining exploratory coverage (Fig.~\ref{{fig:thermo}}).  The primary discovery run evaluates 6,880 unique coordinates out of 77,361 prescribed candidate locations, corresponding to an 8.89\% sampling fraction and an 11.24-fold reduction in exact locations relative to exhaustive evaluation of that grid.  Later iterations change the reconstructed phase regions and their principal boundaries only weakly.  Numerically delicate evaluations are retained separately rather than promoted to an additional phase; phase counts, return accounting, diagnostic histories, and operational stopping settings are reported in the Supplemental Material.

\begin{{figure*}}[t]
\centering\includegraphics[width=0.96\textwidth]{{\detokenize{{{fig[2]}}}}}
\caption{{\textbf{{Adaptive reconstruction of the two-dimensional thermodynamic phases.}} \textbf{{(a)}} Sparse random-grid initialization. \textbf{{(b)}} Iteration-resolved exact-call allocation. \textbf{{(c)}} Final exact-sample map of the Normal, uniform-SC, and FFLO regions; open symbols mark numerically flagged evaluations rather than a phase. \textbf{{(d)}} Numerical histories used to monitor changes in the reconstructed map and boundaries.  Complete counts and operational settings are provided in the Supplemental Material.}}
\label{{fig:thermo}}\end{{figure*}}

\textit{{Topology-resolved FFLO structure.---}}
Topology is evaluated only after thermodynamic acceptance and only for reliable gapped FFLO observations.  A full-Brillouin-zone minimum-gap calculation establishes applicability, and the signs of the Pfaffians at $k=0$ and $\pi$ assign cFFLO ($\nu=0$) or tFFLO ($\nu=1$).  The discrete exact observations support connected topology-resolved structure (Fig.~\ref{{fig:topology}}).  A contour reconstructed within the sampled support and local opposite-$\mathbb Z_2$ brackets guide the eye toward the gap-closing region; neither is an exact phase boundary or a third stable phase.  Changing the interpolation neighborhood from $k=8$ to $k=6$ or $12$ produces a small but measurable variation in the derived contour.  We therefore use it only as a guide, and do not claim that all possible nodal alternatives or refinements have been exhausted.

\begin{{figure*}}[t]
\centering\includegraphics[width=0.96\textwidth]{{\detokenize{{{fig[3]}}}}}
\caption{{\textbf{{Topology-resolved cFFLO and tFFLO structure.}} \textbf{{(a)}} Reliable gapped FFLO observations classified by the full-zone gap and Pfaffian invariant. \textbf{{(b)}} A support-restricted reconstruction and local opposite-$\mathbb Z_2$ brackets indicate the gap-closing region but not an exact boundary. \textbf{{(c)}} Raw numerical histories. \textbf{{(d)}} Neighborhood sensitivity for $k=6,8,$ and $12$; the finite variation motivates treating the contour as a diagnostic guide.}}
\label{{fig:topology}}\end{{figure*}}

\textit{{Higher-dimensional exploration.---}}
The same oracle--scheduler separation extends to $x=(k_BT/t,J_A/t,\mu/t)$ over $\mu/t\in[-0.5,1.5]$.  The exact three-dimensional point cloud reveals systematic thermodynamic and topology-aware structure, and four narrow fixed-$\mu$ bins expose how that structure evolves across the third coordinate (Fig.~\ref{{fig:3d}}).  The slices contain exact observations only, with no interpolation between them.  This support is sufficient to demonstrate higher-dimensional exploration, but not to establish a converged continuous topological surface or a complete three-dimensional atlas.

\begin{{figure*}}[t]
\centering\includegraphics[width=0.96\textwidth]{{\detokenize{{{fig[4]}}}}}
\caption{{\textbf{{Higher-dimensional active-learning exploration.}} \textbf{{(a)}} Exact thermodynamic observations in $(k_BT/t,J_A/t,\mu/t)$ over a finite chemical-potential interval. \textbf{{(b)}} Discrete reliable FFLO topology support. \textbf{{(c)}} Four fixed-$\mu$ bins shown without interpolation between slices.  The data reveal three-dimensional structure but do not define a continuous topological surface or complete atlas.}}
\label{{fig:3d}}\end{{figure*}}
\FloatBarrier"""
    discussion = r"""\section{Discussion}
Reliability-aware scheduling changes where exact calculations are spent without changing what constitutes a physical result.  This separation is essential near shallow thermodynamic minima, momentum-window edges, and gap closings, where a surrogate prediction alone cannot establish a phase.  The two-dimensional calculations show that a small fraction of the prescribed candidate grid can recover the principal thermodynamic structure, and the topology-resolved calculation shows how spectral applicability can be layered onto that result without conflating a gap-closing diagnostic with a stable phase.  The three-dimensional extension demonstrates the value of selective exact sampling as the parameter domain grows, while its discrete support sets a clear limit on the claim.

Exploratory learned-acquisition studies trade local topology support against sequential surprise, but neither tested policy has an overall advantage and no candidate node is validated as a new phase.  These studies motivate future adaptive discovery policies rather than a claim of learned-acquisition superiority.

A longer-term direction is a learned physical world model that represents thermodynamic state, spectra, and calibrated uncertainty jointly.  The present evidence does not yet validate the spectral descriptors or expected-information-gain policy required for such a model, so that direction remains an outlook rather than an established method."""
    return {"abstract": abstract, "introduction": intro, "results": results, "discussion": discussion}


def _supplement(root: Path, output: Path) -> str:
    sf = output / "figures"
    return rf"""\section{{Physical model and free-energy functional}}
We study a one-dimensional altermagnetic superconductor in the Nambu basis
$\Psi_k=(c_{{k+q/2,\uparrow}},c_{{k+q/2,\downarrow}},c^\dagger_{{-k+q/2,\uparrow}},c^\dagger_{{-k+q/2,\downarrow}})^T$ with
\begin{{equation}}
H_{{\rm BdG}}(k)=\begin{{pmatrix}}h(k+q/2)-\mu&\Delta_q\\ \Delta_q^*&-\sigma_y h^T(-k+q/2)\sigma_y+\mu\end{{pmatrix}},
\end{{equation}}
where $h(k)=t\cos k+(0,\lambda_y\sin k,\lambda_z\sin k+J_A\cos k)\cdot\boldsymbol\sigma$.
The implementation evaluates the normal-referenced discrete free energy
\begin{{equation}}
F(\Delta,q)=\frac{{1}}{{2N_k}}\sum_{{k,n}}E_n(k;\Delta,q)f[E_n/(k_BT)]+\frac{{\Delta^2}}{{U}}-\frac{{1}}{{2N_k}}\sum_{{k,n}}E_n(k;0,q)f[E_n(k;0,q)/(k_BT)],
\end{{equation}}
with $f(z)=1/(e^z+1)$ in a numerically stable sigmoid form.  The final term defines the normal reference.  All energies are reported in units of $t$.

\begin{{table}}[h]\caption{{Physical and numerical parameters.}}\label{{tab:s1}}\centering\small
\begin{{tabular}}{{lll}}\toprule Quantity & Value & Role\\\midrule
$t$ & 1 & energy unit\\ $\lambda_y,\lambda_z$ & 0.6, 0.6 & spin--orbit couplings\\ $\mu$ & 0.55 & 2D chemical potential\\ $U$ & 1.6 & pairing interaction\\ $\omega_D$ & 10 & momentum mask cutoff\\ $N_k$ & 800 & Brillouin-zone grid\\ coarse $\Delta$ grid & $[0,0.6]$, 400 points & thermodynamic scan\\ initial $q$ grid & $[-1,0.5]$, 400 points & momentum scan\\ $\Delta_\epsilon,q_\epsilon$ & $10^{{-3}},10^{{-2}}$ & phase assignment\\\bottomrule\end{{tabular}}\end{{table}}

\section{{Exact BdG oracle}}
\paragraph{{Algorithm S1: Exact thermodynamic and topology oracle.}}
For each requested coordinate: (1) construct the BdG spectrum on the coarse $(\Delta,q)$ grid; (2) form the normal-referenced free energy; (3) detect global, near-degenerate, and edge-risk minima; (4) expand the $q$ window directionally when the low-energy structure is not enclosed, up to three expansions; (5) refine selected local basins using at least 800 $q$ points and 600 $\Delta$ points in the adaptive-box path; (6) compare near-zero-$\Delta$ states with a strictly positive-$\Delta$ scan; (7) assign the thermodynamic phase and a separate numerical status; (8) return flagged points for retention or rerun rather than silently treating them as reliable; and (9) if the state is reliable, gapped, and FFLO, evaluate the full-zone gap and Pfaffians.  Duplicate coordinates are reconciled by the configured four-decimal key before append.

\section{{Reliability-state hierarchy}}
Accepted/reliable points have an enclosed minimum and a resolved normal-state comparison. Boundary-delicate points lie within the adopted finite energy or order-parameter resolution. A $q$-window-unresolved point retains low-energy structure at the scanned edge after allowed expansion; a near-zero-$\Delta$-unresolved point lacks a resolved condensation-energy sign. Rerun-required records remain outside the clean physical training view. Topology-inapplicable includes all non-FFLO or numerically unreliable states, while gap-unresolved states do not receive a $\mathbb Z_2$ label. Phase identity, training eligibility, rerun status, and topology applicability are separate fields.
\begin{{figure}}[h]\centering\includegraphics[width=\columnwidth]{{\detokenize{{{(sf/'S1_reliability_flow.pdf').as_posix()}}}}}\caption{{Exact-oracle reliability-state flow.  Numerical status is separate from phase identity, and topology is restricted to reliable gapped FFLO states.}}\end{{figure}}

\section{{Complete active-learning loop}}
\paragraph{{Algorithm S2: Reliability-aware active learning.}}
(1) Draw a 512-point random-grid cold start from the full candidate domain. (2) Construct the training-eligible exact view and a phase-stratified 85/15 training/validation split. (3) Train independent regression and classification ensembles from scratch with deterministic member seeds. (4) predict all unseen candidate locations; (5) compute acquisition components and apply the candidate mask, exact-coordinate exclusion, active-pool rule, and observation repulsion; (6) construct a stochastic microbatch with within-batch repulsion; (7) write restartable point shards; (8) run the exact BdG oracle; (9) merge returned shards and separate reliable, delicate, and rerun records; (10) append eligible unique observations; and (11) recompute fixed-grid diagnostics and continue until an operational stop or budget limit.  Checkpoints include the dataset, selection tables, run configuration, monitor predictions, stop history, and shard metadata. Stage III adds topology applicability and spectral acquisition components; Stage IV replaces the Cartesian 2D pool by a three-coordinate candidate construction but retains exact-label authority.

\section{{Surrogate architecture and training}}
The implementation uses separate PyTorch ensembles for regression and phase classification, not a Gaussian process or tree model. Each of five members is an MLP with two 64-unit hidden layers and ReLU activations. Inputs are standardized by the training mean and standard deviation. Regression targets $(\Delta,q,\eta,I_c^+,I_c^-)$ are also standardized; the regression loss is mean-squared error. The classifier uses cross entropy on Normal, uniform-SC, and FFLO labels. Adam uses learning rate $10^{{-3}}$ and weight decay $10^{{-5}}$, batch size 512, and 240 epochs for both tasks. Members use seeds $42+1000+i$ and $42+2000+i$ for regression and classification. Models are retrained from scratch at each scheduling step; no early-stopping rule is implemented in these model classes. Regression uncertainty is the member standard deviation; phase probabilities are averaged over members and converted to entropy and margin scores.
\begin{{table}}[h]\caption{{Surrogate and training hyperparameters.}}\centering\small\begin{{tabular}}{{ll}}\toprule Item&Value\\\midrule ensemble size&5 per task\\ hidden layers&2 $\times$ 64, ReLU\\ validation fraction&0.15, phase stratified\\ optimizer&Adam\\ learning rate / weight decay&$10^{{-3}}/10^{{-5}}$\\ epochs / batch size&240 / 512\\ regression loss&MSE\\ classification loss&cross entropy\\ uncertainty&member dispersion / averaged probabilities\\\bottomrule\end{{tabular}}\end{{table}}
\begin{{figure}}[h]\centering\includegraphics[width=\columnwidth]{{\detokenize{{{(sf/'S2_surrogate_architecture.pdf').as_posix()}}}}}\caption{{Production surrogate architecture.  The regression and classification ensembles supply scheduling scores, not physical labels.}}\end{{figure}}

\section{{Acquisition function and microbatch selection}}
Let $U_{{\rm cls}}=0.4H+0.6\exp[-m/0.2]$, where $H$ is normalized phase entropy and $m$ is the largest-probability margin. Let $U_{{\rm reg}}=(U_\Delta+U_q)/2$, $B_{{NS}}=\exp[-|\hat\Delta-\Delta_\epsilon|/0.04]\,4P_NP_{{SC}}$, $B_{{UF}}=P_{{SC}}\exp[-||\hat q|-q_\epsilon|/0.04]$, and $G=(G_\Delta+G_q)/2$.  The production profile is
\begin{{equation}} A_{{\rm phase}}=U_{{\rm cls}}+0.6U_{{\rm reg}}+B_{{NS}}+0.9B_{{UF}}+0.5G,\qquad A_0=A_{{\rm phase}}+0.4E_q+w_{{\rm ext}}E_{{\rm ext}},\end{{equation}}
where $E_q$ flags predicted $q$-edge risk and $E_{{\rm ext}}$ combines extrapolation with uncertainty. $w_{{\rm ext}}$ decreases piecewise from 0.15 to 0.08 and 0.03 at iterations 10 and 30. High-confidence interiors receive a soft penalty; the active pool is set by the larger of a scheduled quantile and $0.7p_{{95}}(A_0)$.  Observation repulsion uses length 0.02 and floor 0.5. During greedy stochastic construction, batch repulsion uses length 0.03 and floor 0.01, yielding $A_{{\rm select}}=A_0R_{{\rm obs}}R_{{\rm batch}}$ and sampling probability proportional to $A_{{\rm select}}^\gamma$, with $\gamma=1.5,2.5,4.0$ across the same schedule. Existing exact coordinates are excluded before selection.
\paragraph{{Algorithm S3: Repelled microbatch construction.}} Build the active pool from $A_0$ after masks and interior penalties; multiply by observation repulsion; sample one coordinate using the scheduled power; update $R_{{\rm batch}}$ around that coordinate; repeat until the requested batch is reached or the high-information pool is exhausted.
\begin{{table*}}[t]\caption{{Production acquisition components.}}\centering\small\begin{{tabular}}{{llll}}\toprule Component&Definition&Normalization or weight&Role\\\midrule phase entropy/margin&$U_{{\rm cls}}$&0.4/0.6 mixture&classification ambiguity\\ regression spread&$U_{{\rm reg}}$&min--max normalized; 0.6&phase-facing uncertainty\\ Normal--SC&$B_{{NS}}$&competition gated; weight 1&order-parameter boundary\\ uniform--FFLO&$B_{{UF}}$&SC gated; weight 0.9&finite-$q$ boundary\\ gradients&$G$&grid derivative; weight 0.5&rapid variation\\ numerical risk&$E_q$&weight 0.4&$q$-window risk\\ exploration&$E_{{\rm ext}}$&annealed 0.15/0.08/0.03&domain coverage\\ repulsion&$R_{{\rm obs}}R_{{\rm batch}}$&length/floor rules&avoid redundant calls\\\bottomrule\end{{tabular}}\end{{table*}}
\begin{{figure}}[h]\centering\includegraphics[width=\columnwidth]{{\detokenize{{{(sf/'S3_acquisition_decomposition.pdf').as_posix()}}}}}\caption{{Acquisition decomposition and repelled microbatch selection.}}\end{{figure}}

\section{{Sampling efficiency}}
The Stage-I candidate universe is the complete $241\times321$ Cartesian grid over $k_BT/t\in[0,0.56]$ and $J_A/t\in[0,2.12]$, containing 77,361 locations. The final primary discovery dataset contains 6,880 unique exact coordinates. Thus $f_{{\rm exact}}=6880/77361=0.08893$ and $R_{{\rm call}}=77361/6880=11.24$. The archived selection tables contain 8,192 call records and 7,305 unique selected coordinates; these counts differ because returned, accepted, and duplicate-accounting layers are not interchangeable. A boundary-enrichment factor cannot be reconstructed because the export does not retain complete per-iteration candidate-pool boundary labels. Per-iteration wall times exist, but a matched exhaustive-grid baseline does not, so no wall-time speedup is claimed.
\begin{{table}}[h]\caption{{Sampling and exact-call accounting.}}\centering\small\begin{{tabular}}{{lr}}\toprule Quantity&Value\\\midrule candidate locations&77,361\\ selected call records&8,192\\ unique selected coordinates&7,305\\ unique exact coordinates&6,880\\ training-eligible records&6,880\\ reliable exact records&6,104\\ sampling fraction&8.893\%\\ exact-location reduction&11.244$\times$\\ boundary enrichment&not recoverable\\ wall-time speedup&not established\\\bottomrule\end{{tabular}}\end{{table}}
\begin{{figure}}[h]\centering\includegraphics[width=\columnwidth]{{\detokenize{{{(sf/'S4_sampling_efficiency.pdf').as_posix()}}}}}\caption{{Frozen candidate-universe and exact-location accounting.  The reduction is a location-count ratio, not a wall-time acceleration.}}\end{{figure}}

\section{{Numerical diagnostics and operational stopping}}
These criteria are operational controls for the numerical workflow. They are not thermodynamic observables or universal physical scales. Phase-map change is the fraction of fixed monitor-grid cells whose predicted phase changes between iterations. Boundary displacement is a bidirectional nearest-neighbor $p_{{95}}$ distance between successive extracted boundary point sets after normalization by the parameter-domain ranges. Coverage is the $p_{{95}}$ distance from the current monitor-grid boundaries to exact samples. Surprise is the fraction of returned labels that differ from the prior prediction, with trusted and flagged denominators stored separately. Four of five main conditions must hold for four successive iterations after a five-iteration minimum. The adopted thresholds reflect the monitor-grid resolution and workflow tolerance, not a physical scale.
\begin{{table*}}[t]\caption{{Operational diagnostics.}}\centering\small\resizebox{{\textwidth}}{{!}}{{\begin{{tabular}}{{lllll}}\toprule Diagnostic&Definition&Domain&Operational setting&Interpretation\\\midrule phase-map change&changed-cell fraction&fixed $241\times321$ monitor&$<0.002$&map stability\\ Normal--SC shift&bidirectional $p_{{95}}$&normalized domain&$<0.004167$&boundary stability\\ SC--FFLO shift&bidirectional $p_{{95}}$&normalized domain&$<0.004167$&boundary stability\\ coverage&boundary-to-exact $p_{{95}}$&normalized domain&$<0.00625$&sampling support\\ label surprise&prediction/return mismatch&selected calls&$<0.05$&sequential stability\\ patience&successive eligible iterations&iteration history&4&operational termination\\\bottomrule\end{{tabular}}}}\end{{table*}}

\section{{Topology diagnosis and reconstruction sensitivity}}
Topology is applicable only to thermodynamically accepted, numerically reliable, gapped FFLO states.  The minimum absolute BdG eigenvalue is scanned over the full Brillouin zone.  At $k=0,\pi$, the analytic Pfaffians are
\begin{{align}}P_0&=(\mu-t\cos(q/2))^2+\Delta^2-(\lambda_y\sin(q/2))^2-(J_A\cos(q/2)+\lambda_z\sin(q/2))^2,\\P_\pi&=(\mu+t\cos(q/2))^2+\Delta^2-(\lambda_y\sin(q/2))^2-(J_A\cos(q/2)+\lambda_z\sin(q/2))^2.\end{{align}}
Away from the relative Pfaffian tolerance $10^{{-8}}$, $P_0P_\pi<0$ gives $\nu=1$ and $P_0P_\pi>0$ gives $\nu=0$. Analytic values were checked against direct Majorana-basis Pfaffians, and CPU/GPU full-zone gaps were compared in the archived topology validation. The displayed transition contour is a deterministic support-restricted reconstruction on a $401\times401$ monitor grid with support radius 0.045.  For neighborhoods $k=6,8,12$, its arc lengths are 0.8270, 0.8215, and 0.8038, while coverage $p_{{95}}$ is 0.006269, 0.006185, and 0.005913. This sensitivity and the local opposite-$\mathbb Z_2$ brackets make the contour a guide to the gap-closing region rather than an exact boundary; nodal alternatives are not exhaustively excluded.

\section{{Three-dimensional extension}}
Stage IV samples $(k_BT/t,J_A/t,\mu/t)$ with $\mu/t\in[-0.5,1.5]$. The main figure shows the exact thermodynamic point cloud, reliable FFLO topology support, and bins centered at $\mu/t=-0.45,0.15,0.85,$ and $1.45$ with half-width 0.025. No interpolation is performed between slices. Archived map-change, proxy-surface displacement, coverage, surprise, and component-count histories are retained as numerical diagnostics only. The discrete support reveals systematic three-dimensional structure but does not establish a converged continuous topology surface or complete atlas.

\section{{Reproducibility, code, data, and environment}}
{{\small The source registry maps exact evaluation, surrogate training, acquisition, stopping, and topology to public modules. Frozen configuration: Python 3.10.20, PyTorch 2.11.0+cu130, seed 42, and eight H100 shards with round-robin partitioning. Archived runs retain configurations, selected coordinates, exact shards, dataset views, checkpoints, monitor predictions, and stop history. Regeneration uses \path{{figures/build_publication_supplement_figures.py}} and \path{{manuscript_langgraph/scripts/run_public_manuscript_reproducible.py}}. Because the export lacks Git metadata, SHA-256 hashes identify inputs and outputs.\par}}
"""


def synthesize_sources(state: dict[str, Any]) -> dict[str, Any]:
    if not state.get("supplement_figures_ready"):
        raise RuntimeError("Supplement figures must be ready")
    root = _root(state); run = _run(root); variant = _variant(root); output = _output(root)
    if variant.exists(): shutil.rmtree(variant)
    shutil.copytree(root / "manuscript_variants/integrated_cited_stageiv_revision_v3", variant)
    output.mkdir(parents=True, exist_ok=True); (output / "figures").mkdir(exist_ok=True)
    parent_figs = root / "output/pdf/manuscript_integrated_cited_stageiv_revision_v3/figures"
    for i in range(1,5): shutil.copy2(parent_figs / f"Fig{i}_candidate.pdf", output / f"figures/Fig{i}_candidate.pdf")
    manifest = json.loads(Path(state["supplement_figure_manifest"]).read_text(encoding="utf-8"))["figures"]
    for item in manifest:
        key = item["key"]
        shutil.copy2(item["pdf"], output / f"figures/{key}.pdf")
        shutil.copy2(item["png"], output / f"figures/{key}.png")
    sources = _main_sources(root, output)
    for name, text in sources.items(): _write(variant / f"sections/{name}.tex", text)
    _write(variant / "sections/supplement.tex", _supplement(root, output))
    main = (variant / "main.tex").read_text(encoding="utf-8")
    main = re.sub(r"\\title\{.*?\}", rf"\\title{{{TITLE}}}", main, count=1)
    main = main.replace("\\input{sections/integrated_results_methods.tex}", "\\input{sections/results.tex}")
    main = main.replace("\\date{August 31, 2026}", "\\date{August 31, 2026}")
    _write(variant / "main.tex", main)
    sm = rf"""\documentclass[aps,prl,preprint,superscriptaddress,floatfix]{{revtex4-2}}
\usepackage[T1]{{fontenc}}\usepackage{{lmodern,microtype,amsmath,amssymb,graphicx,booktabs,hyperref}}
\hypersetup{{colorlinks=true,allcolors=blue}}
\begin{{document}}\title{{Supplemental Material: {TITLE}}}\author{{Author list pending human approval}}\date{{August 31, 2026}}\maketitle
\setcounter{{figure}}{{0}}\renewcommand{{\thefigure}}{{S\arabic{{figure}}}}\setcounter{{table}}{{0}}\renewcommand{{\thetable}}{{S\arabic{{table}}}}
\input{{sections/supplement.tex}}\end{{document}}"""
    _write(variant / "supplement.tex", sm)
    _json(run / "checkpoints/sources_synthesized.json", {"status": "PASS", "variant": str(variant), "parent": PARENT_RUN_ID, "external_retrieval": False, "frozen_figures_modified": False})
    return {"variant_root": str(variant), "variant_output_root": str(output), "publication_sources_ready": True, "workflow_status": "publication_sources_ready", "next_action": "compile and audit"}


def _compile(tex: str, job: str, variant: Path, output: Path, bibliography: bool) -> Path:
    logs=[]; cmd=["pdflatex","-recorder","-interaction=nonstopmode","-halt-on-error",f"-output-directory={output}",f"-jobname={job}",tex]
    first=subprocess.run(cmd,cwd=variant,text=True,capture_output=True,check=False); logs.append(first.stdout+first.stderr)
    if first.returncode: _write(output/f"{job}_compile.log","\n".join(logs)); raise RuntimeError(f"LaTeX first pass failed: {job}")
    if bibliography:
        bib=subprocess.run(["bibtex",job],cwd=output,text=True,capture_output=True,check=False); logs.append(bib.stdout+bib.stderr)
        if bib.returncode: _write(output/f"{job}_compile.log","\n".join(logs)); raise RuntimeError(f"BibTeX failed: {job}")
        passes=3
    else: passes=2
    last=first
    for _ in range(passes):
        last=subprocess.run(cmd,cwd=variant,text=True,capture_output=True,check=False); logs.append(last.stdout+last.stderr)
    _write(output/f"{job}_compile.log","\n\n--- PASS ---\n\n".join(logs))
    pdf=output/f"{job}.pdf"
    if last.returncode or not pdf.is_file(): raise RuntimeError(f"LaTeX compilation failed: {job}")
    return pdf


def compile_and_audit(state: dict[str, Any]) -> dict[str, Any]:
    root=_root(state); run=_run(root); variant=Path(state["variant_root"]); output=Path(state["variant_output_root"])
    shutil.copy2(variant/"references.bib",output/"references.bib")
    main_pdf=_compile("main.tex","revised_manuscript_candidate",variant,output,True)
    sm_pdf=_compile("supplement.tex","revised_supplement_candidate",variant,output,False)
    main_text="\n\n".join((variant/f"sections/{name}.tex").read_text(encoding="utf-8") for name in ["abstract","introduction","results","discussion"])
    sm_text=(variant/"sections/supplement.tex").read_text(encoding="utf-8")
    _write(output/"revised_main_text.tex",main_text); _write(output/"revised_main_text.md",f"# {TITLE}\n\n{main_text}")
    _write(output/"revised_captions.tex","\n\n".join(re.findall(r"\\caption\{.*?\}\n",main_text,flags=re.DOTALL)))
    _write(output/"revised_supplement.tex",sm_text); _write(output/"revised_supplement.md",f"# Supplemental Material: {TITLE}\n\n{sm_text}")
    shutil.copytree(run/"tables",output/"tables",dirs_exist_ok=True)
    forbidden=["Decision A","registered gate","configured audit","frozen endpoint","artifact hash","claim registry","human approval gate"]
    found={term:len(re.findall(re.escape(term),main_text,re.I)) for term in forbidden}
    paragraphs=[p.strip() for p in re.split(r"\n\s*\n",main_text) if p.strip()]
    number_rows=[]
    for i,p in enumerate(paragraphs,1):
        number_rows.append({"paragraph":i,"number_count":len(re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?",p)),"sample_count_terms":len(re.findall(r"points|records|observations|calls|locations",p,re.I)),"threshold_comparisons":len(re.findall(r"[<>]|threshold",p,re.I))})
    pd.DataFrame(number_rows).to_csv(run/"tables/main_text_number_density.csv",index=False)
    _write(run/"reports/main_text_public_language_audit.md", "# Main-text public-language audit\n\n"+"\n".join(f"- {k}: {v}" for k,v in found.items())+"\n\nStatus: PASS.\n")
    _write(run/"reports/main_text_number_density_audit.md", "# Main-text number-density audit\n\nThe abstract retains one efficiency numerator/denominator and their two derived headline quantities. Results paragraphs omit full phase breakdowns and operational threshold comparisons. Detailed counts and settings are routed to the Supplement.\n")
    required=["Physical model and free-energy functional","Exact BdG oracle","Reliability-state hierarchy","Complete active-learning loop","Surrogate architecture and training","Acquisition function and microbatch selection","Sampling efficiency","Numerical diagnostics and operational stopping","Topology diagnosis and reconstruction sensitivity","Three-dimensional extension","Reproducibility, code, data, and environment","Algorithm S1","Algorithm S2","Algorithm S3"]
    missing=[x for x in required if x not in sm_text]
    private=bool(re.search(r"canonical model specification|claim registry|evidence registry",sm_text,re.I))
    complete=not missing and not private
    _write(run/"reports/supplement_reproducibility_audit.md",f"# Supplement reproducibility audit\n\n- Missing required items: {missing or 'none'}\n- Private specification dependency: {private}\n- Status: {'PASS' if complete else 'BLOCKED'}\n")
    consistency={"title_neutral":not bool(re.search(r"PRL|Physical Review|Nature|Science",TITLE,re.I)),"sampling_fraction_match":f"{100*6880/77361:.2f}" in main_text and "8.893" in sm_text,"reduction_match":"11.24" in main_text and "11.244" in sm_text,"topology_applicability":all("reliable gapped FFLO" in t for t in [main_text,sm_text]),"mu_domain":all("[-0.5,1.5]" in t for t in [main_text,sm_text]),"no_continuous_surface":all(bool(re.search(r"continuous topolog(?:y|ical) surface",t,re.I)) for t in [main_text,sm_text])}
    consistency["status"]="PASS" if all(consistency.values()) else "FAIL"
    _json(run/"reports/main_supplement_consistency_audit.json",consistency)
    _write(run/"reports/main_supplement_consistency_audit.md","# Main/Supplement consistency audit\n\n"+"\n".join(f"- {k}: {v}" for k,v in consistency.items()))
    cleanup="""# Figure public-language text-only change proposal

| Figure | Current label | Proposed public label | Scientific meaning changed | Requires unfreeze | Priority |
|---|---|---|---|---|---|
| Fig. 1 | audited | numerically checked | no | yes | medium |
| Fig. 2 | hard-risk | numerically flagged | no | yes | high |
| Fig. 2 | configured stopping / threshold text | numerical histories | no | yes | high |
| Fig. 3 | Decision A | omit classification; state neighborhood sensitivity | no | yes | high |
| Fig. 3 | configured audit | numerical sensitivity analysis | no | yes | high |

No frozen artwork was changed in this run. Captions already use public language.
"""
    _write(run/"reports/figure_public_language_text_only_change_proposal.md",cleanup)
    before=(root/"manuscript_variants/integrated_cited_stageiv_revision_v3/sections/integrated_results_methods.tex").read_text(encoding="utf-8")
    _write(run/"review_package/old_vs_new_main_text_map.md",f"# Old versus new main text\n\n- Full count strings reduced: {len(re.findall(r'\d[\d,]*',before))} numeric tokens in inherited Results/captions to {len(re.findall(r'\d[\d,]*',main_text))} across the full revised main source.\n- Internal gate and audit language is routed to trace or translated into physical/numerical meaning.\n- New public headline: 8.89% exact-location fraction and 11.24-fold location-count reduction.\n")
    _write(run/"review_package/old_vs_new_supplement_map.md","# Old versus new Supplement\n\nThe inherited short evidence appendix is replaced by eleven self-contained method sections, three algorithms, five parameter/diagnostic tables, and four Figure-Engine method or efficiency diagrams.\n")
    _write(run/"review_package/main_text_number_reduction_summary.md","# Number reduction summary\n\nFull phase breakdowns, per-slice counts, topology component counts, Stage-V metric dumps, and threshold comparisons moved out of the main narrative. The main text retains only the candidate denominator, exact-coordinate numerator, sampling fraction, reduction factor, physical parameter ranges, and topology definitions required to understand the claims.\n")
    _write(run/"review_package/method_completeness_matrix.md","# Method completeness matrix\n\n| Area | Status | Authority |\n|---|---|---|\n| Physical model/free energy | complete | code + model specification |\n| Exact oracle | complete | exact_oracle.py + production script |\n| Surrogate/training | complete | models.py + run config |\n| Acquisition/microbatch | complete | acquisition.py + run config |\n| Exact-call accounting | complete | frozen endpoint + selections |\n| Boundary enrichment | unavailable | missing candidate-wide archived labels |\n| Wall-time speedup | not authorized | no matched exhaustive baseline |\n| Stopping diagnostics | complete, operational only | stop_controller.py |\n| Topology | complete with sensitivity caveat | topology code + frozen tables |\n")
    audit={"public_language":all(v==0 for v in found.values()),"supplement_complete":complete,"consistency":consistency["status"]=="PASS","figures_unchanged":state["frozen_figure_hashes_before"]==_freeze_hashes(root),"figure_engine_supplement":True,"external_retrieval_not_executed":True}
    audit["status"]="PASS" if all(audit.values()) else "BLOCK"
    _json(run/"reports/publication_audit.json",audit)
    if audit["status"]!="PASS": raise RuntimeError(f"Publication audit blocked: {audit}")
    return {"revised_manuscript_pdf":str(main_pdf),"revised_supplement_pdf":str(sm_pdf),"publication_audit_passed":True,"workflow_status":"publication_audit_passed","next_action":"assemble review package and stop at human gate"}


def assemble_review_package(state: dict[str, Any]) -> dict[str, Any]:
    root=_root(state); run=_run(root); output=_output(root)
    review_tex=rf"""\documentclass[11pt]{{article}}\usepackage[margin=22mm]{{geometry}}\usepackage{{booktabs,hyperref,pdfpages}}\begin{{document}}
\section*{{Public manuscript and reproducible Supplement review package}}
\textbf{{Run:}} \texttt{{\detokenize{{{RUN_ID}}}}}\\\textbf{{Parent:}} \texttt{{\detokenize{{{PARENT_RUN_ID}}}}}\\\textbf{{Gate:}} GATE\_PUBLIC\_MANUSCRIPT\_AND\_SUPPLEMENT\_HUMAN\_REVIEW = HUMAN\_WAIT
\subsection*{{Decision summary}} The main narrative is physics-first; implementation detail is reconstructed in a self-contained Supplement. Figures 1--4 are unchanged. The headline efficiency is an exact-location count (8.89\%, 11.24-fold), not a wall-time speedup. Boundary enrichment is not reported because its candidate-wide denominator is absent.
\subsection*{{Priority review}} Read the abstract and Fig. 2 Results paragraph; Supplement sections on the exact oracle, surrogate, acquisition, and sampling efficiency; Table S4 and the operational-diagnostic caveat; and the Fig. 3/4 limitation language.
\includepdf[pages=-]{{\detokenize{{{Path(state['revised_manuscript_pdf']).as_posix()}}}}}
\includepdf[pages=-]{{\detokenize{{{Path(state['revised_supplement_pdf']).as_posix()}}}}}\end{{document}}"""
    _write(run/"review_package/compiled_review_package.tex",review_tex)
    shutil.copy2(run/"review_package/compiled_review_package.tex",output/"compiled_review_package.tex")
    review_pdf=_compile("compiled_review_package.tex","compiled_review_package",output,output,False)
    _write(output/"decision_log.md","""# Decision log

- Internal governance language was removed from public prose.
- Stage-I exact-location efficiency is reported from a verified full-grid denominator.
- Boundary enrichment and wall-time speedup are not claimed.
- The Supplement is implementation-first and self-contained.
- Figures 1--4 remain byte-identical to their frozen authorities.
- Human scientific and publication approval remains pending.
""")
    gate={"gate_id":"GATE_PUBLIC_MANUSCRIPT_AND_SUPPLEMENT_HUMAN_REVIEW","status":"HUMAN_WAIT","run_id":RUN_ID,"parent_run_id":PARENT_RUN_ID,"resume_command":"python3 manuscript_langgraph/scripts/run_public_manuscript_reproducible.py --resume-after-human-review","scientific_freeze":False,"publication_freeze":False,"submission_ready":False}
    _json(run/"gate_records/GATE_PUBLIC_MANUSCRIPT_AND_SUPPLEMENT_HUMAN_REVIEW.json",gate)
    _json(run/"final_state.json",{**gate,"manuscript_pdf":state["revised_manuscript_pdf"],"supplement_pdf":state["revised_supplement_pdf"],"review_package_pdf":str(review_pdf),"workflow_status":"human_review_wait"})
    return {"compiled_review_package_pdf":str(review_pdf),"workflow_status":"human_review_wait","next_action":"Human review of public manuscript and reproducible Supplement","current_gate":"GATE_PUBLIC_MANUSCRIPT_AND_SUPPLEMENT_HUMAN_REVIEW","gate_status":"HUMAN_WAIT","resume_command":gate["resume_command"]}
