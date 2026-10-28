from __future__ import annotations

import csv
import hashlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT_ROOT / "manuscript_langgraph/audits/manuscript_artifact_manifest.csv"
TABLE_OUTPUT = PROJECT_ROOT / "manuscript_langgraph/tables/manuscript_artifact_manifest.csv"


def entry(stage, run_id, path, artifact_type, maturity, use, notes=""):
    return {
        "stage": stage,
        "run_id": run_id,
        "path": path,
        "artifact_type": artifact_type,
        "scientific_maturity": maturity,
        "manuscript_use": use,
        "notes": notes,
    }


USAGE_MAP = {
    "methods": "main_text|supplementary",
    "methods_and_supplement": "main_text|supplementary",
    "audit": "supplementary|archive_only",
    "discussion_and_scope": "discussion|archive_only",
    "main_text": "main_text",
    "main_text_and_supplement": "main_text|supplementary",
    "supplement_or_discussion": "supplementary|discussion",
    "scope": "archive_only",
    "discussion": "discussion",
    "supplement": "supplementary",
    "outlook": "discussion",
}


def reproducibility_status(spec: dict[str, str], present: bool) -> str:
    if not present:
        return "blocked_missing_local_artifact"
    if spec["artifact_type"] == "frozen_dataset":
        return "frozen_data_available"
    if spec["artifact_type"] == "hpc_package":
        return "reproduction_package_available"
    if spec["artifact_type"] in {"table", "figure"}:
        return "generated_artifact_available_source_audit_required"
    if spec["artifact_type"] == "report_pdf":
        return "report_level_only"
    return "source_or_report_available"


ENTRIES = [
    entry("all", "canonical_model", "MODEL_SPEC.md", "specification", "authoritative_definition", "methods"),
    entry("all", "canonical_numerics", "docs/NUMERICS_SPEC.md", "specification", "authoritative_definition", "methods_and_supplement"),
    entry("all", "decision_history", "docs/DECISIONS.md", "decision_log", "authoritative_decisions", "audit"),
    entry("all", "project_memory", "docs/PROJECT_SUMMARY.md", "project_memory", "authoritative_latest_record", "audit"),
    entry("I-V", "stage_i_v_compilation", "docs/StageI_V_Report.md", "planning_and_report_prompt", "supporting_history", "discussion_and_scope"),
    entry("II", "phase2_robust_al_final_202606", "docs/PHASE2_FINAL_AUDIT_DECISION_LOG.md", "decision_log", "frozen_supported", "main_text"),
    entry("II", "phase2_robust_al_final_202606", "docs/report_qa/20260620_phase2_final_audit_and_report.md", "audit_note", "frozen_supported", "main_text_and_supplement"),
    entry("II", "active_boundary_discovery_rankcap_k3_tail_surprise_continuation_v1", "manuscript_langgraph/recovered_artifacts/stageII/dataset_iter035.npz", "frozen_dataset", "frozen_supported", "main_text", "recovered 2026-08-13; SHA-256 matches the Phase II reproduction manifest"),
    entry("III", "topology_pass_dataset_iter035_v1", "docs/report_qa/20260620_stageiii_topology_pass_dataset_iter035.md", "validation_note", "supporting_validation", "methods_and_supplement"),
    entry("III", "topology_convergence_audit_full_loop_v2", "docs/report_qa/20260623_topology_convergence_audit_full_loop_v2.md", "convergence_audit", "frozen_supported_with_caveat", "main_text"),
    entry("III", "active_phase_topology_from_scratch_full_loop_v2", "manuscript_langgraph/recovered_artifacts/stageIII/dataset_iter018.npz", "frozen_dataset", "frozen_supported_with_caveat", "main_text", "recovered 2026-08-13; all NPZ arrays match the paired CSV; decision JSON stores the CSV hash under an NPZ-labeled source field"),
    entry("III", "topology_convergence_audit_full_loop_v2", "manuscript_langgraph/recovered_artifacts/stageIII/topology_convergence_audit_full_loop_v2/topology_convergence_audit_report.pdf", "report_pdf", "frozen_supported_with_caveat", "main_text_and_supplement", "recovered report, tables, figures, decision JSON, and source; intermediate cumulative NPZ checkpoints were intentionally omitted by the archive"),
    entry("IV", "active_phase_topology_3d_t_ja_mu_from_scratch_v1", "docs/report_qa/20260628_stageiv_3d_audit_summary.md", "audit_note", "supported_with_caveat_not_strictly_converged", "main_text"),
    entry("IV", "stageiv_scope", "docs/StageIV_MultiDim_Phase_Diagram.md", "planning_and_report_prompt", "supporting_history", "scope"),
    entry("IV", "active_phase_topology_3d_t_ja_mu_from_scratch_v1_identity_guard_hpc_20260624", "manuscript_langgraph/recovered_artifacts/stageIV/dataset_iter025.npz", "frozen_dataset", "supported_with_caveat_not_strictly_converged", "main_text", "recovered 2026-08-13; SHA-256 matches the Stage IV reproduction manifest"),
    entry("IV", "stageiv_3d_return_report_local", "manuscript_langgraph/recovered_artifacts/stageIV/stageiv_3d_return_report_local/stageiv_3d_return_report.pdf", "report_pdf", "supported_with_caveat_not_strictly_converged", "main_text_and_supplement", "recovered report, tables, reproduction manifest, audits, and original figures"),
    entry("IV", "stageiv_3d_return_report_local", "manuscript_langgraph/recovered_artifacts/stageIV/figures/phase_3d_jakt_view.png", "figure", "supported_with_caveat_not_strictly_converged", "main_text", "recovered candidate Figure 4 source; diagnostic smoothing and strict non-convergence caveats remain"),
    entry("V3", "stagev_v3_sparse_arc_boundary_support_3d_v1", "downloads/stagev_v3_v4_matched_audit_raw_20260712/v3/ML_Phase_StageV_V3_SparseArc/active_runs/stagev_v3_sparse_arc_boundary_support_3d_v1/dataset_iter097.npz", "frozen_dataset", "complete_not_converged", "discussion"),
    entry("V4", "stagev_v4_phase_structure_graph_acquisition_3d_v1", "downloads/stagev_v4_min_report_audit_20260711/extracted/ML_Phase_StageV_V4_GraphAcq/active_runs/stagev_v4_phase_structure_graph_acquisition_3d_v1/dataset_iter097.npz", "frozen_dataset", "complete_not_converged", "discussion"),
    entry("V", "stagev_v3_v4_matched_offline_audit", "output/pdf/stagev_v3_v4_matched_audit/stagev_v3_v4_matched_audit_report.md", "matched_audit", "negative_comparative_result", "discussion"),
    entry("V", "stagev_v3_v4_matched_offline_audit", "output/pdf/stagev_v3_v4_matched_audit/stagev_v3_v4_matched_audit_report.pdf", "report_pdf", "negative_comparative_result", "discussion"),
    entry("V", "stagev_v3_v4_matched_offline_audit", "output/pdf/stagev_v3_v4_matched_audit/tables/stagev_v3_v4_surface_metrics.csv", "table", "negative_comparative_result", "discussion"),
    entry("V", "stagev_v3_v4_matched_offline_audit", "output/pdf/stagev_v3_v4_matched_audit/figures/figure_20_matched_decision_summary.png", "figure", "negative_comparative_result", "discussion"),
    entry("V3", "stagev_v3_sparse_arc_boundary_support_3d_v1", "hpc_packages/stagev_v3_sparse_arc_boundary_support_3d_v1_hpc.tar.gz", "hpc_package", "reproducibility_package", "supplement"),
    entry("V4", "stagev_v4_phase_structure_graph_acquisition_3d_v1", "hpc_packages/stagev_v4_phase_structure_graph_acquisition_3d_v1_hpc.tar.gz", "hpc_package", "reproducibility_package", "supplement"),
    entry("VI-0", "stage_vi0_hierarchical_world_model_validation", "output/pdf/stage_vi0_hierarchical_world_model_validation/hierarchical_world_model_validation_report.md", "validation_report", "decision_c", "outlook"),
    entry("VI-0", "stage_vi0_hierarchical_world_model_validation", "output/pdf/stage_vi0_hierarchical_world_model_validation/decision.json", "decision", "decision_c", "outlook"),
    entry("VI-0.2", "stage_vi0_2_hierarchical_world_model_v2", "output/pdf/stage_vi0_2_hierarchical_world_model_v2/hierarchical_world_model_v2_report.md", "validation_report", "decision_d", "outlook"),
    entry("VI-0.2", "stage_vi0_2_hierarchical_world_model_v2", "output/pdf/stage_vi0_2_hierarchical_world_model_v2/decision.json", "decision", "decision_d", "outlook"),
    entry("II", "phase2_robust_al_final_202606", "scripts/build_phase2_final_audit_and_report.py", "report_and_figure_source", "reproducibility_source", "supplement"),
    entry("II", "phase2_robust_al_final_202606", "scripts/plot_discovery_exact_phase_map.py", "phase_map_plotting_source", "reproducibility_source", "supplement"),
    entry("III", "topology_convergence_audit_full_loop_v2", "scripts/build_topology_convergence_audit.py", "convergence_and_figure_source", "reproducibility_source", "supplement"),
    entry("IV", "active_phase_topology_3d_t_ja_mu_from_scratch_v1", "ml_phase/stageiv_3d.py", "higher_dimensional_active_learning_source", "reproducibility_source", "methods_and_supplement"),
    entry("IV", "stageiv_3d_return_report_local", "scripts/build_stageiv_3d_return_report.py", "report_and_figure_source", "reproducibility_source", "supplement"),
    entry("IV", "stageiv_3d_convergence_audit", "scripts/stageiv_3d_convergence_audit.py", "convergence_audit_source", "reproducibility_source", "supplement"),
    entry("V", "stagev_v3_v4_matched_offline_audit", "scripts/stagev_v3_v4_matched_audit.py", "audit_and_plotting_source", "reproducibility_source", "supplement"),
    entry("V", "stagev_v3_v4_matched_offline_audit", "output/pdf/stagev_v3_v4_matched_audit/stagev_v3_v4_matched_audit_report.tex", "latex_report_source", "negative_comparative_result", "discussion"),
    entry("V4", "stagev_v4_phase_structure_graph_acquisition_3d_v1", "output/pdf/stagev_v4_combined_report_audit/figures/fig01_final_phase_topology_projection.png", "phase_topology_map", "complete_not_converged", "discussion"),
    entry("V4", "stagev_v4_phase_structure_graph_acquisition_3d_v1", "scripts/build_stagev_v4_combined_report.py", "report_and_figure_source", "reproducibility_source", "supplement"),
    entry("manuscript", "evidence_first_prl_langgraph", "docs/report_qa/20260807_evidence_first_prl_langgraph.md", "ai_assisted_workflow_document", "workflow_provenance", "audit"),
    entry("manuscript", "evidence_first_prl_langgraph", "manuscript_langgraph/prompts/publication_oriented_manuscript_langgraph_prompt.md", "recorded_prompt", "workflow_authority", "audit"),
    entry("manuscript", "evidence_first_prl_langgraph", "manuscript_langgraph/review_agent/graphs/prl_manuscript_graph.py", "langgraph_source", "workflow_implementation", "audit"),
    entry("manuscript", "evidence_first_prl_langgraph", "manuscript_langgraph/manuscript_langgraph_implementation_report.tex", "latex_report_source", "workflow_report", "audit"),
    entry("manuscript", "evidence_first_prl_langgraph", "manuscript_langgraph/audits/scope_approval_record.yaml", "approval_record", "human_approved_scope", "audit"),
    entry("manuscript", "reliability_aware_3d_scope_revision", "manuscript_langgraph/audits/scope_revision_record_20260812.yaml", "approval_record", "human_approved_scope_revision", "audit"),
    entry("manuscript", "claim_approval_20260812", "manuscript_langgraph/audits/claim_approval_record_20260812.yaml", "approval_record", "human_approved_claim_registry", "audit"),
    entry("manuscript", "evidence_first_prl_langgraph", "manuscript_langgraph/audits/gate_status.yaml", "gate_status", "authoritative_workflow_state", "audit"),
    entry("manuscript", "frozen_artifact_recovery_20260813", "manuscript_langgraph/audits/artifact_recovery_report.md", "recovery_audit", "recovery_complete_for_requested_groups", "audit"),
    entry("manuscript", "frozen_artifact_recovery_20260813", "manuscript_langgraph/audits/recovery_search_manifest.csv", "recovery_search_manifest", "recovery_complete_for_requested_groups", "audit"),
    entry("manuscript", "transferred_package_inventory_20260813", "manuscript_langgraph/audits/transferred_package_inventory.md", "data_presence_audit", "evidence_archive_verified", "audit"),
    entry("manuscript", "transferred_package_inventory_20260813", "manuscript_langgraph/audits/transferred_package_data_presence.csv", "table", "evidence_archive_verified", "audit"),
    entry("manuscript", "windows_archive_intake_20260813", "manuscript_langgraph/audits/mac_archive_intake_report.md", "archive_intake", "checksum_verified", "audit"),
    entry("manuscript", "windows_archive_intake_20260813", "manuscript_langgraph/audits/manuscript_input_manifest.csv", "artifact_index", "checksum_verified", "audit"),
    entry("manuscript", "windows_archive_intake_20260813", "manuscript_langgraph/audits/artifact_path_map.yaml", "path_map", "canonical_selection_resolved", "audit"),
    entry("manuscript", "claim_approval_20260813", "manuscript_langgraph/audits/claim_approval_record_20260813.yaml", "approval_record", "human_approved_with_modifications", "audit"),
    entry("manuscript", "active_learning_prl_v1", "manuscript_langgraph/section_outline.yaml", "section_contract", "evidence_traced_candidate", "audit"),
    entry("manuscript", "active_learning_prl_v1", "manuscript_figures/final/Fig1_candidate.pdf", "figure", "candidate_unapproved", "main_text"),
    entry("II", "active_learning_prl_v1", "manuscript_figures/final/Fig2_candidate.pdf", "figure", "candidate_unapproved", "main_text"),
    entry("III", "active_learning_prl_v1", "manuscript_figures/final/Fig3_candidate.pdf", "figure", "candidate_unapproved", "main_text"),
    entry("IV", "active_learning_prl_v1", "manuscript_figures/final/Fig4_candidate.pdf", "figure", "candidate_unapproved_with_strict_nonconvergence_caveat", "main_text"),
    entry("II-IV", "active_learning_prl_v1", "manuscript_figures/final/SuppFig_S1_candidate.pdf", "figure", "candidate_unapproved", "supplement"),
    entry("manuscript", "active_learning_prl_v1", "manuscript_figures/scientific_validation.json", "validation", "pass", "audit"),
    entry("manuscript", "active_learning_prl_v1", "manuscript_langgraph/GATE_FIGURE_APPROVAL_PACKAGE/figure_approval_package.pdf", "approval_package", "pending_human_approval", "audit"),
    entry("manuscript", "active_learning_prl_v1", "manuscript_langgraph/runs/active_learning_prl_v1/checkpoints/manuscript.sqlite", "langgraph_checkpoint", "paused_at_figure_gate", "audit"),
    entry("manuscript", "active_learning_prl_v1", "manuscript_langgraph/runs/active_learning_prl_v1/final_state.json", "graph_state", "paused_at_figure_gate", "audit"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build() -> list[dict[str, str]]:
    rows = []
    for index, spec in enumerate(ENTRIES, start=1):
        path = PROJECT_ROOT / spec["path"]
        present = path.is_file()
        rows.append(
            {
                "artifact_id": f"ART-{index:03d}",
                "path": spec["path"],
                "type": spec["artifact_type"],
                "source_stage": spec["stage"],
                "run_id": spec["run_id"],
                "scientific_role": spec["scientific_maturity"],
                "provenance_status": (
                    "present_local_verified"
                    if present
                    else "unresolved_path"
                    if spec["path"].startswith("UNRESOLVED_PATH/")
                    else "missing_local"
                ),
                "reproducibility_status": reproducibility_status(spec, present),
                "possible_usage": USAGE_MAP[spec["manuscript_use"]],
                "sha256": sha256(path) if present else "",
                "status": "present" if present else "missing",
                "notes": spec["notes"],
            }
        )
    return rows


def write(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    result = build()
    write(result, OUTPUT)
    write(result, TABLE_OUTPUT)
    print(f"wrote {len(result)} rows to {OUTPUT.relative_to(PROJECT_ROOT)}")
