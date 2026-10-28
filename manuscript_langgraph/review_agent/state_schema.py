from __future__ import annotations

from typing import Any, Literal, TypedDict


GateStatus = Literal["PASS", "RECHECK", "BLOCK", "HUMAN_WAIT"]
ClaimStatus = Literal[
    "frozen_supported",
    "supported_with_caveat",
    "provisional",
    "discussion_only",
    "future_work",
]


class GateRecord(TypedDict):
    gate_id: str
    status: GateStatus
    evidence: list[str]
    findings: list[str]
    next_action: str


class ParagraphMetadata(TypedDict):
    paragraph_id: str
    observation: str
    interpretation: str
    evidence_source: list[str]
    figure_reference: list[str]
    claims_used: list[str]
    evidence_used: list[str]
    figure_panels_cited: list[str]
    caveats: list[str]
    unsupported_sentence_flags: list[str]


class ManuscriptGraphState(TypedDict, total=False):
    project_root: str
    manuscript_root: str
    artifact_manifest_path: str
    scope_path: str
    readiness_report_path: str
    claim_registry_path: str
    evidence_registry_path: str
    figure_registry_path: str
    figure_approval_package_path: str
    checkpoint_path: str
    run_id: str
    supplement_contract_path: str
    artifact_audit_complete: bool
    scope_approved: bool
    claim_approved: bool
    figure_approved: bool
    results_human_approved: bool
    results_correction_approved: bool
    final_manuscript_approved: bool
    claims_frozen: bool
    figure_contracts_approved: bool
    results_claims_frozen: bool
    vertical_slice_approved: bool
    external_retrieval_approved: bool
    latex_merge_approved: bool
    final_human_approved: bool
    active_figure_id: str
    paragraph_metadata: list[ParagraphMetadata]
    section_drafts: dict[str, str]
    gate_results: list[GateRecord]
    audit_findings: list[str]
    next_action: str
    workflow_status: str
    results_package_path: str
    results_review_checkpoint_path: str
    downstream_package_path: str
    methods_draft_complete: bool
    supplement_draft_complete: bool
    discussion_draft_complete: bool
    outlook_draft_complete: bool
    introduction_prerequisites_path: str
    results_correction_proposal_path: str
    metadata: dict[str, Any]
    publication_profile_path: str
    publication_profile_id: str
    variant_run_id: str
    parent_run_id: str
    variant_root: str
    variant_output_root: str
    method_packets_path: str
    integrated_manuscript_tex: str
    integrated_manuscript_pdf: str
    integrated_manuscript_md: str
    semantic_equivalence_trace_path: str
    inline_figure_placement_manifest_path: str
    prl_length_budget_path: str
    prl_integrated_candidate_complete: bool
    introduction_retrieval_authorized: bool
    verified_literature_registry_path: str
    citation_matrix_path: str
    bibliography_path: str
    cited_introduction_path: str
    citation_audit_passed: bool
    cited_manuscript_tex: str
    cited_manuscript_pdf: str
    compiled_citation_audit_passed: bool
    stageiv_scope_gate_passed: bool
    revision_sources_ready: bool
    revision_compiled: bool
    revision_audit_passed: bool
    revised_manuscript_pdf: str
    revised_supplement_pdf: str
    repository_discovery_complete: bool
    frozen_figure_hashes_before: dict[str, str]
    publication_layer_installed: bool
    method_evidence_complete: bool
    efficiency_evidence: dict[str, Any]
    supplement_figures_ready: bool
    supplement_figure_manifest: str
    publication_sources_ready: bool
    publication_audit_passed: bool
    compiled_review_package_pdf: str
    current_gate: str
    gate_status: str
    resume_command: str
    parent_source_hashes_before: dict[str, str]
    baseline_word_counts: dict[str, int]
    scientific_consistency_audits_complete: bool
    free_energy_consistency_status: str
    count_semantics_status: str
    count_semantics: dict[str, Any]
    depth_layer_installed: bool
    interpretation_packets_ready: bool
    descriptive_statistics_manifest: str
    selected_layout: str
    layout_metrics: dict[str, Any]
    expanded_manuscript_pdf: str
    fallback_manuscript_pdf: str
    main_word_counts: dict[str, int]
    main_words_added: int
    single_column_sources_ready: bool
    selected_single_column_layout: str
    single_column_layout_metrics: dict[str, Any]
    single_column_manuscript_pdf: str
    single_column_alternative_pdf: str
    single_column_compiled: bool
