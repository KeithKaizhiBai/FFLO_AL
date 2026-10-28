from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .results_package import build_results_review_package
from .downstream_package import (
    build_discussion_package,
    audit_fig4_caption_consistency,
    build_introduction_prerequisites,
    build_methods_package,
    build_supplement_package,
    compile_intermediate_review,
)
from .state_schema import GateRecord, ManuscriptGraphState


def _root(state: ManuscriptGraphState) -> Path:
    return Path(state.get("manuscript_root", "manuscript_langgraph")).resolve()


def _append_gate(
    state: ManuscriptGraphState,
    gate_id: str,
    status: str,
    findings: list[str],
    next_action: str,
    evidence: list[str] | None = None,
) -> list[GateRecord]:
    gates = [gate for gate in state.get("gate_results", []) if gate.get("gate_id") != gate_id]
    gates.append(
        {
            "gate_id": gate_id,
            "status": status,
            "evidence": evidence or [],
            "findings": findings,
            "next_action": next_action,
        }
    )
    return gates


def artifact_provenance_audit(state: ManuscriptGraphState) -> dict[str, Any]:
    manifest = _root(state) / "audits/manuscript_artifact_manifest.csv"
    findings: list[str] = []
    complete = manifest.exists()
    if complete:
        with manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        missing = [
            row["path"]
            for row in rows
            if row.get("provenance_status") in {"missing_local", "unresolved_path"}
        ]
        findings.extend([f"manifest_rows={len(rows)}", f"missing_rows={len(missing)}"])
    else:
        findings.append("artifact manifest is absent")
    return {
        "artifact_audit_complete": complete,
        "artifact_manifest_path": str(manifest),
        "audit_findings": findings,
        "gate_results": _append_gate(
            state,
            "ARTIFACT_PROVENANCE_AUDIT",
            "PASS" if complete else "BLOCK",
            findings,
            "build or review the artifact manifest",
            [str(manifest)],
        ),
    }


def publication_readiness_gate(state: ManuscriptGraphState) -> dict[str, Any]:
    matrix = _root(state) / "audits/publication_gap_matrix.csv"
    blockers: list[str] = []
    if matrix.exists():
        with matrix.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["blocks_main_claim"].lower() == "true" and row["status"] != "closed":
                    blockers.append(row["gap_id"])
    else:
        blockers.append("missing_publication_gap_matrix")
    return {
        "readiness_report_path": str(_root(state) / "audits/publication_readiness_report.md"),
        "gate_results": _append_gate(
            state,
            "PUBLICATION_READINESS_GATE",
            "BLOCK" if blockers else "PASS",
            blockers or ["no blocking gap recorded"],
            "close blockers or narrow the approved scope",
            [str(matrix)],
        ),
    }


def claim_freeze(state: ManuscriptGraphState) -> dict[str, Any]:
    registry = _root(state) / "registries/claim_registry.yaml"
    scope = _root(state) / "manuscript_scope.yaml"
    payload = json.loads(registry.read_text(encoding="utf-8")) if registry.exists() else {}
    claims = payload.get("claims", [])
    allowed_statuses = set(payload.get("allowed_statuses", []))
    required_fields = {
        "claim_id",
        "statement",
        "evidence_ids",
        "scientific_status",
        "allowed_language",
        "forbidden_language",
        "limitations",
    }
    structurally_valid = bool(claims) and bool(allowed_statuses) and all(
        required_fields <= set(claim)
        and claim["scientific_status"] in allowed_statuses
        for claim in claims
    )
    scope_approved = bool(state.get("scope_approved", False))
    claim_approved = bool(state.get("claim_approved", False))
    gates = _append_gate(
        state,
        "GATE_SCOPE_APPROVAL",
        "PASS" if scope_approved else "HUMAN_WAIT",
        [
            "approved physics-first scope is machine-readable"
            if scope_approved
            else "recommended scope is machine-readable but requires scientific approval"
        ],
        "review claim_registry.yaml" if scope_approved else "human reviews manuscript_scope.yaml",
        [str(scope)],
    )
    if not scope_approved:
        return {
            "scope_path": str(scope),
            "claim_registry_path": str(registry),
            "claims_frozen": False,
            "workflow_status": "awaiting_scope_approval",
            "next_action": "review manuscript_scope.yaml",
            "gate_results": gates,
        }
    claim_status = "BLOCK" if not structurally_valid else "PASS" if claim_approved else "HUMAN_WAIT"
    gates = _append_gate(
        {**state, "gate_results": gates},
        "GATE_CLAIM_APPROVAL",
        claim_status,
        ["claim registry schema is valid" if structurally_valid else "claim registry schema is invalid"],
        "human reviews claim_registry.yaml" if structurally_valid else "repair the claim registry",
        [str(registry)],
    )
    frozen = structurally_valid and claim_approved
    return {
        "scope_path": str(scope),
        "claim_registry_path": str(registry),
        "claims_frozen": frozen,
        "workflow_status": "claim_freeze_complete" if frozen else "awaiting_claim_approval",
        "next_action": "build figure contracts" if frozen else "review claim_registry.yaml",
        "gate_results": gates,
    }


def figure_evidence_contract(state: ManuscriptGraphState) -> dict[str, Any]:
    registry = _root(state) / "registries/figure_registry.yaml"
    redesign = _root(state) / "figure_redesign_request.yaml"
    present = registry.exists() and redesign.exists()
    return {
        "figure_registry_path": str(registry),
        "gate_results": _append_gate(
            state,
            "FIGURE_EVIDENCE_CONTRACT",
            "PASS" if present else "BLOCK",
            ["figure registry and redesign requests are present" if present else "figure contract artifact is missing"],
            "generate the Stage III approval package and review source gaps",
            [str(registry), str(redesign)],
        ),
    }


def results_by_figure_writer(state: ManuscriptGraphState) -> dict[str, Any]:
    package_path = Path(
        state.get(
            "figure_approval_package_path",
            _root(state) / "GATE_FIGURE_APPROVAL_PACKAGE",
        )
    )
    candidate_path = package_path / "paragraph_metadata.json"
    metadata_path = (
        candidate_path
        if candidate_path.exists()
        else _root(state) / "vertical_slice_stageIII/paragraph_metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else []
    results_package_path = state.get("results_package_path")
    if results_package_path and bool(state.get("figure_approved", False)):
        package = build_results_review_package(
            Path(state.get("project_root", Path.cwd())),
            Path(results_package_path),
            compile_pdf=bool(state.get("metadata", {}).get("compile_results_review_pdf", False)),
            scientifically_frozen=bool(
                state.get("results_claims_frozen", False)
                or (
                    state.get("results_human_approved", False)
                    and state.get("results_correction_approved", False)
                )
            ),
        )
        metadata = package["paragraph_metadata"]
        metadata_path = Path(results_package_path) / "paragraph_metadata.json"
    required = {"observation", "interpretation", "evidence_source", "figure_reference"}
    malformed = [item.get("paragraph_id", "unknown") for item in metadata if not required <= set(item)]
    unsupported = [flag for item in metadata for flag in item.get("unsupported_sentence_flags", [])]
    references = {ref for item in metadata for ref in item.get("figure_reference", [])}
    expected_references = {
        "FIG-01-PIPELINE",
        "FIG-02-THERMO",
        "FIG-03-STAGEIII-TOPOLOGY",
        "FIG-04-STAGEIV-3D-EXTENSION",
    }
    if results_package_path:
        expected_panel_prefixes = {"1", "2", "3", "4"}
        cited_prefixes = {
            panel[:1]
            for item in metadata
            for panel in item.get("figure_panels_cited", [])
            if panel[:1].isdigit()
        }
        complete_main_set = expected_panel_prefixes <= cited_prefixes
    else:
        complete_main_set = expected_references <= references if metadata_path == candidate_path else True
    trace_valid = (
        bool(metadata)
        and not unsupported
        and not malformed
        and complete_main_set
        and bool(state.get("claims_frozen"))
    )
    approved = bool(state.get("figure_approved", False))
    gate_status = "BLOCK" if not trace_valid else "PASS" if approved else "HUMAN_WAIT"
    findings = unsupported + [f"malformed:{item}" for item in malformed]
    if not complete_main_set:
        findings.append(f"missing_figure_references:{sorted(expected_references - references)}")
    if not findings:
        findings = ["Figures 1-4 captions and Results candidates are evidence-traceable"]
    return {
        "active_figure_id": "FIG-01-04-MAIN-TEXT-CONTRACTS",
        "paragraph_metadata": metadata,
        "results_claims_frozen": False,
        "figure_contracts_approved": trace_valid and approved,
        "results_package_path": str(results_package_path) if results_package_path else "",
        "section_drafts": {
            **state.get("section_drafts", {}),
            "results": package["results_markdown"] if results_package_path else "",
        },
        "workflow_status": "results_candidate_generated" if trace_valid and approved else "awaiting_figure_approval",
        "next_action": "human reviews Results and captions" if trace_valid and approved else "review the Figures 1-4 approval package",
        "gate_results": _append_gate(
            state,
            "GATE_FIGURE_APPROVAL",
            gate_status,
            findings,
            "human reviews the Figures 1-4 approval package",
            [str(metadata_path)],
        ),
    }


def results_human_review_gate(state: ManuscriptGraphState) -> dict[str, Any]:
    package_path = Path(state.get("results_package_path", ""))
    required = [
        "results_candidate.tex",
        "results_candidate.md",
        "figure_captions_candidate.tex",
        "results_sentence_evidence_trace.yaml",
        "figure_caption_evidence_trace.yaml",
        "results_claim_evidence_audit.md",
        "results_observation_interpretation_audit.md",
        "results_forbidden_claim_audit.md",
        "results_numerical_consistency_audit.md",
        "manuscript_conflict_ledger.yaml",
        "changed_files_manifest.yaml",
    ]
    if bool(state.get("metadata", {}).get("compile_results_review_pdf", False)):
        required.append("results_review_packet.pdf")
    missing = [name for name in required if not package_path or not (package_path / name).exists()]
    unsupported = [
        flag
        for item in state.get("paragraph_metadata", [])
        for flag in item.get("unsupported_sentence_flags", [])
    ]
    structurally_valid = bool(state.get("figure_contracts_approved")) and not missing and not unsupported
    approved = bool(state.get("results_human_approved", False))
    status = "BLOCK" if not structurally_valid else "PASS" if approved else "HUMAN_WAIT"
    findings = [f"missing:{name}" for name in missing] + [f"unsupported:{flag}" for flag in unsupported]
    if not findings:
        findings = ["Results, captions, traces, and automated audits are ready for scientific review"]
    return {
        "results_claims_frozen": structurally_valid and approved,
        "workflow_status": "results_scientific_freeze" if structurally_valid and approved else "awaiting_results_human_review",
        "next_action": "resume to Methods" if structurally_valid and approved else "human reviews Results, captions, and central caveats",
        "gate_results": _append_gate(
            state,
            "GATE_RESULTS_HUMAN_REVIEW",
            status,
            findings,
            "explicit human approval is required before Methods",
            [str(package_path / name) for name in required if package_path and (package_path / name).exists()],
        ),
    }


def methods_writer(state: ManuscriptGraphState) -> dict[str, Any]:
    allowed = bool(state.get("claims_frozen")) and bool(state.get("results_claims_frozen"))
    package_path = state.get("downstream_package_path")
    built = None
    if allowed and package_path:
        built = build_methods_package(Path(state.get("project_root", Path.cwd())), Path(package_path))
    return {
        "methods_draft_complete": bool(built),
        "section_drafts": {**state.get("section_drafts", {}), "methods": built["methods_tex"] if built else ""},
        "gate_results": _append_gate(
            state,
            "METHODS_WRITER_PREREQUISITES",
            "PASS" if allowed else "BLOCK",
            ["Methods are downstream of approved scope, claims, and figure-traced Results"],
            "approve scope, claims, and the vertical slice before drafting Methods",
        )
    }


def supplement_builder(state: ManuscriptGraphState) -> dict[str, Any]:
    contract = _root(state) / "contracts/supplement_structure.yaml"
    allowed = bool(state.get("results_claims_frozen")) and contract.exists()
    package_path = state.get("downstream_package_path")
    built = None
    if allowed and package_path:
        built = build_supplement_package(Path(state.get("project_root", Path.cwd())), Path(package_path))
    return {
        "supplement_contract_path": str(contract),
        "supplement_draft_complete": bool(built),
        "section_drafts": {**state.get("section_drafts", {}), "supplement": built["supplement_tex"] if built else ""},
        "gate_results": _append_gate(
            state,
            "SUPPLEMENT_BUILDER_PREREQUISITES",
            "PASS" if allowed else "BLOCK",
            ["S1-S7 structure is present" if contract.exists() else "Supplement structure is missing"],
            "populate only evidence-linked Supplement sections after approvals",
            [str(contract)],
        ),
    }


def discussion_scope_guard(state: ManuscriptGraphState) -> dict[str, Any]:
    allowed = bool(state.get("scope_approved")) and bool(state.get("results_claims_frozen"))
    package_path = state.get("downstream_package_path")
    built = None
    if allowed and package_path:
        built = build_discussion_package(Path(state.get("project_root", Path.cwd())), Path(package_path))
    return {
        "discussion_draft_complete": bool(built),
        "outlook_draft_complete": bool(built),
        "section_drafts": {
            **state.get("section_drafts", {}),
            "discussion": built["discussion_tex"] if built else "",
            "outlook": built["outlook_tex"] if built else "",
        },
        "gate_results": _append_gate(
            state,
            "DISCUSSION_SCOPE_GUARD",
            "PASS" if allowed else "BLOCK",
            ["Stage IV is main-text supported-with-caveat; Stage V is Discussion-only; Stage VI is future work"],
            "retain claim destinations from the registry",
        )
    }


def results_figure_consistency_gate(state: ManuscriptGraphState) -> dict[str, Any]:
    package_path = state.get("downstream_package_path")
    results_path = state.get("results_package_path")
    if not package_path or not results_path:
        # Backward-compatible contracts that do not request a downstream review
        # package cannot run the artwork/text audit and are not blocked here.
        audit = {"status": "NOT_REQUESTED", "proposal": "", "results_caption_consistent": True}
    else:
        audit = audit_fig4_caption_consistency(
            Path(state.get("project_root", Path.cwd())),
            Path(package_path),
            Path(results_path),
        )
    consistent = bool(audit.get("results_caption_consistent", False))
    status = "PASS" if consistent else "BLOCK"
    return {
        "results_correction_proposal_path": str(audit.get("proposal", "")),
        "workflow_status": "results_scientific_freeze" if consistent else "blocked_by_manuscript_figure_source_mismatch",
        "next_action": "continue to Introduction retrieval permission gate" if consistent else "restore the registered Figure Engine / strict-data sources",
        "gate_results": _append_gate(
            state,
            "GATE_MANUSCRIPT_FIGURE_SOURCE_REPAIR",
            status,
            ["manuscript figure source or Fig. 4 four-slice text does not match the registered scientific freeze"]
            if not consistent else
            ["Figures 1--4 resolve to the registered Figure Engine / strict-data freezes and Fig. 4 text matches its four slices"],
            "repair figure-source authority" if not consistent else "continue",
            [str(audit.get("proposal", ""))],
        ),
    }


def intro_citation_loop(state: ManuscriptGraphState) -> dict[str, Any]:
    allowed = (
        bool(state.get("scope_approved"))
        and bool(state.get("results_claims_frozen"))
        and bool(state.get("external_retrieval_approved"))
    )
    package_path = state.get("downstream_package_path")
    prereq = None
    compiled = None
    if bool(state.get("results_claims_frozen")) and package_path:
        project_root = Path(state.get("project_root", Path.cwd()))
        output_dir = Path(package_path)
        prereq = build_introduction_prerequisites(project_root, output_dir)
        if bool(state.get("metadata", {}).get("compile_downstream_review_pdfs", False)):
            compiled = compile_intermediate_review(project_root, output_dir, Path(state["results_package_path"]))
    findings = ["Results are frozen; Introduction retrieval still requires explicit external-retrieval approval"]
    if prereq:
        findings.append("claim needs, citation requirements, bibliography audit, and retrieval plan were prepared without external access")
    if compiled:
        findings.append("intermediate main-text and Supplemental Material review PDFs were compiled")
    return {
        "introduction_prerequisites_path": prereq["introduction_prerequisites_path"] if prereq else "",
        "workflow_status": "awaiting_introduction_retrieval_permission" if not allowed else "introduction_retrieval_authorized",
        "next_action": "human authorizes external Introduction retrieval" if not allowed else "run claim-linked Introduction retrieval",
        "gate_results": _append_gate(
            state,
            "INTRO_CITATION_LOOP_GATE",
            "PASS" if allowed else "HUMAN_WAIT",
            findings,
            "approve external retrieval separately after Results stabilization",
        )
    }


def abstract_title_generator(state: ManuscriptGraphState) -> dict[str, Any]:
    allowed = bool(state.get("scope_approved")) and bool(state.get("results_claims_frozen"))
    return {
        "gate_results": _append_gate(
            state,
            "ABSTRACT_TITLE_BOUNDARY",
            "RECHECK" if allowed else "BLOCK",
            ["title and abstract are generated last and remain human-controlled"],
            "generate candidates only after section approval",
        )
    }


def cross_section_consistency_audit(state: ManuscriptGraphState) -> dict[str, Any]:
    return {
        "gate_results": _append_gate(
            state,
            "CROSS_SECTION_CONSISTENCY_AUDIT",
            "RECHECK",
            ["full sections do not yet exist; Stage III consistency stub is available"],
            "rerun after sections are drafted",
        )
    }


def citation_audit(state: ManuscriptGraphState) -> dict[str, Any]:
    return {
        "gate_results": _append_gate(
            state,
            "CITATION_AUDIT",
            "BLOCK",
            ["live citation retrieval was not authorized or run in this task"],
            "run permission-gated retrieval after Results approval",
        )
    }


def prl_length_claim_gate(state: ManuscriptGraphState) -> dict[str, Any]:
    return {
        "gate_results": _append_gate(
            state,
            "PRL_LENGTH_CLAIM_GATE",
            "BLOCK",
            ["no complete manuscript exists to measure"],
            "run only after all approved sections exist",
        )
    }


def latex_integration(state: ManuscriptGraphState) -> dict[str, Any]:
    allowed = bool(state.get("claims_frozen")) and bool(state.get("figure_contracts_approved"))
    return {
        "gate_results": _append_gate(
            state,
            "LATEX_INTEGRATION_BOUNDARY",
            "RECHECK" if allowed else "BLOCK",
            ["production files remain absent; only the isolated Stage III fragment has been built"],
            "assemble an isolated candidate only after all approved sections and figures exist",
        )
    }


def final_human_approval(state: ManuscriptGraphState) -> dict[str, Any]:
    approved = bool(state.get("final_manuscript_approved", state.get("final_human_approved", False)))
    return {
        "workflow_status": "approved" if approved else "awaiting_human_approval",
        "next_action": "release the approved manuscript candidate" if approved else "review the isolated manuscript candidate and all audits",
        "gate_results": _append_gate(
            state,
            "GATE_FINAL_MANUSCRIPT_APPROVAL",
            "PASS" if approved else "HUMAN_WAIT",
            ["final scientific approval cannot be automated"],
            "human decision required",
        ),
    }
