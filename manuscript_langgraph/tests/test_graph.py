from __future__ import annotations

import json
from pathlib import Path

from review_agent.graphs.prl_manuscript_graph import NODE_SEQUENCE, graph
from review_agent.nodes import results_human_review_gate


ROOT = Path(__file__).resolve().parents[1]


def test_generation_order_is_evidence_first():
    names = [name for name, _ in NODE_SEQUENCE]
    assert names.index("results_by_figure_writer") < names.index("methods_writer")
    assert names.index("results_by_figure_writer") < names.index("results_human_review_gate")
    assert names.index("results_human_review_gate") < names.index("methods_writer")
    assert names.index("methods_writer") < names.index("supplement_builder")
    assert names.index("supplement_builder") < names.index("discussion_scope_guard")
    assert names.index("discussion_scope_guard") < names.index("intro_citation_loop")
    assert names.index("discussion_scope_guard") < names.index("results_figure_consistency_gate")
    assert names.index("results_figure_consistency_gate") < names.index("intro_citation_loop")
    assert names.index("intro_citation_loop") < names.index("abstract_title_generator")
    assert names.index("abstract_title_generator") < names.index("latex_integration")


def invoke(**approvals):
    state = {
        "manuscript_root": str(ROOT),
        "scope_approved": False,
        "claim_approved": False,
        "figure_approved": False,
        "results_human_approved": False,
        "external_retrieval_approved": False,
        "final_manuscript_approved": False,
        "gate_results": [],
    }
    state.update(approvals)
    return graph.invoke(state)


def test_unapproved_graph_stops_at_scope_gate():
    result = invoke()
    gates = {item["gate_id"]: item["status"] for item in result["gate_results"]}
    assert gates["GATE_SCOPE_APPROVAL"] == "HUMAN_WAIT"
    assert "GATE_CLAIM_APPROVAL" not in gates
    assert result["workflow_status"] == "awaiting_scope_approval"


def test_scope_approval_reaches_claim_gate_and_stops():
    result = invoke(scope_approved=True)
    gates = {item["gate_id"]: item["status"] for item in result["gate_results"]}
    assert gates["GATE_SCOPE_APPROVAL"] == "PASS"
    assert gates["GATE_CLAIM_APPROVAL"] == "HUMAN_WAIT"
    assert "GATE_FIGURE_APPROVAL" not in gates


def test_persisted_gate_records_stop_at_introduction_permission_gate():
    scope = json.loads((ROOT / "manuscript_scope.yaml").read_text(encoding="utf-8"))
    claims = json.loads((ROOT / "registries/claim_registry.yaml").read_text(encoding="utf-8"))
    figure = json.loads((ROOT / "runs/active_learning_prl_v1/gate_records/figure_approval_record.json").read_text(encoding="utf-8"))
    results = json.loads((ROOT / "runs/active_learning_prl_v1/gate_records/results_human_approval_record.json").read_text(encoding="utf-8"))
    repair = json.loads((ROOT / "runs/active_learning_prl_v1/gate_records/manuscript_figure_source_repair_approval.json").read_text(encoding="utf-8"))
    gates = json.loads((ROOT / "audits/gate_status.yaml").read_text(encoding="utf-8"))

    assert scope["GATE_SCOPE_APPROVAL"] == "approved"
    assert claims["approval_status"] == "approved_with_modifications"
    assert figure["decision"] == "approve_with_edits"
    assert results["decision"] == "approved"
    assert repair["decision"] == "approved_and_applied"
    assert gates["current_stop"] == "INTRO_CITATION_LOOP_GATE"
    assert gates["gates"]["GATE_FIGURE_APPROVAL"] == "approved_with_fig4_nonclosure_caveat"
    assert gates["gates"]["GATE_RESULTS_HUMAN_REVIEW"] == "approved_with_fig4_editorial_amendment"
    assert gates["gates"]["GATE_RESULTS_FIG4_CAPTION_CORRECTION"] == "superseded_by_framework_figure_source_repair"
    assert gates["gates"]["GATE_MANUSCRIPT_FIGURE_SOURCE_REPAIR"] == "approved_and_applied"
    assert gates["gates"]["INTRO_CITATION_LOOP_GATE"] == "permission_not_granted"


def test_claim_approval_reaches_figure_gate_and_stops():
    result = invoke(scope_approved=True, claim_approved=True)
    gates = {item["gate_id"]: item["status"] for item in result["gate_results"]}
    assert gates["GATE_CLAIM_APPROVAL"] == "PASS"
    assert gates["GATE_FIGURE_APPROVAL"] == "HUMAN_WAIT"
    assert result["workflow_status"] == "awaiting_figure_approval"


def test_figure_approval_without_results_package_is_blocked_before_methods():
    result = invoke(scope_approved=True, claim_approved=True, figure_approved=True)
    gates = {item["gate_id"]: item["status"] for item in result["gate_results"]}
    assert gates["GATE_FIGURE_APPROVAL"] == "PASS"
    assert gates["GATE_RESULTS_HUMAN_REVIEW"] == "BLOCK"
    assert "METHODS_WRITER_PREREQUISITES" not in gates


def test_results_package_reaches_human_wait(tmp_path):
    result = invoke(
        project_root=str(ROOT.parent),
        scope_approved=True,
        claim_approved=True,
        figure_approved=True,
        results_package_path=str(tmp_path / "results"),
    )
    gates = {item["gate_id"]: item["status"] for item in result["gate_results"]}
    assert gates["GATE_RESULTS_HUMAN_REVIEW"] == "HUMAN_WAIT"
    assert result["results_claims_frozen"] is False
    assert "METHODS_WRITER_PREREQUISITES" not in gates


def test_repeated_gate_evaluation_replaces_stale_record(tmp_path):
    package = tmp_path / "results"
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
    package.mkdir()
    for name in required:
        (package / name).write_text("fixture\n", encoding="utf-8")
    state = {
        "results_package_path": str(package),
        "figure_contracts_approved": True,
        "results_human_approved": False,
        "paragraph_metadata": [],
        "gate_results": [
            {
                "gate_id": "GATE_RESULTS_HUMAN_REVIEW",
                "status": "BLOCK",
                "evidence": [],
                "findings": ["stale"],
                "next_action": "stale",
            }
        ],
    }
    result = results_human_review_gate(state)
    matching = [gate for gate in result["gate_results"] if gate["gate_id"] == "GATE_RESULTS_HUMAN_REVIEW"]
    assert len(matching) == 1
    assert matching[0]["status"] == "HUMAN_WAIT"


def test_full_traversal_retains_final_human_gate(tmp_path):
    result = invoke(
        project_root=str(ROOT.parent),
        scope_approved=True,
        claim_approved=True,
        figure_approved=True,
        results_human_approved=True,
        results_package_path=str(tmp_path / "results"),
        external_retrieval_approved=True,
    )
    gates = {item["gate_id"]: item["status"] for item in result["gate_results"]}
    assert gates["GATE_FINAL_MANUSCRIPT_APPROVAL"] == "HUMAN_WAIT"
    assert result["workflow_status"] == "awaiting_human_approval"
