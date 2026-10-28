from __future__ import annotations

from review_agent.langgraph_compat import END, START, StateGraph
from review_agent.nodes import artifact_provenance_audit, claim_freeze, figure_evidence_contract, publication_readiness_gate
from review_agent.prl_cited_variant import (
    assemble_cited_candidate,
    audit_cited_candidate,
    build_verified_literature_bundle,
    citation_coverage_audit,
    citation_retrieval_authorization,
    cited_manuscript_human_review_gate,
    synthesize_physics_first_introduction,
)
from review_agent.state_schema import ManuscriptGraphState


NODE_SEQUENCE = [
    ("artifact_provenance_audit", artifact_provenance_audit),
    ("publication_readiness_gate", publication_readiness_gate),
    ("claim_freeze", claim_freeze),
    ("figure_evidence_contract", figure_evidence_contract),
    ("citation_retrieval_authorization", citation_retrieval_authorization),
    ("build_verified_literature_bundle", build_verified_literature_bundle),
    ("synthesize_physics_first_introduction", synthesize_physics_first_introduction),
    ("citation_coverage_audit", citation_coverage_audit),
    ("assemble_cited_candidate", assemble_cited_candidate),
    ("audit_cited_candidate", audit_cited_candidate),
    ("cited_manuscript_human_review_gate", cited_manuscript_human_review_gate),
]


def _after_claim_freeze(state: ManuscriptGraphState) -> str:
    return "continue" if state.get("claims_frozen") else "stop"


def _after_figure_contract(state: ManuscriptGraphState) -> str:
    gate = next((item for item in state.get("gate_results", []) if item.get("gate_id") == "FIGURE_EVIDENCE_CONTRACT"), {})
    return "continue" if gate.get("status") == "PASS" and state.get("figure_approved") else "stop"


def _after_retrieval_gate(state: ManuscriptGraphState) -> str:
    return "continue" if state.get("introduction_retrieval_authorized") else "stop"


def build_graph():
    builder = StateGraph(ManuscriptGraphState)
    for name, node in NODE_SEQUENCE:
        builder.add_node(name, node)
    builder.add_edge(START, "artifact_provenance_audit")
    builder.add_edge("artifact_provenance_audit", "publication_readiness_gate")
    builder.add_edge("publication_readiness_gate", "claim_freeze")
    builder.add_conditional_edges("claim_freeze", _after_claim_freeze, {"continue": "figure_evidence_contract", "stop": END})
    builder.add_conditional_edges("figure_evidence_contract", _after_figure_contract, {"continue": "citation_retrieval_authorization", "stop": END})
    builder.add_conditional_edges("citation_retrieval_authorization", _after_retrieval_gate, {"continue": "build_verified_literature_bundle", "stop": END})
    downstream = [
        "build_verified_literature_bundle",
        "synthesize_physics_first_introduction",
        "citation_coverage_audit",
        "assemble_cited_candidate",
        "audit_cited_candidate",
        "cited_manuscript_human_review_gate",
    ]
    for left, right in zip(downstream, downstream[1:]):
        builder.add_edge(left, right)
    builder.add_edge("cited_manuscript_human_review_gate", END)
    return builder.compile()


graph = build_graph()
