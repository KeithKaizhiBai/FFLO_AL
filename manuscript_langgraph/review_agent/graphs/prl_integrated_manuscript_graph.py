from __future__ import annotations

from review_agent.langgraph_compat import END, START, StateGraph
from review_agent.nodes import (
    artifact_provenance_audit,
    claim_freeze,
    figure_evidence_contract,
    publication_readiness_gate,
)
from review_agent.prl_integrated_variant import (
    assemble_prl_integrated_candidate,
    audit_prl_length_budget,
    audit_semantic_equivalence,
    build_method_packets,
    integrated_human_review_gate,
    plan_inline_figure_placement,
    select_publication_profile,
    weave_results_and_methods,
)
from review_agent.state_schema import ManuscriptGraphState


NODE_SEQUENCE = [
    ("artifact_provenance_audit", artifact_provenance_audit),
    ("publication_readiness_gate", publication_readiness_gate),
    ("claim_freeze", claim_freeze),
    ("figure_evidence_contract", figure_evidence_contract),
    ("select_publication_profile", select_publication_profile),
    ("build_method_packets", build_method_packets),
    ("weave_results_and_methods", weave_results_and_methods),
    ("plan_inline_figure_placement", plan_inline_figure_placement),
    ("assemble_prl_integrated_candidate", assemble_prl_integrated_candidate),
    ("audit_semantic_equivalence", audit_semantic_equivalence),
    ("audit_prl_length_budget", audit_prl_length_budget),
    ("integrated_human_review_gate", integrated_human_review_gate),
]


def _after_claim_freeze(state: ManuscriptGraphState) -> str:
    return "continue" if state.get("claims_frozen") else "stop"


def _after_figure_contract(state: ManuscriptGraphState) -> str:
    contract = next(
        (gate for gate in state.get("gate_results", []) if gate.get("gate_id") == "FIGURE_EVIDENCE_CONTRACT"),
        {},
    )
    return "continue" if contract.get("status") == "PASS" and state.get("figure_approved") else "stop"


def build_graph():
    builder = StateGraph(ManuscriptGraphState)
    for name, node in NODE_SEQUENCE:
        builder.add_node(name, node)
    builder.add_edge(START, "artifact_provenance_audit")
    builder.add_edge("artifact_provenance_audit", "publication_readiness_gate")
    builder.add_edge("publication_readiness_gate", "claim_freeze")
    builder.add_conditional_edges("claim_freeze", _after_claim_freeze, {"continue": "figure_evidence_contract", "stop": END})
    builder.add_conditional_edges("figure_evidence_contract", _after_figure_contract, {"continue": "select_publication_profile", "stop": END})
    downstream = [
        "select_publication_profile",
        "build_method_packets",
        "weave_results_and_methods",
        "plan_inline_figure_placement",
        "assemble_prl_integrated_candidate",
        "audit_semantic_equivalence",
        "audit_prl_length_budget",
        "integrated_human_review_gate",
    ]
    for left, right in zip(downstream, downstream[1:]):
        builder.add_edge(left, right)
    builder.add_edge("integrated_human_review_gate", END)
    return builder.compile()


graph = build_graph()
