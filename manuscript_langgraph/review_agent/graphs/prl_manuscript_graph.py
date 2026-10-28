from __future__ import annotations

from review_agent.langgraph_compat import END, START, StateGraph
from review_agent.nodes import (
    abstract_title_generator,
    artifact_provenance_audit,
    citation_audit,
    claim_freeze,
    cross_section_consistency_audit,
    discussion_scope_guard,
    figure_evidence_contract,
    final_human_approval,
    intro_citation_loop,
    latex_integration,
    methods_writer,
    prl_length_claim_gate,
    publication_readiness_gate,
    results_by_figure_writer,
    results_human_review_gate,
    results_figure_consistency_gate,
    supplement_builder,
)
from review_agent.state_schema import ManuscriptGraphState


NODE_SEQUENCE = [
    ("artifact_provenance_audit", artifact_provenance_audit),
    ("publication_readiness_gate", publication_readiness_gate),
    ("claim_freeze", claim_freeze),
    ("figure_evidence_contract", figure_evidence_contract),
    ("results_by_figure_writer", results_by_figure_writer),
    ("results_human_review_gate", results_human_review_gate),
    ("methods_writer", methods_writer),
    ("supplement_builder", supplement_builder),
    ("discussion_scope_guard", discussion_scope_guard),
    ("results_figure_consistency_gate", results_figure_consistency_gate),
    ("intro_citation_loop", intro_citation_loop),
    ("abstract_title_generator", abstract_title_generator),
    ("cross_section_consistency_audit", cross_section_consistency_audit),
    ("citation_audit", citation_audit),
    ("prl_length_claim_gate", prl_length_claim_gate),
    ("latex_integration", latex_integration),
    ("final_human_approval", final_human_approval),
]


def _after_claim_freeze(state: ManuscriptGraphState) -> str:
    return "continue" if state.get("claims_frozen") else "stop"


def _after_results_writer(state: ManuscriptGraphState) -> str:
    return "continue" if state.get("figure_contracts_approved") else "stop"


def _after_results_human_review(state: ManuscriptGraphState) -> str:
    return "continue" if state.get("results_claims_frozen") else "stop"


def _after_results_figure_consistency(state: ManuscriptGraphState) -> str:
    gate = next(
        (item for item in state.get("gate_results", []) if item.get("gate_id") == "GATE_MANUSCRIPT_FIGURE_SOURCE_REPAIR"),
        {},
    )
    return "continue" if gate.get("status") == "PASS" else "stop"


def _after_intro_citation_loop(state: ManuscriptGraphState) -> str:
    return "continue" if state.get("external_retrieval_approved") else "stop"


def build_graph(*, checkpointer=None, interrupt_after=None):
    builder = StateGraph(ManuscriptGraphState)
    for name, function in NODE_SEQUENCE:
        builder.add_node(name, function)

    builder.add_edge(START, "artifact_provenance_audit")
    builder.add_edge("artifact_provenance_audit", "publication_readiness_gate")
    builder.add_edge("publication_readiness_gate", "claim_freeze")
    builder.add_conditional_edges(
        "claim_freeze",
        _after_claim_freeze,
        {"continue": "figure_evidence_contract", "stop": END},
    )
    builder.add_edge("figure_evidence_contract", "results_by_figure_writer")
    builder.add_conditional_edges(
        "results_by_figure_writer",
        _after_results_writer,
        {"continue": "results_human_review_gate", "stop": END},
    )
    builder.add_conditional_edges(
        "results_human_review_gate",
        _after_results_human_review,
        {"continue": "methods_writer", "stop": END},
    )

    builder.add_edge("methods_writer", "supplement_builder")
    builder.add_edge("supplement_builder", "discussion_scope_guard")
    builder.add_edge("discussion_scope_guard", "results_figure_consistency_gate")
    builder.add_conditional_edges(
        "results_figure_consistency_gate",
        _after_results_figure_consistency,
        {"continue": "intro_citation_loop", "stop": END},
    )
    builder.add_conditional_edges(
        "intro_citation_loop",
        _after_intro_citation_loop,
        {"continue": "abstract_title_generator", "stop": END},
    )

    downstream = [
        "abstract_title_generator",
        "cross_section_consistency_audit",
        "citation_audit",
        "prl_length_claim_gate",
        "latex_integration",
        "final_human_approval",
    ]
    for left, right in zip(downstream, downstream[1:]):
        builder.add_edge(left, right)
    builder.add_edge("final_human_approval", END)
    compile_kwargs = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    if interrupt_after is not None:
        compile_kwargs["interrupt_after"] = interrupt_after
    return builder.compile(**compile_kwargs)


graph = build_graph()
