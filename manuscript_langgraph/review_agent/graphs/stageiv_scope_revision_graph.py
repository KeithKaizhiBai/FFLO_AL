from __future__ import annotations

from review_agent.langgraph_compat import END, START, StateGraph
from review_agent.stageiv_scope_revision import (
    audit_revision,
    compile_revision,
    figure_scope_gate,
    prepare_revision_sources,
    revision_human_gate,
)
from review_agent.state_schema import ManuscriptGraphState


NODE_SEQUENCE = [
    ("figure_scope_gate", figure_scope_gate),
    ("prepare_revision_sources", prepare_revision_sources),
    ("compile_revision", compile_revision),
    ("audit_revision", audit_revision),
    ("revision_human_gate", revision_human_gate),
]


def build_graph():
    builder = StateGraph(ManuscriptGraphState)
    for name, node in NODE_SEQUENCE:
        builder.add_node(name, node)
    builder.add_edge(START, "figure_scope_gate")
    for (left, _), (right, _) in zip(NODE_SEQUENCE, NODE_SEQUENCE[1:]):
        builder.add_edge(left, right)
    builder.add_edge("revision_human_gate", END)
    return builder.compile()


graph = build_graph()
