from __future__ import annotations

from review_agent.langgraph_compat import END, START, StateGraph
from review_agent.scientific_depth_revision import (
    assemble_review_package,
    audit_revision,
    build_interpretation_packets,
    compile_layout_candidates,
    install_depth_layer,
    repository_discovery,
    scientific_consistency_audits,
    synthesize_sources,
)
from review_agent.state_schema import ManuscriptGraphState


NODE_SEQUENCE = [
    ("repository_discovery", repository_discovery),
    ("scientific_consistency_audits", scientific_consistency_audits),
    ("install_depth_layer", install_depth_layer),
    ("build_interpretation_packets", build_interpretation_packets),
    ("synthesize_sources", synthesize_sources),
    ("compile_layout_candidates", compile_layout_candidates),
    ("audit_revision", audit_revision),
    ("assemble_review_package", assemble_review_package),
]


def build_graph():
    builder = StateGraph(ManuscriptGraphState)
    for name, node in NODE_SEQUENCE:
        builder.add_node(name, node)
    builder.add_edge(START, NODE_SEQUENCE[0][0])
    for (left, _), (right, _) in zip(NODE_SEQUENCE, NODE_SEQUENCE[1:]):
        builder.add_edge(left, right)
    builder.add_edge(NODE_SEQUENCE[-1][0], END)
    return builder.compile()


graph = build_graph()
