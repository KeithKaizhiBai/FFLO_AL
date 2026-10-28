from __future__ import annotations

from review_agent.langgraph_compat import END, START, StateGraph
from review_agent.single_column_placement import audit_and_gate, build_sources, compile_and_select, repository_discovery
from review_agent.state_schema import ManuscriptGraphState


NODE_SEQUENCE = [
    ("repository_discovery", repository_discovery),
    ("build_sources", build_sources),
    ("compile_and_select", compile_and_select),
    ("audit_and_gate", audit_and_gate),
]


def build_graph(profile="legacy_v6"):
    nodes = NODE_SEQUENCE
    if profile == "synchronized_v7":
        from review_agent import synchronized_review as paired
        nodes = [(name, getattr(paired, name)) for name, _ in NODE_SEQUENCE]
    elif profile == "reader_revision_v8":
        from review_agent import reader_revision as paired
        nodes = [(name, getattr(paired, name)) for name, _ in NODE_SEQUENCE]
    elif profile == "authorship_v9":
        from review_agent import authorship_revision as paired
        nodes = [(name, getattr(paired, name)) for name, _ in NODE_SEQUENCE]
    elif profile != "legacy_v6":
        raise ValueError(f"Unknown manuscript profile: {profile}")
    builder = StateGraph(ManuscriptGraphState)
    for name, node in nodes:
        builder.add_node(name, node)
    builder.add_edge(START, nodes[0][0])
    for (left, _), (right, _) in zip(nodes, nodes[1:]):
        builder.add_edge(left, right)
    builder.add_edge(nodes[-1][0], END)
    return builder.compile()


graph = build_graph()
