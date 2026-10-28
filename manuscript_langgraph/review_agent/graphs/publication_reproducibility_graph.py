from __future__ import annotations

from review_agent.langgraph_compat import END, START, StateGraph
from review_agent.publication_reproducibility import (
    assemble_review_package,
    build_supplement_figures,
    compile_and_audit,
    ingest_evidence,
    install_publication_layer,
    repository_discovery,
    synthesize_sources,
)
from review_agent.state_schema import ManuscriptGraphState


NODE_SEQUENCE = [
    ("repository_discovery", repository_discovery),
    ("install_publication_layer", install_publication_layer),
    ("ingest_evidence", ingest_evidence),
    ("build_supplement_figures", build_supplement_figures),
    ("synthesize_sources", synthesize_sources),
    ("compile_and_audit", compile_and_audit),
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

