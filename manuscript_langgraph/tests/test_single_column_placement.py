from review_agent.graphs.single_column_placement_graph import NODE_SEQUENCE
from review_agent.single_column_placement import GATE_ID, PARENT_RUN_ID, RUN_ID


def test_single_column_continuation_identity():
    assert RUN_ID == "active_learning_prl_single_column_v6"
    assert PARENT_RUN_ID == "active_learning_prl_scientific_depth_v5"
    assert GATE_ID == "GATE_SINGLE_COLUMN_FIGURE_PLACEMENT_HUMAN_REVIEW"


def test_single_column_graph_is_minimal_layout_continuation():
    assert [name for name, _ in NODE_SEQUENCE] == [
        "repository_discovery",
        "build_sources",
        "compile_and_select",
        "audit_and_gate",
    ]
