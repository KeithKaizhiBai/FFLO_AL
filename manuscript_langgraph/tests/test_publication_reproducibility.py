from pathlib import Path

from review_agent.graphs.publication_reproducibility_graph import NODE_SEQUENCE
from review_agent.publication_reproducibility import PARENT_RUN_ID, RUN_ID, TITLE


ROOT = Path(__file__).resolve().parents[1]


def test_publication_branch_has_parent_and_neutral_title():
    assert RUN_ID == "active_learning_public_reproducible_v4"
    assert PARENT_RUN_ID == "active_learning_manuscript_stageiv_revision_v3"
    assert all(word.lower() not in TITLE.lower() for word in ["prl", "physical review", "nature", "science"])


def test_publication_graph_is_downstream_extension():
    names = [name for name, _ in NODE_SEQUENCE]
    assert names == ["repository_discovery", "install_publication_layer", "ingest_evidence", "build_supplement_figures", "synthesize_sources", "compile_and_audit", "assemble_review_package"]


def test_parent_and_frozen_figures_exist():
    assert (ROOT.parent / "manuscript_variants/integrated_cited_stageiv_revision_v3/main.tex").is_file()
    for path in ["figures/fig1_engine_v3/fig1_engine_v3.pdf", "figures/fig2_strict_data/fig2_strict_data.pdf", "figures/fig3_strict_data/fig3_strict_data.pdf", "figures/fig4_strict_data_v2/fig4_strict_data_v2.pdf"]:
        assert (ROOT.parent / path).is_file()
