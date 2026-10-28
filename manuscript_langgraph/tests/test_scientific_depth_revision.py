from pathlib import Path

from review_agent.graphs.scientific_depth_revision_graph import NODE_SEQUENCE
from review_agent.scientific_depth_revision import PARENT_RUN_ID, PROFILE_ID, RUN_ID, TITLE


ROOT = Path(__file__).resolve().parents[1]


def test_scientific_depth_branch_identity_and_neutral_title():
    assert RUN_ID == "active_learning_prl_scientific_depth_v5"
    assert PARENT_RUN_ID == "active_learning_public_reproducible_v4"
    assert PROFILE_ID == "PUBLIC_PHYSICS_NARRATIVE_WITH_SCIENTIFIC_DEPTH"
    assert all(word.lower() not in TITLE.lower() for word in ["prl", "physical review", "nature", "science"])


def test_scientific_depth_graph_is_minimal_downstream_extension():
    assert [name for name, _ in NODE_SEQUENCE] == [
        "repository_discovery",
        "scientific_consistency_audits",
        "install_depth_layer",
        "build_interpretation_packets",
        "synthesize_sources",
        "compile_layout_candidates",
        "audit_revision",
        "assemble_review_package",
    ]


def test_parent_sources_and_frozen_figures_exist():
    parent = ROOT.parent / "output/pdf/manuscript_public_reproducible_v4"
    assert (parent / "revised_main_text.tex").is_file()
    assert (parent / "revised_supplement.tex").is_file()
    for path in [
        "figures/fig1_engine_v3/fig1_engine_v3.pdf",
        "figures/fig2_strict_data/fig2_strict_data.pdf",
        "figures/fig3_strict_data/fig3_strict_data.pdf",
        "figures/fig4_strict_data_v2/fig4_strict_data_v2.pdf",
    ]:
        assert (ROOT.parent / path).is_file()


def test_read_only_descriptive_statistics_builder_exists():
    script = ROOT / "scripts/build_scientific_depth_descriptive_stats.py"
    assert script.is_file()
    text = script.read_text(encoding="utf-8")
    assert "np.save" not in text
    assert "to_csv" in text
    assert "descriptive_statistics_manifest.json" in text
