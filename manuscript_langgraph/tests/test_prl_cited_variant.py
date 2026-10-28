from __future__ import annotations

import json
import re
from pathlib import Path

from review_agent.figure_authority import CANONICAL_MAIN_FIGURES, sha256
from review_agent.graphs.prl_integrated_cited_graph import NODE_SEQUENCE


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VARIANT_ROOT = PROJECT_ROOT / "manuscript_variants/prl_integrated_cited_v2"


def test_cited_graph_contains_real_retrieval_synthesis_and_audit_nodes():
    names = [name for name, _ in NODE_SEQUENCE]
    assert "citation_retrieval_authorization" in names
    assert "build_verified_literature_bundle" in names
    assert "synthesize_physics_first_introduction" in names
    assert "citation_coverage_audit" in names
    assert names.index("citation_coverage_audit") < names.index("assemble_cited_candidate")


def test_cited_profile_is_non_destructive_and_retrieval_approved():
    profile = json.loads((PROJECT_ROOT / "manuscript_langgraph/publication_profiles/prl_integrated_cited_main_text_v2.yaml").read_text(encoding="utf-8"))
    assert profile["parent_run_id"] == "active_learning_prl_integrated_cited_v1"
    assert profile["literature_retrieval_authorized"] is True
    assert profile["frozen_figures_modified"] is False
    assert profile["citation_policy"]["target_reference_count"] == 50
    assert profile["citation_policy"]["uncited_bibliography_padding_forbidden"] is True


def test_cited_introduction_has_background_and_complete_bib_when_present():
    intro_path = VARIANT_ROOT / "sections/introduction.tex"
    bib_path = VARIANT_ROOT / "references.bib"
    if not intro_path.exists() or not bib_path.exists():
        return
    intro = intro_path.read_text(encoding="utf-8")
    bib = bib_path.read_text(encoding="utf-8")
    cited = {key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", intro) for key in group.split(",")}
    defined = set(re.findall(r"@article\{([^,]+),", bib))
    assert "The computational problem considered here" not in intro
    assert intro.index("Superconductivity with nonzero") < intro.index("Here we develop")
    assert cited == defined
    assert 45 <= len(cited) <= 55
    assert len(cited) == 50


def test_expanded_registry_is_unique_tiered_and_fully_mapped_when_present():
    registry_path = VARIANT_ROOT / "verified_literature_registry.yaml"
    matrix_path = VARIANT_ROOT / "introduction_claim_to_citation_matrix.yaml"
    if not registry_path.exists() or not matrix_path.exists():
        return
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    sources = registry["sources"]
    keys = [item["key"] for item in sources]
    dois = [item["doi"].lower() for item in sources if item["doi"]]
    matrix_keys = {key for claim in matrix["claims"] for key in claim["keys"]}
    assert len(sources) == registry["target_reference_count"] == 50
    assert len(keys) == len(set(keys))
    assert len(dois) == len(set(dois))
    assert set(keys) == matrix_keys
    assert sum(value for key, value in registry["verification_tier_counts"].items() if key.startswith("T1")) == 12
    assert registry["verification_tier_counts"]["T2_crossref_doi_metadata"] == 38


def test_cited_variant_preserves_frozen_physics_when_present():
    body_path = VARIANT_ROOT / "sections/integrated_results_methods.tex"
    if not body_path.exists():
        return
    body = body_path.read_text(encoding="utf-8")
    assert "0.006269" in body
    assert "not all closed" in body
    assert "not a complete three-dimensional atlas" in body
    for record in CANONICAL_MAIN_FIGURES.values():
        assert sha256(PROJECT_ROOT / record["source_pdf"]) == record["pdf_sha256"]


def test_cited_variant_stops_at_human_gate_when_present():
    gate_path = PROJECT_ROOT / "manuscript_langgraph/runs/active_learning_prl_integrated_cited_v2/gate_records/cited_introduction_manuscript_human_review_gate.json"
    if not gate_path.exists():
        return
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert gate["status"] == "HUMAN_WAIT"
    assert gate["retrieval_executed"] is True
    assert gate["scientific_freeze"] is False
    assert gate["publication_freeze"] is False
    assert gate["frozen_results_modified"] is False
    assert gate["frozen_figures_modified"] is False
