from __future__ import annotations

import json
from pathlib import Path

from review_agent.figure_authority import CANONICAL_MAIN_FIGURES, sha256
from review_agent.graphs.prl_integrated_manuscript_graph import NODE_SEQUENCE


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_integrated_graph_is_a_profile_extension():
    names = [name for name, _ in NODE_SEQUENCE]
    assert names[:4] == [
        "artifact_provenance_audit",
        "publication_readiness_gate",
        "claim_freeze",
        "figure_evidence_contract",
    ]
    assert "select_publication_profile" in names
    assert "weave_results_and_methods" in names
    assert "audit_semantic_equivalence" in names


def test_profile_keeps_frozen_authority_and_new_run():
    profile = json.loads((PROJECT_ROOT / "manuscript_langgraph/publication_profiles/prl_integrated_main_text.yaml").read_text(encoding="utf-8"))
    assert profile["run_id"] == "active_learning_prl_integrated_v1"
    assert profile["parent_run_id"] == "active_learning_prl_v1"
    assert profile["standalone_methods_section"] is False
    assert profile["standalone_outlook_section"] is False
    assert profile["inline_figures"] is True
    assert profile["external_retrieval_authorized"] is False


def test_canonical_figures_are_unchanged():
    for record in CANONICAL_MAIN_FIGURES.values():
        assert sha256(PROJECT_ROOT / record["source_pdf"]) == record["pdf_sha256"]


def test_built_integrated_candidate_has_required_structure_when_present():
    main = PROJECT_ROOT / "manuscript_variants/prl_integrated/main.tex"
    if not main.exists():
        return
    source = main.read_text(encoding="utf-8")
    body = (main.parent / "sections/integrated_results_methods.tex").read_text(encoding="utf-8")
    discussion = (main.parent / "sections/discussion.tex").read_text(encoding="utf-8")
    assert "\\section{Results}" not in source + body
    assert "\\section{Methods}" not in source + body
    assert "\\section{Outlook}" not in source + discussion
    assert body.count("\\begin{figure*}[t]") == 4
    assert "0.006269" in body
    assert "not all closed" in body


def test_built_integrated_candidate_stops_at_human_wait_when_present():
    gate_path = PROJECT_ROOT / "manuscript_langgraph/runs/active_learning_prl_integrated_v1/gate_records/prl_integrated_manuscript_human_review_gate.json"
    if not gate_path.exists():
        return
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert gate["status"] == "HUMAN_WAIT"
    assert gate["scientific_freeze"] is False
    assert gate["publication_freeze"] is False
