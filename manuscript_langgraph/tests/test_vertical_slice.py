from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SLICE = ROOT / "vertical_slice_stageIII"


def test_paragraphs_are_evidence_traced():
    rows = json.loads((SLICE / "paragraph_metadata.json").read_text(encoding="utf-8"))
    assert len(rows) == 3
    for row in rows:
        assert row["observation"]
        assert row["interpretation"]
        assert row["evidence_source"]
        assert row["figure_reference"]
        assert row["claims_used"]
        assert row["evidence_used"]
        assert row["figure_panels_cited"]
        assert row["unsupported_sentence_flags"] == []


def test_sensitivity_caveat_is_in_results_caption_and_latex():
    for name in ["results_candidate.md", "caption_candidate.md", "integrated_fragment.tex"]:
        text = (SLICE / name).read_text(encoding="utf-8")
        assert "0.006269" in text


def test_latex_fragment_warns_against_unapproved_merge():
    text = (SLICE / "integrated_fragment.tex").read_text(encoding="utf-8")
    assert "DO NOT merge" in text
    assert "GATE_FIGURE_APPROVAL" in text
    assert "Approval-package placeholder only" in text


def test_vertical_slice_contract_uses_prompt_vocabulary():
    claim = json.loads((SLICE / "claim_contract.yaml").read_text(encoding="utf-8"))
    figure = json.loads((SLICE / "figure_contract.yaml").read_text(encoding="utf-8"))
    assert claim["scientific_status"] == "supported_with_caveat"
    assert claim["human_gate"] == "GATE_FIGURE_APPROVAL"
    assert figure["approval_gate"] == "GATE_FIGURE_APPROVAL"
    assert figure["approval_status"] == "pending_human_approval"
