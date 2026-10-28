from __future__ import annotations

import json
from pathlib import Path

from review_agent.figure_authority import CANONICAL_MAIN_FIGURES, sha256


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRIAL_ROOT = PROJECT_ROOT / "manuscript_variants/prl_integrated_singlecol_trial"


def test_single_column_trial_preserves_frozen_figures():
    for record in CANONICAL_MAIN_FIGURES.values():
        assert sha256(PROJECT_ROOT / record["source_pdf"]) == record["pdf_sha256"]


def test_single_column_trial_structure_when_present():
    body_path = TRIAL_ROOT / "sections/integrated_results_methods.tex"
    if not body_path.exists():
        return
    body = body_path.read_text(encoding="utf-8")
    assert body.count(r"\begin{figure*}[t]") == 1
    assert body.count(r"\begin{figure}[t]") == 3
    assert body.count(r"width=\columnwidth") == 3
    assert "0.006269" in body
    assert "not all closed" in body


def test_single_column_trial_is_non_destructive_when_present():
    manifest_path = TRIAL_ROOT / "single_column_trial_manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["frozen_figures_modified"] is False
    assert manifest["captions_modified"] is False
    assert manifest["scientific_text_modified"] is False
    assert manifest["original_candidate_modified"] is False
    assert manifest["gate_effect"] == "none"
    assert manifest["layout"] == {
        "fig1": "double_column",
        "fig2": "single_column",
        "fig3": "single_column",
        "fig4": "single_column",
    }


def test_single_column_trial_does_not_approve_manuscript_gate():
    gate_path = PROJECT_ROOT / "manuscript_langgraph/runs/active_learning_prl_integrated_v1/gate_records/prl_integrated_manuscript_human_review_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert gate["status"] == "HUMAN_WAIT"
    assert gate["scientific_freeze"] is False
    assert gate["publication_freeze"] is False
