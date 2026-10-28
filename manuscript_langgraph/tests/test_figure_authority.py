from __future__ import annotations

from pathlib import Path

from review_agent.figure_authority import (
    CANONICAL_MAIN_FIGURES,
    sha256,
    stage_canonical_main_figures,
    verify_canonical_frozen_artifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_canonical_main_figures_match_registered_freeze_hashes():
    rows = verify_canonical_frozen_artifacts(PROJECT_ROOT)
    assert len(rows) == 12
    assert all(row["status"] == "PASS" for row in rows)


def test_manuscript_staging_is_byte_identical_to_canonical_sources(tmp_path):
    audit = stage_canonical_main_figures(PROJECT_ROOT, tmp_path / "figures", tmp_path)
    assert audit["status"] == "PASS"
    assert audit["legacy_candidate_bundle_used"] is False
    for record in CANONICAL_MAIN_FIGURES.values():
        staged = tmp_path / "figures" / record["staged_pdf"]
        assert sha256(staged) == record["pdf_sha256"]
