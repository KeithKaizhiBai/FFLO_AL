"""Identity-only revision guards; no physical calculations."""
import json
import csv
from pathlib import Path
from pypdf import PdfReader
from review_agent import authorship_revision as revision
from review_agent.synchronized_review import sha

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / revision.OUTPUT


def test_identity_in_all_three_pdfs():
    for job in revision.JOBS:
        pdf = PdfReader(OUT / (job + ".pdf"))
        text = " ".join(" ".join(p.extract_text() for p in pdf.pages).split())
        assert all(value in text for value in (revision.AUTHOR, revision.EMAIL, revision.AFFILIATION))


def test_science_and_v8_files_unchanged():
    manifest = json.loads((OUT / "input_manifest.json").read_text())
    assert all(sha(ROOT / p) == digest for p, digest in manifest["parent_hashes"].items())
    for p in (ROOT / revision.PARENT / "source/sections").glob("*.tex"):
        assert revision.update_availability(p.read_text(), p.name) == (OUT / "source/sections" / p.name).read_text()


def test_identity_approval_does_not_approve_science():
    state = json.loads((OUT / "build_state.json").read_text())
    assert state["author_identity_approved"] is True
    assert state["gate_status"] == "HUMAN_WAIT"
    assert state["physics_changed"] is False


def test_current_placement_table_matches_current_pdf():
    checks = json.loads((OUT / "source/built/validation.json").read_text())
    with (OUT / "tables/figure_placements.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == sum(len(data["embedded_forms"]) for data in checks.values())
    for row in rows:
        form = checks[row["document"]]["embedded_forms"][int(row["figure"])-1]
        assert int(row["page"]) == form["page"]
        assert float(row["width_mm"]) == form["rendered_width_mm"]
