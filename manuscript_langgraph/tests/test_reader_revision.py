"""Document-workflow guards only; no physics calculations or reconstruction."""
import hashlib
import json
from pathlib import Path

from review_agent import reader_revision
from review_agent.graphs.single_column_placement_graph import build_graph

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / reader_revision.OUTPUT


def test_reader_revision_is_separate_and_builds_existing_graph():
    assert reader_revision.PARENT != reader_revision.OUTPUT
    assert build_graph("reader_revision_v8") is not None


def test_protected_review_inputs_and_frozen_artwork_unchanged():
    manifest = json.loads((OUT / "input_manifest.json").read_text())
    for path, expected in manifest["hashes"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path


def test_author_choices_not_automatically_approved():
    state = json.loads((OUT / "build_state.json").read_text())
    assert state["gate_status"] == "HUMAN_WAIT"
    assert state["author_dependent_items"] == ["R01", "R02", "R03 Ansatz/basis", "R11", "R12", "R13"]
    assert state["physics_changed"] is False


def test_method_corrections_and_atomic_algorithm_units():
    text = (OUT / "source/sections/supplement.tex").read_text()
    assert "cooldown masks are disabled" in text
    assert "quantiles of the penalized" in text
    assert text.count(r"\refstepcounter{algorithm}") == 3
    assert text.count(r"\noindent\begin{minipage}{\textwidth}") == 3
    assert r"\partial_\tau" in text


def test_new_artwork_linted_but_still_waits_for_human_review():
    folders = [ROOT / "figures/fig2_reader_revision_v8"]
    folders += list((ROOT / "figures/supplementary_publication_methods_v8").glob("S*"))
    assert len(folders) == 3
    for folder in folders:
        report = json.loads((folder / "figure_validation_report.json").read_text())
        assert report["status"] == "pass"
        assert report["hard_failure_count"] == 0
        assert report["gate"]["state"] == "HUMAN_WAIT"
