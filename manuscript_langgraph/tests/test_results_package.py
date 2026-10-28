from __future__ import annotations

import yaml

from review_agent.results_package import build_results_review_package


def test_graph_results_package_is_evidence_traced_and_caveated(tmp_path):
    # Resolve the actual repository root from the test module location.
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    output = tmp_path / "results"
    package = build_results_review_package(project_root, output, compile_pdf=False)

    assert package["audit_status"] == "PASS"
    assert (output / "results_candidate.tex").exists()
    assert (output / "figure_captions_candidate.tex").exists()
    results = (output / "results_candidate.tex").read_text(encoding="utf-8")
    captions = (output / "figure_captions_candidate.tex").read_text(encoding="utf-8")
    assert "0.006269" in results
    assert "not all closed" in results
    assert "continuous topology surface" in results
    assert "0.006269" in captions
    assert "[-0.5,1.5]" in captions
    assert "Not scientifically frozen" in (output / "results_candidate.md").read_text(encoding="utf-8")
    review = (output / "results_review_packet.tex").read_text(encoding="utf-8")
    assert "figures/fig4_strict_data/fig4_strict_data.pdf" in review
    assert "manuscript_figures/final/Fig4_candidate.pdf" not in review
    assert "Four explicit narrow fixed-$\\mu$ bins" in results
    assert "$\\mu/t=-0.45,0.15,0.85,$ and $1.45$" in captions

    trace = yaml.safe_load((output / "results_sentence_evidence_trace.yaml").read_text(encoding="utf-8"))
    observations = [item for item in trace["sentences"] if item["sentence_role"] == "OBSERVATION"]
    interpretations = [item for item in trace["sentences"] if item["sentence_role"] == "INTERPRETATION"]
    assert observations and all(item["evidence_ids"] for item in observations)
    assert interpretations and all(item["depends_on_observations"] for item in interpretations)


def test_results_package_freeze_status_is_explicit(tmp_path):
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    output = tmp_path / "frozen-results"
    build_results_review_package(
        project_root,
        output,
        compile_pdf=False,
        scientifically_frozen=True,
    )

    results = (output / "results_candidate.md").read_text(encoding="utf-8")
    review = (output / "results_review_packet.tex").read_text(encoding="utf-8")
    assert "Scientifically frozen after the approved Figure Engine source repair" in results
    assert "INTRO\\_CITATION\\_LOOP\\_GATE = HUMAN\\_WAIT" in review
