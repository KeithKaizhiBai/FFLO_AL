from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "manuscript_langgraph"
RUN_ROOT = PACKAGE_ROOT / "runs/active_learning_prl_v1"
RESULTS_ROOT = PROJECT_ROOT / "output/pdf/manuscript_results_review_v1"
DOWNSTREAM_ROOT = PROJECT_ROOT / "output/pdf/manuscript_methods_sm_discussion_review_v1"
sys.path.insert(0, str(PACKAGE_ROOT))

from review_agent.figure_authority import CANONICAL_MAIN_FIGURES, sha256  # noqa: E402
from review_agent.graphs.prl_manuscript_graph import graph  # noqa: E402


def main() -> None:
    result = graph.invoke({
        "project_root": str(PROJECT_ROOT),
        "manuscript_root": str(PACKAGE_ROOT),
        "figure_approval_package_path": str(PACKAGE_ROOT / "GATE_FIGURE_APPROVAL_PACKAGE"),
        "results_package_path": str(RESULTS_ROOT),
        "downstream_package_path": str(DOWNSTREAM_ROOT),
        "scope_approved": True,
        "claim_approved": True,
        "figure_approved": True,
        "results_human_approved": True,
        # Kept for state-schema compatibility; the operative gate is now the
        # manuscript-wide canonical figure-source audit.
        "results_correction_approved": True,
        "external_retrieval_approved": False,
        "final_manuscript_approved": False,
        "gate_results": [],
        "metadata": {
            "compile_results_review_pdf": True,
            "compile_downstream_review_pdfs": True,
            "new_physics_permission": False,
            "external_retrieval_approved": False,
            "figure_source_authority": "Figure Engine / strict-data scientific freeze",
        },
    })
    gates = {item["gate_id"]: item["status"] for item in result["gate_results"]}
    if gates.get("GATE_MANUSCRIPT_FIGURE_SOURCE_REPAIR") != "PASS":
        raise SystemExit("canonical manuscript figure-source gate did not pass")
    if gates.get("INTRO_CITATION_LOOP_GATE") != "HUMAN_WAIT":
        raise SystemExit("workflow did not stop at the external Introduction retrieval permission gate")
    forbidden = {"ABSTRACT_TITLE_BOUNDARY", "LATEX_INTEGRATION_BOUNDARY", "GATE_FINAL_MANUSCRIPT_APPROVAL"}
    if forbidden & gates.keys():
        raise SystemExit("workflow crossed the external Introduction retrieval permission gate")

    staged = {}
    for figure_id, record in CANONICAL_MAIN_FIGURES.items():
        path = DOWNSTREAM_ROOT / "figures" / record["staged_pdf"]
        actual = sha256(path)
        if actual != record["pdf_sha256"]:
            raise SystemExit(f"staged manuscript figure hash mismatch: {figure_id}")
        staged[figure_id] = {
            "source": record["source_pdf"],
            "staged": str(path.relative_to(PROJECT_ROOT)),
            "sha256": actual,
        }

    record_path = RUN_ROOT / "gate_records/manuscript_figure_source_repair_gate.json"
    record_path.write_text(json.dumps({
        "schema_version": 1,
        "gate_id": "GATE_MANUSCRIPT_FIGURE_SOURCE_REPAIR",
        "status": "PASS",
        "decision": "approved_and_applied",
        "evaluated_at": str(date.today()),
        "staged_figures": staged,
        "results_pdf": {
            "path": str((RESULTS_ROOT / "results_review_packet.pdf").relative_to(PROJECT_ROOT)),
            "sha256": sha256(RESULTS_ROOT / "results_review_packet.pdf"),
        },
        "main_pdf": {
            "path": str((DOWNSTREAM_ROOT / "manuscript_intermediate_review.pdf").relative_to(PROJECT_ROOT)),
            "sha256": sha256(DOWNSTREAM_ROOT / "manuscript_intermediate_review.pdf"),
        },
        "supplement_pdf": {
            "path": str((DOWNSTREAM_ROOT / "supplement_review.pdf").relative_to(PROJECT_ROOT)),
            "sha256": sha256(DOWNSTREAM_ROOT / "supplement_review.pdf"),
        },
        "external_retrieval_executed": False,
        "physics_recomputed": False,
    }, indent=2) + "\n", encoding="utf-8")
    (RUN_ROOT / "introduction_gate_state.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (RUN_ROOT / "logs/framework_figure_rewrite.log").write_text(
        "LANGGRAPH_FRAMEWORK_FIGURE_REWRITE_PASS\n"
        "current_stop=INTRO_CITATION_LOOP_GATE\n"
        "figures_1_4=canonical_scientific_freezes\n"
        "external_retrieval=false\n"
        "new_physics=false\n",
        encoding="utf-8",
    )
    print("LANGGRAPH_FRAMEWORK_FIGURE_REWRITE_PASS")
    print(f"results_pdf={RESULTS_ROOT / 'results_review_packet.pdf'}")
    print(f"main_pdf={DOWNSTREAM_ROOT / 'manuscript_intermediate_review.pdf'}")
    print(f"supplement_pdf={DOWNSTREAM_ROOT / 'supplement_review.pdf'}")


if __name__ == "__main__":
    main()
